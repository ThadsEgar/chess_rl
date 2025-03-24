#!/usr/bin/env python3
"""
Chess Reinforcement Learning using Ray RLlib.
This implementation uses RLlib's built-in distributed training capabilities with environment-based reward shaping.
"""
# Standard library imports
import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple

# Third-party imports
import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import PolicyID
from ray.rllib.utils.annotations import override
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical


# Local imports
from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper
import gc
import os
# Framework-specific imports
torch, nn = try_import_torch()

# Constant for masking invalid actions
MASK_VALUE = -1e9  # Large negative value to mask out invalid actions

# Define agent IDs for white and black players (global constants)
WHITE = "white"
BLACK = "black"

# Metrics-only callback
class MultiAgentDebugCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        # Print training metrics
        print(f"\n=== MULTI-AGENT TRAINING DEBUG ===")
        print(f"Training iteration: {result['training_iteration']}")
        
        # Check for policy-specific metrics
        if "policy_reward_mean" in result:
            for policy_id, reward in result["policy_reward_mean"].items():
                print(f"Policy {policy_id} - Mean reward: {reward:.4f}")
        
        # Check for episode-level metrics
        if "episode_len_mean" in result:
            print(f"Mean episode length: {result['episode_len_mean']:.2f}")
        
        # Print advantage info if available
        if "info" in result and "learner" in result["info"]:
            for policy_id, stats in result["info"]["learner"].items():
                if "stats" in stats and "vf_explained_var" in stats["stats"]:
                    print(f"Policy {policy_id} - Value function explained variance: {stats['stats']['vf_explained_var']:.4f}")
                    print(f"This indicates how well advantages are being calculated (closer to 1.0 is better)")
        
        print(f"=== END MULTI-AGENT DEBUG ===\n")
    
    def on_episode_end(self, *, episode, **kwargs):
        # Extract episode info from the last step
        rewards = {agent_id: episode.agent_rewards.get(agent_id, 0) for agent_id in episode.get_agents()}
        print(f"Episode {episode.episode_id} ended with rewards: {rewards}")
        
        # Check for specific outcomes
        if len(episode.get_last_infos()) > 0:
            for agent_id, info in episode.get_last_infos().items():
                if "agent_outcome" in info:
                    print(f"Agent {agent_id} outcome: {info['agent_outcome']}")

class ChessCombinedCallback(MultiAgentDebugCallback):
    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeV2, "EpisodeType"],
        env_runner: Optional[Any] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        **kwargs,
    ) -> None:
        # Call parent method for debug info
        super().on_episode_end(episode=episode, **kwargs)
        
        # Extract episode info from the last step
        agent_infos = episode.get_last_infos()
        
        # Track metrics for each agent
        for agent_id, info in agent_infos.items():
            # Skip non-agent keys
            if agent_id not in [WHITE, BLACK]:
                continue
                
            # Calculate metrics based on agent outcome
            win = 1.0 if info.get("agent_outcome") == "win" else 0.0
            loss = 1.0 if info.get("agent_outcome") == "loss" else 0.0
            draw = 1.0 if info.get("agent_outcome") == "draw" else 0.0
            
            # Use the metrics_logger to properly record these metrics with a sliding window
            if metrics_logger is not None:
                # Each metric is logged with a sliding window of 100 episodes
                metrics_logger.log_value(f"{agent_id}_win", win, window=100)
                metrics_logger.log_value(f"{agent_id}_loss", loss, window=100)
                metrics_logger.log_value(f"{agent_id}_draw", draw, window=100)
                
                # Track game stats
                if "termination_reason" in info:
                    checkmate = 1.0 if info["termination_reason"] == "checkmate" else 0.0
                    stalemate = 1.0 if info["termination_reason"] == "stalemate" else 0.0
                    metrics_logger.log_value("checkmate", checkmate, window=100)
                    metrics_logger.log_value("stalemate", stalemate, window=100)

class ChessMaskingPPOModule(PPOTorchRLModule):
    """RLModule that extends PPOTorchRLModule and implements action masking for chess environment.
    
    This extends the default PPO torch module to handle action masking for the chess environment.
    By inheriting from PPOTorchRLModule, we get proper advantage calculation while adding our masking logic.
    """
    
    @override(PPOTorchRLModule)
    def get_train_action_dist_cls(self):
        return TorchCategorical

    @override(PPOTorchRLModule)
    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    @override(PPOTorchRLModule)
    def get_inference_action_dist_cls(self):
        return TorchCategorical
    
    @override(PPOTorchRLModule)
    def setup(self):
        """Initialize neural network modules."""
        # Extract observation space information
        self.board_shape = self.observation_space["board"].shape
        self.board_channels = self.board_shape[0]
        self.action_space_n = self.action_space.n

        # Create feature extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(self.board_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 832),
            nn.ReLU(),
        )
        
        # Policy and value function heads
        self.pi = nn.Linear(832, self.action_space_n)
        self.vf = nn.Linear(832, 1)
    
    def _mask_action_dist_inputs(self, outputs, action_mask):
        """Apply action masking to action distribution inputs."""
        if Columns.ACTION_DIST_INPUTS in outputs and action_mask is not None:
            # Get logits and apply action masking
            logits = outputs[Columns.ACTION_DIST_INPUTS]
            # Make sure mask is on the same device
            action_mask = action_mask.to(logits.device)
            # Adjust shape if necessary
            if len(action_mask.shape) == 1 and len(logits.shape) > 1:
                action_mask = action_mask.unsqueeze(0)
            # Apply the mask
            outputs[Columns.ACTION_DIST_INPUTS] = logits + (action_mask - 1) * MASK_VALUE
        return outputs

    @override(PPOTorchRLModule)
    def _forward_inference(self, batch, **kwargs):
        # Get observations
        obs = batch.get(Columns.OBS, batch)
        # Extract action mask
        action_mask = obs.get("action_mask", None)
        # Call parent class forward method
        outputs = super()._forward_inference(batch, **kwargs)
        # Apply masking
        return self._mask_action_dist_inputs(outputs, action_mask)

    @override(PPOTorchRLModule)
    def _forward_exploration(self, batch, **kwargs):
        # Get observations
        obs = batch.get(Columns.OBS, batch)
        # Extract action mask
        action_mask = obs.get("action_mask", None)
        # Call parent class forward method
        outputs = super()._forward_exploration(batch, **kwargs)
        # Apply masking
        return self._mask_action_dist_inputs(outputs, action_mask)

    @override(PPOTorchRLModule)
    def _forward_train(self, batch, **kwargs):
        # Get observations
        obs = batch.get(Columns.OBS, batch)
        # Extract action mask if available in this format
        action_mask = obs.get("action_mask", None) if isinstance(obs, dict) else None
        # Call parent class forward method
        outputs = super()._forward_train(batch, **kwargs)
        # Apply masking if we have a mask
        if action_mask is not None:
            return self._mask_action_dist_inputs(outputs, action_mask)
        return outputs
    
    @override(PPOTorchRLModule)
    def _forward(self, batch, **kwargs):
        """Custom forward method to process board observations."""
        # Extract board from observations
        obs = batch.get(Columns.OBS, batch)
        board = obs["board"]
        
        # Handle single observations vs batches
        if len(board.shape) == 3:
            board = board.unsqueeze(0)
            
        # Extract features using encoder
        features = self.encoder(board)
        
        # Create outputs dict
        outputs = {}
        
        # Policy head outputs
        outputs[Columns.ACTION_DIST_INPUTS] = self.pi(features)
        
        # If action mask is available, apply it
        if "action_mask" in obs:
            action_mask = obs["action_mask"]
            if len(action_mask.shape) == 1:
                action_mask = action_mask.unsqueeze(0)
            action_mask = action_mask.to(features.device)
            # Apply masking
            outputs[Columns.ACTION_DIST_INPUTS] = outputs[Columns.ACTION_DIST_INPUTS] + (action_mask - 1) * MASK_VALUE
        
        return outputs
    
    @override(PPOTorchRLModule)
    def compute_values(self, batch, embeddings=None):
        """Compute value function predictions."""
        # Extract observations
        obs = batch.get(Columns.OBS, batch)
        
        # Handle different observation formats
        if isinstance(obs, dict) and "board" in obs:
            board = obs["board"]
        else:
            board = obs
            
        # Ensure proper batch dimension
        if len(board.shape) == 3:
            board = board.unsqueeze(0)
            
        # Extract features
        features = self.encoder(board)
        
        # Compute and return value predictions
        values = self.vf(features).squeeze(-1)
        return values

def create_rllib_chess_env(config):
    env = ChessEnv()
    env = ActionMaskWrapper(env)
    
    # Use the global WHITE and BLACK constants
    
    class ChessMultiAgentWrapper(gym.Wrapper):
        """MultiAgent wrapper for chess environment.
        
        This wrapper converts a standard chess environment into a multi-agent environment
        where white and black are separate agents taking turns.
        """
        def __init__(self, env):
            super().__init__(env)
            # Define observation spaces for each agent
            self.observation_space = gym.spaces.Dict({
                "board": gym.spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32),
                "action_mask": gym.spaces.Box(low=0, high=1, shape=(env.action_space.n,), dtype=np.float32),
                "white_to_move": gym.spaces.Discrete(2)
            })
            self.action_space = env.action_space
            self.steps = 0
            self.current_agent = WHITE  # Start with white
            print(f"ChessMultiAgentWrapper initialized with action space: {self.action_space}")
            
            # Define agent IDs
            self.agents = [WHITE, BLACK]
            self.possible_agents = self.agents.copy()

        def reset(self, **kwargs):
            self.steps = 0
            self.current_agent = WHITE  # Reset to white's turn
            print(f"Environment reset - Current agent: {self.current_agent}")
            obs, info = self.env.reset(**kwargs)
            wrapped_obs = self._wrap_observation(obs)
            
            # Create a multi-agent format dictionary with observations only for the current agent
            multi_agent_obs = {
                self.current_agent: wrapped_obs
            }
            multi_agent_info = {
                self.current_agent: info
            }
            
            return multi_agent_obs, multi_agent_info

        def step(self, action_dict):
            # Extract action for the current agent
            action = action_dict[self.current_agent]
            self.steps += 1
            
            # Take a step in the environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Terminal-only rewards with player perspective flipping
            white_reward = 0.0
            black_reward = 0.0
            
            # Only assign rewards at termination
            if terminated and "outcome" in info:
                outcome = info["outcome"]
                
                if outcome == "white_win":
                    white_reward = 1.0
                    black_reward = -1.0
                elif outcome == "black_win":
                    white_reward = -1.0
                    black_reward = 1.0
                elif outcome == "draw":
                    white_reward = 0.0
                    black_reward = 0.0
            
            # Switch the current agent
            next_agent = BLACK if self.current_agent == WHITE else WHITE
            self.current_agent = next_agent
            
            # Wrap observation for the next agent if the game is not over
            wrapped_obs = self._wrap_observation(obs)
            
            # Create multi-agent formatted outputs
            # Only include the next agent in observations if the game is not terminated
            multi_agent_obs = {}
            if not terminated and not truncated:
                multi_agent_obs[self.current_agent] = wrapped_obs
            
            # Assign rewards to both agents
            multi_agent_reward = {
                WHITE: white_reward,
                BLACK: black_reward
            }
            
            # Create termination and truncation dicts
            multi_agent_terminated = {
                WHITE: terminated,
                BLACK: terminated,
                "__all__": terminated  # Required by RLlib
            }
            
            multi_agent_truncated = {
                WHITE: truncated,
                BLACK: truncated,
                "__all__": truncated  # Required by RLlib
            }
            
            # Create info dict for both agents
            multi_agent_info = {
                WHITE: info.copy(),
                BLACK: info.copy()
            }
            
            # Add agent-specific information
            if "outcome" in info:
                multi_agent_info[WHITE]["agent_outcome"] = "win" if info["outcome"] == "white_win" else "loss" if info["outcome"] == "black_win" else "draw"
                multi_agent_info[BLACK]["agent_outcome"] = "win" if info["outcome"] == "black_win" else "loss" if info["outcome"] == "white_win" else "draw"
            
            return multi_agent_obs, multi_agent_reward, multi_agent_terminated, multi_agent_truncated, multi_agent_info

        def _wrap_observation(self, obs):
            if not isinstance(obs, dict) or "board" not in obs or "action_mask" not in obs:
                #print(f"Warning: Invalid observation format: {type(obs)}")
                raise ValueError(f"Invalid observation format: {type(obs)}")
            return {
                "board": np.asarray(obs["board"], dtype=np.float32),
                "action_mask": np.asarray(obs["action_mask"], dtype=np.float32),
                "white_to_move": int(obs.get("white_to_move", 1))
            }

    return ChessMultiAgentWrapper(env)

def train(args):
    if args.device == "cuda" and not args.force_cpu:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPUs")
    else:
        gpu_count = 0

    ray.init(
        address=args.head_address if args.distributed else None,
        ignore_reinit_error=True,
        include_dashboard=args.dashboard,
        _redis_password=args.redis_password,
        num_cpus=118,
        num_gpus=gpu_count,
    )

    # Resource configuration
    num_env_runners = 1
    driver_gpus = 0.5
    num_cpus_per_env_runner = 4
    num_gpus_per_env_runner = 0.5
    num_envs_per_env_runner = 4
    num_learners = 3
    num_gpus_per_learner = 1

    # Register the environment
    tune.register_env("chess_env", create_rllib_chess_env)

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the observation and action spaces for the agents
    observation_space = gym.spaces.Dict({
        "board": gym.spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32),
        "action_mask": gym.spaces.Box(low=0, high=1, shape=(20480,), dtype=np.float32),
        "white_to_move": gym.spaces.Discrete(2)
    })
    action_space = gym.spaces.Discrete(20480)

    # Create an RLModuleSpec instance for our masked PPO module
    module_spec = RLModuleSpec(
        module_class=ChessMaskingPPOModule,
        observation_space=observation_space,
        action_space=action_space,
        model_config={},
    )

    # Define the policy mapping function for multi-agent
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # We'll use the same policy for both agents (self-play)
        return "shared_policy"

    # Configure PPO with Ray 2.43.0 API conventions and multi-agent support
    config = (
        PPOConfig()
        .environment("chess_env")
        .framework("torch")
        # Resources configuration
        .resources(
            num_gpus=driver_gpus,
            num_cpus_for_main_process=8,
        )
        # Learner configuration
        .learners(
            num_learners=num_learners,
            num_gpus_per_learner=num_gpus_per_learner,
        )
        # Environment runners configuration
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_cpus_per_env_runner=num_cpus_per_env_runner,
            num_gpus_per_env_runner=num_gpus_per_env_runner,
            rollout_fragment_length="auto",
            batch_mode="complete_episodes",  # Critical for correct advantage calculation
            preprocessor_pref=None,
            sample_timeout_s=300,  # Increase timeout for complex chess environment
        )
        # Training configuration
        .training(
            train_batch_size_per_learner=4096,
            minibatch_size=256,
            num_epochs=1,  # In Ray 2.43.0, this replaces num_epochs
            lr=5e-5,
            grad_clip=1.0,
            gamma=1.0,
            use_gae=True,  # Explicitly enable GAE
            lambda_=0.95,
            vf_loss_coeff=0.5,
            entropy_coeff=args.entropy_coeff,
            clip_param=0.2,
            kl_coeff=0.2,
            vf_share_layers=False,
        )
        # Callback configuration
        .callbacks(ChessCombinedCallback)
        # Multi-agent configuration
        .multi_agent(
            policies={
                "shared_policy": (
                    None,  # Policy class (None = use default)
                    observation_space,
                    action_space,
                    {},  # Configuration dict
                    module_spec,  # RLModuleSpec for this policy
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"],
        )
    )

    # Convert config to dict and add explicit settings for advantage calculation
    config_dict = config.to_dict()
    
    # Ensure these parameters are explicitly set for GAE calculation
    config_dict.update({
        "batch_mode": "complete_episodes",
        "use_gae": True,
        "lambda": 0.95,
        "gamma": 1.0,
        "_enable_rl_module_api": True,
        "sample_async": False,  # Synchronous sampling for more stability
    })

    print(f"Training with {num_learners} learners, {num_env_runners} env runners")
    print(f"Multi-agent configuration: {config_dict.get('multiagent', {})}")

    # Using the newer Tuner API instead of the older tune.run
    tuner = tune.Tuner(
        PPO,
        run_config=tune.RunConfig(
            stop={"training_iteration": args.max_iterations},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=25,
                checkpoint_at_end=True,
            ),
            storage_path=checkpoint_dir,
            verbose=3,
        ),
        param_space=config_dict,  # Use dict instead of config object
    )
    results = tuner.fit()
    
    # Get the best checkpoint from the results
    return results.get_best_result().checkpoint

def main():
    parser = argparse.ArgumentParser(description="Chess RL training using RLlib")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--head_address", type=str, default=None)
    parser.add_argument("--redis_password", type=str, default=None)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--checkpoint_dir", type=str, default="rllib_checkpoints")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--entropy_coeff", type=float, default=0.05)
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
if __name__ == "__main__":
    main()