#!/usr/bin/env python3
"""
Chess Reinforcement Learning using Ray RLlib.
This implementation uses RLlib's built-in distributed training capabilities with ConnectorV2 for reward adjustment.
"""
# Standard library imports
import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Third-party imports
import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.connector_pipeline_v2 import ConnectorPipelineV2
from ray.rllib.connectors.learner.add_one_ts_to_episodes_and_truncate import AddOneTsToEpisodesAndTruncate
from ray.rllib.connectors.common.add_observations_from_episodes_to_batch import AddObservationsFromEpisodesToBatch
from ray.rllib.connectors.learner.add_next_observations_from_episodes_to_train_batch import AddNextObservationsFromEpisodesToTrainBatch
from ray.rllib.connectors.learner.general_advantage_estimation import GeneralAdvantageEstimation
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import PolicyID
from ray.tune.utils import merge_dicts
from ray.rllib.core.columns import Columns

# Local imports
from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper
import gc
import os
import platform
import tracemalloc
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import gymnasium as gym

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import (
    OldAPIStack,
    override,
    OverrideToImplementCustomLogic,
    PublicAPI,
)
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import AgentID, EnvType, EpisodeType, PolicyID
from ray.tune.callback import _CallbackMeta

# Framework-specific imports
torch, nn = try_import_torch()

# Custom ConnectorV2 for reward shaping
class ChessRewardShapingConnector(ConnectorV2):
    """Shapes rewards based on player perspective for chess environment."""
    
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        metrics: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> dict:
        """
        Shape rewards according to player perspective.
        - White's rewards stay as-is
        - Black's rewards are flipped (multiplied by -1)
        """
        if not batch or "rewards" not in batch:
            return batch
        
        rewards = batch["rewards"]
        if len(rewards) == 0:
            return batch
            
        # Create a mask to identify which steps belong to black (odd indices)
        # Since we're not directly tracking player turns in the batch,
        # we'll use a heuristic based on step indices within episodes
        shaped_rewards = rewards.copy()
        
        # Process each episode
        for episode in episodes:
            # Get the episode's rewards
            episode_rewards = episode.rewards
            
            # Shape rewards based on player turns (even indices for white, odd for black)
            for i in range(len(episode_rewards)):
                player = i % 2  # 0 for White, 1 for Black
                
                # For black's rewards, flip the sign for non-terminal states
                if player == 1 and i < len(episode_rewards) - 1:
                    episode_rewards[i] = -episode_rewards[i]
            
            # Shape terminal rewards if the episode is done
            if episode.length > 0 and (episode.terminated[-1] or episode.truncated[-1]):
                last_idx = episode.length - 1
                info = episode.infos[last_idx] if episode.infos else {}
                
                if "outcome" in info:
                    outcome = info["outcome"]
                    last_player = last_idx % 2  # 0 for White, 1 for Black
                    
                    if outcome == "white_win":
                        episode_rewards[last_idx] = 1.0 if last_player == 0 else -1.0
                        if last_idx > 0:
                            # Previous player lost
                            episode_rewards[last_idx - 1] = -1.0 if last_player == 0 else 1.0
                    elif outcome == "black_win":
                        episode_rewards[last_idx] = 1.0 if last_player == 1 else -1.0
                        if last_idx > 0:
                            # Previous player lost
                            episode_rewards[last_idx - 1] = -1.0 if last_player == 1 else 1.0
                    elif outcome == "draw":
                        episode_rewards[last_idx] = 0.0
                        if last_idx > 0:
                            episode_rewards[last_idx - 1] = 0.0
        
        # Update the batch with shaped rewards from episodes
        batch["rewards"] = shaped_rewards
        return batch

# Define the DeepMind-style learner connector pipeline with GAE
def build_deepmind_learner_connector(
    input_observation_space=None,
    input_action_space=None,
    device=None,
):
    """
    Build a DeepMind-style learner connector pipeline.
    
    This includes:
    1. Reward shaping for chess (white/black perspectives)
    2. Adding one timestep to episodes for proper GAE
    3. Adding observations from episodes to the batch
    4. Adding next observations for computing TD errors
    5. Generalized Advantage Estimation (GAE)
    """
    pipeline = ConnectorPipelineV2(
        description="DeepMind-style connector pipeline with GAE",
        input_observation_space=input_observation_space,
        input_action_space=input_action_space,
    )
    
    # First, shape rewards based on player perspective
    pipeline.append(ChessRewardShapingConnector())
    
    # Then, add one timestep to episodes for proper GAE calculation
    pipeline.append(AddOneTsToEpisodesAndTruncate())
    
    # Next, add observations to the batch
    pipeline.append(AddObservationsFromEpisodesToBatch())
    
    # Add next observations for TD learning
    pipeline.append(AddNextObservationsFromEpisodesToTrainBatch())
    
    # Finally, compute GAE advantages (gamma and lambda set in training config)
    pipeline.append(GeneralAdvantageEstimation(
        gamma=0.99,       # Discount factor
        lambda_=0.95,     # GAE lambda parameter
    ))
    
    return pipeline

# Metrics-only callback
class ChessCombinedCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        # TODO (sven): Deprecate these args.
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        metrics = {
            "white_win": 0.0,
            "black_win": 0.0,
            "draw": 0.0,
            "checkmate": 0.0,
            "stalemate": 0.0,
        }
        infos = episode.get_infos()
        info = infos[-1] if infos else {}
        if "outcome" in info:
            outcome = info["outcome"]
            if outcome == "white_win":
                metrics["white_win"] = 1.0
            elif outcome == "black_win":
                metrics["black_win"] = 1.0
            elif outcome == "draw":
                metrics["draw"] = 1.0
        if "termination_reason" in info:
            reason = info["termination_reason"]
            if reason == "checkmate":
                metrics["checkmate"] = 1.0
            elif reason == "stalemate":
                metrics["stalemate"] = 1.0
        for key, value in metrics.items():
            episode.custom_metrics[key] = value

class ChessMaskingRLModule(TorchRLModule):
    def __init__(self, observation_space=None, action_space=None, model_config=None, inference_only=False, catalog_class=None, **kwargs):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config or {},
            inference_only=inference_only,
            catalog_class=catalog_class,
            **kwargs
        )
        
        self.board_shape = observation_space["board"].shape
        self.board_channels = self.board_shape[0]
        self.action_space_n = action_space.n

        self.features_extractor = nn.Sequential(
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
        
        self.policy_head = nn.Linear(832, self.action_space_n)
        self.value_head = nn.Linear(832, 1)
        
    def _forward(self, batch, **kwargs):
        obs = batch.get("obs", batch)
        board = obs["board"]
        action_mask = obs["action_mask"]
        
        if len(board.shape) == 3:
            board = board.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
            
        batch_size = board.shape[0]
        features = self.features_extractor(board) + 1e-8
        action_logits = self.policy_head(features)
        
        # Apply action masking
        action_mask = action_mask.to(action_logits.device)
        masked_logits = action_logits + (action_mask - 1) * 1e9  # Large negative for invalid actions
        value = self.value_head(features).view(batch_size, 1)
        
        return {
            Columns.ACTIONS: torch.distributions.Categorical(logits=masked_logits).sample(),
            Columns.ACTION_DIST_INPUTS: masked_logits,
            Columns.ACTION_LOGP: torch.distributions.Categorical(logits=masked_logits).log_prob(
                torch.distributions.Categorical(logits=masked_logits).sample()
            ),
            Columns.VF_PREDS: value
        }

def create_rllib_chess_env(config):
    env = ChessEnv()
    env = ActionMaskWrapper(env)
    
    class DictObsWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Dict({
                "board": gym.spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32),
                "action_mask": gym.spaces.Box(low=0, high=1, shape=(env.action_space.n,), dtype=np.float32),
                "white_to_move": gym.spaces.Discrete(2)
            })
            self.action_space = env.action_space
            self.steps = 0
            print(f"DictObsWrapper initialized with action space: {self.action_space}")

        def reset(self, **kwargs):
            self.steps = 0
            print(f"Environment reset")
            obs, info = self.env.reset(**kwargs)
            wrapped_obs = self._wrap_observation(obs)
            print(f"Reset obs: {type(wrapped_obs)}, info: {info}")
            return wrapped_obs, info

        def step(self, action):
            self.steps += 1
            print(f"Step {self.steps}, action: {action}")
            obs, reward, terminated, truncated, info = self.env.step(action)
            print(f"Raw step result - reward: {reward}, terminated: {terminated}, info: {info}")
            
            # Shape the reward based on player perspective
            player = (self.steps - 1) % 2  # 0 for White, 1 for Black
            shaped_reward = reward
            
            # For intermediate rewards: White as-is, Black flipped
            if player == 1 and not terminated:  # Black's turn, non-terminal
                shaped_reward = -reward
                
            # For terminal rewards
            if terminated and "outcome" in info:
                outcome = info["outcome"]
                last_player = (self.steps - 1) % 2  # 0 for White, 1 for Black
                
                if outcome == "white_win":
                    shaped_reward = 1.0 if last_player == 0 else -1.0
                elif outcome == "black_win":
                    shaped_reward = 1.0 if last_player == 1 else -1.0
                elif outcome == "draw":
                    shaped_reward = 0.0
            
            print(f"Shaped reward: {shaped_reward} (original: {reward})")
            wrapped_obs = self._wrap_observation(obs)
            print(f"Wrapped obs type: {type(wrapped_obs)}")
            return wrapped_obs, shaped_reward, terminated, truncated, info

        def _wrap_observation(self, obs):
            if not isinstance(obs, dict) or "board" not in obs or "action_mask" not in obs:
                print(f"Warning: Invalid observation format: {type(obs)}")
                return {
                    "board": np.zeros((13, 8, 8), dtype=np.float32),
                    "action_mask": np.ones(self.action_space.n, dtype=np.float32),
                    "white_to_move": 1
                }
            return {
                "board": np.asarray(obs["board"], dtype=np.float32),
                "action_mask": np.asarray(obs["action_mask"], dtype=np.float32),
                "white_to_move": int(obs.get("white_to_move", 1))
            }

    return DictObsWrapper(env)

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

    observation_space = gym.spaces.Dict({
        "board": gym.spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32),
        "action_mask": gym.spaces.Box(low=0, high=1, shape=(20480,), dtype=np.float32),
        "white_to_move": gym.spaces.Discrete(2)
    })
    action_space = gym.spaces.Discrete(20480)

    rl_module_spec = RLModuleSpec(
        module_class=ChessMaskingRLModule,
        observation_space=observation_space,
        action_space=action_space,
        model_config={},
    )

    config = (
        PPOConfig()
        .environment("chess_env")
        .framework("torch")
        .resources(
            num_gpus=driver_gpus,
            num_cpus_for_main_process=8,
        )
        .learners(
            num_learners=num_learners,
            num_gpus_per_learner=num_gpus_per_learner,
            connector_fn=build_deepmind_learner_connector,  # Use the DeepMind-style learner connector
        )
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_cpus_per_env_runner=num_cpus_per_env_runner,
            num_gpus_per_env_runner=num_gpus_per_env_runner,
            sample_timeout_s=None,
        )
        .training(
            train_batch_size_per_learner=4096,
            minibatch_size=256,
            num_epochs=10,
            lr=5e-5,
            grad_clip=1.0,
            gamma=0.99,           # Discount factor for GAE
            use_gae=True,         # Enable GAE
            lambda_=0.95,         # GAE lambda parameter
            vf_loss_coeff=0.5,    # Value function loss coefficient
            entropy_coeff=args.entropy_coeff,
            clip_param=0.2,       # PPO clipping parameter
            kl_coeff=0.2,         # KL divergence coefficient
            vf_share_layers=False,
        )
        .callbacks(ChessCombinedCallback)
        .rl_module(rl_module_spec=rl_module_spec)
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
    )

    print(f"Training with {num_learners} learners, {num_env_runners} env runners")

    analysis = tune.run(
        "PPO",  # Use default PPO since CustomPPO was for old API debugging
        stop={"training_iteration": args.max_iterations},
        checkpoint_freq=25,
        checkpoint_at_end=True,
        storage_path=checkpoint_dir,
        verbose=3,
        config=config.to_dict(),  # Convert to dict for tune.run
        resume="AUTO",
        restore=args.checkpoint if args.checkpoint and os.path.exists(args.checkpoint) else None,
    )

    return analysis.best_checkpoint

def evaluate(args):
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print(f"Error: Valid checkpoint path required for evaluation, got: {args.checkpoint}")
        return

    ray.init(ignore_reinit_error=True, include_dashboard=args.dashboard)
    tune.register_env("chess_env", create_rllib_chess_env)

    observation_space = gym.spaces.Dict({
        "board": gym.spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32),
        "action_mask": gym.spaces.Box(low=0, high=1, shape=(20480,), dtype=np.float32),
        "white_to_move": gym.spaces.Discrete(2)
    })
    action_space = gym.spaces.Discrete(20480)

    rl_module_spec = RLModuleSpec(
        module_class=ChessMaskingRLModule,
        observation_space=observation_space,
        action_space=action_space,
    )

    config = (
        PPOConfig()
        .environment("chess_env")
        .framework("torch")
        .resources(num_gpus=1 if args.device == "cuda" else 0)
        .env_runners(num_env_runners=0, num_cpus_per_env_runner=1)
        .rl_module(rl_module_spec=rl_module_spec)
        .exploration(explore=False)
        .training(lambda_=0.95, num_epochs=0)
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
    )

    algo = PPO(config=config)
    algo.restore(args.checkpoint)

    env = create_rllib_chess_env({})
    num_episodes = 50
    total_rewards = []
    outcomes = {"white_win": 0, "black_win": 0, "draw": 0, "unknown": 0}

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = truncated = False
        episode_reward = 0

        while not (done or truncated):
            action = algo.compute_single_action(obs, explore=False)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            if args.render:
                env.render()

        outcomes[info.get("outcome", "unknown")] += 1
        total_rewards.append(episode_reward)

    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Outcomes: {outcomes}")
    env.close()
    ray.shutdown()

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
    else:
        evaluate(args)

if __name__ == "__main__":
    main()