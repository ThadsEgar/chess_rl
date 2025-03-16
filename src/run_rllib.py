#!/usr/bin/env python3
"""
Chess Reinforcement Learning using Ray RLlib.
This implementation uses RLlib's built-in distributed training capabilities with ConnectorV2 for reward adjustment.
"""
# Standard library imports
import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Union

# Third-party imports
import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.connectors.connector_v2 import ConnectorV2
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

# Framework-specific imports
torch, nn = try_import_torch()

# Custom ConnectorV2 for reward adjustment
class ChessRewardConnector(ConnectorV2):
    def __call__(
        self,
        *,
        rl_module: "RLModule",
        data: dict,
        episodes: list["EpisodeType"],
        **kwargs
    ) -> dict:
        """
        Adjust rewards in the data batch based on player perspective.
        - Intermediate rewards alternate (White: as-is, Black: flipped).
        - Terminal rewards: winner +1, loser -1.
        """
        print(f"{data}")
        rewards = data["rewards"]
        dones = data["dones"]
        infos = data["infos"]

        # Adjust all rewards
        adjusted_rewards = rewards.copy()

        # Alternate intermediate rewards
        for i in range(len(rewards)):
            player = i % 2  # 0 for White, 1 for Black
            if player == 1 and not dones[i]:  # Black's turn, non-terminal
                adjusted_rewards[i] = -adjusted_rewards[i]  # Flip sign

        # Adjust terminal rewards
        terminal_indices = np.where(dones)[0]
        if len(terminal_indices) > 0:
            terminal_index = terminal_indices[-1]  # Last terminal step
            info = infos[terminal_index]

            if "outcome" in info:
                outcome = info["outcome"]
                last_player = terminal_index % 2  # 0 for White, 1 for Black

                if outcome == "white_win":
                    if last_player == 0:  # White's winning move
                        adjusted_rewards[terminal_index] = 1.0
                        if terminal_index > 0:
                            adjusted_rewards[terminal_index - 1] = -1.0  # Black loses
                    else:  # Rare case (White wins on Black's turn)
                        adjusted_rewards[terminal_index] = -1.0
                        if terminal_index > 0:
                            adjusted_rewards[terminal_index - 1] = 1.0
                elif outcome == "black_win":
                    if last_player == 1:  # Black's winning move
                        adjusted_rewards[terminal_index] = 1.0
                        if terminal_index > 0:
                            adjusted_rewards[terminal_index - 1] = -1.0  # White loses
                    else:  # Rare case (Black wins on White's turn)
                        adjusted_rewards[terminal_index] = -1.0
                        if terminal_index > 0:
                            adjusted_rewards[terminal_index - 1] = 1.0
                elif outcome == "draw":
                    adjusted_rewards[terminal_index] = 0.0
                    if terminal_index > 0:
                        adjusted_rewards[terminal_index - 1] = 0.0

        # Update the data batch
        data["rewards"] = adjusted_rewards
        return data

# Metrics-only callback
class ChessCombinedCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: dict[PolicyID, "Policy"],
        episode: "SingleAgentEpisode",
        **kwargs
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

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self._wrap_observation(obs), info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            return self._wrap_observation(obs), reward, terminated, truncated, info

        def _wrap_observation(self, obs):
            if not isinstance(obs, dict) or "board" not in obs or "action_mask" not in obs:
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

# Define a function to create an env-to-module connector pipeline using the ChessRewardConnector
def make_chess_reward_connector(env):
    # Return a single ConnectorV2 object or a list of ConnectorV2 objects
    return [ChessRewardConnector()]

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
        )
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_cpus_per_env_runner=num_cpus_per_env_runner,
            num_gpus_per_env_runner=num_gpus_per_env_runner,
            sample_timeout_s=None,
            env_to_module_connector=make_chess_reward_connector
        )
        .training(
            train_batch_size_per_learner=4096,
            minibatch_size=256,
            num_epochs=10,
            lr=5e-5,
            grad_clip=1.0,
            gamma=0.99,
            use_gae=True,
            lambda_=0.95,
            vf_loss_coeff=0.5,
            entropy_coeff=args.entropy_coeff,
            clip_param=0.2,
            kl_coeff=0.2,
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
        verbose=2,
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