#!/usr/bin/env python3
"""
Chess Reinforcement Learning using Ray RLlib.
This implementation uses RLlib's built-in distributed training capabilities.
"""
# Import custom environment
from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper
# Import RLModule for the new API stack
import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch

# Import custom environment
from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper

torch, nn = try_import_torch()


class ChessMetricsCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker,           # Typically a RolloutWorker
        base_env,         # Typically a BaseEnv
        policies,         # Typically a Dict[str, Policy]
        episode,          # Typically an Episode object
        env_index=None,   # Optional, may be passed in some contexts
        **kwargs
    ) -> None:
        # Access the last info dictionary from the episode
        info = episode.infos[-1]  # Your current approach
        if "outcome" in info:
            outcome = info["outcome"]
            if outcome == "white_win":
                episode.custom_metrics["white_win"] = 1.0
                episode.custom_metrics["black_win"] = 0.0
                episode.custom_metrics["draw"] = 0.0
            elif outcome == "black_win":
                episode.custom_metrics["white_win"] = 0.0
                episode.custom_metrics["black_win"] = 1.0
                episode.custom_metrics["draw"] = 0.0
            elif outcome == "draw":
                episode.custom_metrics["white_win"] = 0.0
                episode.custom_metrics["black_win"] = 0.0
                episode.custom_metrics["draw"] = 1.0
            
            # Record termination reason if available
            if "termination_reason" in info:
                reason = info["termination_reason"]
                if reason == "checkmate":
                    episode.custom_metrics["checkmate"] = 1.0
                elif reason == "stalemate":
                    episode.custom_metrics["stalemate"] = 1.0
                elif reason == "insufficient_material":
                    episode.custom_metrics["insufficient_material"] = 1.0
                elif reason == "fifty_moves":
                    episode.custom_metrics["fifty_moves"] = 1.0
                elif reason == "repetition":
                    episode.custom_metrics["repetition"] = 1.0
                elif reason == "move_limit_exceeded":
                    episode.custom_metrics["move_limit_exceeded"] = 1.0


class ChessMaskedRLModule(TorchRLModule):
    """Custom RLModule for Chess that supports action masking - compatible with the new RLlib API stack"""
    
    def __init__(self, observation_space=None, action_space=None, inference_only=False, learner_only=False, model_config=None, **kwargs):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config or {}
        )
        
        # Get spaces
        obs_space = observation_space
        
        # Print observation space details for debugging
        print(f"Initializing ChessMaskedRLModule with observation space: {obs_space}")
        if isinstance(obs_space, gym.spaces.Dict):
            print(f"Dict keys: {obs_space.spaces.keys()}")
            for key, space in obs_space.spaces.items():
                print(f"  {key}: {space}")
        
        # Handle both Dict observation spaces and non-Dict spaces
        if isinstance(obs_space, gym.spaces.Dict):
            # Get feature dimensions from observation space
            self.board_shape = obs_space["board"].shape
            self.action_mask_shape = obs_space["action_mask"].shape
        else:
            # If we don't have a Dict space, assume it's a Box with the expected board dimensions
            # Log a warning but continue instead of raising an error
            print(f"WARNING: Expected Dict observation space but got {type(obs_space)}. Assuming standard dimensions.")
            # Assume standard chess board dimensions: 13 channels for pieces, 8x8 board
            self.board_shape = (13, 8, 8)
            self.action_mask_shape = (action_space.n,)  # Standard action mask size
            
        self.board_channels = self.board_shape[0]  # 13 channels (6 piece types x 2 colors + empty)
        self.board_size = self.board_shape[1]  # 8x8 board
        
        # Feature extractor CNN
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
        
        # Policy and value heads
        self.policy_head = nn.Linear(832, action_space.n)
        self.value_head = nn.Linear(832, 1)
        
        # Epsilon-greedy exploration parameters
        self.initial_epsilon = 0.1  # 10% random action probability
        self.final_epsilon = 0.02   # 2% random action probability
        self.epsilon_timesteps = 1_000_000  # Decay over 1M timesteps
        self.current_epsilon = self.initial_epsilon
        self.random_exploration = True  # Set to False to disable exploration
        self.timesteps = 0  # Track timesteps for epsilon decay
        
        # Check if we're in evaluation mode from model config
        model_config_dict = model_config or {}
        if model_config_dict.get("evaluation_mode", False):
            self.random_exploration = False
            print("Model initialized in evaluation mode, exploration disabled")
        else:
            print(f"Initialized epsilon-greedy exploration with initial_epsilon={self.initial_epsilon}, final_epsilon={self.final_epsilon}")
    
    def set_evaluation_mode(self, evaluation_mode=True):
        """Explicitly set whether this model is in evaluation mode (disables exploration)"""
        previous_mode = not self.random_exploration
        if evaluation_mode:
            self.random_exploration = False
            print("Model set to evaluation mode, exploration disabled")
        else:
            self.random_exploration = True
            print("Model set to training mode, exploration enabled")
        return previous_mode
    
    def update_epsilon(self, num_steps=None):
        """Update exploration epsilon based on timesteps or the provided step count"""
        if num_steps is not None:
            self.timesteps = num_steps
        else:
            self.timesteps += 1
            
        # Linear decay from initial_epsilon to final_epsilon over epsilon_timesteps
        if self.timesteps < self.epsilon_timesteps:
            self.current_epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * (self.timesteps / self.epsilon_timesteps)
        else:
            self.current_epsilon = self.final_epsilon
            
        return self.current_epsilon
    
    def _forward_inference(self, batch):
        """Forward pass for inference (no exploration)"""
        # Extract tensor components
        if isinstance(batch, dict) and "board" in batch:
            board = batch["board"]
            action_mask = batch["action_mask"]
            print(f"Inference with dict batch: board shape={board.shape}, action_mask shape={action_mask.shape}")
        else:
            # Handle non-dict inputs (should be rare with the wrapper)
            device = next(self.parameters()).device
            # Create a default batch size of 1 if batch is not a tensor with shape
            batch_size = getattr(batch, "shape", [1])[0] if hasattr(batch, "shape") else 1
            print(f"Inference with non-dict batch: type={type(batch)}, batch_size={batch_size}")
            board = torch.zeros((batch_size, 13, 8, 8), device=device)
            action_mask = torch.zeros((batch_size, 20480), device=device)
        
        # Ensure we have a batch dimension
        if len(board.shape) == 3:  # If missing batch dimension (13, 8, 8)
            board = board.unsqueeze(0)  # Add batch dimension -> (1, 13, 8, 8)
            
        if len(action_mask.shape) == 1:  # If missing batch dimension (20480,)
            action_mask = action_mask.unsqueeze(0)  # Add batch dimension -> (1, 20480)
            
        # Get batch size from board tensor
        batch_size = board.shape[0]
        
        # Process through CNN feature extractor
        features = self.features_extractor(board)
        
        # Add small epsilon to features to avoid NaN issues
        features = features + 1e-8
        
        # Get raw action outputs (logits)
        action_logits = self.policy_head(features)
        
        # Apply action mask by setting illegal actions to a large negative number
        if action_mask is not None:
            # Apply the mask to the logits
            inf_mask = torch.clamp(1 - action_mask, min=0, max=1) * -1000.0
            masked_logits = action_logits + inf_mask
            
            # Apply gradient clipping to logits
            masked_logits = torch.clamp(masked_logits, min=-50.0, max=50.0)
            
            # Get the most likely action (for deterministic policy)
            actions = torch.argmax(masked_logits, dim=-1)
            
            # Print shapes for debugging
            print(f"Output shapes - masked_logits: {masked_logits.shape}, actions: {actions.shape}")
            
            return {
                "action_dist": masked_logits,
                "actions": actions
            }
        
        # If no action mask available, return unmasked logits (with clipping)
        action_logits = torch.clamp(action_logits, min=-50.0, max=50.0)
        actions = torch.argmax(action_logits, dim=-1)
        
        # Print shapes for debugging
        print(f"Output shapes - action_logits: {action_logits.shape}, actions: {actions.shape}")
            
        return {
            "action_dist": action_logits,
            "actions": actions
        }
    
    def _forward_exploration(self, batch, **kwargs):
        if 'obs' not in batch:
            raise ValueError("Expected 'obs' key in batch")
        
        obs = batch['obs']
        
        if not isinstance(obs, dict) or 'board' not in obs or 'action_mask' not in obs:
            raise ValueError("Expected 'obs' to be a dict with 'board' and 'action_mask'")
        
        board = obs['board']
        action_mask = obs['action_mask']
        batch_size = board.shape[0]
        device = board.device

        if board.shape[1:] != (13, 8, 8) or action_mask.shape[1:] != (20480,):
            raise ValueError(f"Unexpected shapes: board={board.shape}, action_mask={action_mask.shape}")

        features = self.features_extractor(board)
        features = features + 1e-8
        action_logits = self.policy_head(features)
        value = self.value_head(features)

        inf_mask = torch.clamp(1 - action_mask, min=0, max=1) * -1000.0
        masked_logits = action_logits + inf_mask

        if self.random_exploration:
            self.update_epsilon()
            random_samples = torch.rand(batch_size, device=device) < self.current_epsilon
            for i in range(batch_size):
                if random_samples[i]:
                    legal_actions = torch.where(action_mask[i] > 0)[0]
                    if len(legal_actions) == 0:
                        raise ValueError(f"No legal actions in batch index {i}, action_mask sum: {action_mask[i].sum()}")
                    random_action_idx = legal_actions[torch.randint(0, len(legal_actions), (1,), device=device)].item()
                    masked_logits[i] = torch.full_like(masked_logits[i], -5.0)
                    masked_logits[i, random_action_idx] = 5.0

        masked_logits = torch.clamp(masked_logits, min=-50.0, max=50.0)
        actions = torch.argmax(masked_logits, dim=-1)
        value = value.view(batch_size, 1)  # Ensure value matches batch size

        return {
            'action_dist': masked_logits,
            'vf': value,
            'actions': actions
        }
    
    def forward(self, batch, **kwargs):
        """Forward method for both training and inference"""
        # Check if we're in inference mode
        deterministic = kwargs.get("deterministic", False)
        
        if deterministic:
            return self._forward_inference(batch)
        else:
            return self._forward_exploration(batch, **kwargs)


def create_rllib_chess_env(config):
    """Factory function to create chess environment for RLlib"""
    try:
        # Create the base environment
        env = ChessEnv()
        
        # First wrap with ActionMaskWrapper 
        env = ActionMaskWrapper(env)
        
        # Extra validation and wrapping for RLlib compatibility
        # This is critical for ensuring Dict observation space in all contexts
        from gymnasium import spaces
        import numpy as np
        
        # ALWAYS use the Dict wrapper regardless of current observation space
        # Print current observation space for debugging
        print(f"Wrapping environment with DictObsWrapper, original obs space: {type(env.observation_space)}")
        
        # Define a custom wrapper right here to ensure Dict observation space
        class DictObsWrapper(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                self.observation_space = spaces.Dict({
                    'board': spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32),
                    'action_mask': spaces.Box(low=0, high=1, shape=(env.action_space.n,), dtype=np.float32),
                    'white_to_move': spaces.Discrete(2)
                })
                self.action_space = env.action_space

            def reset(self, **kwargs):
                result = self.env.reset(**kwargs)
                if isinstance(result, tuple):
                    obs, info = result
                else:
                    obs, info = result, {}
                return self._wrap_observation(obs), self._ensure_info(info)

            def step(self, action):
                result = self.env.step(action)
                if len(result) == 4:
                    obs, reward, done, info = result
                    terminated, truncated = done, False
                else:
                    obs, reward, terminated, truncated, info = result
                wrapped_obs = self._wrap_observation(obs)
                info = self._ensure_info(info)
                # Ensure action mask has at least one legal move if not done
                if not (terminated or truncated):
                    assert wrapped_obs['action_mask'].sum() > 0, "No legal actions available in non-terminal state"
                return wrapped_obs, reward, terminated, truncated, info

            def _wrap_observation(self, obs):
                if obs is None or not isinstance(obs, dict) or 'board' not in obs or 'action_mask' not in obs:
                    print(f"Warning: Invalid observation {type(obs)}, using placeholder")
                    return {
                        'board': np.zeros((13, 8, 8), dtype=np.float32),
                        'action_mask': np.ones(self.env.action_space.n, dtype=np.float32),
                        'white_to_move': 1
                    }
                board = np.asarray(obs['board'], dtype=np.float32)
                action_mask = np.asarray(obs['action_mask'], dtype=np.float32)
                white_to_move = int(obs.get('white_to_move', 1))
                # Enforce correct shapes
                if board.shape != (13, 8, 8):
                    print(f"Warning: Fixing board shape from {board.shape} to (13, 8, 8)")
                    board = np.zeros((13, 8, 8), dtype=np.float32)
                if action_mask.shape != (self.action_space.n,):
                    print(f"Warning: Fixing action_mask shape from {action_mask.shape}")
                    action_mask = np.ones(self.action_space.n, dtype=np.float32)
                return {
                    'board': board,
                    'action_mask': action_mask,
                    'white_to_move': white_to_move
                }

            def _ensure_info(self, info):
                if not isinstance(info, dict):
                    info = {}
                if 'outcome' not in info:
                    info['outcome'] = 'unknown'
                if 'termination_reason' not in info:
                    info['termination_reason'] = 'unknown'
                return info
                    
        # Always apply the wrapper to ensure consistent observation format
        env = DictObsWrapper(env)
        
        return env
        
    except Exception as e:
        # Log any errors during environment creation
        import traceback
        print(f"Error creating environment: {e}")
        print(traceback.format_exc())
        # Return a placeholder environment with the expected Dict observation space
        from gymnasium import spaces
        import numpy as np
        
        class PlaceholderEnv(gym.Env):
            def __init__(self):
                super().__init__()
                # Define action and observation spaces
                self.action_space = spaces.Discrete(20480)  # Same as ChessEnv
                
                # Define observation space with same structure as ChessEnv and DictObsWrapper
                self.observation_space = spaces.Dict({
                    'board': spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32),
                    'action_mask': spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.float32),
                    'white_to_move': spaces.Discrete(2)  # Boolean flag: 0=False (Black's turn), 1=True (White's turn)
                })
            
            def reset(self, **kwargs):
                # Return a default observation with the correct structure
                obs = {
                    'board': np.zeros((13, 8, 8), dtype=np.float32),
                    'action_mask': np.ones(self.action_space.n, dtype=np.float32),
                    'white_to_move': 1  # Default to White's turn
                }
                # Set empty squares in board representation (channel 12)
                obs['board'][12, :, :] = 1.0
                
                return obs, {}
            
            def step(self, action):
                # Return a default observation with the correct Gymnasium API format
                # (obs, reward, terminated, truncated, info)
                obs = {
                    'board': np.zeros((13, 8, 8), dtype=np.float32),
                    'action_mask': np.ones(self.action_space.n, dtype=np.float32),
                    'white_to_move': 1  # Default to White's turn
                }
                # Set empty squares in board representation (channel 12)
                obs['board'][12, :, :] = 1.0
                
                return obs, 0.0, True, False, {}
        
        return PlaceholderEnv()


def train(args):
    """Main training function using RLlib PPO"""
    # Import required modules
    import os
    
    # Optimize PyTorch memory usage for CUDA
    if args.device == "cuda" and not args.force_cpu:
        # Enable TensorFloat32 for faster computation on Ampere GPUs
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        # Set memory allocation strategy to reduce fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
        print("🚀 Enabled PyTorch CUDA optimizations")
    
    # Initialize Ray with fixed number of CPUs and GPUs
    if args.redis_password:
        ray.init(
            address=args.head_address,
            ignore_reinit_error=True, 
            include_dashboard=args.dashboard,
            _redis_password=args.redis_password,
            num_cpus=118,
        )
    else:
        ray.init(
            address="auto" if args.distributed else None,
            ignore_reinit_error=True,
            include_dashboard=args.dashboard,
            num_cpus=118,
        )
    
    # Hardware configuration - using fixed allocation as requested
    print("\n===== Hardware Configuration =====")
    print(f"Ray runtime resources: {ray.available_resources()}")
    
    # Fixed resource allocation as requested:
    # - 3:3 GPU split
    # - 20 workers with 4 CPUs each
    driver_gpus = .99999          # Fixed at 3 GPUs for driver
    worker_gpus = 3         # Fixed at 3 GPUs for ivworkers
    num_workers = 12          # Fixed at 20 workers
    cpus_per_worker = 4       # Fixed at 4 CPUs per worker
    driver_cpus = 8       # Fixed at 8 CPUs for driver
    num_envs = 4            # Environments per worker
    
    # Calculate exact GPU allocation per worker
    if num_workers > 0 and worker_gpus > 0:
        # If workers > GPUs, give each worker a fixed fraction (no remainder)
        if num_workers >= worker_gpus:
            # Integer workers per GPU
            workers_per_gpu = num_workers // worker_gpus
            
            # Distribute GPUs evenly, ensuring each worker gets the same amount
            # (e.g., with 3 GPUs and 20 workers, each worker gets 3/20 = 0.15 GPU)
            gpus_per_worker = worker_gpus / num_workers
            
            # Round to 2 decimal places to avoid floating point issues
            gpus_per_worker = round(worker_gpus / num_workers, 6)
        else:
            # If GPUs > workers, each worker gets at least 1 GPU
            gpus_per_worker = worker_gpus // num_workers
    else:
        gpus_per_worker = 0
    
    # Use fixed batch sizes
    train_batch_size = 32768 * 4
    sgd_minibatch_size = 4096 * 4
    
    total_cpu_request = driver_cpus + (cpus_per_worker * num_workers)
    print(f"CPU allocation: {driver_cpus} (driver) + {cpus_per_worker}*{num_workers} (workers) = {total_cpu_request}")
    
    # Print GPU allocation
    print(f"GPU allocation: {driver_gpus} (driver) + {gpus_per_worker}*{num_workers} (workers) = {driver_gpus + (gpus_per_worker * num_workers)}")
    print(f"Batch size: {train_batch_size} (train) / {sgd_minibatch_size} (SGD)")
    print("==================================\n")
    
    # Test environment creation to diagnose observation space issues
    print("\n===== Environment Test =====")
    env = create_rllib_chess_env({})
    print(f"Environment observation space: {env.observation_space}")
    print(f"Environment action space: {env.action_space}")
    obs, _ = env.reset()
    print(f"Observation structure: {type(obs)}")
    if isinstance(obs, dict):
        print(f"Observation keys: {obs.keys()}")
        for key, value in obs.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {type(value)} with shape {value.shape}")
            else:
                print(f"  {key}: {type(value)} with value {value}")
    print("============================\n")

    # Register only the environment (we don't need ModelCatalog with RLModule API)
    tune.register_env("chess_env", create_rllib_chess_env)
    
    # Create an absolute path for checkpoint directory
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Using checkpoint directory: {checkpoint_dir}")
    
    # Print exploration strategy
    print(f"\n===== Exploration Strategy =====")
    print(f"Initial entropy coefficient: {args.entropy_coeff}")
    print(f"Annealing schedule: 100% → 50% → 10% over {args.max_iterations} iterations")
    print(f"This will encourage the agent to explore diverse moves early and exploit its knowledge later.")
    print(f"================================\n")
    
    # Set up checkpoint restoration if provided
    restore_path = None
    if args.checkpoint:
        # Validate checkpoint path
        if os.path.exists(args.checkpoint):
            restore_path = args.checkpoint
            print(f"Will restore from checkpoint: {restore_path}")
        else:
            print(f"Warning: Checkpoint path {args.checkpoint} does not exist. Starting fresh.")

    # Modify config to use learner API properly with all GPUs
    # Reassign GPU resources - crucial for fixing utilization
    config = {
        "env": "chess_env",
        "framework": "torch",
        "disable_env_checking": True,
        "_enable_rl_module_api": True,
        "_enable_learner_api": True,
        "num_workers": 12,
        "num_cpus_per_env_runner": 4,
        "num_gpus_per_learner": 1.0,
        "num_learners": 4,
        "train_batch_size": 16384,
        "sgd_minibatch_size": 2048,
        "lr": 5e-5,
        "grad_clip": 1.0,
        "entropy_coeff": args.entropy_coeff,
        "sample_timeout_s": 600,  # Increase timeout to 20 seconds
        "rollout_fragment_length": "auto",  # Reduce fragment length for faster sampling
        "num_envs_per_env_runner": 4,  # Use a single environment per worker
        "batch_mode": "truncate_episodes",  # Use truncated episodes for faster sampling
        "callbacks": ChessMetricsCallback,  # Add metrics callback
    }
    
    # Create a proper RLModuleSpec instance
    config["rl_module_spec"] = RLModuleSpec(
        module_class=ChessMaskedRLModule,
        model_config={"evaluation_mode": False}
    )
    
    print("\n===== Multi-GPU Learner Configuration =====")
    print(f"GPU allocation: 4 learners with 1.0 GPU each")
    print(f"Workers focus on data collection without GPU allocation")
    print("===========================================\n")
    
    analysis = tune.run(
        "PPO",
        stop={"training_iteration": args.max_iterations},
        checkpoint_freq=25,  # Save checkpoint every 25 iterations (increased frequency)
        checkpoint_at_end=True,
        storage_path=checkpoint_dir,
        verbose=2,  # Detailed output
        metric=None,
        mode="max",
        resume="AUTO",  # AUTO mode: resume if checkpoint exists, otherwise start fresh
        restore=restore_path,  # Add this for restoring from specific checkpoint
        config=config,
    )
    
    # Get best checkpoint
    best_checkpoint = analysis.best_checkpoint
    print(f"Best checkpoint: {best_checkpoint}")
    
    return best_checkpoint


def evaluate(args):
    """Evaluate a trained policy"""
    if not args.checkpoint:
        print("Error: Checkpoint path required for evaluation mode")
        return

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint path '{args.checkpoint}' does not exist")
        return
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, include_dashboard=args.dashboard)
    
    # Register only the environment (we don't need ModelCatalog with RLModule API)
    tune.register_env("chess_env", create_rllib_chess_env)
    
    # Create evaluation configuration
    config = {
        "env": "chess_env",
        "framework": "torch",
        "num_workers": 0,  # Single worker for evaluation
        "num_gpus": 1 if args.device == "cuda" else 0,  # Use exact integers for GPU allocation
        # Use RLModule API with proper RLModuleSpec
        "_enable_rl_module_api": True,
        # Critical: Ensure we're using deterministic actions (no exploration)
        "explore": False,
    }
    
    # Create a proper RLModuleSpec instance for evaluation
    config["rl_module_spec"] = RLModuleSpec(
        module_class=ChessMaskedRLModule,
        model_config={"evaluation_mode": True}
    )
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    # Create algorithm and restore from checkpoint
    algo = PPO(config=config)
    algo.restore(args.checkpoint)
    
    # Create environment for evaluation
    env = create_rllib_chess_env({})
    
    # Run evaluation
    num_episodes = 50  # Number of games to evaluate
    total_rewards = []
    outcomes = {"white_win": 0, "black_win": 0, "draw": 0, "unknown": 0}
    
    print(f"Evaluating model over {num_episodes} episodes...")
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (done or truncated):
            # Get action from policy, using deterministic=True to disable exploration
            action = algo.compute_single_action(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Optionally render if requested
            if args.render:
                env.render()
        
        # Record outcome
        outcome = info.get("outcome", "unknown")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward}, Steps = {steps}, Outcome = {outcome}")
    
    # Print evaluation results
    print("\n----- Evaluation Results -----")
    print(f"Average Episode Reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"Win %: White={outcomes['white_win']/num_episodes*100:.1f}%, Black={outcomes['black_win']/num_episodes*100:.1f}%, Draw={outcomes['draw']/num_episodes*100:.1f}%")
    print(f"Total Outcomes: White Wins: {outcomes['white_win']}, Black Wins: {outcomes['black_win']}, Draws: {outcomes['draw']}")
    print("-----------------------------\n")
    
    # Close environment
    env.close()
    
    # Shutdown Ray
    ray.shutdown()


def main():
    """Parse arguments and run appropriate mode"""
    parser = argparse.ArgumentParser(description="Chess RL training using RLlib")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", 
                        help="Mode: train or evaluate")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", 
                        help="Device to use (cpu or cuda)")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint to load")
    parser.add_argument("--force_cpu", action="store_true", 
                        help="Force CPU use even if CUDA is available")
    parser.add_argument("--distributed", action="store_true", 
                        help="Enable distributed training")
    parser.add_argument("--head", action="store_true", 
                        help="This node is the head node")
    parser.add_argument("--head_address", type=str, default=None, 
                        help="Ray head node address (for worker nodes)")
    parser.add_argument("--redis_password", type=str, default=None, 
                        help="Redis password for Ray cluster")
    parser.add_argument("--dashboard", action="store_true", 
                        help="Enable Ray dashboard")
    parser.add_argument("--num_workers", type=int, default=2, 
                        help="Number of worker processes")
    parser.add_argument("--max_iterations", type=int, default=10000, 
                        help="Maximum number of training iterations")
    parser.add_argument("--checkpoint_dir", type=str, default="rllib_checkpoints", 
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=10, 
                        help="How often to save checkpoints (iterations)")
    parser.add_argument("--render", action="store_true", 
                        help="Render environment during evaluation")
    parser.add_argument("--inference_mode", choices=["cpu", "gpu"], default="gpu", 
                        help="Inference mode: cpu or gpu")
    parser.add_argument("--entropy_coeff", type=float, default=0.05, 
                        help="Entropy coefficient for PPO")
    
    args = parser.parse_args()
    
    # Run in appropriate mode
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)


if __name__ == "__main__":
    main() 