import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
import gym
from collections import deque

from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper
from src.cnn import CNNMCTSActorCriticPolicy, MCTS, Node

# Simple meter collection replacement
class AverageMeterCollection:
    def __init__(self):
        self.meters = {}
        
    def update(self, metrics):
        for k, v in metrics.items():
            if k not in self.meters:
                self.meters[k] = []
            self.meters[k].append(v)
            
    def summary(self):
        return {k: np.mean(v[-100:]) if v else 0 for k, v in self.meters.items()}

# Make environment creation function
def make_env(seed=0, simple_test=False, white_advantage=None):
    env = ChessEnv(simple_test=simple_test, white_advantage=white_advantage)
    env = ActionMaskWrapper(env)
    env.reset(seed=seed)
    return env

# Function to get available resources on the system
def get_available_resources():
    """Get available GPU and CPU resources on the system"""
    try:
        gpu_count = torch.cuda.device_count()
    except:
        gpu_count = 0
    
    cpu_count = os.cpu_count() or 16  # Default to 16 if can't detect
    
    return {
        "gpu_count": gpu_count,
        "cpu_count": cpu_count
    }

# Actor class - collects experiences and sends gradients
@ray.remote(num_gpus=0.05)
class A3CActor:
    def __init__(self, actor_id, model_cls, obs_space, action_space, config):
        self.actor_id = actor_id
        self.config = config
        
        # Create local model
        self.local_model = model_cls(
            obs_space,
            action_space,
            lambda _: config["learning_rate"]
        ).to(config["device"])
        
        # Initialize local environments
        self.envs = []
        for i in range(config["n_envs_per_actor"]):
            env = make_env(
                seed=actor_id * 1000 + i, 
                simple_test=config.get("simple_test", False),
                white_advantage=config.get("white_advantage", None)
            )
            self.envs.append(env)
        
        # Setup MCTS if needed
        if config["mcts_sims"] > 0:
            self.mcts = MCTS(self.local_model, num_simulations=config["mcts_sims"], c_puct=2.0)
            self.use_mcts = True
        else:
            self.use_mcts = False
        
        # Statistics
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        
        # Game statistics
        self.game_outcomes = {
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
            "games_completed": 0,
            "move_counts": []  # Track number of moves per game
        }
        self.current_game_moves = [0] * len(self.envs)
    
    def get_weights(self):
        """Return model weights as numpy arrays"""
        return {k: v.cpu().numpy() for k, v in self.local_model.state_dict().items()}
    
    def set_weights(self, weights):
        """Set model weights from numpy arrays"""
        weights_dict = {k: torch.tensor(v).to(self.config["device"]) for k, v in weights.items()}
        self.local_model.load_state_dict(weights_dict)
    
    def compute_gradients(self, weights):
        """Compute gradients based on collected experiences"""
        # Set local model weights
        self.set_weights(weights)
        
        # Collect trajectory
        trajectory = self.collect_trajectory(max_steps=self.config["trajectory_length"])
        
        # Compute gradients
        grads = self.compute_gradients_from_trajectory(trajectory)
        
        # Return gradients and stats
        info = {
            "actor_id": self.actor_id,
            "steps": self.total_steps,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
        }
        
        return grads, info
    
    def collect_trajectory(self, max_steps=128):
        """Collect a trajectory of experiences"""
        states = []
        actions = []
        values = []
        log_probs = []
        rewards = []
        masks = []
        action_masks = []
        
        # Start state for each environment
        obs_list = [env.reset()[0] for env in self.envs]  # [0] to get observation, not info
        dones = [False] * len(self.envs)
        
        for _ in range(max_steps):
            # Collect batch of observations
            batch_obs = {
                'board': np.stack([o['board'] for o in obs_list]),
                'action_mask': np.stack([o['action_mask'] for o in obs_list])
            }
            
            # Store action masks
            action_masks.append(batch_obs['action_mask'])
            
            # Convert to tensors
            board_tensor = torch.tensor(batch_obs['board'], dtype=torch.float32).to(self.config["device"])
            mask_tensor = torch.tensor(batch_obs['action_mask'], dtype=torch.float32).to(self.config["device"])
            obs_dict = {'board': board_tensor, 'action_mask': mask_tensor}
            
            batch_actions = []
            batch_values = []
            batch_log_probs = []
            
            # Get actions using policy
            with torch.no_grad():
                if self.use_mcts and np.random.random() < self.config["mcts_freq"]:
                    # Use MCTS for some environments
                    for i in range(len(self.envs)):
                        if dones[i]:
                            # For done environments, use placeholder values
                            batch_actions.append(0)
                            batch_values.append(0)
                            batch_log_probs.append(0)
                            continue
                            
                        state = self.envs[i].board.copy()
                        player = 1 if state.turn else -1
                        root = Node(state=state, root_player=player)
                        
                        # Run MCTS search
                        action, value = self.mcts.search(root)
                        
                        # Get log_prob from policy
                        single_board = board_tensor[i:i+1]
                        single_mask = mask_tensor[i:i+1]
                        single_obs = {'board': single_board, 'action_mask': single_mask}
                        
                        logits = self.local_model._get_action_dist(single_obs)[0]
                        action_dist = torch.distributions.Categorical(logits=logits)
                        log_prob = action_dist.log_prob(torch.tensor(action, device=self.config["device"]))
                        
                        batch_actions.append(action)
                        batch_values.append(value)
                        batch_log_probs.append(log_prob.item())
                else:
                    # Use standard policy for all environments
                    logits, values_tensor = self.local_model._get_action_dist_and_value(obs_dict)
                    
                    # Create action distributions
                    if hasattr(torch.distributions, 'Categorical'):
                        action_dists = [torch.distributions.Categorical(logits=logits[i]) for i in range(len(self.envs))]
                    else:
                        # For older PyTorch versions
                        action_dists = []
                        for i in range(len(self.envs)):
                            if dones[i]:
                                action_dists.append(None)
                            else:
                                probs = F.softmax(logits[i], dim=0)
                                action_dists.append(torch.distributions.Categorical(probs=probs))
                    
                    # Sample actions
                    for i in range(len(self.envs)):
                        if dones[i]:
                            batch_actions.append(0)
                            batch_values.append(0)
                            batch_log_probs.append(0)
                        else:
                            action = action_dists[i].sample().item()
                            log_prob = action_dists[i].log_prob(torch.tensor(action, device=self.config["device"])).item()
                            batch_actions.append(action)
                            batch_values.append(values_tensor[i].item())
                            batch_log_probs.append(log_prob)
            
            # Store batch data
            states.append(batch_obs)
            actions.append(batch_actions)
            values.append(batch_values)
            log_probs.append(batch_log_probs)
            
            # Take actions in environments
            next_obs_list = []
            reward_list = []
            done_list = []
            info_list = []
            
            for i, env in enumerate(self.envs):
                if dones[i]:
                    # Skip done environments
                    next_obs_list.append(obs_list[i])
                    reward_list.append(0)
                    done_list.append(True)
                    info_list.append({})
                else:
                    # Step environment
                    next_obs, reward, terminated, truncated, info = env.step(batch_actions[i])
                    done = terminated or truncated
                    
                    # Track game moves
                    self.current_game_moves[i] += 1
                    
                    next_obs_list.append(next_obs)
                    reward_list.append(reward)
                    done_list.append(done)
                    info_list.append(info)
                    
                    # Track statistics
                    self.total_steps += 1
                    
                    # Record game outcomes
                    if done:
                        self.game_outcomes["games_completed"] += 1
                        self.game_outcomes["move_counts"].append(self.current_game_moves[i])
                        self.current_game_moves[i] = 0
                        
                        if info.get('white_won', False):
                            self.game_outcomes["white_wins"] += 1
                        elif info.get('black_won', False):
                            self.game_outcomes["black_wins"] += 1
                        else:
                            self.game_outcomes["draws"] += 1
                    
                    # Record episode rewards
                    if done:
                        if 'episode' in info and 'r' in info['episode']:
                            self.episode_rewards.append(info['episode']['r'])
                        elif hasattr(env, 'rewards'):
                            self.episode_rewards.append(sum(env.rewards))
                        else:
                            # Estimate episode reward
                            self.episode_rewards.append(reward)
            
            # Store rewards and dones
            rewards.append(reward_list)
            masks.append([1 - int(done) for done in done_list])
            
            # Reset done environments
            for i, done in enumerate(done_list):
                if done:
                    obs_list[i] = self.envs[i].reset()[0]
                    dones[i] = False
                else:
                    obs_list[i] = next_obs_list[i]
                    dones[i] = done_list[i]
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        
        # Get final value estimate for bootstrapping
        final_obs = {
            'board': np.stack([o['board'] for o in obs_list]),
            'action_mask': np.stack([o['action_mask'] for o in obs_list])
        }
        
        with torch.no_grad():
            board_tensor = torch.tensor(final_obs['board'], dtype=torch.float32).to(self.config["device"])
            mask_tensor = torch.tensor(final_obs['action_mask'], dtype=torch.float32).to(self.config["device"])
            final_obs_dict = {'board': board_tensor, 'action_mask': mask_tensor}
            
            _, final_values = self.local_model._get_action_dist_and_value(final_obs_dict)
            final_values = final_values.cpu().numpy()
        
        # Compute GAE for each environment
        for env_idx in range(len(self.envs)):
            env_returns = []
            env_advantages = []
            
            gae = 0
            next_value = final_values[env_idx]
            
            # Process trajectory in reverse for this environment
            for t in reversed(range(len(rewards))):
                mask = masks[t][env_idx]
                reward = rewards[t][env_idx]
                value = values[t][env_idx]
                
                # TD error
                delta = reward + self.config["gamma"] * next_value * mask - value
                
                # GAE
                gae = delta + self.config["gamma"] * self.config["gae_lambda"] * mask * gae
                
                # Return = advantage + value
                ret = gae + value
                
                env_returns.insert(0, ret)
                env_advantages.insert(0, gae)
                
                next_value = value
            
            returns.append(env_returns)
            advantages.append(env_advantages)
        
        # Transpose returns and advantages to match trajectory format
        returns = list(map(list, zip(*returns)))
        advantages = list(map(list, zip(*advantages)))
        
        return {
            'states': states,
            'actions': actions,
            'returns': returns,
            'advantages': advantages,
            'log_probs': log_probs,
            'action_masks': action_masks
        }
    
    def compute_gradients_from_trajectory(self, trajectory):
        """Compute gradients from a collected trajectory"""
        # Extract data
        states = trajectory['states']
        actions = trajectory['actions']
        returns = trajectory['returns']
        advantages = trajectory['advantages']
        old_log_probs = trajectory['log_probs']
        
        # Zero gradients
        self.local_model.zero_grad()
        
        # Initialize losses
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        
        # Process each step in the trajectory
        for t in range(len(states)):
            # Get batch data
            state = states[t]
            action = np.array(actions[t])
            ret = np.array(returns[t])
            advantage = np.array(advantages[t])
            old_log_prob = np.array(old_log_probs[t])
            
            # Convert to tensors
            board_tensor = torch.tensor(state['board'], dtype=torch.float32).to(self.config["device"])
            mask_tensor = torch.tensor(state['action_mask'], dtype=torch.float32).to(self.config["device"])
            obs_dict = {'board': board_tensor, 'action_mask': mask_tensor}
            
            action_tensor = torch.tensor(action, dtype=torch.long).to(self.config["device"])
            return_tensor = torch.tensor(ret, dtype=torch.float32).to(self.config["device"])
            advantage_tensor = torch.tensor(advantage, dtype=torch.float32).to(self.config["device"])
            old_log_prob_tensor = torch.tensor(old_log_prob, dtype=torch.float32).to(self.config["device"])
            
            # Normalize advantages
            advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std() + 1e-8)
            
            # Forward pass
            logits, values = self.local_model._get_action_dist_and_value(obs_dict)
            
            # Calculate action probabilities and entropy
            action_dists = []
            for i in range(len(logits)):
                if hasattr(torch.distributions, 'Categorical'):
                    action_dists.append(torch.distributions.Categorical(logits=logits[i]))
                else:
                    probs = F.softmax(logits[i], dim=0)
                    action_dists.append(torch.distributions.Categorical(probs=probs))
            
            # Calculate log probabilities
            log_probs = torch.stack([dist.log_prob(action_tensor[i]) for i, dist in enumerate(action_dists)])
            
            # Calculate entropy
            entropy = torch.stack([dist.entropy() for dist in action_dists]).mean()
            
            # Calculate policy loss (PPO style with clipping)
            ratio = torch.exp(log_probs - old_log_prob_tensor)
            policy_loss_1 = -advantage_tensor * ratio
            policy_loss_2 = -advantage_tensor * torch.clamp(ratio, 1.0 - self.config["clip_range"], 1.0 + self.config["clip_range"])
            batch_policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
            
            # Calculate value loss
            batch_value_loss = F.mse_loss(values.squeeze(-1), return_tensor)
            
            # Accumulate losses
            policy_loss += batch_policy_loss
            value_loss += batch_value_loss
            entropy_loss -= entropy  # Negative because we want to maximize entropy
        
        # Calculate average losses
        policy_loss /= len(states)
        value_loss /= len(states)
        entropy_loss /= len(states)
        
        # Total loss
        total_loss = policy_loss + self.config["vf_coef"] * value_loss + self.config["ent_coef"] * entropy_loss
        
        # Backpropagation
        total_loss.backward()
        
        # Get gradients as numpy arrays
        grads = {}
        for name, param in self.local_model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.cpu().numpy()
            else:
                grads[name] = np.zeros_like(param.data.cpu().numpy())
        
        return grads
    
    def get_stats(self):
        """Return current statistics"""
        # Calculate win rates
        total_games = max(1, self.game_outcomes["games_completed"])
        white_win_rate = self.game_outcomes["white_wins"] / total_games * 100
        black_win_rate = self.game_outcomes["black_wins"] / total_games * 100
        draw_rate = self.game_outcomes["draws"] / total_games * 100
        
        # Calculate average moves per game
        avg_moves = np.mean(self.game_outcomes["move_counts"]) if self.game_outcomes["move_counts"] else 0
        
        return {
            "steps": self.total_steps,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "num_episodes": len(self.episode_rewards),
            "white_wins": self.game_outcomes["white_wins"],
            "black_wins": self.game_outcomes["black_wins"],
            "draws": self.game_outcomes["draws"],
            "white_win_rate": white_win_rate,
            "black_win_rate": black_win_rate,
            "draw_rate": draw_rate,
            "avg_moves_per_game": avg_moves,
            "games_completed": self.game_outcomes["games_completed"]
        }

# Learner class - updates global model using gradients from actors
class A3CLearner:
    def __init__(self, model_cls, obs_space, action_space, config):
        self.config = config
        
        # Create global model
        self.model = model_cls(
            obs_space,
            action_space,
            lambda _: config["learning_rate"]
        ).to(config["device"])
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        
        # Statistics
        self.total_updates = 0
        self.metrics = AverageMeterCollection()
    
    def get_weights(self):
        """Return model weights as numpy arrays"""
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
    
    def apply_gradients(self, gradients_list, infos):
        """Apply gradients from multiple actors"""
        # Zero optimizer gradients
        self.optimizer.zero_grad()
        
        # Average gradients from all actors
        avg_grads = {}
        for name, param in self.model.named_parameters():
            # Initialize with zeros
            avg_grads[name] = torch.zeros_like(param.data)
            
            # Sum gradients from all actors
            for grads in gradients_list:
                if name in grads:
                    grad_tensor = torch.tensor(grads[name], device=self.config["device"])
                    avg_grads[name] += grad_tensor
            
            # Average the gradients
            avg_grads[name] /= len(gradients_list)
            
            # Apply gradient to parameter
            param.grad = avg_grads[name]
        
        # Apply gradient clipping if needed
        if self.config["max_grad_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
        
        # Update model
        self.optimizer.step()
        
        # Update statistics
        self.total_updates += 1
        
        # Update metrics
        for info in infos:
            if "mean_reward" in info:
                self.metrics.update({"mean_reward": info["mean_reward"]})
        
        return self.total_updates, self.metrics.summary()
    
    def save_model(self, path):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

# A3C Trainer class
class A3CTrainer:
    def __init__(self, config):
        self.config = config
        
        # Auto-adjust number of actors based on available resources
        if config.get("auto_adjust_actors", True):
            resources = get_available_resources()
            gpu_count = resources["gpu_count"]
            cpu_count = resources["cpu_count"]
            
            # If we have GPUs, limit actors based on GPU resources
            if gpu_count > 0:
                # Calculate how many actors we can fit with GPU fraction
                gpu_fraction = 0.05  # GPU fraction per actor
                max_gpu_actors = int(gpu_count / gpu_fraction)
                
                # Also consider CPU constraints (1 CPU per actor)
                max_cpu_actors = max(1, cpu_count - 2)  # Leave 2 CPUs for system
                
                # Take the minimum of the two constraints
                max_actors = min(max_gpu_actors, max_cpu_actors)
                
                # Limit actors to what's available
                adjusted_actors = min(config["num_actors"], max_actors)
                
                if adjusted_actors < config["num_actors"]:
                    print(f"Auto-adjusting actors from {config['num_actors']} to {adjusted_actors} based on available resources")
                    config["num_actors"] = adjusted_actors
            else:
                # CPU-only mode
                max_actors = max(1, cpu_count - 2)  # Leave 2 CPUs for system
                
                if config["num_actors"] > max_actors:
                    print(f"Auto-adjusting actors from {config['num_actors']} to {max_actors} based on available CPU resources")
                    config["num_actors"] = max_actors
        
        # Initialize Ray if not already done
        ray_address = config.get("ray_address")
        if not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True)
        
        # Create test environment to get spaces
        env = make_env()
        self.obs_space = env.observation_space
        self.action_space = env.action_space
        
        # Create learner
        self.learner = A3CLearner(
            CNNMCTSActorCriticPolicy,
            self.obs_space,
            self.action_space,
            config
        )
        
        # Create actors
        self.actors = []
        for i in range(config["num_actors"]):
            actor = A3CActor.remote(
                i,
                CNNMCTSActorCriticPolicy,
                self.obs_space,
                self.action_space,
                config
            )
            self.actors.append(actor)
        
        # Statistics
        self.start_time = time.time()
        self.total_steps = 0
        self.updates = 0
        
        # Game statistics
        self.game_stats = {
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
            "games_completed": 0,
            "avg_moves_per_game": 0
        }
        
        # Create metrics directory
        os.makedirs("data/metrics", exist_ok=True)
        self.metrics_file = os.path.join("data/metrics", "training_metrics.csv")
        
        # Initialize metrics file with header if it doesn't exist
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w") as f:
                f.write("timestamp,steps,updates,white_wins,black_wins,draws,white_win_rate,black_win_rate,draw_rate,avg_moves_per_game,games_completed,mean_reward\n")
    
    def update_game_stats(self, actor_stats_list):
        """Update aggregated game statistics from actor stats"""
        # Reset counters
        white_wins = 0
        black_wins = 0
        draws = 0
        games_completed = 0
        move_counts = []
        
        # Aggregate stats from all actors
        for stats in actor_stats_list:
            white_wins += stats.get("white_wins", 0)
            black_wins += stats.get("black_wins", 0)
            draws += stats.get("draws", 0)
            games_completed += stats.get("games_completed", 0)
            if "avg_moves_per_game" in stats and stats["avg_moves_per_game"] > 0:
                move_counts.append(stats["avg_moves_per_game"])
        
        # Update global stats
        self.game_stats["white_wins"] = white_wins
        self.game_stats["black_wins"] = black_wins
        self.game_stats["draws"] = draws
        self.game_stats["games_completed"] = games_completed
        self.game_stats["avg_moves_per_game"] = np.mean(move_counts) if move_counts else 0
        
        # Log to file
        self.log_metrics_to_file()
        
    def log_metrics_to_file(self):
        """Log metrics to CSV file"""
        # Calculate rates
        total_games = max(1, self.game_stats["games_completed"])
        white_win_rate = self.game_stats["white_wins"] / total_games * 100
        black_win_rate = self.game_stats["black_wins"] / total_games * 100
        draw_rate = self.game_stats["draws"] / total_games * 100
        
        # Get mean reward from learner
        mean_reward = self.learner.metrics.summary().get("mean_reward", 0)
        
        # Write to file
        with open(self.metrics_file, "a") as f:
            f.write(f"{time.time()},{self.total_steps},{self.updates},{self.game_stats['white_wins']},{self.game_stats['black_wins']},{self.game_stats['draws']},{white_win_rate},{black_win_rate},{draw_rate},{self.game_stats['avg_moves_per_game']},{self.game_stats['games_completed']},{mean_reward}\n")
    
    def print_game_stats(self):
        """Print game statistics to console"""
        total_games = max(1, self.game_stats["games_completed"])
        white_win_rate = self.game_stats["white_wins"] / total_games * 100
        black_win_rate = self.game_stats["black_wins"] / total_games * 100
        draw_rate = self.game_stats["draws"] / total_games * 100
        
        print("\n===== Game Statistics =====")
        print(f"Games completed: {self.game_stats['games_completed']}")
        print(f"White wins: {self.game_stats['white_wins']} ({white_win_rate:.2f}%)")
        print(f"Black wins: {self.game_stats['black_wins']} ({black_win_rate:.2f}%)")
        print(f"Draws: {self.game_stats['draws']} ({draw_rate:.2f}%)")
        print(f"Avg moves per game: {self.game_stats['avg_moves_per_game']:.1f}")
        print("==========================\n")

    def train(self):
        """Run training loop"""
        print(f"Starting A3C training with {len(self.actors)} actors")
        print(f"Training will run for {self.config['max_steps']} steps or until stopped")
        
        # Get initial weights
        weights = self.learner.get_weights()
        
        # Get initial stats from all actors
        stats_futures = [actor.get_stats.remote() for actor in self.actors]
        initial_stats = ray.get(stats_futures)
        self.update_game_stats(initial_stats)
        
        # Training loop
        try:
            while self.total_steps < self.config["max_steps"]:
                # Send weights to actors and get gradients asynchronously
                gradient_futures = [actor.compute_gradients.remote(weights) for actor in self.actors]
                
                # Use ray.wait to process gradients as they become available
                ready_gradient_futures, remaining_gradient_futures = ray.wait(
                    gradient_futures, num_returns=max(1, len(gradient_futures) // 2)
                )
                
                # Get gradients and infos from ready futures
                gradients_and_infos = ray.get(ready_gradient_futures)
                gradients_list = [g for g, _ in gradients_and_infos]
                infos = [info for _, info in gradients_and_infos]
                
                # Apply gradients
                update_id, metrics = self.learner.apply_gradients(gradients_list, infos)
                
                # Update weights
                weights = self.learner.get_weights()
                
                # Track progress
                actor_steps = sum(info.get("steps", 0) for info in infos)
                self.total_steps += actor_steps
                self.updates += 1
                
                # Log progress
                if self.updates % self.config["log_interval"] == 0:
                    # Get updated stats from all actors
                    stats_futures = [actor.get_stats.remote() for actor in self.actors]
                    all_stats = ray.get(stats_futures)
                    self.update_game_stats(all_stats)
                    
                    # Calculate training speed
                    elapsed = time.time() - self.start_time
                    steps_per_sec = self.total_steps / elapsed if elapsed > 0 else 0
                    
                    # Print progress
                    print(f"Update {self.updates}, Steps: {self.total_steps}, "
                          f"Steps/sec: {steps_per_sec:.1f}, "
                          f"Mean reward: {metrics.get('mean_reward', 0):.2f}")
                    
                    # Print game stats every 5 log intervals
                    if self.updates % (self.config["log_interval"] * 5) == 0:
                        self.print_game_stats()
                
                # Save model checkpoint
                if self.updates % self.config["save_interval"] == 0:
                    checkpoint_path = os.path.join(
                        self.config["save_dir"],
                        f"a3c_chess_model_{self.total_steps}_steps.pt"
                    )
                    self.learner.save_model(checkpoint_path)
                    print(f"Saved model checkpoint to {checkpoint_path}")
                    
                    # Also save a CSV with current game stats
                    stats_path = os.path.join(
                        self.config["save_dir"],
                        f"game_stats_{self.total_steps}_steps.csv"
                    )
                    with open(stats_path, "w") as f:
                        f.write("metric,value\n")
                        f.write(f"white_wins,{self.game_stats['white_wins']}\n")
                        f.write(f"black_wins,{self.game_stats['black_wins']}\n")
                        f.write(f"draws,{self.game_stats['draws']}\n")
                        f.write(f"games_completed,{self.game_stats['games_completed']}\n")
                        f.write(f"white_win_rate,{self.game_stats['white_wins']/max(1,self.game_stats['games_completed'])*100}\n")
                        f.write(f"black_win_rate,{self.game_stats['black_wins']/max(1,self.game_stats['games_completed'])*100}\n")
                        f.write(f"draw_rate,{self.game_stats['draws']/max(1,self.game_stats['games_completed'])*100}\n")
                        f.write(f"avg_moves_per_game,{self.game_stats['avg_moves_per_game']}\n")
        
        except KeyboardInterrupt:
            print("Training interrupted.")
        
        # Save final model
        final_path = os.path.join(self.config["save_dir"], "a3c_chess_model_final.pt")
        self.learner.save_model(final_path)
        
        # Get final stats from all actors
        stats_futures = [actor.get_stats.remote() for actor in self.actors]
        all_stats = ray.get(stats_futures)
        self.update_game_stats(all_stats)
        
        # Print final statistics
        self.print_game_stats()
        
        print("\nTraining completed:")
        print(f"Total steps: {self.total_steps}")
        print(f"Mean reward: {self.learner.metrics.summary().get('mean_reward', 0):.2f}")
        print(f"Total updates: {self.updates}")
        print(f"Total time: {time.time() - self.start_time:.1f} seconds")
    
    def shutdown(self):
        """Shutdown Ray"""
        ray.shutdown()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a chess model using Ray A3C')
    
    # Ray settings
    parser.add_argument('--ray_address', type=str, default=None,
                      help='Ray address (None for local, "auto" for cluster)')
    parser.add_argument('--num_actors', type=int, default=4,
                      help='Number of actor processes')
    parser.add_argument('--n_envs_per_actor', type=int, default=4,
                      help='Number of environments per actor')
    parser.add_argument('--auto_adjust_actors', action='store_true', default=True,
                      help='Automatically adjust number of actors based on available resources')
    
    # Training parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (cuda or cpu)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                      help='GAE lambda parameter')
    parser.add_argument('--ent_coef', type=float, default=0.1,
                      help='Entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                      help='Value function coefficient')
    parser.add_argument('--clip_range', type=float, default=0.2,
                      help='PPO clipping parameter')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                      help='Maximum norm for gradients')
    parser.add_argument('--trajectory_length', type=int, default=128,
                      help='Length of trajectory to collect before update')
    parser.add_argument('--max_steps', type=int, default=10000000,
                      help='Total number of steps to train for')
    parser.add_argument('--save_interval', type=int, default=50,
                      help='Interval (in updates) between model saves')
    parser.add_argument('--log_interval', type=int, default=5,
                      help='Interval (in updates) between logging')
    parser.add_argument('--save_dir', type=str, default='data/models',
                      help='Directory to save models')
    
    # MCTS parameters
    parser.add_argument('--mcts_sims', type=int, default=100,
                      help='Number of MCTS simulations (0 to disable)')
    parser.add_argument('--mcts_freq', type=float, default=0.2,
                      help='Frequency of using MCTS (0-1)')
    
    # Environment parameters
    parser.add_argument('--simple_test', action='store_true',
                      help='Use simplified chess positions for testing')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Create config dictionary from arguments
    config = vars(args)
    
    # Create save directory if it doesn't exist
    os.makedirs(config["save_dir"], exist_ok=True)
    
    # Create trainer
    trainer = A3CTrainer(config)
    
    # Run training
    try:
        trainer.train()
    finally:
        # Ensure Ray is shut down
        trainer.shutdown() 