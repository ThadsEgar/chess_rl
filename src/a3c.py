import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import time
from collections import deque
import argparse

from custom_gym.chess_gym import ChessEnv, ActionMaskWrapper
from src.cnn import CNNMCTSActorCriticPolicy, MCTS, Node
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import explained_variance

# Global shared model
def make_env(rank, seed=0, simple_test=False, white_advantage=None):
    def _init():
        env = ChessEnv(simple_test=simple_test, white_advantage=white_advantage)
        env = ActionMaskWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init

class A3CWorker(mp.Process):
    def __init__(self, global_model, optimizer, rank, args, device='cpu'):
        super(A3CWorker, self).__init__()
        self.global_model = global_model
        self.optimizer = optimizer
        self.rank = rank
        self.args = args
        self.device = device
        
        # Local model
        self.local_model = CNNMCTSActorCriticPolicy(
            global_model.observation_space,
            global_model.action_space,
            lambda _: args.learning_rate  # Constant LR for worker
        ).to(device)
        
        # Copy parameters from global model
        self.sync_with_global()
        
        # Local environment
        self.env = SubprocVecEnv([make_env(rank, seed=rank) for _ in range(args.n_envs_per_worker)])
        
        # MCTS setup if needed
        if args.mcts_sims > 0:
            self.mcts = MCTS(self.local_model, num_simulations=args.mcts_sims, c_puct=2.0)
            self.use_mcts = True
        else:
            self.use_mcts = False
            
        # Tracking metrics
        self.episode_rewards = []
        self.total_steps = 0
        
    def sync_with_global(self):
        """Copy parameters from global model to local model"""
        self.local_model.load_state_dict(self.global_model.state_dict())
    
    def update_global(self, loss):
        """Update global model with gradients from local loss"""
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Backpropagate loss
        loss.backward()
        
        # Clip gradients
        if self.args.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.args.max_grad_norm)
        
        # Copy gradients to global model
        for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad.clone()
            else:
                global_param.grad += local_param.grad.clone()
        
        # Update global model
        self.optimizer.step()
        
        # Sync back to local model
        self.sync_with_global()
    
    def collect_trajectory(self, max_steps=200):
        """Collect a trajectory of experiences"""
        states = []
        actions = []
        values = []
        log_probs = []
        rewards = []
        masks = []
        action_masks = []
        
        # Reset environment
        obs = self.env.reset()
        
        for step in range(max_steps):
            # Convert observation to tensor
            board_tensor = torch.as_tensor(obs['board']).to(self.device)
            mask_tensor = torch.as_tensor(obs['action_mask']).to(self.device)
            obs_dict = {'board': board_tensor, 'action_mask': mask_tensor}
            
            # Store action mask
            action_masks.append(obs['action_mask'])
            
            # Get action, value, and log probability
            if self.use_mcts and np.random.random() < self.args.mcts_freq:
                # Use MCTS for action selection
                mcts_actions = []
                mcts_values = []
                mcts_log_probs = []
                
                for i in range(self.env.num_envs):
                    state = self.env.envs[i].unwrapped.board.copy()
                    player = 1 if state.turn else -1
                    root = Node(state=state, root_player=player)
                    
                    # Run MCTS search
                    action, value = self.mcts.search(root)
                    
                    # Get log_prob for the action from the policy network
                    single_board = obs_dict['board'][i:i+1]
                    single_mask = obs_dict['action_mask'][i:i+1]
                    single_obs = {'board': single_board, 'action_mask': single_mask}
                    
                    with torch.no_grad():
                        logits = self.local_model._get_action_dist(single_obs)[0]
                        action_dist = Categorical(logits=logits)
                        log_prob = action_dist.log_prob(torch.tensor(action, device=self.device))
                    
                    mcts_actions.append(action)
                    mcts_values.append(value)
                    mcts_log_probs.append(log_prob.item())
                
                action = np.array(mcts_actions)
                value = np.array(mcts_values)
                log_prob = np.array(mcts_log_probs)
            else:
                # Use standard policy network
                with torch.no_grad():
                    logits, values_tensor = self.local_model._get_action_dist_and_value(obs_dict)
                    action_dists = [Categorical(logits=logits[i]) for i in range(self.env.num_envs)]
                    actions_tensor = torch.stack([dist.sample() for dist in action_dists])
                    log_probs_tensor = torch.stack([dist.log_prob(actions_tensor[i]) for i, dist in enumerate(action_dists)])
                
                action = actions_tensor.cpu().numpy()
                value = values_tensor.cpu().numpy()
                log_prob = log_probs_tensor.cpu().numpy()
            
            # Store experience
            states.append(obs)
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Store reward and mask
            rewards.append(reward)
            masks.append(1 - done)
            
            # Update observation
            obs = next_obs
            
            # Track total steps
            self.total_steps += self.env.num_envs
            
            # If episode is done, record reward
            for i, d in enumerate(done):
                if d:
                    self.episode_rewards.append(info[i].get('episode', {}).get('r', 0))
                    
            # Check if max steps reached or all environments done
            if step == max_steps - 1 or np.all(done):
                break
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        
        # Get final value for bootstrapping
        if len(obs) > 0:
            board_tensor = torch.as_tensor(obs['board']).to(self.device)
            mask_tensor = torch.as_tensor(obs['action_mask']).to(self.device)
            obs_dict = {'board': board_tensor, 'action_mask': mask_tensor}
            
            with torch.no_grad():
                _, next_value = self.local_model._get_action_dist_and_value(obs_dict)
                next_value = next_value.cpu().numpy()
        else:
            next_value = np.zeros(self.env.num_envs)
        
        # Calculate returns and advantages using GAE
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.args.gamma * next_value * masks[i] - values[i]
            gae = delta + self.args.gamma * self.args.gae_lambda * masks[i] * gae
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
            next_value = values[i]
        
        return {
            'states': states,
            'actions': actions,
            'returns': returns,
            'advantages': advantages,
            'log_probs': log_probs,
            'action_masks': action_masks
        }
    
    def train(self, trajectory):
        """Train on collected trajectory and update global model"""
        states = trajectory['states']
        actions = trajectory['actions']
        returns = trajectory['returns']
        advantages = trajectory['advantages']
        old_log_probs = trajectory['log_probs']
        action_masks = trajectory['action_masks']
        
        # Convert to tensors
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Calculate loss for each mini-batch
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            return_val = returns_tensor[i]
            advantage = advantages_tensor[i]
            old_log_prob = old_log_probs_tensor[i]
            
            # Convert observation to tensor
            board_tensor = torch.as_tensor(state['board']).to(self.device)
            mask_tensor = torch.as_tensor(state['action_mask']).to(self.device)
            obs_dict = {'board': board_tensor, 'action_mask': mask_tensor}
            
            # Forward pass
            logits, value = self.local_model._get_action_dist_and_value(obs_dict)
            
            # Calculate action probabilities
            action_dist = Categorical(logits=logits)
            entropy = action_dist.entropy().mean()
            
            # Convert action to tensor
            action_tensor = torch.tensor(action, dtype=torch.long).to(self.device)
            
            # Calculate log probability
            log_prob = action_dist.log_prob(action_tensor)
            
            # Calculate policy loss (PPO style with clipping)
            ratio = torch.exp(log_prob - old_log_prob)
            policy_loss_1 = -advantage * ratio
            policy_loss_2 = -advantage * torch.clamp(ratio, 1.0 - self.args.clip_range, 1.0 + self.args.clip_range)
            policy_loss = torch.maximum(policy_loss_1, policy_loss_2).mean()
            
            # Calculate value loss
            value_loss = F.mse_loss(value, return_val)
            
            # Append losses
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropy_losses.append(-entropy)  # Negative because we want to maximize entropy
        
        # Calculate total loss
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy_loss = torch.stack(entropy_losses).mean()
        
        total_loss = policy_loss + self.args.vf_coef * value_loss + self.args.ent_coef * entropy_loss
        
        # Update global model
        self.update_global(total_loss)
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def run(self):
        """Main worker loop"""
        print(f"Worker {self.rank} starting")
        
        while self.total_steps < self.args.max_steps:
            # Collect trajectory
            trajectory = self.collect_trajectory(max_steps=self.args.trajectory_length)
            
            # Train on trajectory
            loss_stats = self.train(trajectory)
            
            # Log progress periodically
            if self.rank == 0 and len(self.episode_rewards) > 0 and len(self.episode_rewards) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                print(f"Step {self.total_steps}: Mean reward: {mean_reward:.2f}, "
                      f"Policy loss: {loss_stats['policy_loss']:.4f}, "
                      f"Value loss: {loss_stats['value_loss']:.4f}, "
                      f"Entropy: {-loss_stats['entropy_loss']:.4f}")
        
        print(f"Worker {self.rank} finished after {self.total_steps} steps")

def init_shared_model(env, args):
    """Initialize the global shared model"""
    model = CNNMCTSActorCriticPolicy(
        env.observation_space,
        env.action_space,
        lambda _: args.learning_rate
    ).to(args.device)
    
    model.share_memory()  # Enable shared memory for multiprocessing
    return model

def train_a3c(args):
    """Main function to train using A3C"""
    # Set up environment for getting spaces
    env = make_env(0)()
    
    # Create global model
    global_model = init_shared_model(env, args)
    
    # Create optimizer
    optimizer = optim.Adam(global_model.parameters(), lr=args.learning_rate)
    
    # Set up distributed training with multiprocessing
    processes = []
    
    # Create workers
    for rank in range(args.num_workers):
        worker = A3CWorker(
            global_model, 
            optimizer, 
            rank, 
            args,
            device=args.device
        )
        worker.daemon = True  # Terminate worker when main process terminates
        processes.append(worker)
        worker.start()
    
    # Wait for all workers to finish
    for p in processes:
        p.join()
    
    # Save the trained model
    torch.save(global_model.state_dict(), os.path.join(args.save_dir, "a3c_chess_model.pt"))
    print("Training completed and model saved")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a chess model using A3C')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--n_envs_per_worker', type=int, default=4,
                        help='Number of environments per worker')
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
    parser.add_argument('--mcts_sims', type=int, default=100,
                        help='Number of MCTS simulations (0 to disable)')
    parser.add_argument('--mcts_freq', type=float, default=0.2,
                        help='Frequency of using MCTS (0-1)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='data/models',
                        help='Directory to save model')
    return parser.parse_args()

if __name__ == "__main__":
    # Initialize multiprocessing support
    mp.set_start_method('spawn')
    
    # Parse arguments
    args = parse_arguments()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Train using A3C
    train_a3c(args) 