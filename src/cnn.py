import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
    
    

class CNNMCTSActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CNNMCTSActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )

        self.board_channels = 13
        self.board_size = 8
        self.board_flat_size = self.board_channels * self.board_size * self.board_size

        self.features_extractor = self._build_cnn_extractor()
        self.features_dim = 832
        self.policy_net = self._build_policy_net(self.features_dim)
        self.value_net = self._build_value_net(self.features_dim)

        self.mcts = None
        self.training = True

    def _build_cnn_extractor(self):
        cnn = nn.Sequential(
            nn.Conv2d(self.board_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        fc = nn.Sequential(
            nn.Linear(256, 832),
            nn.ReLU()
        )
        return nn.Sequential(cnn, fc)

    def _build_policy_net(self, features_dim):
        return nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_space.n)
        )

    def _build_value_net(self, features_dim):
        return nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def extract_features(self, obs, features_extractor=None):
        if features_extractor is None:
            features_extractor = self.features_extractor

        if isinstance(obs, dict):
            board_flat = obs["board"]
        else:
            board_flat = obs[:, :self.board_flat_size]

        batch_size = board_flat.shape[0]
        board_3d = board_flat.reshape(batch_size, self.board_channels, self.board_size, self.board_size)

        return features_extractor(board_3d)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi = self.policy_net(features)
        values = self.value_net(features)

        action_mask = self._get_action_mask(obs)
        illegal_actions_mask = (action_mask < 0.5)
        logits = latent_pi.masked_fill(illegal_actions_mask, -1e8)

        distribution = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(distribution.probs, dim=1)
        else:
            actions = distribution.sample()
        log_probs = distribution.log_prob(actions)

        return actions, values, log_probs

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi = self.policy_net(features)
        values = self.value_net(features)

        action_mask = self._get_action_mask(obs)
        illegal_actions_mask = (action_mask < 0.5)
        logits = latent_pi.masked_fill(illegal_actions_mask, -1e8)

        distribution = torch.distributions.Categorical(logits=logits)
        log_probs = distribution.log_prob(actions.squeeze())
        entropy = distribution.entropy()

        return values, log_probs, entropy

    def _get_action_mask(self, obs):
        return obs["action_mask"]

    def predict_values(self, obs):
        features = self.extract_features(obs)
        values = self.value_net(features)
        return values


class Node:
    def __init__(self, state, prior=1.0, parent=None, root_player=None):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.mean_value = 0
        self.is_terminal = state.is_terminal()
        self.root_player = root_player if root_player is not None else state.current_player()

    def select_child(self, c_puct=1.0):
        best_score = float('-inf')
        best_action = None
        
        for action, child in self.children.items():
            score = child.mean_value + c_puct * child.prior * \
                   math.sqrt(self.visit_count) / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action, self.children[best_action]

    def expand(self, policy_net):
        from custom_gym.chess_gym import canonical_encode_board_for_cnn
        
        # Get board representation
        board_obs = canonical_encode_board_for_cnn(self.state)
        board_obs_flat = board_obs.flatten()
        
        # Get legal actions
        legal_actions = self.state.legal_actions()
        
        # Create action mask
        mask = np.zeros(4672, dtype=np.float32)
        mask[legal_actions] = 1.0
        
        # Combine for full observation
        obs = np.concatenate([board_obs_flat, mask])
        
        # Get policy evaluation
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(policy_net.device)
            actions, values, _ = policy_net(obs_tensor, deterministic=False)
            
            # Get probabilities
            action_mask = obs_tensor[:, policy_net.board_flat_size:]
            features = policy_net.extract_features(obs_tensor)
            latent_pi, _ = policy_net.mlp_extractor(features)
            logits = policy_net.action_net(latent_pi)
            illegal_mask = (action_mask < 0.5)
            logits = logits.masked_fill(illegal_mask, -1e8)
            probs = F.softmax(logits, dim=-1).squeeze(0)
        
        # Create child nodes
        for action in legal_actions:
            if action not in self.children:
                next_state = self.state.clone()
                next_state.apply_action(action)
                self.children[action] = Node(
                    state=next_state,
                    prior=probs[action].item(),
                    parent=self,
                    root_player=self.root_player
                )
        
        return values.item()

    def backpropagate(self, value):
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count
        if self.parent:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, policy_net, num_simulations=100, c_puct=1.0):
        self.policy_net = policy_net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def search(self, state):
        root = Node(state)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection phase
            while node.children and not node.is_terminal:
                action, node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # Expansion and evaluation
            if not node.is_terminal:
                value = node.expand(self.policy_net)
                if node.state.current_player() != root.root_player:
                    value = -value
            else:
                value = node.state.rewards()[root.root_player]
            
            # Backpropagation
            for node in reversed(search_path):
                node.backpropagate(value)
        
        # Choose move based on visit counts
        visits = {action: child.visit_count for action, child in root.children.items()}
        if not visits:
            legal_actions = state.legal_actions()
            if legal_actions:
                return np.random.choice(legal_actions)
            return 0
            
        top_k = sorted(visits.items(), key=lambda x: x[1], reverse=True)[:5]
        actions, counts = zip(*top_k)
        probs = np.array(counts) / sum(counts)
        return np.random.choice(actions, p=probs)


class MCTSPPO(PPO):
    """
    PPO with MCTS integration and self-play for chess
    Includes custom rollout collection logic
    """
    def __init__(self, policy, env, **kwargs):
        super().__init__(policy, env, **kwargs)
        
        self.last_agent_step = [None] * self.n_envs
        
        if self.n_envs % 2 == 0:
            half = self.n_envs // 2
            self.agent_record_player = np.array([0] * half + [1] * half)
        else:
            half = self.n_envs // 2
            extra = np.random.choice([0, 1])
            self.agent_record_player = np.array([0] * half + [1] * half + [extra])
        
        self.opponent_policies = []
        self.current_opponent_policies = [self.policy] * self.n_envs
        self.opponent_pool_prob = kwargs.get('opponent_pool_prob', 0.5)
    
    def collect_rollouts(self, env, callback, rollout_buffer: RolloutBuffer, n_rollout_steps):
        assert self._last_obs is not None, "No previous observation"
        n_steps = 0
        rollout_buffer.reset()
        
        infos = [{} for _ in range(self.n_envs)]
        current_players = [0 for _ in range(self.n_envs)]
        
        if hasattr(env, 'env_method'):
            try:
                current_players = env.env_method('get_current_player')
            except Exception:
                pass

        while n_steps < n_rollout_steps:
            actions = np.zeros(self.n_envs, dtype=int)
            batch_values = torch.zeros(self.n_envs, device=self.device)
            batch_log_probs = torch.zeros(self.n_envs, device=self.device)

            # Identify turns
            agent_turn_indices = [e for e in range(self.n_envs) if current_players[e] == self.agent_record_player[e]]
            opponent_turn_indices = [e for e in range(self.n_envs) if current_players[e] != self.agent_record_player[e]]

            # Agent's turns
            if agent_turn_indices:
                # Extract board and action mask for agent's turns from batched arrays
                agent_boards = self._last_obs['board'][agent_turn_indices]  # Shape: (num_agent_turns, 832)
                agent_masks = self._last_obs['action_mask'][agent_turn_indices]  # Shape: (num_agent_turns, 4672)
                
                # Convert to tensors
                board_tensor = torch.as_tensor(agent_boards).to(self.device)
                mask_tensor = torch.as_tensor(agent_masks).to(self.device)
                obs_dict = {'board': board_tensor, 'action_mask': mask_tensor}
                
                with torch.no_grad():
                    agent_actions, agent_values, agent_log_probs = self.policy(obs_dict)
                
                agent_actions = agent_actions.cpu().numpy()
                for i, e in enumerate(agent_turn_indices):
                    actions[e] = agent_actions[i]
                    batch_values[e] = agent_values[i]
                    batch_log_probs[e] = agent_log_probs[i]

            # Opponent's turns
            for e in opponent_turn_indices:
                opponent_policy = self.current_opponent_policies[e]
                # Extract observation for this specific environment
                obs_e = {
                    'board': self._last_obs['board'][e:e+1],  # Shape: (1, 832)
                    'action_mask': self._last_obs['action_mask'][e:e+1]  # Shape: (1, 4672)
                }
                with torch.no_grad():
                    obs_tensor = {
                        'board': torch.as_tensor(obs_e['board']).to(self.device),
                        'action_mask': torch.as_tensor(obs_e['action_mask']).to(self.device)
                    }
                    action, _, _ = opponent_policy(obs_tensor, deterministic=False)
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                actions[e] = action.item() if hasattr(action, 'item') else action

            # Step environment
            new_obs, rewards, dones, infos = env.step(actions)
            if not isinstance(infos, list):
                infos = [infos] * self.n_envs
                    
            current_players = [info.get("current_player", 0) for info in infos]

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            # Fill batch data
            # Use self._last_obs directly as it's in the correct dictionary-of-arrays format
            batch_obs = self._last_obs
            batch_actions = actions
            batch_rewards = rewards
            batch_episode_starts = self._last_episode_starts
            batch_dones = dones

            rollout_buffer.add(
                batch_obs,
                batch_actions.reshape(self.n_envs, 1),
                batch_rewards,
                batch_episode_starts,
                batch_values,
                batch_log_probs
            )

            for e in range(self.n_envs):
                if dones[e]:
                    if self.last_agent_step[e] is not None:
                        last_pos = self.last_agent_step[e]
                        if infos[e].get('is_draw', False):
                            rollout_buffer.rewards[last_pos, e] = 0
                        elif infos[e].get('white_won', False):
                            rollout_buffer.rewards[last_pos, e] = 1 if self.agent_record_player[e] == 1 else -1
                        elif infos[e].get('black_won', False):
                            rollout_buffer.rewards[last_pos, e] = 1 if self.agent_record_player[e] == 0 else -1
                        else:
                            rollout_buffer.rewards[last_pos, e] = 0
                        
                        new_obs_e, new_info = env.reset(indices=[e])
                        # Update self._last_obs for the reset environment
                        current_players[e] = 1
                        self._last_obs[e] = new_obs[e]  # Use the reset obs from step
                        self.last_agent_step[e] = None
                        current_players[e] = new_info.get("current_player", 0)
                    else:
                        current_players[e] = infos[e].get("current_player", current_players[e])           
                    
                    if self.opponent_policies and np.random.rand() < self.opponent_pool_prob:
                        self.current_opponent_policies[e] = np.random.choice(self.opponent_policies)
                    else:
                        self.current_opponent_policies[e] = self.policy

            self._last_obs = new_obs
            self._last_episode_starts = batch_dones
            self.num_timesteps += self.n_envs
            n_steps += 1

            if rollout_buffer.pos >= n_rollout_steps:
                break

        with torch.no_grad():
            obs_tensor = {
                key: torch.as_tensor(value, device=self.device)
                for key, value in self._last_obs.items()
            }
            # Now pass the tensor dictionary to predict_values
            last_values = self.policy.predict_values(obs_tensor)
        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)

        return True


def create_cnn_mcts_ppo(env, tensorboard_log, device='cpu', checkpoint=None):
    if checkpoint:
        print(f'Checkpoint: {checkpoint} loaded')
        model = MCTSPPO.load(
            checkpoint, 
            env=env, 
            policy=CNNMCTSActorCriticPolicy,
            tensorboard_log=tensorboard_log, 
            verbose=1,
            learning_rate=3e-5,
            n_steps=2048,
            batch_size=128,
            n_epochs=4,
            gamma=0.99,
            device=device,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
        )
    else:
        model = MCTSPPO(
            CNNMCTSActorCriticPolicy,
            env,
            tensorboard_log=tensorboard_log,
            policy_kwargs=None,
            verbose=1,
            learning_rate=3e-5,
            n_steps=2048,
            batch_size=128,
            n_epochs=4,
            gamma=0.99,
            device=device,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
        )
    
    return model