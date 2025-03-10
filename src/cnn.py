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
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
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
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space.n)
        )

    def _build_value_net(self, features_dim):
        return nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
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
        self.device = getattr(policy_net, 'device', 'cpu')
        
    def search(self, state):
        root = Node(state)
        
        # Batch size for parallel evaluations
        batch_size = 8  # Can be increased based on GPU memory
        
        for sim_batch in range(0, self.num_simulations, batch_size):
            # Actual batch size for this iteration
            current_batch_size = min(batch_size, self.num_simulations - sim_batch)
            
            # Selection and expansion phases - prepare batch
            leaf_nodes = []
            for _ in range(current_batch_size):
                node = root
                search_path = [node]
                
                # Selection phase - find leaf node
                while not node.is_terminal and node.children:
                    action, node = node.select_child(self.c_puct)
                    search_path.append(node)
                
                # If leaf node is not terminal and not expanded, add to batch
                if not node.is_terminal and not node.children:
                    leaf_nodes.append((node, search_path))
            
            # Skip if no leaf nodes to evaluate
            if not leaf_nodes:
                continue
                
            # Prepare batch for network evaluation
            batch_obs = []
            for node, _ in leaf_nodes:
                try:
                    from custom_gym.chess_gym import canonical_encode_board_for_cnn
                    board_obs = canonical_encode_board_for_cnn(node.state)
                    board_obs_flat = board_obs.flatten()
                    legal_actions = node.state.legal_actions()
                    mask = np.zeros(node.state.num_actions(), dtype=np.float32)
                    mask[legal_actions] = 1.0
                    obs = np.concatenate([board_obs_flat, mask])
                    batch_obs.append(obs)
                except Exception as e:
                    print(f"Error preparing observation: {e}")
                    continue
            
            if not batch_obs:
                continue
                
            # Convert to tensor and evaluate batch
            try:
                with torch.no_grad():
                    batch_tensor = torch.FloatTensor(np.array(batch_obs)).to(self.device)
                    batch_values = []
                    batch_probs = []
                    
                    # Process in smaller chunks if needed
                    for i in range(0, len(batch_obs), 4):  # Process 4 at a time to avoid memory issues
                        chunk = batch_tensor[i:i+4]
                        
                        # Extract features
                        features = self.policy_net.extract_features(chunk)
                        
                        # Get values
                        values = self.policy_net.value_net(features)
                        batch_values.extend(values.cpu().numpy())
                        
                        # Get action probabilities
                        latent_pi = self.policy_net.policy_net(features)
                        
                        # Apply action masking
                        board_flat_size = self.policy_net.board_flat_size
                        action_masks = chunk[:, board_flat_size:]
                        illegal_masks = (action_masks < 0.5)
                        logits = latent_pi.masked_fill(illegal_masks, -1e8)
                        probs = F.softmax(logits, dim=-1)
                        
                        batch_probs.append(probs.cpu())
                    
                    if batch_probs:
                        batch_probs = torch.cat(batch_probs, dim=0)
            except Exception as e:
                print(f"Error in batch evaluation: {e}")
                continue
            
            # Expansion and backpropagation for each leaf node
            for i, (node, search_path) in enumerate(leaf_nodes):
                if i >= len(batch_values):  # Safety check
                    continue
                    
                value = batch_values[i]
                probs = batch_probs[i] if i < len(batch_probs) else None
                
                # Expand node with evaluated probabilities
                try:
                    legal_actions = node.state.legal_actions()
                    for action in legal_actions:
                        if action not in node.children and probs is not None:
                            next_state = node.state.clone()
                            next_state.apply_action(action)
                            node.children[action] = Node(
                                state=next_state,
                                prior=probs[action].item(),
                                parent=node,
                                root_player=node.root_player
                            )
                except Exception as e:
                    print(f"Error expanding node: {e}")
                    continue
                
                # Backpropagate value
                for path_node in reversed(search_path):
                    path_node.visit_count += 1
                    path_node.total_value += value
                    path_node.mean_value = path_node.total_value / path_node.visit_count
                    
                    # Flip value for opponent's perspective
                    value = -value
        
        # Select best action from root
        if not root.children:
            # If no children, return a random legal action
            legal_actions = root.state.legal_actions()
            if legal_actions:
                return np.random.choice(legal_actions)
            return 0  # Fallback
            
        # Use visit count to determine best action (most robust)
        best_action = None
        most_visits = -1
        
        for action, child in root.children.items():
            if child.visit_count > most_visits:
                most_visits = child.visit_count
                best_action = action
                
        return best_action


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
        
        # Initialize opponent policies list with a random policy
        self.opponent_policies = []
        
        # Add a random policy to the opponent pool
        try:
            # Create a random policy that selects random legal moves
            class RandomPolicy:
                def __init__(self, device='cpu'):
                    self.device = device
                
                def __call__(self, obs, deterministic=False):
                    # Extract action mask from observation
                    if isinstance(obs, dict) and 'action_mask' in obs:
                        action_mask = obs['action_mask']
                    else:
                        # Fallback for non-dictionary observations
                        board_size = 832  # 13*8*8
                        action_mask = obs[:, board_size:]
                    
                    # Find legal actions (where mask is 1)
                    batch_size = action_mask.shape[0]
                    actions = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
                    values = torch.zeros(batch_size, device=self.device)
                    log_probs = torch.zeros(batch_size, device=self.device)
                    
                    # For each observation in the batch
                    for i in range(batch_size):
                        legal_actions = torch.where(action_mask[i] > 0.5)[0]
                        if len(legal_actions) > 0:
                            # Choose a random legal action
                            random_idx = torch.randint(0, len(legal_actions), (1,), device=self.device)
                            actions[i] = legal_actions[random_idx]
                    
                    return actions, values, log_probs
                
                def set_training_mode(self, mode):
                    # Random policy doesn't need training mode
                    pass
            
            # Create and add the random policy
            random_policy = RandomPolicy(device=self.device)
            self.opponent_policies.append(random_policy)
            print("Added random policy to opponent pool")
        except Exception as e:
            print(f"Error creating random policy: {e}")
        
        # Initialize current opponent policies only if policy is available
        # This handles the case when loading from a checkpoint
        if hasattr(self, 'policy') and self.policy is not None:
            self.current_opponent_policies = [self.policy] * self.n_envs
        else:
            # Will be initialized later when policy becomes available
            self.current_opponent_policies = None
            
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
            except Exception as e:
                print(f"Warning: Could not get current player: {e}")
                
        # Initialize policy if not already done
        if self.policy.mcts is None:
            self.policy.mcts = MCTS(self.policy, num_simulations=50, c_puct=2.0)
            
        # Initialize current_opponent_policies if it's None
        if self.current_opponent_policies is None:
            self.current_opponent_policies = [self.policy] * self.n_envs
            
        use_mcts_in_training = False

        while n_steps < n_rollout_steps:
            actions = np.zeros(self.n_envs, dtype=int)
            batch_values = torch.zeros(self.n_envs, device=self.device)
            batch_log_probs = torch.zeros(self.n_envs, device=self.device)

            agent_turn_indices = [e for e in range(self.n_envs) if current_players[e] == self.agent_record_player[e]]
            opponent_turn_indices = [e for e in range(self.n_envs) if current_players[e] != self.agent_record_player[e]]

            if agent_turn_indices:
                if not use_mcts_in_training:
                    try:
                        agent_boards = self._last_obs['board'][agent_turn_indices]
                        agent_masks = self._last_obs['action_mask'][agent_turn_indices]
                        
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
                            self.last_agent_step[e] = rollout_buffer.pos
                    except Exception as ex:
                        print(f"Error in agent policy evaluation: {ex}")
                        for e in agent_turn_indices:
                            legal_actions = env.get_attr('state', indices=[e])[0].legal_actions()
                            if legal_actions:
                                actions[e] = np.random.choice(legal_actions)
                else:
                    # MCTS implementation (unchanged)
                    pass

            if opponent_turn_indices:
                try:
                    policy_groups = {}
                    for e in opponent_turn_indices:
                        policy = self.current_opponent_policies[e]
                        if policy not in policy_groups:
                            policy_groups[policy] = []
                        policy_groups[policy].append(e)
                    
                    for policy, indices in policy_groups.items():
                        opp_boards = self._last_obs['board'][indices]
                        opp_masks = self._last_obs['action_mask'][indices]
                        
                        board_tensor = torch.as_tensor(opp_boards).to(self.device)
                        mask_tensor = torch.as_tensor(opp_masks).to(self.device)
                        obs_dict = {'board': board_tensor, 'action_mask': mask_tensor}
                        
                        with torch.no_grad():
                            opp_actions, _, _ = policy(obs_dict, deterministic=False)
                            opp_actions = opp_actions.cpu().numpy()
                        
                        for i, e in enumerate(indices):
                            actions[e] = opp_actions[i]
                except Exception as ex:
                    print(f"Error in opponent policy evaluation: {ex}")
                    for env_idx in opponent_turn_indices:
                        try:
                            legal_actions = env.get_attr('state', indices=[env_idx])[0].legal_actions()
                            if legal_actions:
                                actions[env_idx] = np.random.choice(legal_actions)
                        except Exception:
                            actions[env_idx] = 0

            try:
                new_obs, rewards, dones, infos = env.step(actions)
                if not isinstance(infos, list):
                    infos = [infos] * self.n_envs
                        
                current_players = [info.get("current_player", 0) for info in infos]
            except Exception as ex:
                print(f"Error stepping environment: {ex}")
                try:
                    new_obs = env.reset()
                    rewards = np.zeros(self.n_envs)
                    dones = np.ones(self.n_envs, dtype=bool)
                    infos = [{} for _ in range(self.n_envs)]
                    current_players = [0 for _ in range(self.n_envs)]
                except Exception as reset_error:
                    print(f"Error resetting environment: {reset_error}")
                    return False

            callback.update_locals(locals())
            if not callback.on_step():
                return False

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

            for env_idx in range(self.n_envs):
                if dones[env_idx]:
                    if self.last_agent_step[env_idx] is not None:
                        last_pos = self.last_agent_step[env_idx]
                        if infos[env_idx].get('is_draw', False):
                            rollout_buffer.rewards[last_pos, env_idx] = 0
                        elif infos[env_idx].get('white_won', False):
                            rollout_buffer.rewards[last_pos, env_idx] = 1 if self.agent_record_player[env_idx] == 1 else -1
                        elif infos[env_idx].get('black_won', False):
                            rollout_buffer.rewards[last_pos, env_idx] = 1 if self.agent_record_player[env_idx] == 0 else -1
                        else:
                            rollout_buffer.rewards[last_pos, env_idx] = 0
                        
                        try:
                            # Fix for VecMonitor.reset() indices error
                            single_env_reset = env.env_method('reset', indices=[env_idx])
                            if single_env_reset and len(single_env_reset) > 0:
                                new_obs_e = single_env_reset[0]
                                new_info = {}
                                if isinstance(new_obs_e, tuple) and len(new_obs_e) > 1:
                                    new_obs_e, new_info = new_obs_e
                                
                                # Update observation for this environment
                                for key in self._last_obs:
                                    if key in new_obs_e:
                                        self._last_obs[key][env_idx] = new_obs_e[key]
                                
                                self.last_agent_step[env_idx] = None
                                current_players[env_idx] = new_info.get("current_player", 0)
                        except Exception as ex:
                            print(f"Error resetting environment {env_idx}: {ex}")
                    else:
                        current_players[env_idx] = infos[env_idx].get("current_player", current_players[env_idx])           
                    
                    # Fix for UnboundLocalError
                    if self.opponent_policies and np.random.rand() < self.opponent_pool_prob:
                        self.current_opponent_policies[env_idx] = np.random.choice(self.opponent_policies)
                    else:
                        self.current_opponent_policies[env_idx] = self.policy

            self._last_obs = new_obs
            self._last_episode_starts = batch_dones
            self.num_timesteps += self.n_envs
            n_steps += 1

            if rollout_buffer.pos >= n_rollout_steps:
                break

        try:
            with torch.no_grad():
                obs_tensor = {
                    key: torch.as_tensor(value, device=self.device)
                    for key, value in self._last_obs.items()
                }
                last_values = self.policy.predict_values(obs_tensor)
            rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
        except Exception as ex:
            print(f"Error computing returns and advantage: {ex}")
            last_values = torch.zeros(self.n_envs, device=self.device)
            rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)

        return True


def create_cnn_mcts_ppo(env, tensorboard_log, device='cpu', checkpoint=None):
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        
    policy_kwargs = {
        'activation_fn': nn.ReLU,
        'net_arch': None,  # Use custom CNN architecture
        'normalize_images': False,  # Already normalized in our preprocessing
    }
    
    n_steps = 2048
    n_envs = env.num_envs
    total_steps = n_steps * n_envs
    
    batch_size = min(2048, n_steps)
    
    print(f"Training with {n_envs} environments, {n_steps} steps per environment")
    print(f"Total steps per iteration: {total_steps}, batch size: {batch_size}")
    
    if checkpoint:
        print(f'Checkpoint: {checkpoint} loaded')
        model = MCTSPPO.load(
            checkpoint, 
            env=env, 
            policy=CNNMCTSActorCriticPolicy,
            tensorboard_log=tensorboard_log, 
            verbose=1,
            learning_rate=3e-5,
            n_steps=n_steps,        # Steps to collect per environment
            batch_size=batch_size,  # Batch size for updates
            n_epochs=4,             # Number of passes through the batch
            gamma=0.99,
            device=device,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            policy_kwargs=policy_kwargs,
        )
    else:
        model = MCTSPPO(
            CNNMCTSActorCriticPolicy,
            env,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=3e-5,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=4, 
            gamma=0.99,
            device=device,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
        )
    
    return model