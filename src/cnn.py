import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance
import time
from gymnasium import spaces

# Custom Layer Normalization for 2D convolutional features
class LayerNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_features, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + self.eps
        normalized = (x - mean) / std
        return self.weight * normalized + self.bias

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
        
        # Initialize weights properly for training without normalization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Kaiming initialization for better training without normalization"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)

    def _build_cnn_extractor(self):
        cnn = nn.Sequential(
            nn.Conv2d(self.board_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
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

    def init_mcts(self, num_simulations=100, c_puct=2.0):
        """Initialize MCTS for policy improvement"""
        if self.mcts is None:
            self.mcts = MCTS(self, num_simulations=num_simulations, c_puct=c_puct)
            print(f"MCTS initialized with {num_simulations} simulations")
        return self.mcts


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
        
        try:
            # Get board representation
            board_obs = canonical_encode_board_for_cnn(self.state)
            board_obs_flat = board_obs.flatten()
            
            # Get legal actions
            legal_actions = self.state.legal_actions()
            
            # Create action mask (fixed size for chess: 4672)
            mask = np.zeros(4672, dtype=np.float32)
            mask[legal_actions] = 1.0
            
            # Combine for full observation
            obs = np.concatenate([board_obs_flat, mask])
            
            # Get policy evaluation
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(policy_net.device)
                # Create Dict observation for policy
                obs_dict = {
                    'board': obs_tensor[:, :832],  # First 832 elements are the board representation
                    'action_mask': obs_tensor[:, 832:]  # Rest is the action mask
                }
                
                # Get action distribution and value
                actions, values, _ = policy_net(obs_dict, deterministic=False)
                
                # Calculate probabilities for each action
                logits = policy_net.policy_net(policy_net.extract_features(obs_dict))
                illegal_actions_mask = (obs_dict['action_mask'] < 0.5)
                logits = logits.masked_fill(illegal_actions_mask, -1e8)
                probs = F.softmax(logits, dim=-1).squeeze(0)
                
                # For legal actions, create child nodes
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
        except Exception as e:
            print(f"Error in Node.expand: {str(e)}")
            return 0.0

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
        print(f"MCTS initialized with {num_simulations} simulations using device: {self.device}")
    
    def batch_search(self, states):
        """
        Perform MCTS search on multiple states in parallel.
        
        Args:
            states: List of game states to search
            
        Returns:
            List of best actions for each state
        """
        try:
            if not states:
                return []
                
            start_time = time.time()
            num_states = len(states)
            if num_states > 1:
                print(f"Starting batch MCTS search for {num_states} states with {self.num_simulations} simulations each")
                
            # Create a root node for each state
            roots = [Node(state) for state in states]
            
            # For efficiency, use a larger batch size when processing many states
            eval_batch_size = min(32, max(16, num_states * 4))
            
            # Perform the specified number of simulations
            for sim_iter in range(self.num_simulations):
                # Gather leaf nodes from all search trees
                all_leaves = []
                all_paths = []
                
                # Selection phase - done separately for each tree
                for root_idx, root in enumerate(roots):
                    if root.is_terminal:
                        continue
                        
                    node = root
                    search_path = [node]
                    
                    # Find a leaf node using MCTS selection
                    while not node.is_terminal and node.children:
                        action, node = node.select_child(self.c_puct)
                        search_path.append(node)
                    
                    # If we found an unexpanded leaf, add it to the batch
                    if not node.is_terminal and not node.children:
                        all_leaves.append((root_idx, node, search_path))
                
                # Log progress for large batches
                if num_states > 4 and sim_iter % 10 == 0 and sim_iter > 0:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / sim_iter) * (self.num_simulations - sim_iter)
                    print(f"MCTS progress: {sim_iter}/{self.num_simulations} simulations ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")
                
                if not all_leaves:
                    continue
                
                # Process leaves in smaller batches to avoid memory issues
                for batch_start in range(0, len(all_leaves), eval_batch_size):
                    batch_end = min(batch_start + eval_batch_size, len(all_leaves))
                    current_leaves = all_leaves[batch_start:batch_end]
                    
                    # Prepare batch for network evaluation
                    batch_obs = []
                    for _, node, _ in current_leaves:
                        try:
                            from custom_gym.chess_gym import canonical_encode_board_for_cnn
                            board_obs = canonical_encode_board_for_cnn(node.state)
                            board_obs_flat = board_obs.flatten()
                            legal_actions = node.state.legal_actions()
                            mask = np.zeros(4672, dtype=np.float32)
                            mask[legal_actions] = 1.0
                            obs = np.concatenate([board_obs_flat, mask])
                            batch_obs.append(obs)
                        except Exception as e:
                            print(f"Error preparing batch observation: {e}")
                            # Add a dummy observation to maintain alignment
                            batch_obs.append(np.zeros(4672 + 832, dtype=np.float32))
                    
                    if not batch_obs:
                        continue
                        
                    # Evaluate all leaf nodes in a single batch (much faster!)
                    try:
                        with torch.no_grad():
                            # Move data to GPU in an optimized way
                            if self.device == 'cuda':
                                # Use pinned memory for faster CPU->GPU transfer
                                batch_tensor = torch.FloatTensor(np.array(batch_obs)).pin_memory().to(self.device, non_blocking=True)
                            else:
                                batch_tensor = torch.FloatTensor(np.array(batch_obs)).to(self.device)
                                
                            batch_board = batch_tensor[:, :832]  # Board representation is first 832 elements
                            batch_mask = batch_tensor[:, 832:]  # Action mask is the rest
                            
                            # Create batch of dict observations
                            batch_dict_obs = {
                                'board': batch_board,
                                'action_mask': batch_mask
                            }
                            
                            # Get values from policy network
                            _, batch_values, _ = self.policy_net(batch_dict_obs)
                            batch_values = batch_values.cpu().detach()
                            
                            # Get probabilities for expansion
                            logits = self.policy_net.policy_net(self.policy_net.extract_features(batch_dict_obs))
                            illegal_actions_mask = (batch_dict_obs['action_mask'] < 0.5)
                            logits = logits.masked_fill(illegal_actions_mask, -1e8)
                            batch_probs = F.softmax(logits, dim=-1).cpu().detach()
                    except Exception as e:
                        print(f"Error in batch evaluation: {e}")
                        continue
                    
                    # Expansion and backpropagation for each leaf
                    for i, (root_idx, node, search_path) in enumerate(current_leaves):
                        if i >= len(batch_values):
                            continue
                            
                        try:
                            value = batch_values[i].item()
                            probs = batch_probs[i]
                            
                            # Expand node with evaluated probabilities
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
                                    
                            # Backpropagate value through the tree
                            for path_node in reversed(search_path):
                                path_node.visit_count += 1
                                path_node.total_value += value
                                path_node.mean_value = path_node.total_value / path_node.visit_count
                                
                                # Flip value for opponent's perspective
                                value = -value
                        except Exception as e:
                            print(f"Error in expansion/backpropagation: {e}")
            
            if num_states > 1:
                elapsed = time.time() - start_time
                print(f"Completed batch MCTS search in {elapsed:.2f}s ({elapsed/num_states:.3f}s per state)")
                
            # Select best action for each root based on visit counts
            best_actions = []
            for root in roots:
                if not root.children:
                    # If no children, return a random legal action
                    legal_actions = root.state.legal_actions()
                    if legal_actions:
                        best_actions.append(np.random.choice(legal_actions))
                    else:
                        best_actions.append(0)  # Fallback
                else:
                    # Use visit count to determine best action
                    best_action = None
                    most_visits = -1
                    
                    for action, child in root.children.items():
                        if child.visit_count > most_visits:
                            most_visits = child.visit_count
                            best_action = action
                            
                    best_actions.append(best_action if best_action is not None else 0)
            
            return best_actions
            
        except Exception as e:
            print(f"MCTS batch search error: {e}")
            # Return random legal actions as fallback
            return [self._get_random_action(state) for state in states]
    
    def _get_random_action(self, state):
        """Get a random legal action for the given state."""
        try:
            legal_actions = state.legal_actions()
            if legal_actions:
                return np.random.choice(legal_actions)
        except:
            pass
        return 0
        
    def search(self, state):
        """Single state search - now implemented as a wrapper around batch_search"""
        try:
            return self.batch_search([state])[0]
        except Exception as e:
            print(f"MCTS search error: {e}")
            return self._get_random_action(state)


class MCTSPPO(PPO):
    """
    PPO with MCTS integration and self-play for chess
    Includes custom rollout collection logic
    """
    def __init__(self, policy, env, **kwargs):
        # Extract custom parameters
        self.mcts_sims = kwargs.pop('mcts_sims', 100)
        self.mcts_eval_only = kwargs.pop('mcts_eval_only', False)
        self.mcts_freq = kwargs.pop('mcts_freq', 1.0)
        self.mcts_envs_frac = kwargs.pop('mcts_envs_frac', 1.0)
        self.separate_color_learning = kwargs.pop('separate_color_learning', False)
        
        if self.mcts_freq < 1.0:
            print(f"Using MCTS with {self.mcts_sims} simulations at {self.mcts_freq*100:.1f}% frequency")
            
        if self.mcts_envs_frac < 1.0:
            print(f"Limiting MCTS to at most {self.mcts_envs_frac*100:.1f}% of environments per step")
            
        if self.separate_color_learning:
            print("Using separate experience buffers for black and white pieces")
        
        super().__init__(policy, env, **kwargs)
        
        self.last_agent_step = [None] * self.n_envs
        
        # Set up separate buffers for white and black if requested
        if self.separate_color_learning:
            # Original buffer will be used for white
            self.white_buffer = self.rollout_buffer
            
            # Create a separate buffer with identical parameters for black
            self.black_buffer = RolloutBuffer(
                buffer_size=self.rollout_buffer.buffer_size,
                observation_space=self.rollout_buffer.observation_space,
                action_space=self.rollout_buffer.action_space,
                device=self.rollout_buffer.device,
                gae_lambda=self.rollout_buffer.gae_lambda,
                gamma=self.rollout_buffer.gamma,
                n_envs=self.rollout_buffer.n_envs,
            )
        
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
        
        # Reset appropriate buffers
        if self.separate_color_learning:
            # Reset both color-specific buffers
            self.white_buffer.reset()
            self.black_buffer.reset()
            # Point to the buffers for easier reference later
            white_buffer = self.white_buffer
            black_buffer = self.black_buffer
        else:
            # Use the provided rollout buffer for all experiences
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
            self.policy.init_mcts(num_simulations=self.mcts_sims, c_puct=2.0)
            
        # Initialize current_opponent_policies if it's None
        if self.current_opponent_policies is None:
            self.current_opponent_policies = [self.policy] * self.n_envs
            
        use_mcts_in_training = not self.mcts_eval_only  # Only use MCTS during training if not eval_only mode

        while n_steps < n_rollout_steps:
            actions = np.zeros(self.n_envs, dtype=int)
            batch_values = torch.zeros(self.n_envs, device=self.device)
            batch_log_probs = torch.zeros(self.n_envs, device=self.device)

            agent_turn_indices = [e for e in range(self.n_envs) if current_players[e] == self.agent_record_player[e]]
            opponent_turn_indices = [e for e in range(self.n_envs) if current_players[e] != self.agent_record_player[e]]

            # ===== Collect both agent and opponent experiences =====
            # We'll track whose turn it is but collect experiences from both sides
            acting_indices = agent_turn_indices + opponent_turn_indices
            
            # Process all observations in a single batch for efficiency
            if acting_indices:
                try:
                    all_boards = self._last_obs['board'][acting_indices]
                    all_masks = self._last_obs['action_mask'][acting_indices]
                    
                    board_tensor = torch.as_tensor(all_boards).to(self.device)
                    mask_tensor = torch.as_tensor(all_masks).to(self.device)
                    obs_dict = {'board': board_tensor, 'action_mask': mask_tensor}
                    
                    with torch.no_grad():
                        # Use agent policy for all observations for consistent learning
                        all_actions, all_values, all_log_probs = self.policy(obs_dict)
                    
                    all_actions = all_actions.cpu().numpy()
                    
                    # Identify which environments need MCTS
                    mcts_indices = []
                    if use_mcts_in_training and self.policy.mcts is not None:
                        # Apply frequency filter - only use MCTS on a subset of states based on frequency
                        if self.mcts_freq >= 1.0 or np.random.random() < self.mcts_freq:
                            # Get all potential MCTS candidates
                            potential_mcts_indices = [e for i, e in enumerate(acting_indices) 
                                                    if e in agent_turn_indices]
                            
                            # Apply environment fraction limit if needed
                            if self.mcts_envs_frac < 1.0 and potential_mcts_indices:
                                # Calculate max number of environments to use MCTS on
                                max_mcts_envs = max(1, int(len(potential_mcts_indices) * self.mcts_envs_frac))
                                # Randomly select subset of environments
                                if max_mcts_envs < len(potential_mcts_indices):
                                    mcts_indices = np.random.choice(
                                        potential_mcts_indices, 
                                        size=max_mcts_envs, 
                                        replace=False
                                    ).tolist()
                                else:
                                    mcts_indices = potential_mcts_indices
                            else:
                                mcts_indices = potential_mcts_indices
                    
                    # If we have environments that need MCTS, process them in parallel
                    if mcts_indices:
                        try:
                            # Get all states from these environments
                            env_states = env.get_attr('state', indices=mcts_indices)
                            
                            # Run batch MCTS search on all states at once
                            mcts_actions = self.policy.mcts.batch_search(env_states)
                            
                            # Process MCTS results for each environment
                            for idx, e in enumerate(mcts_indices):
                                if idx < len(mcts_actions):
                                    # Use the MCTS action
                                    actions[e] = mcts_actions[idx]
                                    
                                    # Find position in the original acting_indices list
                                    pos = acting_indices.index(e)
                                    
                                    # Re-evaluate to get value and log_prob
                                    single_obs = {
                                        'board': board_tensor[pos:pos+1],
                                        'action_mask': mask_tensor[pos:pos+1]
                                    }
                                    
                                    with torch.no_grad():
                                        _, values, _ = self.policy(single_obs)
                                        action_tensor = torch.tensor([mcts_actions[idx]], device=self.device)
                                        _, log_probs, _ = self.policy.evaluate_actions(single_obs, action_tensor)
                                    
                                    batch_values[e] = values[0]
                                    if log_probs.dim() == 0:
                                        batch_log_probs[e] = log_probs
                                    else:
                                        batch_log_probs[e] = log_probs[0]
                                    
                                    # Record position for rollout buffer
                                    self.last_agent_step[e] = rollout_buffer.pos
                        except Exception as mcts_ex:
                            print(f"Error using batch MCTS: {mcts_ex}, falling back to policy")
                            # For any environments that failed, use regular policy actions
                            for e in mcts_indices:
                                pos = acting_indices.index(e)
                                actions[e] = all_actions[pos]
                                batch_values[e] = all_values[pos]
                                batch_log_probs[e] = all_log_probs[pos]
                    
                    # For all non-MCTS environments, use regular policy
                    for i, e in enumerate(acting_indices):
                        if e not in mcts_indices:
                            actions[e] = all_actions[i] 
                            batch_values[e] = all_values[i]
                            batch_log_probs[e] = all_log_probs[i]
                            
                            # Record position for rollout buffer
                            self.last_agent_step[e] = rollout_buffer.pos
                except Exception as ex:
                    print(f"Error in policy evaluation: {ex}")
                    for e in acting_indices:
                        legal_actions = env.get_attr('state', indices=[e])[0].legal_actions()
                        if legal_actions:
                            actions[e] = np.random.choice(legal_actions)
            
            # For opponent indices, override actions with opponent policy
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
                            # Replace with opponent policy action 
                            # (but keep the value/log_prob from agent policy)
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

            # Add all transitions to buffer, not just agent's turns
            if self.separate_color_learning:
                # Separate white and black experiences
                white_indices = []
                black_indices = []
                
                # Check the player color from the channel 12 of the board observation
                for env_idx in range(self.n_envs):
                    # Check the color of the current player (channel 12 in the observation)
                    # 1.0 means white's turn, 0.0 means black's turn
                    board_flat = self._last_obs['board'][env_idx]
                    # Reconstruct the 3D board to get the player channel (channel 12)
                    # Shape is (13, 8, 8), channel 12 indicates the current player
                    board_3d = board_flat.reshape(13, 8, 8)
                    player_channel = board_3d[12, 0, 0]  # Check first element of player channel
                    
                    if player_channel > 0.5:  # White's turn
                        white_indices.append(env_idx)
                    else:  # Black's turn
                        black_indices.append(env_idx)
                
                # Add white player experiences to white buffer
                if white_indices:
                    self.white_buffer.add(
                        {k: v[white_indices] for k, v in self._last_obs.items()},
                        actions.reshape(self.n_envs, 1)[white_indices],
                        rewards[white_indices],
                        self._last_episode_starts[white_indices],
                        batch_values[white_indices],
                        batch_log_probs[white_indices]
                    )
                
                # Add black player experiences to black buffer
                if black_indices:
                    self.black_buffer.add(
                        {k: v[black_indices] for k, v in self._last_obs.items()},
                        actions.reshape(self.n_envs, 1)[black_indices],
                        rewards[black_indices],
                        self._last_episode_starts[black_indices],
                        batch_values[black_indices],
                        batch_log_probs[black_indices]
                    )
            else:
                # Original behavior: add all transitions to the same buffer
                rollout_buffer.add(
                    self._last_obs,
                    actions.reshape(self.n_envs, 1),
                    rewards,
                    self._last_episode_starts,
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
                    
                    # Update opponent policy
                    if self.opponent_policies and np.random.rand() < self.opponent_pool_prob:
                        self.current_opponent_policies[env_idx] = np.random.choice(self.opponent_policies)
                    else:
                        self.current_opponent_policies[env_idx] = self.policy

            self._last_obs = new_obs
            self._last_episode_starts = dones
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

    def train(self) -> None:
        """
        Override the train method to handle separate buffers for white and black
        """
        if self.separate_color_learning:
            # Train on white experiences
            if self.white_buffer.pos > 0:  # Check if buffer has data
                self._train_on_buffer(self.white_buffer, "white")
            
            # Train on black experiences
            if self.black_buffer.pos > 0:  # Check if buffer has data
                self._train_on_buffer(self.black_buffer, "black")
        else:
            # Use original PPO training with the combined buffer
            super().train()
    
    def _train_on_buffer(self, buffer, color):
        """
        Train on a specific buffer (white or black)
        """
        # Reference the original rollout buffer temporarily
        original_buffer = self.rollout_buffer
        self.rollout_buffer = buffer
        
        # Print training info
        print(f"Training on {color} experiences with {buffer.pos} steps")
        
        # Original PPO training code from parent class
        self._update_learning_rate(self.policy.optimizer)
        
        # Initialize logging values
        pg_losses, clip_fractions = [], []
        value_losses, entropy_losses = [], []
        
        # Train for n_epochs on the rollout buffer
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -self.clip_range_vf, self.clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            
            self._n_updates += 1
            
        # Calculate statistics
        explained_var = explained_variance(buffer.values.flatten(), buffer.returns.flatten())
        
        # Print training statistics
        print(f"{color.capitalize()} training: loss(p)={np.mean(pg_losses):.5f}, "
              f"loss(v)={np.mean(value_losses):.5f}, loss(e)={np.mean(entropy_losses):.5f}, "
              f"kl={np.mean(approx_kl_divs):.5f}, clip={np.mean(clip_fractions):.3f}, "
              f"explained_var={explained_var:.3f}")
        
        # Restore original buffer
        self.rollout_buffer = original_buffer


def create_cnn_mcts_ppo(env, tensorboard_log, device='cpu', checkpoint=None, learning_rate=3e-5, n_epochs=4, batch_size=256, clip_range=0.2, max_grad_norm=None, use_layer_norm=True, mcts_sims=100, mcts_eval_only=False, mcts_freq=1.0, mcts_envs_frac=1.0, separate_color_learning=False):
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
    
    # Ensure batch_size is not larger than n_steps
    batch_size = min(batch_size, n_steps)
    
    print(f"Training with {n_envs} environments, {n_steps} steps per environment")
    print(f"Total steps per iteration: {total_steps}, batch size: {batch_size}")
    print(f"Initial learning rate: {learning_rate}, epochs: {n_epochs}, clip_range: {clip_range}")
    
    if max_grad_norm is not None:
        print(f"Max gradient norm: {max_grad_norm}")
    else:
        print("Gradient clipping: Disabled (for faster learning)")
        
    print(f"Using learning rate schedule: YES (linear decay)")
    
    # Store MCTS simulations value
    if mcts_eval_only:
        print(f"MCTS enabled during evaluation only with {mcts_sims} simulations")
    else:
        if mcts_freq < 1.0:
            print(f"MCTS enabled during training at {mcts_freq*100:.1f}% frequency with {mcts_sims} simulations")
        else:
            print(f"MCTS enabled during both training and evaluation with {mcts_sims} simulations")
    
    if separate_color_learning:
        print("Using separate learning for white and black pieces")
    
    # Use a higher learning rate for faster training
    if learning_rate > 3e-4 and not checkpoint:
        print(f"NOTE: Reducing initial learning rate to 3e-4 for stability")
        learning_rate = 3e-4
    
    # Create a linear learning rate schedule
    def linear_schedule(progress_remaining):
        """
        Linear learning rate schedule that decreases from initial_value to final_value.
        
        :param progress_remaining: float between 0 and 1
        :return: current learning rate 
        """
        # Start with the specified learning rate and decrease to 10% of initial value
        return progress_remaining * learning_rate + (1 - progress_remaining) * (learning_rate * 0.1)
    
    # Common kwargs for both model creation paths
    model_kwargs = {
        'env': env,
        'policy': CNNMCTSActorCriticPolicy,
        'tensorboard_log': tensorboard_log,
        'verbose': 1,
        'learning_rate': linear_schedule,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'gamma': 0.99,
        'device': device,
        'clip_range': clip_range,
        'ent_coef': 0.1,  # Increased from 0.01 to 0.1 for much more exploration
        'vf_coef': 0.5,
        'policy_kwargs': policy_kwargs,
        'mcts_sims': mcts_sims,  # Pass MCTS simulations count
        'mcts_eval_only': mcts_eval_only,  # Whether to use MCTS only during evaluation
        'mcts_freq': mcts_freq,  # Pass MCTS frequency
        'mcts_envs_frac': mcts_envs_frac,  # Pass MCTS environment fraction
        'separate_color_learning': separate_color_learning,  # Enable separate learning for black and white
    }
    
    # Add max_grad_norm only if it's not None
    if max_grad_norm is not None:
        model_kwargs['max_grad_norm'] = max_grad_norm
    
    if checkpoint:
        print(f'Checkpoint: {checkpoint} loaded')
        model = MCTSPPO.load(
            checkpoint,
            **model_kwargs
        )
    else:
        model = MCTSPPO(
            **model_kwargs
        )
    
    return model