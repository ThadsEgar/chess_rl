import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from collections import deque

# Layer Normalization for 2D convolutional features
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

# Residual block for the CNN
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

# Pure PyTorch CNN model for chess
class ChessCNN(nn.Module):
    def __init__(self, action_space_n=20480):
        super(ChessCNN, self).__init__()
        
        self.board_channels = 13  # Number of channels in the chess board representation
        self.board_size = 8       # Size of the chess board (8x8)
        self.action_space_n = action_space_n
        
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
            nn.ReLU()
        )
        
        # Policy network (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(832, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_n)
        )
        
        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(832, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # MCTS reference (will be set later if needed)
        self.mcts = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _initialize_weights(self):
        """Initialize weights with Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
    
    def extract_features(self, obs):
        """Extract features from the observation"""
        if isinstance(obs, dict):
            board_flat = obs["board"]
        else:
            # Assume board is the first part of the observation
            board_flat = obs[:, :self.board_channels * self.board_size * self.board_size]
        
        batch_size = board_flat.shape[0]
        board_3d = board_flat.reshape(batch_size, self.board_channels, self.board_size, self.board_size)
        
        return self.features_extractor(board_3d)
    
    def forward(self, obs, deterministic=False):
        """Forward pass through the network"""
        # Extract features
        features = self.extract_features(obs)
        
        # Get policy logits and values
        logits = self.policy_net(features)
        values = self.value_net(features)
        
        # Apply action mask if provided
        if isinstance(obs, dict) and "action_mask" in obs:
            action_mask = obs["action_mask"]
            illegal_actions_mask = (action_mask < 0.5)
            logits = logits.masked_fill(illegal_actions_mask, -1e8)
        
        # Create probability distribution
        distribution = torch.distributions.Categorical(logits=logits)
        
        # Sample or take most likely action
        if deterministic:
            actions = torch.argmax(distribution.probs, dim=1)
        else:
            actions = distribution.sample()
            
        # Get log probabilities
        log_probs = distribution.log_prob(actions)
        
        return {
            "actions": actions,
            "values": values.squeeze(-1),
            "log_probs": log_probs,
            "entropy": distribution.entropy(),
            "distribution": distribution
        }
    
    def evaluate_actions(self, obs, actions):
        """Evaluate actions to get values, log probs, and entropy"""
        # Extract features
        features = self.extract_features(obs)
        
        # Get policy logits and values
        logits = self.policy_net(features)
        values = self.value_net(features)
        
        # Apply action mask if provided
        if isinstance(obs, dict) and "action_mask" in obs:
            action_mask = obs["action_mask"]
            illegal_actions_mask = (action_mask < 0.5)
            logits = logits.masked_fill(illegal_actions_mask, -1e8)
        
        # Create probability distribution
        distribution = torch.distributions.Categorical(logits=logits)
        
        # Get log probabilities and entropy
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return {
            "values": values.squeeze(-1),
            "log_probs": log_probs,
            "entropy": entropy
        }
    
    def get_action_distribution(self, obs):
        """Get action distribution from observation"""
        # Extract features
        features = self.extract_features(obs)
        
        # Get policy logits
        logits = self.policy_net(features)
        
        # Apply action mask if provided
        if isinstance(obs, dict) and "action_mask" in obs:
            action_mask = obs["action_mask"]
            illegal_actions_mask = (action_mask < 0.5)
            logits = logits.masked_fill(illegal_actions_mask, -1e8)
        
        return logits
    
    def predict_values(self, obs):
        """Predict values from observation"""
        features = self.extract_features(obs)
        values = self.value_net(features)
        return values.squeeze(-1)
    
    def init_mcts(self, num_simulations=100, c_puct=2.0):
        """Initialize MCTS for policy improvement"""
        self.mcts = MCTS(self, num_simulations=num_simulations, c_puct=c_puct)
        print(f"MCTS initialized with {num_simulations} simulations")
        return self.mcts
    
    def to(self, device):
        """Move model to device"""
        self.device = device
        return super().to(device)


# MCTS Node class
class Node:
    def __init__(self, state, prior=1.0, parent=None, root_player=None):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.mean_value = 0
        self.is_terminal = state.is_game_over() if hasattr(state, 'is_game_over') else False
        self.root_player = root_player if root_player is not None else (1 if state.turn else -1)

    def select_child(self, c_puct=1.0):
        """Select best child according to PUCT formula"""
        best_score = float('-inf')
        best_action = None
        
        for action, child in self.children.items():
            score = child.mean_value + c_puct * child.prior * \
                   math.sqrt(self.visit_count) / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_action = action
                
        if best_action is None:
            raise ValueError("No children to select")
            
        return best_action, self.children[best_action]

    def expand(self, policy_net):
        """Expand node by adding children nodes for all legal moves"""
        try:
            from custom_gym.chess_gym import canonical_encode_board_for_cnn
            
            # Get board representation
            board_obs = canonical_encode_board_for_cnn(self.state)
            board_obs_flat = board_obs.flatten()
            
            # Get legal actions
            legal_moves = list(self.state.legal_moves)
            legal_actions = [self.state.move_to_action(move) for move in legal_moves]
            
            # Create action mask
            action_mask = np.zeros(4672, dtype=np.float32)
            for action in legal_actions:
                if 0 <= action < 4672:
                    action_mask[action] = 1.0
            
            # Create observation dictionary
            obs = {
                'board': torch.FloatTensor(board_obs_flat).unsqueeze(0).to(policy_net.device),
                'action_mask': torch.FloatTensor(action_mask).unsqueeze(0).to(policy_net.device)
            }
            
            # Get action probabilities
            with torch.no_grad():
                logits = policy_net.get_action_distribution(obs)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            # Create children for each legal action
            for action in legal_actions:
                # Skip if child already exists
                if action in self.children:
                    continue
                    
                # Get probability for this action
                prior = probs[action]
                
                # Create new state
                new_state = self.state.copy()
                move = self.state.action_to_move(action)
                new_state.push(move)
                
                # Create child node
                self.children[action] = Node(
                    state=new_state,
                    prior=prior,
                    parent=self,
                    root_player=self.root_player
                )
            
            # Get value of this state
            value = policy_net.predict_values(obs).item()
            return value
            
        except Exception as e:
            print(f"Error in Node.expand: {e}")
            return 0.0

    def backpropagate(self, value):
        """Update values up the tree"""
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count
        
        # Propagate to parent
        if self.parent:
            # Flip value for opponent
            self.parent.backpropagate(-value)


# MCTS search implementation
class MCTS:
    def __init__(self, policy_net, num_simulations=100, c_puct=1.0):
        self.policy_net = policy_net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = policy_net.device
        print(f"MCTS initialized with {num_simulations} simulations using device: {self.device}")
    
    def search(self, state):
        """Perform MCTS search from the given state"""
        # Create root node
        root = Node(state=state, root_player=1 if state.turn else -1)
        
        # If game is already over, return random action
        if root.is_terminal:
            legal_actions = self._get_legal_actions(state)
            if legal_actions:
                return np.random.choice(legal_actions), 0
            return 0, 0
        
        # Expand the root first
        root.expand(self.policy_net)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection phase: select best child until reaching a leaf
            while not node.is_terminal and node.children:
                action, node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # Expansion phase: if not terminal and not fully expanded
            value = 0
            if not node.is_terminal:
                value = node.expand(self.policy_net)
            
            # Backpropagation phase: update values up the tree
            for path_node in reversed(search_path):
                path_node.visit_count += 1
                path_node.total_value += value
                path_node.mean_value = path_node.total_value / path_node.visit_count
                value = -value  # Flip value for opponent
        
        # Select best action based on visit count (not value)
        best_action = None
        most_visits = -1
        
        for action, child in root.children.items():
            if child.visit_count > most_visits:
                most_visits = child.visit_count
                best_action = action
        
        if best_action is None:
            legal_actions = self._get_legal_actions(state)
            if legal_actions:
                return np.random.choice(legal_actions), 0
            return 0, 0
            
        # Return best action and its value
        return best_action, root.children[best_action].mean_value
    
    def batch_search(self, states):
        """Run search on multiple states and return best actions"""
        if not isinstance(states, list):
            states = [states]
            
        return [self.search(state)[0] for state in states]
    
    def _get_legal_actions(self, state):
        """Get legal actions for the given state"""
        try:
            legal_moves = list(state.legal_moves)
            return [state.move_to_action(move) for move in legal_moves]
        except:
            return [] 