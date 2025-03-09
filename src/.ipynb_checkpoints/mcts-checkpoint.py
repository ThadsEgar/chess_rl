import os
import numpy as np
import torch
import math
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import explained_variance
import pyspiel

from custom_gym.chess_gym import ChessEnv, canonical_encode_board

piece_mapping = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
}

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
        board_obs = canonical_encode_board(self.state).flatten()
        legal_actions = self.state.legal_actions()
        mask = np.zeros(4672, dtype=np.float32)
        mask[legal_actions] = 1.0
        obs = np.concatenate([board_obs, mask])
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(policy_net.device)
            logits, values, _ = policy_net.forward(obs_tensor, return_logits=True)
            action_probs = torch.softmax(logits, dim=-1).squeeze(0)
        
        for action in legal_actions:
            if action not in self.children:
                next_state = self.state.clone()
                next_state.apply_action(action)
                self.children[action] = Node(
                    state=next_state,
                    prior=action_probs[action].item(),
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
            
            while node.children and not node.is_terminal:
                action, node = node.select_child(self.c_puct)
                search_path.append(node)
            
            if not node.is_terminal:
                value = node.expand(self.policy_net)
                if node.state.current_player() != root.root_player:
                    value = -value
            else:
                value = node.state.rewards()[root.root_player]
            
            for node in reversed(search_path):
                node.backpropagate(value)
        
        visits = {action: child.visit_count for action, child in root.children.items()}
        top_k = sorted(visits.items(), key=lambda x: x[1], reverse=True)[:5]
        actions, counts = zip(*top_k)
        probs = np.array(counts) / sum(counts)
        return np.random.choice(actions, p=probs)

class MCTSMaskableActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcts = None
        self.training = True

    def forward(self, obs: torch.Tensor, return_logits=False, deterministic=False):
        if torch.isnan(obs).any():
            print("NaN in input obs (forward):", obs[torch.isnan(obs)], flush=True)
        features = self.extract_features(obs)
        if torch.isnan(features).any():
            print("NaN in features (forward):", features[torch.isnan(features)], flush=True)
        latent_pi, latent_vf = self.mlp_extractor(features)
        if torch.isnan(latent_pi).any():
            print("NaN in latent_pi (forward):", latent_pi[torch.isnan(latent_pi)], flush=True)
        
        board_size = 64
        total_actions = self.action_space.n
        mask_obs = obs[:, board_size:board_size + total_actions]
        illegal_mask = (mask_obs < 0.5)
        
        logits = self.action_net(latent_pi)
        if torch.isnan(logits).any():
            print("NaN in logits before clamp (forward):", logits[torch.isnan(logits)], flush=True)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        logits[illegal_mask] = -1e10
        
        if torch.isnan(logits).any():
            print("NaN in logits after clamp (forward):", logits[torch.isnan(logits)], flush=True)
        distribution = torch.distributions.Categorical(logits=logits)
        values = self.value_net(latent_vf)
        
        if deterministic:
            actions = distribution.probs.argmax(dim=-1)
        else:
            actions = distribution.sample()
        
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        log_probs = torch.log_softmax(logits, dim=-1)
        sampled_log_prob = log_probs.gather(1, actions).squeeze(1)
        
        if return_logits:
            return logits, values, log_probs
        return actions.squeeze(1), values, sampled_log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        if torch.isnan(obs).any():
            print("NaN in input obs (evaluate_actions):", obs[torch.isnan(obs)], flush=True)
        
        features = self.extract_features(obs)
        if torch.isnan(features).any():
            print("NaN in features (evaluate_actions):", features[torch.isnan(features)], flush=True)
        
        latent_pi, latent_vf = self.mlp_extractor(features)
        if torch.isnan(latent_pi).any():
            print("NaN in latent_pi (evaluate_actions):", latent_pi[torch.isnan(latent_pi)], flush=True)
        
        board_size = 64
        total_actions = self.action_space.n
        mask_obs = obs[:, board_size:board_size + total_actions]
        illegal_mask = (mask_obs < 0.5)
        
        logits = self.action_net(latent_pi)
        if torch.isnan(logits).any():
            print("NaN in logits before clamp (evaluate_actions):", logits[torch.isnan(logits)], flush=True)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        logits[illegal_mask] = -1e10
        
        if torch.isnan(logits).any():
            print("NaN in logits after clamp (evaluate_actions):", logits[torch.isnan(logits)], flush=True)
        
        distribution = torch.distributions.Categorical(logits=logits)
        values = self.value_net(latent_vf)
        log_prob = distribution.log_prob(actions.squeeze())
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if self.mcts is None:
            self.mcts = MCTS(self, num_simulations=1200, c_puct=2)
        if deterministic:  # Always use MCTS for deterministic in testing
            self.training = False  # Force testing mode
            action = self.mcts.search(state)
            if isinstance(action, np.ndarray):
                action = action.item()
            return action, None
        return super().predict(observation, state, episode_start, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.training = mode

class MCTSPPO(PPO):
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

def collect_rollouts(self, env, callback, rollout_buffer: RolloutBuffer, n_rollout_steps):
    assert self._last_obs is not None, "No previous observation"
    n_steps = 0
    rollout_buffer.reset()

    while n_steps < n_rollout_steps:
        # Prepare to select actions
        actions = np.zeros(self.n_envs, dtype=int)
        batch_values = torch.zeros(self.n_envs, device=self.device)
        batch_log_probs = torch.zeros(self.n_envs, device=self.device)

        # Identify agent's and opponent's turns
        current_players = env.get_attr("current_player")
        if not isinstance(current_players, list):
            current_players = [current_players] * self.n_envs
        agent_turn_indices = [e for e in range(self.n_envs) if current_players[e] == self.agent_record_player[e]]
        opponent_turn_indices = [e for e in range(self.n_envs) if current_players[e] != self.agent_record_player[e]]

        # Compute actions for agent's turns
        if agent_turn_indices:
            agent_obs = self._last_obs[agent_turn_indices]
            with torch.no_grad():
                obs_tensor = torch.as_tensor(agent_obs).to(self.device)
                agent_actions, agent_values, agent_log_probs = self.policy.forward(obs_tensor)
            agent_actions = agent_actions.cpu().numpy()
            for i, e in enumerate(agent_turn_indices):
                actions[e] = agent_actions[i]
                batch_values[e] = agent_values[i]
                batch_log_probs[e] = agent_log_probs[i]

        # Compute actions for opponent's turns
        for e in opponent_turn_indices:
            opponent_policy = self.current_opponent_policies[e]
            obs_e = self._last_obs[e]
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs_e).unsqueeze(0).to(self.device)
                action, _, _ = opponent_policy.forward(obs_tensor, deterministic=False)
            actions[e] = action.item()

        # Step the environment
        new_obs, rewards, dones, infos = env.step(actions)
        if not isinstance(infos, list):
            infos = [infos] * self.n_envs
        current_players = [info.get("current_player", 0) for info in infos]

        # Update callback
        callback.update_locals(locals())
        if not callback.on_step():
            return False

        # Prepare full batch data for all environments
        batch_obs = np.zeros_like(self._last_obs)
        batch_actions = np.zeros(self.n_envs, dtype=int)
        batch_rewards = np.zeros(self.n_envs, dtype=float)
        batch_episode_starts = np.zeros(self.n_envs, dtype=bool)

        # Fill in data where the agent is acting
        for e in range(self.n_envs):
            if current_players[e] == self.agent_record_player[e]:
                batch_obs[e] = self._last_obs[e]
                batch_actions[e] = actions[e]
                batch_rewards[e] = rewards[e]
                batch_episode_starts[e] = self._last_episode_starts[e]
                self.last_agent_step[e] = rollout_buffer.pos

        # Add full batch to rollout buffer
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
                if self.last_agent_step[e] is not None:  # Agent made at least one move
                    last_pos = self.last_agent_step[e]
                    if infos[e].get('is_draw', False):
                        rollout_buffer.rewards[last_pos, e] = 0
                    elif infos[e].get('white_won', False):
                        # Reward depends on who the agent is
                        rollout_buffer.rewards[last_pos, e] = -1 if self.agent_record_player[e] == 0 else 1
                    elif infos[e].get('black_won', False):
                        rollout_buffer.rewards[last_pos, e] = 1 if self.agent_record_player[e] == 0 else -1
                    else:
                        rollout_buffer.rewards[last_pos, e] = 0  # Default
                # Reset environment and update opponent policy
                self._last_obs[e], _ = env.reset([e])
                self.last_agent_step[e] = None
                if self.opponent_policies and np.random.rand() < self.opponent_pool_prob:
                    self.current_opponent_policies[e] = np.random.choice(self.opponent_policies)
                else:
                    self.current_opponent_policies[e] = self.policy

        # Update state
        self._last_obs = new_obs
        self._last_episode_starts = dones
        self.num_timesteps += self.n_envs
        n_steps += 1

        # Prevent buffer overflow
        if rollout_buffer.pos >= n_rollout_steps:
            break

    # Compute returns and advantages
    with torch.no_grad():
        last_values = self.policy.predict_values(torch.as_tensor(self._last_obs).to(self.device))
    rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)

    return True

def create_mcts_ppo(env, tensorboard_log, device='cuda', checkpoint=None):
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512, 256, 256, 128], vf=[512, 512, 256, 256, 128])
    )
    
    if checkpoint:
        print(f'checkpoint: {checkpoint} loaded')
        model = MCTSPPO.load(checkpoint, 
                            env=env, 
                            tensorboard_log=tensorboard_log, 
                            verbose=1,
                            learning_rate=5e-5,
                            n_steps=16384,
                            batch_size=16384,
                            n_epochs=10,
                            gamma=0.99,
                            device=device,
                            clip_range=0.2,
                            ent_coef=0.3,
                            vf_coef=0.5)
    else:
        model = MCTSPPO(
            MCTSMaskableActorCriticPolicy,
            env,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=5e-5,
            n_steps=16384,
            batch_size=16384,
            n_epochs=10,
            gamma=0.99,
            device=device,
            clip_range=0.2,
            ent_coef=0.3,
            vf_coef=0.5
        )
    return model