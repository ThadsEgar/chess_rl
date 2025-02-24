import os
import numpy as np
import torch
import math
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
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
            self.mcts = MCTS(self, num_simulations=400, c_puct=2)
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
        self.root_players = np.random.randint(0, 2, size=env.num_envs)
        self.buffer_white = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs
        )
        self.buffer_black = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs
        )

def collect_rollouts(self, env, callback, n_rollout_steps):
        n_envs = env.num_envs
        self.buffer_white.reset()
        self.buffer_black.reset()
        n_collected = np.zeros(n_envs, dtype=int)

        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            initial_obs, infos = reset_output
        else:
            initial_obs = reset_output
            infos = [{}] * n_envs
        self._last_obs = np.array(initial_obs)
        if self._last_obs.ndim == 1:
            self._last_obs = np.expand_dims(self._last_obs, axis=0)
        self._last_episode_starts = np.ones(n_envs, dtype=bool)

        while np.sum(n_collected) < n_rollout_steps * n_envs:
            current_players = np.array([info.get("current_player", 0) for info in infos])

            with torch.no_grad():
                obs_tensor = torch.FloatTensor(self._last_obs).to(self.device)
                actions, _, _ = self.policy(obs_tensor)
                actions = actions.cpu().numpy()
                _, values, log_probs_full = self.policy(obs_tensor, return_logits=True)
                actions_tensor = torch.LongTensor(actions).to(self.device).unsqueeze(1)
                log_probs = torch.gather(
                    torch.log_softmax(log_probs_full, dim=-1),
                    1, actions_tensor
                ).flatten()
                log_probs = torch.clamp(log_probs, min=-10.0, max=0.0)

            step_result = env.step(actions)
            if len(step_result) == 4:
                new_obs, rewards, dones, infos = step_result
                truncated = np.zeros_like(dones, dtype=bool)
            else:
                new_obs, rewards, dones, truncated, infos = step_result

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            # Separate into White (0) and Black (1) buffers
            for player in [0, 1]:
                player_indices = np.where(current_players == player)[0]
                if len(player_indices) > 0:
                    buffer = self.buffer_white if player == 0 else self.buffer_black
                    buffer.add(
                        self._last_obs[player_indices],
                        actions[player_indices].reshape(-1, 1),
                        rewards[player_indices],
                        self._last_episode_starts[player_indices],
                        values[player_indices],
                        log_probs[player_indices]
                    )
                    n_collected[player_indices] += 1

            self.num_timesteps += n_envs
            self._last_obs = np.array(new_obs)
            if self._last_obs.ndim == 1:
                self._last_obs = np.expand_dims(self._last_obs, axis=0)
            self._last_episode_starts = dones

            for e in range(n_envs):
                if dones[e] or truncated[e]:
                    reset_individual = env.env_method("reset", indices=[e])[0]
                    if isinstance(reset_individual, tuple):
                        obs_e = reset_individual[0]
                    else:
                        obs_e = reset_individual
                    obs_e = np.array(obs_e)
                    if obs_e.ndim == 1:
                        obs_e = np.expand_dims(obs_e, axis=0)[0]
                    self._last_obs[e, :] = obs_e

        with torch.no_grad():
            last_values = self.policy.predict_values(torch.FloatTensor(self._last_obs).to(self.device))
        self.buffer_white.compute_returns_and_advantage(last_values=last_values, dones=dones)
        self.buffer_black.compute_returns_and_advantage(last_values=last_values, dones=dones)
        
        callback.on_rollout_end()
        return True

def create_mcts_ppo(env, tensorboard_log, device='cuda', checkpoint=None):
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512, 256, 256, 128], vf=[512, 512, 256, 256, 128])
    )
    
    if checkpoint and os.path.exists(checkpoint):
        model = MCTSPPO.load(checkpoint, env=env, tensorboard_log=tensorboard_log, device=device)
    else:
        model = MCTSPPO(
            MCTSMaskableActorCriticPolicy,
            env,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=5e-5,
            n_steps=2048,
            batch_size=4096,
            n_epochs=10,
            gamma=0.99,
            device=device,
            clip_range=0.2,
            ent_coef=0.01
        )
    return model