import numpy as np

class RolloutBuffer:
    def __init__(
            self,
            episodes_in_parallel,
            buffer_size,
            obs_size,
    ):
        self.n_episodes = episodes_in_parallel
        self.buffer_size = buffer_size
        self.obs_size = obs_size
        self.internal_index = 0
        self.full = False

        self.values = np.zeros((self.buffer_size, self.n_episodes), dtype="float32")
        self.actions = np.zeros((self.buffer_size, self.n_episodes), dtype="float32")
        self.log_probs = np.zeros((self.buffer_size, self.n_episodes), dtype="float32")
        self.rewards = np.zeros((self.buffer_size, self.n_episodes), dtype="float32")
        self.starts = np.zeros((self.buffer_size + 1, self.n_episodes), dtype="float32")
        self.game_maps = np.zeros((self.buffer_size + 1, self.n_episodes, obs_size), dtype="float32")

        self.spilled_map = None
        self.spilled_dones = None

    def reset(self, maps):
        self.values = np.zeros((self.buffer_size, self.n_episodes), dtype="float32")
        self.actions = np.zeros((self.buffer_size, self.n_episodes), dtype="float32")
        self.log_probs = np.zeros((self.buffer_size, self.n_episodes), dtype="float32")
        self.rewards = np.zeros((self.buffer_size, self.n_episodes), dtype="float32")
        self.starts = np.zeros((self.buffer_size + 1, self.n_episodes), dtype="float32")
        self.game_maps = np.zeros((self.buffer_size + 1, self.n_episodes , self.obs_size), dtype="float32")

        self.game_maps[0] = np.stack(maps)
        self.starts[0] = np.ones(self.n_episodes)
        self.internal_index = 0
        self.full = False

    def restart_from_last(self):
        self.values = np.zeros((self.buffer_size, self.n_episodes), dtype="float32")
        self.actions = np.zeros((self.buffer_size, self.n_episodes), dtype="float32")
        self.log_probs = np.zeros((self.buffer_size, self.n_episodes), dtype="float32")
        self.rewards = np.zeros((self.buffer_size, self.n_episodes), dtype="float32")

        self.spilled_start = self.starts[-1]
        self.spilled_map = self.game_maps[-1]
        self.starts = np.zeros((self.buffer_size + 1, self.n_episodes), dtype="float32")
        self.game_maps = np.zeros((self.buffer_size + 1, self.n_episodes , self.obs_size), dtype="float32")
        self.starts[0] = self.spilled_start
        self.game_maps[0] = self.spilled_map

        self.internal_index = 0
        self.full = False

    def add(self, step, reward, actions, values, log_probs, dones, maps):
        assert step == self.internal_index
        assert not self.full

        self.rewards[step] = reward
        self.actions[step] = actions
        self.values[step] = values
        self.log_probs[step] = log_probs
        self.game_maps[step + 1] = np.stack(maps)
        self.starts[step + 1] = dones
        self.internal_index += 1

        if step == self.buffer_size - 1:
            self.full = True

        return self.full

    def compute_gae(self, last_values, dones, gamma, gae_lam):
        """
        see arXiv:1506.02438v6 and stable-baselines 3 implementation
        """
        advantages = np.zeros((self.buffer_size, self.n_episodes), dtype="float32")
        dones = np.array(dones)
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = ~dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_non_terminal * next_values - self.values[step]
            last_gae_lam = delta + gamma * gae_lam * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
        returns = advantages + self.values
        return advantages, returns
