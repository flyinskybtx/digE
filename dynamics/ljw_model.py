import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from tqdm import trange

from utils.viewer import display_coverage

PARAM_SETS = {
    1: {"A": np.array([[0.9, 0, 0.01, 0],
                       [0, 0.9, 0, 0.01],
                       [0, 0, 0.8891, 0],
                       [0, 0, 0, 0.9587]])
        , "B": np.array([[0, 0],
                         [0, 0],
                         [0.0283, 0.0009],
                         [0.0028, 0.0094]])},
    2: {"A": np.array([[1, 0, 0.01, 0],
                       [0, 1, 0, 0.01],
                       [0, 0, 0.8891, 0],
                       [0, 0, 0, 0.9587]])
        , "B": np.array([[0, 0],
                         [0, 0],
                         [0.0283, 0.0009],
                         [0.0028, 0.0094]])},
}


class LjwModel(gym.Env):
    def __init__(self, version, max_state=2, max_u=10, tolerance=0.2, noise_level=0.01):
        self.max_state = max_state
        self.max_u = max_u
        self.tolerance = tolerance
        self.noise_level = noise_level
        self.__dict__.update(PARAM_SETS[version])
        self.action_space = spaces.Box(low=-self.max_u, high=self.max_u, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.max_state, high=self.max_state, shape=(4,), dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-self.max_state, high=self.max_state,
                                            size=(4,))
        return self._get_obs()

    def step(self, action: np.ndarray):
        self.state = np.dot(self.A, self.state) + np.dot(self.B, action)
        if any(self.state < -(1 + self.tolerance) * self.max_state) or any(
                self.state > (1 + self.tolerance) * self.max_state):
            done = True
        else:
            done = False
        return self._get_obs(), None, done, {}

    def _get_obs(self):
        obs = self.state + self.noise_level * self.np_random.uniform(-self.max_state, self.max_state, size=(4,))
        return obs


if __name__ == '__main__':
    ljw_model = LjwModel(version=1)
    action_space = ljw_model.action_space
    state_space = ljw_model.observation_space

    EPISODE_LEN = 1000

    Uk = []
    Xk = []
    Xk1 = []
    done = True
    for i in trange(EPISODE_LEN):
        if done:
            obs = ljw_model.reset()
        u = action_space.sample()
        new_obs, _, done, _ = ljw_model.step(u)

        Uk.append(u)
        Xk.append(obs)
        Xk1.append(new_obs)

        obs = new_obs

    display_coverage(Uk, (-ljw_model.max_u, ljw_model.max_u), 10, f'Uk')
    display_coverage(Xk, (-ljw_model.max_state, ljw_model.max_state), 10, f'Xk')
