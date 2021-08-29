"""
质量m, 长l 均匀分布的刚性杆，转动惯量I = 1/3 * m * l^2
控制u为作用于轴处的控制量

Defaults:
    max_speed = 8
    max_torque = 2.0
    dt = 0.05
    g = g
    m = 1.0
    l = 1.0
"""
import numpy as np
from gym import spaces
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from tqdm import trange


class CustomPendulum(PendulumEnv):
    def __init__(self, noise_level=0.01, **kwargs):
        super().__init__()
        self.noise_level = noise_level
        for k, v in kwargs.items():
            setattr(self, k, v)

        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def reset(self, state=None):
        """

        :param state: (1-d array): x1:[-pi, pi] x2:[-max_speed, max_speed]
        :return:
        """
        super().reset()
        high = np.array([np.pi, self.max_speed])
        self.state = state or self.state
        assert np.all(-high <= self.state) and np.all(self.state <= high)
        return self._get_obs()

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        newthdot = (
                thdot
                + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l ** 2) * u) * dt
        )
        newth = th + newthdot * dt

        done = False if -self.max_speed < newthdot < self.max_speed else True
        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, done, {}

    def _get_obs(self):
        theta, thetadot = self.state
        theta += self.noise_level * self.np_random.uniform(-np.pi, np.pi)
        thetadot += self.noise_level + self.np_random.uniform(-1, 1)
        return theta, thetadot


if __name__ == '__main__':
    pendulum = CustomPendulum()
    action_space = pendulum.action_space
    state_space = pendulum.observation_space

    EPISODE_LEN = 1000

    Uk = []
    Xk = []
    Xk1 = []
    done = True
    for i in trange(EPISODE_LEN):
        if done:
            obs = pendulum.reset()
        u = action_space.sample()
        new_obs, _, _, _ = pendulum.step(u)

        Uk.append(u)
        Xk.append(obs)
        Xk1.append(new_obs)

        obs = new_obs
