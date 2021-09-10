import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from tqdm import trange

Rs = 0.5
Ld = Lq = 1e-3
Phif = 0.07
Np = 1
J = 50 * 1e-7
B = 0
Tf = 0
Umax = 48
Imax = 20
Omegamax = 500
T = 1e-4


def f(x):
    """
    d_id, d_iq, d_theta_e = f(id, iq, theta_e)
    """
    x1, x2, x3 = x
    dx1 = (- Rs / Ld) * x1 + (Lq / Ld) * x2 * x3
    dx2 = (- Ld / Lq) * x1 * x3 + (-Rs / Lq) * x2 + (-Phif / Lq) * x3
    dx3 = (3 / 2 * Np ** 2 / J) * (Phif * x2 + Ld * x1 * x2 - Lq * x1 * x2) + (-B / J) * x3 + (-Np * Tf / J) * \
          np.sign(x3)
    return np.array([dx1, dx2, dx3])


def g(u):
    """
    d_id, d_iq, d_theta_e = g(ud, uq)
    """
    return np.array([[1 / Ld, 0], [0, 1 / Lq], [0, 0]]) @ u


class PMSMModel(gym.Env):
    def __init__(self, noise_level=0.01):
        """
        state: id, iq, theta_e
        action: ud, uq
        """
        self.action_space = spaces.Box(low=-Umax, high=Umax, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.array([Imax, Imax, Omegamax]),
                                            high=np.array([Imax, Imax, Omegamax]),
                                            shape=(3,), dtype=np.float32)
        self.noise_level = noise_level
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.observation_space.sample()
        return self._get_obs()

    def step(self, action: np.ndarray):
        dxdt = f(self.state) + g(action)
        self.state += dxdt * T

        if not (-Imax < self.state[0] < Imax and
                -Imax < self.state[1] < Imax and
                -Omegamax < self.state[2] < Omegamax):
            done = True
        else:
            done = False
        return self._get_obs(), None, done, {}

    def _get_obs(self):
        obs = self.state + self.noise_level * self.np_random.uniform([-Imax, -Imax, -Omegamax],
                                                                     [Imax, Imax, Omegamax])
        return obs


if __name__ == '__main__':
    ljw_model = PMSMModel()
    action_space = ljw_model.action_space
    state_space = ljw_model.observation_space

    EPISODE_LEN = 10000

    Uk = []
    Xk = []
    Xk1 = []
    done = True
    cnt = 0
    for i in trange(EPISODE_LEN):
        if done:
            cnt += 1
            obs = ljw_model.reset()
        u = action_space.sample()
        new_obs, _, done, _ = ljw_model.step(u)

        Uk.append(u)
        Xk.append(obs)
        Xk1.append(new_obs)

        obs = new_obs

    print(f"Total {cnt} trials.")
