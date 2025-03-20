# trading_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, prices):
        """
        prices: 1D numpy array of normalized prices (dtype can be float64, but we'll cast to float32 in the observations)
        """
        super(TradingEnv, self).__init__()
        self.prices = prices
        self.current_step = 60  # start after the first 60 data points
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(60,), dtype=np.float32)
        self.done = False

    def reset(self, seed=None, options=None):
        self.current_step = 60
        self.done = False
        # Return the last 60 price points as float32
        obs = self.prices[self.current_step - 60:self.current_step].astype(np.float32)
        return obs, {}

    def step(self, action):
        current_price = self.prices[self.current_step]
        previous_price = self.prices[self.current_step - 1]

        reward = 0
        if action == 1:  # Buy
            reward = current_price - previous_price
        elif action == 2:  # Sell
            reward = previous_price - current_price

        self.current_step += 1
        if self.current_step >= len(self.prices):
            self.done = True

        # Get the observation as float32 to match the observation space
        obs = self.prices[self.current_step - 60:self.current_step].astype(np.float32)
        return obs, reward, self.done, False, {}  # False for 'truncated'

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Price: {self.prices[self.current_step]}")
