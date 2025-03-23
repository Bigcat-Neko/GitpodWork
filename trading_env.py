# trading_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, prices=None, window_size=10):
        """
        Parameters:
            prices: Optional 1D numpy array of prices. If None, loads 'close' prices
                    from 'data/forex_data_preprocessed.csv' and normalizes them between 0 and 1.
            window_size: Number of past data points to use as observation.
        """
        super(TradingEnv, self).__init__()
        self.window_size = window_size
        
        # Load and normalize prices if not provided
        if prices is None:
            df = pd.read_csv("data/forex_data_preprocessed.csv", parse_dates=["date", "fetched_at"])
            prices = df["close"].values.reshape(-1, 1)
            self.scaler = MinMaxScaler()
            prices = self.scaler.fit_transform(prices).flatten()
        else:
            self.scaler = None
            prices = np.array(prices)
        self.prices = prices.astype(np.float32)
        
        self.current_step = self.window_size
        # Actions: 0 = Hold, 1 = Buy (long), 2 = Sell (short)
        self.action_space = spaces.Discrete(3)
        # Observations: the last 'window_size' normalized price values
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.window_size,), dtype=np.float32)
        
        self.position = 0  # 0 means no position, 1 means long, -1 means short
        self.done = False

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.position = 0
        self.done = False
        obs = self.prices[self.current_step - self.window_size:self.current_step]
        return obs, {}

    def step(self, action):
        if self.done:
            return self.reset()
        
        # Calculate reward: change in price multiplied by the current position
        previous_price = self.prices[self.current_step - 1]
        current_price = self.prices[self.current_step]
        reward = (current_price - previous_price) * self.position
        
        # Update position based on action:
        # Action 1 (Buy) sets position to 1, Action 2 (Sell) sets position to -1, Action 0 keeps the same position.
        if action == 1:
            self.position = 1
        elif action == 2:
            self.position = -1
        
        self.current_step += 1
        if self.current_step >= len(self.prices):
            self.done = True
        
        obs = self.prices[self.current_step - self.window_size:self.current_step]
        return obs, reward, self.done, False, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Price: {self.prices[self.current_step]}, Position: {self.position}")

if __name__ == "__main__":
    # Simple test run of the environment
    env = TradingEnv()
    obs, _ = env.reset()
    print("Initial observation:", obs)
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
