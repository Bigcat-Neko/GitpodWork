import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

def compute_ATR(prices, window=14):
    # A simple ATR computation using close price differences as a proxy.
    diffs = np.abs(np.diff(prices))
    if len(diffs) < window:
        return np.mean(diffs) if diffs.size > 0 else 0.0
    return pd.Series(diffs).rolling(window=window).mean().iloc[-1]

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, prices=None, window_size=30):
        """
        Initializes the trading environment.
        Observation space will be a vector of the last `window_size` normalized prices.
        """
        super().__init__()
        self.window_size = window_size
        
        # Load or generate price data.
        if prices is None:
            try:
                df = pd.read_csv("data/forex_data_preprocessed.csv", parse_dates=["date", "fetched_at"])
                prices = df["close"].values.astype(np.float32)
                self.scaler = MinMaxScaler()
                prices = self.scaler.fit_transform(prices.reshape(-1, 1)).flatten()
                logger.info("Loaded prices from data file")
            except FileNotFoundError:
                logger.warning("Data file not found, using generated data")
                prices = np.random.rand(1000).astype(np.float32)
                self.scaler = None
        else:
            prices = np.array(prices).astype(np.float32)
            self.scaler = None

        self.prices = prices
        self.current_step = self.window_size
        self.position = 0  # 0: Flat, 1: Long, -1: Short
        self.done = False
        
        # The observation is now simply the most recent window of normalized prices.
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.window_size,), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell

    def _get_obs(self):
        # Return the current window of prices.
        window = self.prices[self.current_step - self.window_size:self.current_step]
        return window

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.position = 0
        self.done = False
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        if self.done:
            return self.reset()
            
        prev_price = self.prices[self.current_step - 1]
        current_price = self.prices[self.current_step]
        # Reward is calculated based on profit from the previous position.
        reward = (current_price - prev_price) * self.position
        
        # Update position based on the action.
        if action == 1:
            self.position = 1
        elif action == 2:
            self.position = -1
        # Otherwise, hold the current position.
            
        self.current_step += 1
        self.done = self.current_step >= len(self.prices)
        
        obs = self._get_obs()
        # Gymnasium's step returns (observation, reward, terminated, truncated, info)
        return obs, reward, self.done, False, {}

    def render(self, mode="human"):
        if mode == "human":
            print(f"Step: {self.current_step}, Price: {self.prices[self.current_step]:.4f}, Position: {self.position}")
