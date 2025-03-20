# train_ppo.py

import gymnasium as gym  # Use Gymnasium instead of Gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os

# -------------------------------
# Define a Trading Environment
# -------------------------------
class TradingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Action space: 0 = HOLD, 1 = BUY, 2 = SELL
        self.action_space = gym.spaces.Discrete(3)
        # Observation space: 60 price points normalized between 0 and 1 (dtype float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(60,), dtype=np.float32)
        self.current_step = 0
        # Create dummy data; replace with your actual market data in production.
        self.data = np.random.random(1000)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        obs = self.data[self.current_step:self.current_step+60].astype(np.float32)
        return obs, {}  # Gymnasium reset returns (observation, info)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= (len(self.data) - 60)
        obs = self.data[self.current_step:self.current_step+60].astype(np.float32)
        # Dummy reward: price difference
        reward = self.data[self.current_step+59] - self.data[self.current_step]
        info = {}
        # Gymnasium step returns: (observation, reward, terminated, truncated, info)
        return obs, reward, done, False, info

# -------------------------------
# Create and Check the Environment
# -------------------------------
env = TradingEnv()
# Check the environment using Gymnasium's checker
check_env(env, warn=True)

# -------------------------------
# Create the PPO Agent
# -------------------------------
model = PPO("MlpPolicy", env, verbose=1)

# -------------------------------
# Train the Model
# -------------------------------
TIMESTEPS = 100000  # Increase this number for longer training
model.learn(total_timesteps=TIMESTEPS)

# -------------------------------
# Save the Model
# -------------------------------
os.makedirs("models", exist_ok=True)
model.save("models/trading_ppo.zip")
print("PPO model trained and saved as 'models/trading_ppo.zip'")
