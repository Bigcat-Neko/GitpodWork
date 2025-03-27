#!/usr/bin/env python
import os
from stable_baselines3 import PPO
from trading_env import TradingEnv
def train_ppo():
    print("\n===== Training PPO Model =====")
    env = TradingEnv()  # Ensure your env is properly defined
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("models/ppo_model")
    print("PPO model saved as models/ppo_model")
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_ppo()
