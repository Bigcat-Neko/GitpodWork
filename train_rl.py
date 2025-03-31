#!/usr/bin/env python
import os
from stable_baselines3 import DQN
from trading_env import TradingEnv

def train_dqn():
    print("\n===== Training DQN Model =====")
    # Set window_size to 30 to match the new observation dimension
    env = TradingEnv(window_size=30)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("models/dqn_model")
    print("DQN model saved as models/dqn_model")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_dqn()
