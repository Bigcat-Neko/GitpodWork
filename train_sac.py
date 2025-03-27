#!/usr/bin/env python
import os
from stable_baselines3 import SAC
from trading_env import TradingEnv
def train_sac():
    print("\n===== Training SAC Model =====")
    env = TradingEnv()  # Ensure your environment supports continuous actions for SAC
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("models/sac_model")
    print("SAC model saved as models/sac_model")
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_sac()
