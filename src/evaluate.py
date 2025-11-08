# src/evaluate.py
import numpy as np
from stable_baselines3 import SAC
from src.envs.cloud_env import CloudEnv
import os

def evaluate(model_path, episodes=10):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    env = CloudEnv()
    model = SAC.load(model_path)

    results = []
    for e in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, truncated, info = env.step(action)
            total_reward += rew
        results.append(total_reward)
    print('Mean reward:', np.mean(results), 'Std:', np.std(results))

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else './models/sac_cloud_final.zip'
    evaluate(path)
