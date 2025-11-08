# src/callbacks.py
import os
import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TrainingLoggingCallback(BaseCallback):
    """
    Logs episode-level metrics (reward, length) and environment-specific stats
    to a CSV file. Works with a DummyVecEnv containing Monitor-wrapped env(s).
    """

    def __init__(self, log_dir="logs", log_file="training_log.csv", verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_file = log_file
        os.makedirs(self.log_dir, exist_ok=True)
        self.filepath = os.path.join(self.log_dir, self.log_file)
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_step", "episode_reward", "episode_length",
                    "mean_recent_throughput", "mean_workers", "mean_queue_len"
                ])
        self.ep_rewards = None
        self.ep_lengths = None

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.ep_rewards = [0.0] * n_envs
        self.ep_lengths = [0] * n_envs

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if rewards is not None:
            for i, r in enumerate(rewards):
                # rewards can be float or array-like
                if isinstance(r, (list, tuple, np.ndarray)):
                    self.ep_rewards[i] += float(np.array(r).sum())
                else:
                    self.ep_rewards[i] += float(r)
                self.ep_lengths[i] += 1

        if dones is not None:
            for i, done in enumerate(dones):
                if done:
                    ep_reward = self.ep_rewards[i]
                    ep_length = self.ep_lengths[i]
                    mean_throughput = 0.0
                    mean_workers = 0.0
                    mean_queue = 0.0
                    try:
                        inner_env = self.training_env.envs[i]
                        hist = getattr(inner_env, "history", None)
                        if hist:
                            mean_throughput = float(np.mean(hist.get("throughput", [0.0])))
                            mean_workers = float(np.mean(hist.get("workers", [inner_env.current_workers])))
                            mean_queue = float(np.mean(hist.get("queue_len", [0.0])))
                    except Exception:
                        pass

                    with open(self.filepath, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            self.num_timesteps,
                            ep_reward,
                            ep_length,
                            mean_throughput,
                            mean_workers,
                            mean_queue
                        ])

                    self.ep_rewards[i] = 0.0
                    self.ep_lengths[i] = 0
        return True
