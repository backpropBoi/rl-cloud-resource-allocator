# src/train.py
"""Train a SAC agent for cloud resource allocation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from src.callbacks import TrainingLoggingCallback
from src.config import load_config
from src.envs.cloud_env import CloudEnv, register_env


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env_fn(env_cfg: dict, rank: int = 0):
    def _init():
        env_seed = env_cfg.get("seed")
        if env_seed is not None:
            env_seed = int(env_seed) + rank
        env = CloudEnv(
            max_workers=env_cfg["max_workers"],
            min_workers=env_cfg["min_workers"],
            arrival_rate=env_cfg["arrival_rate"],
            max_queue=env_cfg["max_queue"],
            episode_length=env_cfg["episode_length"],
            jobs_per_worker=env_cfg["jobs_per_worker"],
            scale_up_delay=env_cfg["scale_up_delay"],
            scale_down_cooldown=env_cfg["scale_down_cooldown"],
            workload=env_cfg["workload"],
            reward_weights=env_cfg.get("reward_weights"),
            seed=env_seed,
        )
        return Monitor(env)

    return _init


def build_vec_env(env_cfg: dict, n_envs: int, seed: int):
    if n_envs <= 1:
        return DummyVecEnv([make_env_fn(env_cfg, 0)])
    return make_vec_env(
        make_env_fn(env_cfg),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
    )


def main():
    parser = argparse.ArgumentParser(description="Train SAC cloud resource allocator")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--n-envs", type=int, default=None, help="Override parallel env count")
    parser.add_argument("--workload", type=str, default=None, choices=["steady", "burst", "spike"])
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    env_cfg = config["env"]
    train_cfg = config["training"]
    paths = config["paths"]

    if args.timesteps is not None:
        train_cfg["total_timesteps"] = args.timesteps
    if args.n_envs is not None:
        train_cfg["n_envs"] = args.n_envs
    if args.workload is not None:
        env_cfg["workload"] = args.workload
    if args.seed is not None:
        train_cfg["seed"] = args.seed
        env_cfg["seed"] = args.seed

    env_cfg["reward_weights"] = config.get("reward", {})

    seed = int(train_cfg["seed"])
    set_global_seed(seed)

    model_dir = Path(os.environ.get("MODEL_DIR", paths["model_dir"]))
    logs_dir = Path(os.environ.get("LOG_DIR", paths["log_dir"]))
    tb_dir = Path(paths["tensorboard_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    register_env()

    n_envs = int(train_cfg["n_envs"])
    env = build_vec_env(env_cfg, n_envs, seed)

    eval_env = DummyVecEnv([make_env_fn(env_cfg, 1000)])
    if train_cfg.get("use_vec_normalize", True):
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env.training = False
        eval_env.norm_reward = False

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=float(train_cfg["learning_rate"]),
        buffer_size=int(train_cfg["buffer_size"]),
        batch_size=int(train_cfg["batch_size"]),
        learning_starts=int(train_cfg["learning_starts"]),
        ent_coef="auto",
        tensorboard_log=str(tb_dir),
        seed=seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(int(train_cfg["checkpoint_freq"]) // n_envs, 1),
        save_path=str(model_dir),
        name_prefix="sac_cloud",
    )
    log_callback = TrainingLoggingCallback(log_dir=str(logs_dir), log_file="training_log.csv")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(logs_dir),
        eval_freq=max(int(train_cfg["eval_freq"]) // n_envs, 1),
        n_eval_episodes=int(train_cfg["eval_episodes"]),
        deterministic=True,
    )

    model.learn(
        total_timesteps=int(train_cfg["total_timesteps"]),
        callback=[checkpoint_callback, log_callback, eval_callback],
        progress_bar=True,
    )

    final_path = model_dir / "sac_cloud_final"
    model.save(str(final_path))
    if isinstance(env, VecNormalize):
        env.save(str(model_dir / "vec_normalize.pkl"))

    env.close()
    eval_env.close()
    print(f"Training complete. Model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
