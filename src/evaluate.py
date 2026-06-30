# src/evaluate.py
"""Evaluate SAC model and baseline policies with detailed metrics."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.baselines import get_baseline_policies
from src.config import load_config
from src.envs.cloud_env import CloudEnv


@dataclass
class EpisodeMetrics:
    total_reward: float = 0.0
    throughput: float = 0.0
    resource_cost: float = 0.0
    sla_violations: float = 0.0
    latency: float = 0.0
    dropped: float = 0.0
    scaling_events: float = 0.0
    steps: int = 0


@dataclass
class PolicyResults:
    policy_name: str
    episodes: int
    mean_reward: float
    std_reward: float
    mean_throughput: float
    mean_resource_cost: float
    mean_sla_violations: float
    mean_latency: float
    mean_dropped: float
    mean_scaling_events: float
    episode_rewards: list[float] = field(default_factory=list)


def run_episode(env: CloudEnv, policy, use_sb3: bool = False) -> EpisodeMetrics:
    obs, _ = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    metrics = EpisodeMetrics()
    done = False
    while not done:
        if use_sb3:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        metrics.total_reward += float(reward)
        metrics.throughput += info.get("finished", 0)
        metrics.resource_cost += info.get("resource_cost", 0.0)
        metrics.sla_violations += info.get("sla_violations", 0)
        metrics.latency += info.get("latency", 0.0)
        metrics.dropped += info.get("dropped", 0)
        metrics.scaling_events += info.get("scaling_events", 0)
        metrics.steps += 1
    return metrics


def aggregate_episodes(episodes: list[EpisodeMetrics], policy_name: str) -> PolicyResults:
    rewards = [e.total_reward for e in episodes]
    return PolicyResults(
        policy_name=policy_name,
        episodes=len(episodes),
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_throughput=float(np.mean([e.throughput for e in episodes])),
        mean_resource_cost=float(np.mean([e.resource_cost for e in episodes])),
        mean_sla_violations=float(np.mean([e.sla_violations for e in episodes])),
        mean_latency=float(np.mean([e.latency for e in episodes])),
        mean_dropped=float(np.mean([e.dropped for e in episodes])),
        mean_scaling_events=float(np.mean([e.scaling_events for e in episodes])),
        episode_rewards=rewards,
    )


def build_env(env_cfg: dict, reward_cfg: dict, workload: str | None = None) -> CloudEnv:
    return CloudEnv(
        max_workers=env_cfg["max_workers"],
        min_workers=env_cfg["min_workers"],
        arrival_rate=env_cfg["arrival_rate"],
        max_queue=env_cfg["max_queue"],
        episode_length=env_cfg["episode_length"],
        jobs_per_worker=env_cfg["jobs_per_worker"],
        scale_up_delay=env_cfg["scale_up_delay"],
        scale_down_cooldown=env_cfg["scale_down_cooldown"],
        workload=workload or env_cfg["workload"],
        reward_weights=reward_cfg,
        seed=env_cfg.get("seed"),
    )


def evaluate_policy(
    policy,
    policy_name: str,
    env_cfg: dict,
    reward_cfg: dict,
    episodes: int = 10,
    use_sb3: bool = False,
    workload: str | None = None,
) -> PolicyResults:
    episode_metrics = []
    for ep in range(episodes):
        env = build_env(env_cfg, reward_cfg, workload=workload)
        env.seed(int(env_cfg.get("seed", 0)) + ep)
        episode_metrics.append(run_episode(env, policy, use_sb3=use_sb3))
        env.close()
    return aggregate_episodes(episode_metrics, policy_name)


def evaluate_all(
    model_path: str | None,
    config_path: str | None = None,
    episodes: int = 10,
    workload: str | None = None,
    output_dir: str | None = None,
) -> list[PolicyResults]:
    config = load_config(config_path)
    env_cfg = config["env"]
    reward_cfg = config.get("reward", {})
    paths = config["paths"]
    out_dir = Path(output_dir or paths["log_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[PolicyResults] = []

    for baseline in get_baseline_policies(seed=env_cfg.get("seed")):
        results.append(
            evaluate_policy(
                baseline,
                baseline.name,
                env_cfg,
                reward_cfg,
                episodes=episodes,
                workload=workload,
            )
        )

    if model_path and os.path.exists(model_path):
        model = SAC.load(model_path)
        vec_norm_path = Path(model_path).parent / "vec_normalize.pkl"
        eval_workload = workload or env_cfg["workload"]
        if workload is not None and workload != env_cfg["workload"] and vec_norm_path.exists():
            print(
                f"Warning: evaluating on workload '{workload}' but VecNormalize statistics "
                f"were fit on '{env_cfg['workload']}'. Observation normalization may be inaccurate."
            )
        sac_results = []
        for ep in range(episodes):
            ep_seed = int(env_cfg.get("seed", 0)) + ep + 100

            def _make_env(_seed=ep_seed):
                env = build_env(env_cfg, reward_cfg, workload=eval_workload)
                env.seed(_seed)
                return env

            venv = DummyVecEnv([_make_env])
            if vec_norm_path.exists():
                venv = VecNormalize.load(str(vec_norm_path), venv)
                venv.training = False
                venv.norm_reward = False

            metrics = EpisodeMetrics()
            obs = venv.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = venv.step(action)
                done = bool(terminated[0]) or bool(truncated[0])
                step_info = info[0]
                metrics.total_reward += float(reward[0])
                metrics.throughput += step_info.get("finished", 0)
                metrics.resource_cost += step_info.get("resource_cost", 0.0)
                metrics.sla_violations += step_info.get("sla_violations", 0)
                metrics.latency += step_info.get("latency", 0.0)
                metrics.dropped += step_info.get("dropped", 0)
                metrics.scaling_events += step_info.get("scaling_events", 0)
                metrics.steps += 1
            sac_results.append(metrics)
            venv.close()
        results.append(aggregate_episodes(sac_results, "sac"))
    elif model_path:
        raise FileNotFoundError(f"Model not found: {model_path}")

    _save_results(results, out_dir, workload or env_cfg["workload"])
    return results


def _save_results(results: list[PolicyResults], out_dir: Path, workload: str) -> None:
    serializable = []
    for r in results:
        row = asdict(r)
        serializable.append(row)

    json_path = out_dir / f"eval_results_{workload}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    csv_path = out_dir / f"eval_results_{workload}.csv"
    headers = [
        "policy_name", "episodes", "mean_reward", "std_reward",
        "mean_throughput", "mean_resource_cost", "mean_sla_violations",
        "mean_latency", "mean_dropped", "mean_scaling_events",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            f.write(
                f"{r.policy_name},{r.episodes},{r.mean_reward:.4f},{r.std_reward:.4f},"
                f"{r.mean_throughput:.4f},{r.mean_resource_cost:.4f},{r.mean_sla_violations:.4f},"
                f"{r.mean_latency:.4f},{r.mean_dropped:.4f},{r.mean_scaling_events:.4f}\n"
            )

    print(f"Saved evaluation results to {json_path} and {csv_path}")


def print_summary(results: list[PolicyResults]) -> None:
    print("\n=== Evaluation Summary ===")
    for r in results:
        print(
            f"{r.policy_name:>10} | reward={r.mean_reward:8.2f} ± {r.std_reward:5.2f} | "
            f"throughput={r.mean_throughput:7.1f} | cost={r.mean_resource_cost:7.1f} | "
            f"sla={r.mean_sla_violations:6.1f} | latency={r.mean_latency:8.1f} | "
            f"dropped={r.mean_dropped:5.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAC and baseline policies")
    parser.add_argument("--model", type=str, default="./models/sac_cloud_final.zip")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--workload", type=str, default=None, choices=["steady", "burst", "spike"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-model", action="store_true", help="Evaluate baselines only")
    args = parser.parse_args()

    model_path = None if args.no_model else args.model
    results = evaluate_all(
        model_path=model_path,
        config_path=args.config,
        episodes=args.episodes,
        workload=args.workload,
        output_dir=args.output_dir,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
