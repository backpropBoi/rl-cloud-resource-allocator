# src/envs/cloud_env.py
"""Simulated cloud resource allocation environment for RL training."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

WORKLOAD_SCENARIOS = ("steady", "burst", "spike")


class CloudEnv(gym.Env):
    """
    A simulated cloud resource allocation environment.

    Continuous action in [0, 1] maps to a desired worker count in
    [min_workers, max_workers]. Actual workers change after scale-up delay
    or scale-down cooldown to mimic real autoscaling behavior.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_workers: int = 20,
        min_workers: int = 1,
        arrival_rate: float = 3.0,
        max_queue: int = 100,
        episode_length: int = 500,
        jobs_per_worker: int = 2,
        scale_up_delay: int = 3,
        scale_down_cooldown: int = 5,
        workload: str = "steady",
        reward_weights: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        if workload not in WORKLOAD_SCENARIOS:
            raise ValueError(f"workload must be one of {WORKLOAD_SCENARIOS}")

        self.max_workers = max_workers
        self.min_workers = min_workers
        self.base_arrival_rate = arrival_rate
        self.arrival_rate = arrival_rate
        self.max_queue = max_queue
        self.episode_length = episode_length
        self.jobs_per_worker = jobs_per_worker
        self.scale_up_delay = scale_up_delay
        self.scale_down_cooldown = scale_down_cooldown
        self.workload = workload

        default_weights = {
            "throughput_weight": 1.0,
            "resource_cost_per_worker": 0.05,
            "sla_penalty_weight": 0.1,
            "latency_penalty_weight": 0.01,
            "overflow_penalty_weight": 0.5,
            "scaling_penalty_weight": 0.02,
            "sla_queue_threshold_ratio": 0.5,
        }
        self.reward_weights = {**default_weights, **(reward_weights or {})}

        # queue_len, avg_demand, workers, throughput, queue_delta, utilization, pending_scale, time
        obs_low = np.zeros(8, dtype=np.float32)
        obs_high = np.ones(8, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.rng = np.random.default_rng(seed)
        self._reset_internal()

    def _reset_internal(self) -> None:
        self.time = 0
        self.queue: list[dict[str, float | int]] = []
        self.prev_queue_len = 0
        self.desired_workers = self.min_workers
        self.current_workers = self.min_workers
        self.pending_scale_steps = 0
        self.cooldown_remaining = 0
        self.recent_throughput = 0
        self.jobs_dropped = 0
        self.history: dict[str, list[float | int]] = {
            "throughput": [],
            "workers": [],
            "queue_len": [],
            "cpu": [],
            "mem": [],
            "latency": [],
            "reward_components": [],
        }

    def seed(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed(seed)
        if options and "workload" in options:
            workload = options["workload"]
            if workload not in WORKLOAD_SCENARIOS:
                raise ValueError(f"workload must be one of {WORKLOAD_SCENARIOS}")
            self.workload = workload
        self._reset_internal()
        return self._get_obs(), {}

    def _effective_arrival_rate(self) -> float:
        t = self.time
        if self.workload == "steady":
            return self.base_arrival_rate
        if self.workload == "burst":
            # Periodic bursts every ~100 steps
            phase = t % 100
            return self.base_arrival_rate * (3.0 if 40 <= phase <= 70 else 1.0)
        # spike: large spikes lasting 10 steps, occurring every ~150 steps
        phase = t % 150
        if 0 < phase <= 10:
            return self.base_arrival_rate * 5.0
        return self.base_arrival_rate

    def _action_to_desired_workers(self, action: np.ndarray | float) -> int:
        if isinstance(action, (list, tuple, np.ndarray)):
            target = float(np.clip(action[0], 0.0, 1.0))
        else:
            target = float(np.clip(action, 0.0, 1.0))
        span = self.max_workers - self.min_workers
        return int(round(self.min_workers + target * span))

    def _apply_scaling(self, desired: int) -> int:
        """Return scaling cost (number of worker changes initiated this step)."""
        scaling_cost = 0
        if desired > self.current_workers:
            if self.pending_scale_steps <= 0:
                self.pending_scale_steps = self.scale_up_delay
                self.desired_workers = desired
                scaling_cost = desired - self.current_workers
        elif desired < self.current_workers:
            if self.cooldown_remaining <= 0:
                self.desired_workers = desired
                self.cooldown_remaining = self.scale_down_cooldown
                scaling_cost = self.current_workers - desired
            else:
                self.desired_workers = self.current_workers
        else:
            self.desired_workers = desired

        if self.pending_scale_steps > 0:
            self.pending_scale_steps -= 1
            if self.pending_scale_steps == 0:
                # Apply full desired count once scale-up delay elapses
                self.current_workers = self.desired_workers

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining == 0 and self.desired_workers < self.current_workers:
                # Apply full desired count once cooldown elapses
                self.current_workers = self.desired_workers

        self.current_workers = int(np.clip(self.current_workers, self.min_workers, self.max_workers))
        return scaling_cost

    def _simulated_utilization(self) -> tuple[float, float, float]:
        capacity = max(1, self.current_workers * self.jobs_per_worker)
        active = min(len(self.queue), capacity)
        cpu_usage = min(100.0, (active / capacity) * 100.0 + self.rng.uniform(0, 5))
        mem_usage = min(
            100.0,
            20.0 + len(self.queue) * 0.5 + self.current_workers * 2.0,
        )
        utilization = active / capacity
        return utilization, cpu_usage, mem_usage

    def _get_obs(self) -> np.ndarray:
        q_len = len(self.queue) / max(1, self.max_queue)
        avg_demand = min((np.mean([j["demand"] for j in self.queue]) / 10.0) if self.queue else 0.0, 1.0)
        workers_norm = (self.current_workers - self.min_workers) / max(1, self.max_workers - self.min_workers)
        thr_norm = min(self.recent_throughput / 10.0, 1.0)
        queue_delta = np.clip((len(self.queue) - self.prev_queue_len) / max(1, self.max_queue), -1.0, 1.0)
        queue_delta_norm = (queue_delta + 1.0) / 2.0
        utilization, _, _ = self._simulated_utilization()
        pending = abs(self.desired_workers - self.current_workers) / max(1, self.max_workers - self.min_workers)
        time_norm = min(self.time / max(1, self.episode_length), 1.0)
        obs = np.array(
            [q_len, avg_demand, workers_norm, thr_norm, queue_delta_norm, utilization, pending, time_norm],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)

    def step(self, action):
        desired = self._action_to_desired_workers(action)
        scaling_events = self._apply_scaling(desired)

        arrivals = self.rng.poisson(self._effective_arrival_rate())
        dropped = 0
        for _ in range(arrivals):
            demand = max(0.1, float(self.rng.normal(loc=1.0, scale=0.5)))
            service_time = max(1, int(np.ceil(demand * (1 + float(self.rng.random())))))
            # arrived_at lets us compute true queue wait regardless of whether
            # the job is currently being processed
            job = {"demand": float(demand), "remaining": int(service_time), "arrived_at": self.time}
            if len(self.queue) < self.max_queue:
                self.queue.append(job)
            else:
                dropped += 1

        finished = 0
        capacity = self.current_workers * self.jobs_per_worker
        slots = capacity
        i = 0
        while i < len(self.queue) and slots > 0:
            job = self.queue[i]
            job["remaining"] = int(job["remaining"]) - 1
            if job["remaining"] <= 0:
                finished += 1
                self.queue.pop(i)
            else:
                i += 1
            slots -= 1

        sla_threshold = int(self.reward_weights["sla_queue_threshold_ratio"] * self.max_workers)
        # True wait = steps elapsed since the job arrived
        wait_times = [self.time - j["arrived_at"] for j in self.queue]
        sla_violations = sum(1 for w in wait_times if w > sla_threshold)
        resource_cost = self.reward_weights["resource_cost_per_worker"] * float(self.current_workers)
        # Latency: mean elapsed wait across all queued jobs, plus baseline network jitter
        mean_wait = float(np.mean(wait_times)) if wait_times else 0.0
        latency = mean_wait * 5.0 + float(self.rng.uniform(5, 15))
        utilization, cpu_usage, mem_usage = self._simulated_utilization()

        rw = self.reward_weights
        reward = (
            rw["throughput_weight"] * float(finished)
            - resource_cost
            - rw["sla_penalty_weight"] * float(sla_violations)
            - rw["latency_penalty_weight"] * (latency / 10.0)
            - rw["overflow_penalty_weight"] * float(dropped)
            - rw["scaling_penalty_weight"] * float(scaling_events)
        )

        reward_components = {
            "throughput": float(finished),
            "resource_cost": resource_cost,
            "sla_violations": float(sla_violations),
            "latency": latency,
            "dropped": float(dropped),
            "scaling_events": float(scaling_events),
            "total_reward": reward,
        }

        self.recent_throughput = finished
        self.jobs_dropped += dropped
        self.prev_queue_len = len(self.queue)
        self.history["throughput"].append(finished)
        self.history["workers"].append(self.current_workers)
        self.history["queue_len"].append(len(self.queue))
        self.history["cpu"].append(cpu_usage)
        self.history["mem"].append(mem_usage)
        self.history["latency"].append(latency)
        self.history["reward_components"].append(reward_components)

        self.time += 1
        terminated = False
        truncated = self.time >= self.episode_length

        obs = self._get_obs()
        info = {
            "finished": finished,
            "resource_cost": resource_cost,
            "sla_violations": sla_violations,
            "queue_len": len(self.queue),
            "workers": self.current_workers,
            "desired_workers": self.desired_workers,
            "cpu": cpu_usage,
            "memory": mem_usage,
            "latency": latency,
            "utilization": utilization,
            "dropped": dropped,
            "scaling_events": scaling_events,
            "reward_components": reward_components,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(
            f"t={self.time} workers={self.current_workers}/{self.desired_workers} "
            f"queue={len(self.queue)} thr={self.recent_throughput}"
        )

    def close(self):
        pass


def register_env() -> None:
    gym.register(
        id="CloudAlloc-v0",
        entry_point="src.envs.cloud_env:CloudEnv",
        max_episode_steps=500,
    )
