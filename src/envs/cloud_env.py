# src/envs/cloud_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import psutil  # NEW: to track system CPU and memory usage

class CloudEnv(gym.Env):
    """
    A simulated cloud resource allocation environment.
    Continuous action in [0,1] maps to worker count in [min_workers, max_workers].
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 max_workers=20,
                 min_workers=1,
                 arrival_rate=3.0,
                 max_queue=100,
                 seed=None):
        super().__init__()
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.arrival_rate = arrival_rate
        self.max_queue = max_queue

        # Observation: queue_length, avg_demand, current_workers, recent_throughput
        obs_low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Action: continuous scalar [0,1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.rng = np.random.default_rng(seed)
        self._reset_internal()

    def _reset_internal(self):
        self.time = 0
        self.queue = []
        self.current_workers = self.min_workers
        self.recent_throughput = 0
        self.history = {"throughput": [], "workers": [], "queue_len": [], "cpu": [], "mem": [], "latency": []}

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._reset_internal()
        return self._get_obs(), {}

    def _get_obs(self):
        q_len = len(self.queue) / max(1, self.max_queue)
        avg_demand = (np.mean([j['demand'] for j in self.queue]) / 10.0) if self.queue else 0.0
        workers_norm = (self.current_workers - self.min_workers) / max(1, (self.max_workers - self.min_workers))
        thr_norm = min(self.recent_throughput / 10.0, 1.0)
        return np.array([q_len, avg_demand, workers_norm, thr_norm], dtype=np.float32)

    def step(self, action):
        # Map action scalar to worker count
        target = float(np.clip(action, 0.0, 1.0)[0])
        desired_workers = int(round(self.min_workers + target * (self.max_workers - self.min_workers)))
        self.current_workers = desired_workers

        # Job arrivals (Poisson)
        arrivals = self.rng.poisson(self.arrival_rate)
        for _ in range(arrivals):
            demand = max(0.1, float(self.rng.normal(loc=1.0, scale=0.5)))
            service_time = max(1, int(np.ceil(demand * (1 + float(self.rng.random())))))
            job = {"demand": float(demand), "remaining": int(service_time)}
            if len(self.queue) < self.max_queue:
                self.queue.append(job)

        # Process jobs
        finished = 0
        available = self.current_workers
        i = 0
        while i < len(self.queue) and available > 0:
            job = self.queue[i]
            job['remaining'] -= 1
            if job['remaining'] <= 0:
                finished += 1
                self.queue.pop(i)
            else:
                i += 1
            available -= 1

        # SLA violations
        sla_violations = max(0, len(self.queue) - int(0.5 * self.max_workers))

        # Resource cost (workers)
        resource_cost = 0.05 * float(self.current_workers)

        # Simulated latency (higher queue = higher latency)
        latency = len(self.queue) * 5 + np.random.uniform(10, 50)

        # System stats
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().percent

        # Reward: throughput - cost - penalty for SLA violations - latency penalty
        reward = float(finished) - resource_cost - 0.1 * float(sla_violations) - 0.01 * (latency / 10.0)

        # Log info
        self.recent_throughput = finished
        self.history["throughput"].append(finished)
        self.history["workers"].append(self.current_workers)
        self.history["queue_len"].append(len(self.queue))
        self.history["cpu"].append(cpu_usage)
        self.history["mem"].append(memory_usage)
        self.history["latency"].append(latency)

        self.time += 1
        done = self.time >= 500

        obs = self._get_obs()
        info = {
            "finished": finished,
            "resource_cost": resource_cost,
            "sla_violations": sla_violations,
            "queue_len": len(self.queue),
            "workers": self.current_workers,
            "cpu": cpu_usage,
            "memory": memory_usage,
            "latency": latency
        }
        return obs, reward, done, False, info

    def render(self):
        print(f"t={self.time} workers={self.current_workers} queue={len(self.queue)} thr={self.recent_throughput}")

    def close(self):
        pass
