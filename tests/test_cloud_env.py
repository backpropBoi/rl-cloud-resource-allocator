"""Tests for CloudEnv."""

import numpy as np
import pytest

from src.envs.cloud_env import CloudEnv, WORKLOAD_SCENARIOS, register_env


@pytest.fixture
def env():
    e = CloudEnv(seed=42, episode_length=50)
    yield e
    e.close()


def test_reset_clears_state(env):
    env.step(np.array([0.8], dtype=np.float32))
    obs, _ = env.reset()
    assert len(env.queue) == 0
    assert env.time == 0
    assert env.current_workers == env.min_workers
    assert env.jobs_dropped == 0
    assert len(env.history["throughput"]) == 0
    assert obs.shape == (10,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)


def test_observation_bounds(env):
    obs, _ = env.reset(seed=123)
    for _ in range(100):
        assert np.all(obs >= 0.0), obs
        assert np.all(obs <= 1.0), obs
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)


def test_action_maps_to_worker_count(env):
    env.reset()
    low_obs, _, _, _, info_low = env.step(np.array([0.0], dtype=np.float32))
    assert info_low["workers"] == env.min_workers

    env.reset()
    _, _, _, _, info_high = env.step(np.array([1.0], dtype=np.float32))
    assert info_high["workers"] >= env.min_workers
    assert info_high["workers"] <= env.max_workers


def test_sla_violations_reduce_reward(env):
    env.reset(seed=0)
    rewards_with_load = []
    for _ in range(20):
        _, r, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        rewards_with_load.append(r)
    assert np.mean(rewards_with_load) < 0 or len(env.queue) > 0


def test_overflow_penalty(env):
    small_queue_env = CloudEnv(max_queue=2, arrival_rate=20.0, seed=7, episode_length=10)
    small_queue_env.reset()
    total_dropped = 0
    for _ in range(5):
        _, _, _, _, info = small_queue_env.step(np.array([0.1], dtype=np.float32))
        total_dropped += info.get("dropped", 0)
    assert total_dropped > 0
    small_queue_env.close()


def test_truncated_at_episode_length(env):
    env.episode_length = 10
    env.reset()
    truncated = False
    for _ in range(20):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    assert truncated
    assert env.time == 10


@pytest.mark.parametrize("workload", WORKLOAD_SCENARIOS)
def test_workload_scenarios(workload):
    e = CloudEnv(workload=workload, seed=1, episode_length=200)
    e.reset()
    for _ in range(200):
        _, _, _, truncated, _ = e.step(np.array([0.5], dtype=np.float32))
        if truncated:
            break
    e.close()


def test_gym_registration():
    register_env()
    import gymnasium as gym

    env = gym.make("CloudAlloc-v0")
    obs, _ = env.reset()
    assert obs.shape == (10,)
    env.close()


def test_scale_up_delay_applied():
    """Workers should not change immediately on scale-up; they change after the delay."""
    e = CloudEnv(seed=0, episode_length=20, scale_up_delay=3, min_workers=1, max_workers=10)
    e.reset()
    initial_workers = e.current_workers
    # Request max workers
    _, _, _, _, info = e.step(np.array([1.0], dtype=np.float32))
    # Workers should not have jumped to max immediately
    assert info["workers"] == initial_workers
    e.close()


def test_scale_up_completes_after_delay():
    """Workers should reach the desired count after scale_up_delay steps."""
    delay = 3
    e = CloudEnv(seed=0, episode_length=20, scale_up_delay=delay, min_workers=1, max_workers=10)
    e.reset()
    # Issue scale-up request
    e.step(np.array([1.0], dtype=np.float32))
    # Step through the delay
    for _ in range(delay):
        _, _, _, _, info = e.step(np.array([1.0], dtype=np.float32))
    assert info["workers"] == e.max_workers
    e.close()


def test_scale_down_cooldown_applied():
    """Workers should not drop immediately; cooldown must elapse first."""
    cooldown = 5
    e = CloudEnv(seed=0, episode_length=30, scale_up_delay=1, scale_down_cooldown=cooldown,
                 min_workers=1, max_workers=10)
    e.reset()
    # Scale up first
    for _ in range(3):
        e.step(np.array([1.0], dtype=np.float32))
    workers_before_down = e.current_workers
    # Now request scale-down
    e.step(np.array([0.0], dtype=np.float32))
    _, _, _, _, info = e.step(np.array([0.0], dtype=np.float32))
    # Workers should not have dropped yet (cooldown still active)
    assert info["workers"] == workers_before_down or e.cooldown_remaining > 0
    e.close()


def test_spike_workload_has_duration():
    """Spike workload should sustain elevated arrival rate for multiple steps."""
    e = CloudEnv(seed=0, episode_length=200, workload="spike", arrival_rate=3.0)
    e.reset()
    # Force time to just after a spike start boundary (phase 1 of 150-step cycle)
    e.time = 151
    spike_rates = [e._effective_arrival_rate() for _ in range(10)]
    # All 10 steps inside the spike window should be elevated
    assert all(r > 3.0 for r in spike_rates[:10])
    e.close()


def test_latency_increases_with_queue():
    """Latency should be higher when the queue is backed up."""
    e = CloudEnv(seed=1, episode_length=100, arrival_rate=10.0, min_workers=1, max_workers=20)
    # Reset with same seed each time so Poisson draws are identical
    e.reset(seed=1)
    for _ in range(30):
        e.step(np.array([0.0], dtype=np.float32))
    high_latency = np.mean(e.history["latency"][-10:])

    e.reset(seed=1)
    for _ in range(30):
        e.step(np.array([1.0], dtype=np.float32))
    low_latency = np.mean(e.history["latency"][-10:])

    assert high_latency > low_latency
    e.close()
