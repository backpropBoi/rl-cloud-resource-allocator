# app/streamlit_dashboard.py
"""Streamlit dashboard for training metrics, evaluation, and live simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.baselines import FixedWorkersPolicy, ProportionalPolicy, RandomPolicy, ThresholdPolicy
from src.config import load_config
from src.envs.cloud_env import CloudEnv, WORKLOAD_SCENARIOS
from src.evaluate import evaluate_all, print_summary

st.set_page_config(layout="wide", page_title="RL Cloud — Dashboard")
st.title("RL Cloud Resource Allocation — Dashboard (SAC)")

config = load_config()
env_cfg = config["env"]
reward_cfg = config.get("reward", {})
paths = config["paths"]


@st.cache_resource
def load_sac_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        return None, None
    model = SAC.load(str(path))
    vec_norm_path = path.parent / "vec_normalize.pkl"
    vec_norm = str(vec_norm_path) if vec_norm_path.exists() else None
    return model, vec_norm


def make_env(workload: str, seed: int = 0) -> CloudEnv:
    return CloudEnv(
        max_workers=env_cfg["max_workers"],
        min_workers=env_cfg["min_workers"],
        arrival_rate=env_cfg["arrival_rate"],
        max_queue=env_cfg["max_queue"],
        episode_length=env_cfg["episode_length"],
        jobs_per_worker=env_cfg["jobs_per_worker"],
        scale_up_delay=env_cfg["scale_up_delay"],
        scale_down_cooldown=env_cfg["scale_down_cooldown"],
        workload=workload,
        reward_weights=reward_cfg,
        seed=seed,
    )


def run_policy_simulation(policy, policy_name: str, workload: str, steps: int, model=None, vec_norm_path=None):
    env = make_env(workload)
    obs, _ = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    venv = None
    if model is not None:
        _wl = workload
        venv = DummyVecEnv([lambda _w=_wl: make_env(_w)])
        if vec_norm_path and Path(vec_norm_path).exists():
            venv = VecNormalize.load(vec_norm_path, venv)
            venv.training = False
            venv.norm_reward = False
        obs = venv.reset()

    data = {
        "step": [],
        "workers": [],
        "queue_len": [],
        "throughput": [],
        "cpu": [],
        "memory": [],
        "latency": [],
        "reward": [],
        "resource_cost": [],
        "sla_violations": [],
        "dropped": [],
    }

    progress = st.progress(0, text=f"Running {policy_name} simulation...")
    for t in range(steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            step_info = info[0]
            step_reward = float(reward[0])
            done = bool(done[0])
        else:
            action = policy.predict(obs)
            obs, step_reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

        data["step"].append(t)
        data["workers"].append(step_info.get("workers", 0))
        data["queue_len"].append(step_info.get("queue_len", 0))
        data["throughput"].append(step_info.get("finished", 0))
        data["cpu"].append(step_info.get("cpu", 0))
        data["memory"].append(step_info.get("memory", 0))
        data["latency"].append(step_info.get("latency", 0))
        data["reward"].append(step_reward)
        rc = step_info.get("reward_components", {})
        data["resource_cost"].append(rc.get("resource_cost", step_info.get("resource_cost", 0)))
        data["sla_violations"].append(rc.get("sla_violations", step_info.get("sla_violations", 0)))
        data["dropped"].append(rc.get("dropped", step_info.get("dropped", 0)))

        if t % max(steps // 20, 1) == 0:
            progress.progress(min((t + 1) / steps, 1.0), text=f"Running {policy_name} simulation...")

        if done:
            break

    progress.empty()
    if venv is not None:
        venv.close()
    env.close()
    return pd.DataFrame(data)


# Sidebar
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Model path (zip)", "./models/sac_cloud_final.zip")
log_csv = st.sidebar.text_input("Training log CSV", "./logs/training_log.csv")
workload = st.sidebar.selectbox("Workload scenario", WORKLOAD_SCENARIOS, index=0)
sim_steps = st.sidebar.slider("Simulation steps", 100, 3000, value=500, step=100)
compare_policies = st.sidebar.checkbox("Compare all policies", value=True)

model, vec_norm_path = load_sac_model(model_path)

# Training metrics
left, right = st.columns([1, 1])

with left:
    st.subheader("Training Metrics")
    if Path(log_csv).exists():
        try:
            df = pd.read_csv(log_csv)
        except pd.errors.ParserError:
            st.warning(
                "Training log has mixed column formats (old vs new). "
                "Delete the file and retrain, or point to a fresh log."
            )
            df = None
        if df is not None and "episode_reward" in df.columns:
            df["reward_ma"] = df["episode_reward"].rolling(20, min_periods=1).mean()
            st.line_chart(
                df[["episode_reward", "reward_ma"]].rename(
                    columns={"episode_reward": "Episode Reward", "reward_ma": "Reward (MA, 20)"}
                )
            )
            st.dataframe(df.tail(10))
    else:
        st.info("No training log found. Run `python -m src.train` to generate one.")

with right:
    st.subheader("Model Status")
    if model is not None:
        st.success(f"Model loaded: {model_path}")
        if vec_norm_path:
            st.caption(f"Observation normalization: {vec_norm_path}")
    else:
        st.warning("Model not found. Train with `python -m src.train` or update the model path.")

    if st.sidebar.button("Evaluate All Policies"):
        with st.spinner("Evaluating..."):
            results = evaluate_all(
                model_path=model_path if model is not None else None,
                episodes=10,
                workload=workload,
            )
            print_summary(results)
            summary_df = pd.DataFrame([{
                "policy": r.policy_name,
                "mean_reward": r.mean_reward,
                "mean_throughput": r.mean_throughput,
                "mean_cost": r.mean_resource_cost,
                "mean_sla": r.mean_sla_violations,
                "mean_latency": r.mean_latency,
                "mean_dropped": r.mean_dropped,
            } for r in results])
            st.dataframe(summary_df)
            st.bar_chart(summary_df.set_index("policy")[["mean_reward", "mean_throughput"]])

st.markdown("---")
st.subheader("Live Simulation")

policy_choice = st.selectbox(
    "Policy",
    ["sac", "threshold", "proportional", "fixed", "random"] if model else ["threshold", "proportional", "fixed", "random"],
)
run_button = st.button("Run Simulation")

if run_button:
    policies = {
        "threshold": ThresholdPolicy(),
        "proportional": ProportionalPolicy(),
        "fixed": FixedWorkersPolicy(fraction=0.5),
        "random": RandomPolicy(seed=42),
    }

    if compare_policies:
        sim_results = {}
        policy_list = []
        if model is not None:
            policy_list.append(("sac", None))
        policy_list.extend([(name, pol) for name, pol in policies.items()])

        for name, pol in policy_list:
            if name == "sac":
                sim_results[name] = run_policy_simulation(
                    None, name, workload, sim_steps, model=model, vec_norm_path=vec_norm_path
                )
            else:
                sim_results[name] = run_policy_simulation(pol, name, workload, sim_steps)

        st.success("Comparison simulation finished.")
        compare_df = pd.concat(
            [df.assign(policy=name) for name, df in sim_results.items()],
            ignore_index=True,
        )
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.markdown("**Queue length by policy**")
            pivot_q = compare_df.pivot(index="step", columns="policy", values="queue_len")
            st.line_chart(pivot_q)
        with chart_cols[1]:
            st.markdown("**Workers by policy**")
            pivot_w = compare_df.pivot(index="step", columns="policy", values="workers")
            st.line_chart(pivot_w)

        reward_cols = st.columns(2)
        with reward_cols[0]:
            st.markdown("**Cumulative reward by policy**")
            cum_rewards = {name: df["reward"].cumsum() for name, df in sim_results.items()}
            st.line_chart(pd.DataFrame(cum_rewards))
        with reward_cols[1]:
            st.markdown("**Latency by policy**")
            pivot_l = compare_df.pivot(index="step", columns="policy", values="latency")
            st.line_chart(pivot_l)

        summary = compare_df.groupby("policy").agg(
            total_reward=("reward", "sum"),
            avg_workers=("workers", "mean"),
            avg_queue=("queue_len", "mean"),
            total_dropped=("dropped", "sum"),
            avg_latency=("latency", "mean"),
        ).reset_index()
        st.dataframe(summary)

        out_path = Path(paths["log_dir"]) / f"live_simulation_{workload}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        compare_df.to_csv(out_path, index=False)
        st.info(f"Saved comparison log to {out_path}")
    else:
        if policy_choice == "sac" and model is not None:
            result_df = run_policy_simulation(
                None, "sac", workload, sim_steps, model=model, vec_norm_path=vec_norm_path
            )
        else:
            pol = policies[policy_choice]
            result_df = run_policy_simulation(pol, policy_choice, workload, sim_steps)

        st.success(f"Simulation finished ({policy_choice}).")
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(result_df.set_index("step")[["workers", "queue_len", "throughput"]])
        with col2:
            st.line_chart(result_df.set_index("step")[["cpu", "memory", "latency"]])
        st.line_chart(result_df.set_index("step")[["reward", "resource_cost", "sla_violations"]])
        st.write(result_df.describe())
else:
    st.info("Select a workload scenario and click **Run Simulation** to start.")
