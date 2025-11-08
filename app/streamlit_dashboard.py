# app/streamlit_dashboard.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.cloud_env import CloudEnv
import streamlit as st
import numpy as np
import torch
from stable_baselines3 import SAC
import time
import pandas as pd
from pathlib import Path

st.set_page_config(layout="wide", page_title="RL Cloud — Dashboard")
st.title("RL Cloud Resource Allocation — Dashboard (SAC)")

# Sidebar
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Model path (zip)", "./models/sac_cloud_final.zip")
log_csv = st.sidebar.text_input("Training log CSV", "./logs/training_log.csv")
run_sim = st.sidebar.button("Run Live Simulation (model)")
run_random = st.sidebar.button("Run Live Simulation (random policy)")
eval_button = st.sidebar.button("Evaluate Model (10 episodes)")
sim_steps = st.sidebar.slider("Simulation steps", 100, 3000, value=500, step=100)

# --- Training metrics ---
left, right = st.columns([1, 1])

with left:
    st.subheader("Training Metrics")
    if Path(log_csv).exists():
        df = pd.read_csv(log_csv)
        if "episode_reward" in df.columns:
            df["reward_ma"] = df["episode_reward"].rolling(20, min_periods=1).mean()
            st.line_chart(df[["episode_reward", "reward_ma"]].rename(columns={
                "episode_reward": "Episode Reward",
                "reward_ma": "Reward (MA, 20)"
            }))
        st.dataframe(df.tail(10))
    else:
        st.info("No training log found.")

with right:
    st.subheader("Model Evaluation")
    if Path(model_path).exists():
        st.success(f"Model found: {model_path}")
        try:
            model = SAC.load(model_path)
            st.write("✅ Model loaded successfully.")
        except Exception as e:
            st.error(f"Could not load model: {e}")
            model = None
    else:
        st.warning("Model not found.")
        model = None

    if eval_button:
        if model is None:
            st.error("Load a model first.")
        else:
            env = CloudEnv()
            returns = []
            for ep in range(10):
                obs, _ = env.reset()
                done = False
                total = 0.0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, r, done, truncated, info = env.step(action)
                    total += r
                returns.append(total)
            st.metric("Mean return (10 eps)", f"{np.mean(returns):.2f}")
            st.write("Return std:", np.std(returns))

st.markdown("---")
st.subheader("Live Simulation")

col1, col2 = st.columns([1, 1])
placeholder_metrics = col1.empty()
placeholder_chart = col2.empty()

def run_simulation(use_model=True, steps=500):
    env = CloudEnv()
    obs, _ = env.reset()

    data = {
        "workers": [], "queue_len": [], "throughput": [],
        "cpu": [], "memory": [], "latency": []
    }

    for t in range(steps):
        if use_model and model is not None:
            action, _ = model.predict(obs, deterministic=True)
        elif not use_model:
            action = np.array([np.random.rand()])
        else:
            action = np.array([0.5])

        obs, r, done, truncated, info = env.step(action)

        for k in data.keys():
            data[k].append(info.get(k, 0))

        placeholder_metrics.metric("Workers", f"{info['workers']}")
        placeholder_metrics.metric("Queue Len", f"{info['queue_len']}")
        placeholder_metrics.metric("Throughput", f"{info['finished']}")
        placeholder_metrics.metric("CPU (%)", f"{info['cpu']:.1f}")
        placeholder_metrics.metric("Memory (%)", f"{info['memory']:.1f}")
        placeholder_metrics.metric("Latency (ms)", f"{info['latency']:.1f}")

        df_live = pd.DataFrame(data)
        placeholder_chart.line_chart(df_live)
        time.sleep(0.01)
        if done:
            break

    final_df = pd.DataFrame(data)
    st.success("Simulation finished.")
    st.line_chart(final_df)
    st.write(final_df.describe())

    final_df.to_csv("./logs/live_simulation.csv", index=False)
    st.info("Saved live simulation log to ./logs/live_simulation.csv")

if run_sim:
    run_simulation(use_model=True, steps=sim_steps)
elif run_random:
    run_simulation(use_model=False, steps=sim_steps)
else:
    st.info("Click 'Run Live Simulation (model)' or 'Run Live Simulation (random policy)' in the sidebar.")
