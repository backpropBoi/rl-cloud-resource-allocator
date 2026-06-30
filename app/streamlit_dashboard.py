# app/streamlit_dashboard.py
"""Professional Streamlit dashboard for RL Cloud Resource Allocator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.baselines import FixedWorkersPolicy, ProportionalPolicy, RandomPolicy, ThresholdPolicy
from src.config import load_config
from src.envs.cloud_env import CloudEnv, WORKLOAD_SCENARIOS, WORKLOAD_SCENARIOS_WITH_MIXED
from src.evaluate import evaluate_all

st.set_page_config(
    layout="wide",
    page_title="RL Cloud Allocator",
    page_icon="☁️",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0f1117; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e2130, #252a3a);
    border: 1px solid #2d3350;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 4px 0;
}
.metric-label {
    font-size: 0.75rem;
    color: #8b92a5;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1.1;
}
.metric-delta-good  { font-size: 0.8rem; color: #4ade80; margin-top: 4px; }
.metric-delta-bad   { font-size: 0.8rem; color: #f87171; margin-top: 4px; }

/* Section headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 24px 0 4px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #2d3350;
}
.section-sub {
    font-size: 0.85rem;
    color: #8b92a5;
    margin-bottom: 20px;
}

/* Status badges */
.badge-success {
    display: inline-block;
    background: rgba(74,222,128,0.15);
    color: #4ade80;
    border: 1px solid rgba(74,222,128,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
}
.badge-warning {
    display: inline-block;
    background: rgba(251,191,36,0.15);
    color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
}
.badge-info {
    display: inline-block;
    background: rgba(96,165,250,0.15);
    color: #60a5fa;
    border: 1px solid rgba(96,165,250,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
}

/* Config table */
.config-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 8px;
}
.config-item {
    background: #1e2130;
    border: 1px solid #2d3350;
    border-radius: 8px;
    padding: 10px 14px;
}
.config-key   { font-size: 0.72rem; color: #8b92a5; text-transform: uppercase; }
.config-val   { font-size: 1rem; font-weight: 600; color: #e2e8f0; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #13151f; border-right: 1px solid #2d3350; }

/* Hide default streamlit metric styling */
div[data-testid="metric-container"] { display: none; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: #1e2130;
    border-radius: 8px;
    border: 1px solid #2d3350;
    color: #8b92a5;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b4fd8, #6366f1) !important;
    color: white !important;
    border: none !important;
}

/* Buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b4fd8, #6366f1);
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 28px;
}
</style>
""", unsafe_allow_html=True)

# ── Colour palette for policies ───────────────────────────────────────────────
POLICY_COLORS = {
    "sac":          "#6366f1",
    "proportional": "#22d3ee",
    "threshold":    "#4ade80",
    "fixed":        "#f59e0b",
    "random":       "#f87171",
}
POLICY_LABELS = {
    "sac":          "SAC (RL Agent)",
    "proportional": "Proportional",
    "threshold":    "Threshold",
    "fixed":        "Fixed",
    "random":       "Random",
}

config = load_config()
env_cfg   = config["env"]
reward_cfg = config.get("reward", {})
paths     = config["paths"]


# ── helpers ───────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, delta: str = "", good: bool = True) -> str:
    delta_cls  = "metric-delta-good" if good else "metric-delta-bad"
    delta_html = f'<div class="{delta_cls}">{delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>"""


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


def run_policy_simulation(
    policy, policy_name: str, workload: str, steps: int,
    model=None, vec_norm_path=None, progress_slot=None,
) -> pd.DataFrame:
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

    rows = []
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

        rc = step_info.get("reward_components", {})
        rows.append({
            "step": t,
            "workers":       step_info.get("workers", 0),
            "queue_len":     step_info.get("queue_len", 0),
            "throughput":    step_info.get("finished", 0),
            "cpu":           step_info.get("cpu", 0),
            "memory":        step_info.get("memory", 0),
            "latency":       step_info.get("latency", 0),
            "reward":        step_reward,
            "resource_cost": rc.get("resource_cost", step_info.get("resource_cost", 0)),
            "sla_violations":rc.get("sla_violations", step_info.get("sla_violations", 0)),
            "dropped":       rc.get("dropped", step_info.get("dropped", 0)),
        })

        if progress_slot and t % max(steps // 20, 1) == 0:
            progress_slot.progress(min((t + 1) / steps, 1.0), text=f"Simulating {POLICY_LABELS.get(policy_name, policy_name)}…")

        if done:
            break

    if progress_slot:
        progress_slot.empty()
    if venv is not None:
        venv.close()
    env.close()
    return pd.DataFrame(rows)


def plotly_line(df_dict: dict[str, pd.Series], title: str, y_label: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    for name, series in df_dict.items():
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name=POLICY_LABELS.get(name, name),
            line=dict(color=POLICY_COLORS.get(name, "#888"), width=2),
            hovertemplate=f"<b>{POLICY_LABELS.get(name, name)}</b><br>Step: %{{x}}<br>{y_label}: %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,33,48,0.6)",
        font=dict(color="#8b92a5"),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0")),
        xaxis=dict(gridcolor="#2d3350", title="Step"),
        yaxis=dict(gridcolor="#2d3350", title=y_label),
        hovermode="x unified",
    )
    return fig


def plotly_bar(labels: list, values: list, title: str, color_map: dict | None = None) -> go.Figure:
    colors = [color_map.get(l, "#6366f1") for l in labels] if color_map else ["#6366f1"] * len(labels)
    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,33,48,0.6)",
        font=dict(color="#8b92a5"),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor="#2d3350"),
        yaxis=dict(gridcolor="#2d3350"),
        showlegend=False,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# ☁️ RL Cloud Allocator")
    st.markdown('<p style="color:#8b92a5;font-size:0.82rem;margin-top:-10px;">Soft Actor-Critic · Cloud Resource Allocation</p>', unsafe_allow_html=True)
    st.markdown("---")

    model_path = st.text_input("Model path (.zip)", "./models/sac_cloud_final.zip")
    log_csv    = st.text_input("Training log CSV",  "./logs/training_log.csv")

    st.markdown("**Workload**")
    workload = st.selectbox(
        "",
        WORKLOAD_SCENARIOS,
        label_visibility="collapsed",
        help="steady = constant | burst = periodic peaks | spike = sudden surges",
    )
    st.caption({
        "steady": "🟢 Constant arrival rate",
        "burst":  "🟡 Periodic traffic surges every ~100 steps",
        "spike":  "🔴 Sudden 5× spikes lasting 10 steps",
    }.get(workload, ""))

    sim_steps = st.slider("Simulation steps", 100, 1000, 500, 100)
    st.markdown("---")
    st.markdown('<p style="color:#8b92a5;font-size:0.78rem;">Train: <code>python -m src.train</code></p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b92a5;font-size:0.78rem;">Test: <code>pytest tests/ -v</code></p>', unsafe_allow_html=True)

model, vec_norm_path = load_sac_model(model_path)

with st.sidebar:
    st.markdown("---")
    if model:
        st.markdown('<span class="badge-success">✓ Model Loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warning">⚠ No Model Found</span>', unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 8px 0 24px 0;">
  <h1 style="font-size:2rem;font-weight:800;color:#e2e8f0;margin:0;">
    RL Cloud Resource Allocator
  </h1>
  <p style="color:#8b92a5;font-size:0.92rem;margin-top:4px;">
    Soft Actor-Critic agent vs rule-based baselines · Dynamic worker scaling simulation
  </p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊  Training Metrics", "🏁  Policy Comparison", "🔬  Single Policy"])


# ════════════════════════════════════════════════════════════════════════
# TAB 1 — Training Metrics
# ════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-header">Model & Environment</div>', unsafe_allow_html=True)

    col_stat, col_env = st.columns([1, 2])

    with col_stat:
        if model:
            st.markdown('<span class="badge-success">✓ Model Loaded</span>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#8b92a5;font-size:0.82rem;margin-top:8px;">{model_path}</p>', unsafe_allow_html=True)
            if vec_norm_path:
                st.markdown('<span class="badge-info">⚡ VecNormalize active</span>', unsafe_allow_html=True)
                st.markdown(f'<p style="color:#8b92a5;font-size:0.78rem;margin-top:4px;">{vec_norm_path}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-warning">⚠ No model — run python -m src.train</span>', unsafe_allow_html=True)

    with col_env:
        cfg_items = {
            "Max Workers": env_cfg["max_workers"],
            "Min Workers": env_cfg["min_workers"],
            "Arrival Rate": env_cfg["arrival_rate"],
            "Episode Length": env_cfg["episode_length"],
            "Scale-Up Delay": env_cfg["scale_up_delay"],
            "Scale-Down Cooldown": env_cfg["scale_down_cooldown"],
        }
        cols = st.columns(3)
        for i, (k, v) in enumerate(cfg_items.items()):
            cols[i % 3].markdown(
                f'<div class="config-item"><div class="config-key">{k}</div><div class="config-val">{v}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<div class="section-header">Training Progress</div>', unsafe_allow_html=True)

    log_path = Path(log_csv)
    if not log_path.exists():
        st.markdown("""
        <div style="background:#1e2130;border:1px dashed #2d3350;border-radius:12px;padding:40px;text-align:center;">
            <div style="font-size:2rem;">📈</div>
            <div style="color:#e2e8f0;font-weight:600;margin-top:8px;">No training data yet</div>
            <div style="color:#8b92a5;font-size:0.85rem;margin-top:4px;">Run <code>python -m src.train</code> to generate training logs</div>
        </div>""", unsafe_allow_html=True)
    else:
        try:
            df_log = pd.read_csv(log_csv)
        except pd.errors.ParserError:
            st.warning("Training log has mixed formats. Re-run `python -m src.train`.")
            df_log = None

        if df_log is not None and "episode_reward" in df_log.columns and len(df_log) > 0:
            # KPI row
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(metric_card("Total Episodes", f"{len(df_log):,}"), unsafe_allow_html=True)
            k2.markdown(metric_card("Best Reward", f"{df_log['episode_reward'].max():.1f}", "↑ all time high", True), unsafe_allow_html=True)
            k3.markdown(metric_card("Recent Mean (20)", f"{df_log['episode_reward'].tail(20).mean():.1f}"), unsafe_allow_html=True)
            if "episode_length" in df_log.columns:
                k4.markdown(metric_card("Avg Episode Length", f"{df_log['episode_length'].mean():.0f} steps"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Reward curve
            df_log["ma20"] = df_log["episode_reward"].rolling(20, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_log.index, y=df_log["episode_reward"],
                name="Episode Reward", line=dict(color="#6366f1", width=1),
                opacity=0.4, fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
            ))
            fig.add_trace(go.Scatter(
                x=df_log.index, y=df_log["ma20"],
                name="MA-20", line=dict(color="#22d3ee", width=2.5),
            ))
            fig.update_layout(
                title=dict(text="Episode Reward over Training", font=dict(size=15, color="#e2e8f0")),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,33,48,0.6)",
                font=dict(color="#8b92a5"), height=350,
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis=dict(gridcolor="#2d3350", title="Episode"),
                yaxis=dict(gridcolor="#2d3350", title="Reward"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0")),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw log — last 20 rows"):
                st.dataframe(df_log.tail(20).style.background_gradient(subset=["episode_reward"], cmap="RdYlGn"), use_container_width=True)
        else:
            st.markdown("""
            <div style="background:#1e2130;border:1px dashed #2d3350;border-radius:12px;padding:40px;text-align:center;">
                <div style="font-size:2rem;">📈</div>
                <div style="color:#e2e8f0;font-weight:600;margin-top:8px;">Training log is empty</div>
                <div style="color:#8b92a5;font-size:0.85rem;margin-top:4px;">Run <code>python -m src.train</code> to populate it</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — Policy Comparison
# ════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">Policy Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Run all policies side-by-side and compare performance metrics</div>', unsafe_allow_html=True)

    col_mode, col_btn = st.columns([3, 1])
    with col_mode:
        eval_mode = st.radio(
            "Mode",
            ["Live Simulation  —  fast, visual charts", "Full Evaluation  —  10 episodes, statistically rigorous"],
            horizontal=True, label_visibility="collapsed",
        )
    with col_btn:
        run_compare = st.button("▶  Run Comparison", type="primary", use_container_width=True)

    st.markdown("---")

    if run_compare:
        policies = {
            "threshold":    ThresholdPolicy(),
            "proportional": ProportionalPolicy(),
            "fixed":        FixedWorkersPolicy(fraction=0.5),
            "random":       RandomPolicy(seed=42),
        }

        if "Live" in eval_mode:
            sim_results: dict[str, pd.DataFrame] = {}
            policy_list = []
            if model is not None:
                policy_list.append(("sac", None))
            policy_list.extend(list(policies.items()))

            outer_prog = st.progress(0, text="Starting simulations…")
            for i, (name, pol) in enumerate(policy_list):
                outer_prog.progress(i / len(policy_list), text=f"Running {POLICY_LABELS.get(name, name)}…")
                p_slot = st.empty()
                sim_results[name] = run_policy_simulation(
                    pol, name, workload, sim_steps,
                    model=model if name == "sac" else None,
                    vec_norm_path=vec_norm_path if name == "sac" else None,
                    progress_slot=p_slot,
                )
            outer_prog.empty()

            compare_df = pd.concat(
                [df.assign(policy=name) for name, df in sim_results.items()],
                ignore_index=True,
            )

            summary = (
                compare_df.groupby("policy")
                .agg(
                    total_reward=("reward", "sum"),
                    avg_workers=("workers", "mean"),
                    avg_queue=("queue_len", "mean"),
                    total_throughput=("throughput", "sum"),
                    total_dropped=("dropped", "sum"),
                    avg_latency=("latency", "mean"),
                )
                .reset_index()
                .sort_values("total_reward", ascending=False)
            )

            # Winner banner
            winner = summary.iloc[0]["policy"]
            st.markdown(
                f'<div style="background:linear-gradient(135deg,rgba(99,102,241,0.2),rgba(34,211,238,0.1));'
                f'border:1px solid rgba(99,102,241,0.4);border-radius:12px;padding:16px 24px;margin-bottom:20px;">'
                f'<span style="color:#8b92a5;font-size:0.85rem;">BEST POLICY  ·  {workload.upper()} WORKLOAD</span><br>'
                f'<span style="color:#e2e8f0;font-size:1.5rem;font-weight:700;">🏆 {POLICY_LABELS.get(winner, winner)}</span>'
                f'<span style="color:#8b92a5;font-size:0.85rem;margin-left:12px;">Total reward: {summary.iloc[0]["total_reward"]:.0f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # KPI cards for each policy
            kpi_cols = st.columns(len(summary))
            for col, (_, row) in zip(kpi_cols, summary.iterrows()):
                col.markdown(
                    f'<div class="metric-card" style="border-color:{POLICY_COLORS.get(row["policy"],"#444")};'
                    f'border-left:3px solid {POLICY_COLORS.get(row["policy"],"#444")};">'
                    f'<div class="metric-label">{POLICY_LABELS.get(row["policy"], row["policy"])}</div>'
                    f'<div class="metric-value" style="font-size:1.3rem;">{row["total_reward"]:.0f}</div>'
                    f'<div style="color:#8b92a5;font-size:0.75rem;margin-top:4px;">reward · {row["total_throughput"]:.0f} jobs</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Charts row 1
            c1, c2 = st.columns(2)
            with c1:
                cum = {name: df["reward"].cumsum().reset_index(drop=True) for name, df in sim_results.items()}
                st.plotly_chart(plotly_line(cum, "Cumulative Reward", "Reward"), use_container_width=True)
            with c2:
                workers = {name: df.set_index("step")["workers"] for name, df in sim_results.items()}
                st.plotly_chart(plotly_line(workers, "Workers over Time", "Workers"), use_container_width=True)

            # Charts row 2
            c3, c4 = st.columns(2)
            with c3:
                queue = {name: df.set_index("step")["queue_len"] for name, df in sim_results.items()}
                st.plotly_chart(plotly_line(queue, "Queue Length over Time", "Jobs in Queue"), use_container_width=True)
            with c4:
                latency = {name: df.set_index("step")["latency"] for name, df in sim_results.items()}
                st.plotly_chart(plotly_line(latency, "Latency over Time", "Latency (ms)"), use_container_width=True)

            # Charts row 3
            c5, c6 = st.columns(2)
            with c5:
                dropped = {name: df["dropped"].cumsum().reset_index(drop=True) for name, df in sim_results.items()}
                st.plotly_chart(plotly_line(dropped, "Cumulative Jobs Dropped", "Dropped"), use_container_width=True)
            with c6:
                cpu = {name: df.set_index("step")["cpu"] for name, df in sim_results.items()}
                st.plotly_chart(plotly_line(cpu, "CPU Utilization (%)", "CPU %"), use_container_width=True)

            # Summary table
            st.markdown('<div class="section-header" style="margin-top:16px;">Summary Table</div>', unsafe_allow_html=True)
            summary_display = summary.copy()
            summary_display["policy"] = summary_display["policy"].map(lambda p: POLICY_LABELS.get(p, p))
            summary_display.columns = ["Policy", "Total Reward", "Avg Workers", "Avg Queue", "Total Throughput", "Total Dropped", "Avg Latency"]
            summary_display = summary_display.round(2)
            st.dataframe(summary_display.set_index("Policy"), use_container_width=True)

        else:
            # Full evaluation mode
            with st.spinner("Running 10-episode evaluation per policy — this takes ~1 minute…"):
                results = evaluate_all(
                    model_path=model_path if model is not None else None,
                    episodes=10,
                    workload=workload,
                )

            eval_df = pd.DataFrame([{
                "policy":         r.policy_name,
                "mean_reward":    round(r.mean_reward, 1),
                "std_reward":     round(r.std_reward, 1),
                "mean_throughput":round(r.mean_throughput, 1),
                "mean_cost":      round(r.mean_resource_cost, 1),
                "mean_sla":       round(r.mean_sla_violations, 1),
                "mean_latency":   round(r.mean_latency, 1),
                "mean_dropped":   round(r.mean_dropped, 1),
            } for r in results]).sort_values("mean_reward", ascending=False)

            winner = eval_df.iloc[0]["policy"]
            st.markdown(
                f'<div style="background:linear-gradient(135deg,rgba(99,102,241,0.2),rgba(34,211,238,0.1));'
                f'border:1px solid rgba(99,102,241,0.4);border-radius:12px;padding:16px 24px;margin-bottom:20px;">'
                f'<span style="color:#8b92a5;font-size:0.85rem;">BEST POLICY  ·  {workload.upper()} WORKLOAD  ·  10 EPISODES</span><br>'
                f'<span style="color:#e2e8f0;font-size:1.5rem;font-weight:700;">🏆 {POLICY_LABELS.get(winner, winner)}</span>'
                f'<span style="color:#8b92a5;font-size:0.85rem;margin-left:12px;">Mean reward: {eval_df.iloc[0]["mean_reward"]:.1f} ± {eval_df.iloc[0]["std_reward"]:.1f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Bar charts
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.plotly_chart(plotly_bar(
                    [POLICY_LABELS.get(p, p) for p in eval_df["policy"]],
                    eval_df["mean_reward"].tolist(),
                    "Mean Reward",
                    {POLICY_LABELS.get(p, p): POLICY_COLORS.get(p, "#888") for p in eval_df["policy"]},
                ), use_container_width=True)
            with bc2:
                st.plotly_chart(plotly_bar(
                    [POLICY_LABELS.get(p, p) for p in eval_df["policy"]],
                    eval_df["mean_throughput"].tolist(),
                    "Mean Throughput",
                    {POLICY_LABELS.get(p, p): POLICY_COLORS.get(p, "#888") for p in eval_df["policy"]},
                ), use_container_width=True)
            with bc3:
                st.plotly_chart(plotly_bar(
                    [POLICY_LABELS.get(p, p) for p in eval_df["policy"]],
                    eval_df["mean_dropped"].tolist(),
                    "Mean Dropped Jobs",
                    {POLICY_LABELS.get(p, p): POLICY_COLORS.get(p, "#888") for p in eval_df["policy"]},
                ), use_container_width=True)

            st.markdown('<div class="section-header">Evaluation Results</div>', unsafe_allow_html=True)
            display_df = eval_df.copy()
            display_df["policy"] = display_df["policy"].map(lambda p: POLICY_LABELS.get(p, p))
            display_df.columns = ["Policy","Mean Reward","Std Reward","Mean Throughput","Mean Cost","Mean SLA Violations","Mean Latency","Mean Dropped"]
            st.dataframe(display_df.set_index("Policy"), use_container_width=True)

    else:
        st.markdown("""
        <div style="background:#1e2130;border:1px dashed #2d3350;border-radius:12px;padding:60px;text-align:center;">
            <div style="font-size:2.5rem;">🏁</div>
            <div style="color:#e2e8f0;font-weight:600;font-size:1.1rem;margin-top:12px;">Ready to Compare</div>
            <div style="color:#8b92a5;font-size:0.85rem;margin-top:6px;">
                Select a workload in the sidebar, choose a mode, and click <b>▶ Run Comparison</b>
            </div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 3 — Single Policy
# ════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">Single Policy Inspector</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Inspect one policy in detail across all metrics</div>', unsafe_allow_html=True)

    available_policies = (["sac"] if model else []) + ["threshold", "proportional", "fixed", "random"]
    policy_choice = st.selectbox(
        "Policy",
        available_policies,
        format_func=lambda k: {
            "sac":          "🤖  SAC (RL Agent)",
            "threshold":    "📊  Threshold",
            "proportional": "📈  Proportional",
            "fixed":        "📌  Fixed (50%)",
            "random":       "🎲  Random",
        }.get(k, k),
    )
    run_single = st.button("▶  Run", type="primary")

    st.markdown("---")

    if run_single:
        policies = {
            "threshold":    ThresholdPolicy(),
            "proportional": ProportionalPolicy(),
            "fixed":        FixedWorkersPolicy(fraction=0.5),
            "random":       RandomPolicy(seed=42),
        }
        p_slot = st.empty()
        result_df = run_policy_simulation(
            policies.get(policy_choice), policy_choice, workload, sim_steps,
            model=model if policy_choice == "sac" else None,
            vec_norm_path=vec_norm_path if policy_choice == "sac" else None,
            progress_slot=p_slot,
        )

        color = POLICY_COLORS.get(policy_choice, "#6366f1")
        label = POLICY_LABELS.get(policy_choice, policy_choice)

        st.markdown(
            f'<div style="background:linear-gradient(135deg,rgba(99,102,241,0.1),rgba(0,0,0,0));'
            f'border-left:4px solid {color};border-radius:0 8px 8px 0;padding:12px 20px;margin-bottom:20px;">'
            f'<span style="color:{color};font-weight:700;font-size:1.1rem;">{label}</span>'
            f'<span style="color:#8b92a5;font-size:0.85rem;margin-left:12px;">{workload} workload · {len(result_df)} steps</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # KPI cards
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(metric_card("Total Reward",   f"{result_df['reward'].sum():.0f}"),    unsafe_allow_html=True)
        k2.markdown(metric_card("Jobs Completed", f"{result_df['throughput'].sum():.0f}"),unsafe_allow_html=True)
        k3.markdown(metric_card("Avg Workers",    f"{result_df['workers'].mean():.1f}"),  unsafe_allow_html=True)
        k4.markdown(metric_card("Jobs Dropped",   f"{result_df['dropped'].sum():.0f}",
                                "↓ lower is better", result_df['dropped'].sum() == 0),    unsafe_allow_html=True)
        k5.markdown(metric_card("Avg Latency",    f"{result_df['latency'].mean():.1f} ms"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        def single_line(col_name: str, title: str, y_label: str) -> go.Figure:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result_df["step"], y=result_df[col_name],
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
                line=dict(color=color, width=2),
                hovertemplate=f"Step: %{{x}}<br>{y_label}: %{{y:.2f}}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=title, font=dict(size=13, color="#e2e8f0")),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,33,48,0.6)",
                font=dict(color="#8b92a5"), height=240,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(gridcolor="#2d3350", title="Step"),
                yaxis=dict(gridcolor="#2d3350", title=y_label),
                showlegend=False,
            )
            return fig

        r1, r2, r3 = st.columns(3)
        r1.plotly_chart(single_line("workers",    "Workers Allocated",     "Workers"),    use_container_width=True)
        r2.plotly_chart(single_line("queue_len",  "Queue Length",          "Jobs"),       use_container_width=True)
        r3.plotly_chart(single_line("throughput", "Throughput (jobs/step)","Jobs"),       use_container_width=True)

        r4, r5, r6 = st.columns(3)
        r4.plotly_chart(single_line("reward",   "Step Reward",         "Reward"),       use_container_width=True)
        r5.plotly_chart(single_line("latency",  "Latency",             "ms"),           use_container_width=True)
        r6.plotly_chart(single_line("cpu",      "CPU Utilization",     "%"),            use_container_width=True)

        with st.expander("📋 Descriptive Statistics"):
            st.dataframe(result_df.describe().T.round(3), use_container_width=True)

    else:
        st.markdown("""
        <div style="background:#1e2130;border:1px dashed #2d3350;border-radius:12px;padding:60px;text-align:center;">
            <div style="font-size:2.5rem;">🔬</div>
            <div style="color:#e2e8f0;font-weight:600;font-size:1.1rem;margin-top:12px;">Select a policy above</div>
            <div style="color:#8b92a5;font-size:0.85rem;margin-top:6px;">
                Pick a policy from the dropdown and click <b>▶ Run</b>
            </div>
        </div>""", unsafe_allow_html=True)
