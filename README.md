# RL Cloud Resource Allocation (SAC + Streamlit Dashboard)

# Overview

This project implements a Reinforcement Learning (RL) agent based on the Soft Actor-Critic (SAC) algorithm to dynamically allocate cloud computing resources.
The agent learns to optimize cost, latency, and throughput by interacting with a simulated cloud environment.

A Streamlit dashboard is included to visualize training metrics and run live simulations with the trained model.

# Project Structure

rl-cloud-resource-allocator/
│
├── app/
│   └── streamlit_dashboard.py       
│
├── src/
│   ├── envs/
│   │   └── cloud_env.py              
│   ├── train_sac.py                  
│
├── models/
│   └── sac_cloud_final.zip           
│
├── logs/
│   └── training_log.csv              
├── requirements.txt
└── README.md

# Installation & Setup

1.  Clone the repository
git clone https://github.com/backpropBoi/rl-cloud-resource-allocator.git
cd rl-cloud-resource-allocator

2️. Install dependencies
pip install -r requirements.txt

3️. Train the SAC model
python src/train_sac.py

4️. Launch the dashboard
streamlit run app/streamlit_dashboard.py

