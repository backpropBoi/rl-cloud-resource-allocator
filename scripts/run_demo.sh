#!/usr/bin/env bash
set -euo pipefail

echo "Installing package..."
pip install -e ".[dev]" -q

echo "Running tests..."
pytest tests/ -q

echo "Training SAC agent (500k timesteps — use --timesteps 20000 for a quick demo)..."
python -m src.train "$@"

echo "Evaluating model and baselines..."
python -m src.evaluate --model ./models/sac_cloud_final.zip

echo "Done. Launch dashboard with:"
echo "  streamlit run app/streamlit_dashboard.py"
