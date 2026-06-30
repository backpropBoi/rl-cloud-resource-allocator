FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY config ./config
COPY src ./src
COPY app ./app

RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_dashboard.py", "--server.address", "0.0.0.0"]
