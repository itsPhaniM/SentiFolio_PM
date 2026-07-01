# Slim image for the SentiFolio serving layer (FastAPI API + Streamlit dashboard).
# Only the runtime deps are installed; the heavy offline stack (torch/transformers/shap)
# is not needed to serve pre-built features, models and reports.
FROM python:3.12-slim

WORKDIR /app

# system libs LightGBM needs at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY . .

# API on 8000, dashboard on 8501 (see docker-compose.yml for both)
EXPOSE 8000 8501

CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
