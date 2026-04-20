# 1. Use a highly compressed, minimal version of Python 3.11
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install minimal system tools (libgomp1 for ML math, curl for Docker healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the raw MLflow model artifact and our new FastAPI script
COPY ./model_artifact /app/model
COPY ./main.py /app/main.py

# 5. AUTOMATED PYTORCH FIX: Install the latest CPU-only version of PyTorch.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 6. INJECT CUSTOM LIBRARIES: Strip 'torch', then add FastAPI, Uvicorn, and Prometheus
RUN grep -v "torch" /app/model/requirements.txt > /app/model/req_no_torch.txt && \
    echo "fastapi" >> /app/model/req_no_torch.txt && \
    echo "uvicorn[standard]" >> /app/model/req_no_torch.txt && \
    echo "prometheus-fastapi-instrumentator" >> /app/model/req_no_torch.txt && \
    pip install --no-cache-dir -r /app/model/req_no_torch.txt

# 7. Expose the port our API will run on
EXPOSE 8080

# 8. Start the API using Uvicorn (FastAPI's server) instead of MLflow
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]