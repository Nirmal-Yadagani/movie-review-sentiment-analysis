# 🎬 End-to-End MLOps Pipeline: Movie Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![DVC](https://img.shields.io/badge/DVC-Pipeline-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![AWS ECR](https://img.shields.io/badge/AWS-ECR-orange)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Minikube-blue)
![Grafana](https://img.shields.io/badge/Grafana-Observability-orange)

## 📌 Overview
This project is a complete, enterprise-grade Machine Learning Operations (MLOps) pipeline. It automates the entire lifecycle of a sentiment analysis model—from training and mathematical evaluation to containerized CI/CD deployment and Kubernetes observability.

The system features a **FastAPI backend** serving a HuggingFace ONNX/Scikit-Learn model, a **Streamlit frontend** for user interaction, and a strict **Continuous Delivery** pipeline that ensures only mathematically superior models reach production. 

### ✨ Key Enterprise Features
* **Zero-Downtime Rollbacks:** When a new model is promoted, the active production model is automatically safeguarded with a `fallback` MLflow alias, acting as an instant fallback mechanism.
* **Structured Observability:** The entire pipeline utilizes `structlog` to output JSON-formatted logs, ensuring telemetry, parameters, and errors are fully indexed and searchable in production.

## 🏗️ System Workflow

This project follows a strict 5-step MLOps lifecycle:

1. **Experimentation:** Initial model testing and hyperparameter tuning.
2. **Training & Tracking (`model_builder.py` & `model_evaluation.py`):** The best parameters are used to train a fresh model. Metrics (F1, Accuracy) and custom PyFunc artifacts are logged to a remote **DagsHub/MLflow** registry.
3. **Automated Gatekeeping (`model_promotion.py`):** A Python script queries MLflow to compare the new model's performance against both the reigning `Production` and current `Staging` models. If it sets a new high score, it is promoted to `Staging`.
4. **Continuous Delivery (GitHub Actions):** A manually triggered workflow securely connects to MLflow, downloads the `Staging` weights, builds highly-optimized Docker images (Frontend + Backend), and tests the FastAPI `/ping` endpoint. If healthy, images are pushed to **AWS ECR**, the new model is aliased as `Production`, and the old model is archived as `Previous-Production`.
5. **Deployment & Observability (Kubernetes):** The containers are deployed to a local **Minikube Kubernetes cluster**. The cluster is fully instrumented with the **kube-prometheus-stack**, scraping custom FastAPI metrics to visualize live inference traffic on **Grafana** dashboards.

## 💻 Tech Stack
* **Machine Learning:** Scikit-Learn / PyTorch / ONNX, Pandas, NumPy
* **MLOps & Tracking:** MLflow, DagsHub
* **Backend API:** FastAPI, Uvicorn, Pydantic
* **Frontend UI:** Streamlit
* **CI/CD:** GitHub Actions, Docker
* **Cloud Infrastructure:** AWS Elastic Container Registry (ECR), AWS S3
* **Orchestration & Monitoring:** Kubernetes (Minikube), Helm, Prometheus, Grafana

## 🗂️ Data Version Control (DVC) & S3 Integration
This project achieves 100% reproducibility using **DVC**. The 50K IMDB Movie Review dataset is hosted securely in a private **AWS S3** bucket. 

The pipeline is modeled as a Directed Acyclic Graph (DAG) in `dvc.yaml`. DVC tracks data lineage, detects changes in `params.yaml`, and intelligently caches intermediate steps (like tokenization) to save compute time.

To reproduce the exact model locally:
```bash
dvc pull   # Authenticates with AWS and pulls the IMDB dataset from S3
dvc repro  # Re-runs the ML pipeline based on the dvc.yaml graph
```
*(Note: If you do not have IAM access to the S3 bucket, you can download the raw data manually from Kaggle, place it in data/raw/, and update your data paths before running dvc repro.)*

## 🚀 How to Run Locally
### Prerequisites
* **Docker & Docker Compose**

* **Minikube & kubectl**

* **Helm**

### Environment Setup
Before running the project locally, create a .env file in the root directory. You can copy the template provided:

```bash
cp .env.example .env
```
Fill in your specific AWS credentials and DagsHub tracking URIs to ensure the pipeline can authenticate, pull data, and log experiments.

## 1. Run via Docker Compose (Quickstart)
```bash
# Clone the repository
git clone https://github.com/Nirmal-Yadagani/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis

# Build and start the containers
docker-compose up --build
```
Frontend: http://localhost:8501

FastAPI Backend: http://localhost:8080/invocations

## 2. Run via Kubernetes (Full Cluster Simulation)
```bash
# Start Minikube
minikube start --driver=docker

# Apply Kubernetes manifests
kubectl apply -f k8s/backend_minikube_local.yaml
kubectl apply -f k8s/frontend_minikube_local.yaml

# Tunnel to the Streamlit UI
minikube service frontend-loadbalancer
```

## 3. Setup Observability (Prometheus & Grafana)
```bash
# Install the monitoring stack via Helm
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

# Apply the custom ServiceMonitor for FastAPI
kubectl apply -f k8s/service-monitor.yaml

# Port-forward Grafana to localhost:3000
kubectl port-forward svc/monitoring-grafana 3000:80 -n monitoring
```

## 🤖 CI/CD Pipeline Configuration
To enable the GitHub Actions workflow for automated ECR deployments, ensure the following repository secrets are configured in your GitHub settings:

AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY

AWS_REGION & AWS_ACCOUNT_ID

MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, & MLFLOW_TRACKING_PASSWORD