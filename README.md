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

The system features a **FastAPI backend** serving a PyTorch/Scikit-Learn model, a **Streamlit frontend** for user interaction, and a strict **Continuous Delivery** pipeline that ensures only mathematically superior models reach production.

## 🏗️ System Architecture & Workflow

This project follows a strict 5-step MLOps lifecycle:

1. **Experimentation:** Initial model testing and hyperparameter tuning are conducted in Jupyter Notebooks.
2. **Training & Tracking (`model_builder.py` & `model_evaluation.py`):** The best parameters are used to train a fresh model. Metrics (F1, Accuracy) and artifacts are logged to a remote **DagsHub/MLflow** registry without an initial alias.
3. **Automated Gatekeeping (`model_promotion.py`):** A Python script queries MLflow for the reigning `Production` model. It compares the new model's performance against it. If the new model beats production by a defined threshold, it is promoted to `Staging`.
4. **Continuous Delivery (GitHub Actions):** A manually triggered workflow securely connects to MLflow, downloads the `Staging` weights, builds highly-optimized Docker images (Frontend + Backend), and tests the FastAPI `/ping` endpoint. If healthy, images are pushed to **AWS ECR**, and the MLflow alias is officially updated to `Production`.
5. **Deployment & Observability (Kubernetes):** The containers are deployed to a local **Minikube Kubernetes cluster**. The cluster is fully instrumented with the **kube-prometheus-stack**, scraping custom FastAPI metrics to visualize live inference traffic on **Grafana** dashboards.

## 💻 Tech Stack
* **Machine Learning:** Scikit-Learn / PyTorch, Pandas, NumPy
* **MLOps & Tracking:** MLflow, DagsHub
* **Backend API:** FastAPI, Uvicorn, Pydantic
* **Frontend UI:** Streamlit
* **CI/CD:** GitHub Actions, Docker
* **Cloud Infrastructure:** AWS Elastic Container Registry (ECR)
* **Orchestration & Monitoring:** Kubernetes (Minikube), Helm, Prometheus, Grafana

## 🗂️ Data Version Control (DVC) & S3 Integration
This project achieves 100% reproducibility using **DVC**. The 50K IMDB Movie Review dataset is hosted securely in a private **AWS S3** bucket. 

The pipeline is modeled as a Directed Acyclic Graph (DAG) in `dvc.yaml`. DVC tracks data lineage, detects changes in `params.yaml`, and intelligently caches intermediate steps (like tokenization) to save compute time during hyperparameter tuning.

To reproduce the exact model locally:
```bash
dvc pull   # Authenticates with AWS and pulls the IMDB dataset from S3
dvc repro  # Re-runs the ML pipeline based on the dvc.yaml graph
```

## 🚀 How to Run Locally

### Prerequisites
* Docker & Docker Compose
* Minikube & `kubectl`
* Helm

### 1. Run via Docker Compose (Quickstart)
```bash
# Clone the repository
git clone [https://github.com/YourUsername/movie-review-sentiment-analysis.git](https://github.com/YourUsername/movie-review-sentiment-analysis.git)
cd movie-review-sentiment-analysis

# Build and start the containers
docker-compose up --build
```
Frontend: http://localhost:8501

FastAPI Backend: http://localhost:8080/docs


### 2. Run via Kubernetes (Full Cluster Simulation)
```bash
# Start Minikube
minikube start --driver=docker

# Apply Kubernetes manifests
kubectl apply -f k8s/backend-local.yaml
kubectl apply -f k8s/frontend-local.yaml

# Tunnel to the Streamlit UI
minikube service frontend-loadbalancer
```


### 3. Setup Observability (Prometheus & Grafana)
```bash
# Install the monitoring stack via Helm
helm repo add prometheus-community [https://prometheus-community.github.io/helm-charts](https://prometheus-community.github.io/helm-charts)
helm install monitoring prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

# Apply the custom ServiceMonitor for FastAPI
kubectl apply -f k8s/service-monitor.yaml

# Port-forward Grafana to localhost:3000
kubectl port-forward svc/monitoring-grafana 3000:80 -n monitoring
```