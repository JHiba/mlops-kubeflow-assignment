# MLOps Kubeflow Pipeline Assignment

##  Project Overview
This project implements a complete Machine Learning Operations (MLOps) pipeline to predict housing prices (using the California Housing dataset logic). The system is designed to demonstrate **reproducibility**, **automation**, and **orchestration** using industry-standard tools.

The pipeline performs the following steps automatically:
1.  **Data Extraction:** Generates/Loads the dataset.
2.  **Preprocessing:** Cleans, scales, and splits data into training/testing sets.
3.  **Model Training:** Trains a Random Forest Regressor model.
4.  **Evaluation:** Calculates MSE and R2 scores to validate model performance.

## Tools & Technologies
* **Orchestration:** Kubeflow Pipelines (KFP) running on Minikube (Kubernetes).
* **Data Versioning:** DVC (Data Version Control) with local remote storage.
* **Containerization:** Docker (Custom image for offline-compatible execution).
* **CI/CD:** GitHub Actions for automated pipeline compilation and testing.
* **Language:** Python 3.9 (Scikit-learn, Pandas).

---

##  Setup Instructions

### 1. Prerequisites
Ensure the following are installed:
* Docker Desktop
* Minikube
* Python 3.9+
* Git

### 2. Environment Setup
Clone the repository and install dependencies:

git clone [https://github.com/JHiba/mlops-kubeflow-assignment.git](https://github.com/JHiba/mlops-kubeflow-assignment.git)
cd mlops-kubeflow-assignment
pip install -r requirements.txt

Infrastructure Setup (Minikube & Kubeflow)
Start the local Kubernetes cluster and deploy the Kubeflow standalone engine:

# Start Minikube
minikube start --cpus 4 --memory 4000

# Deploy Kubeflow Pipelines (KFP)
kubectl apply -k "[github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=1.8.5](https://github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=1.8.5)"
kubectl apply -k "[github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=1.8.5](https://github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=1.8.5)"

# Expose the API (Required for Python script execution)
kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888
Note: Due to the deprecation of KFP v1.8.5 UI images on public registries, pipeline execution is verified programmatically via the API and Pod Logs.

4. Data Versioning (DVC) Setup
Data is tracked using DVC. To initialize and pull data:

dvc init
# (Optional) Set up local remote if starting fresh
mkdir C:\dvc_storage
dvc remote add -d mylocalremote C:\dvc_storage
dvc pull
⚙️ Pipeline Execution Walkthrough
To ensure reliability despite network restrictions inside Minikube, this project uses a Custom Docker Image approach.

Step 1: Build & Load Custom Image
We build the execution environment locally and load it directly into Minikube:


# Build image containing pandas/scikit-learn
docker build -t docker.io/library/mlops-custom:v1 -f Dockerfile .

# Load into Minikube
minikube image load docker.io/library/mlops-custom:v1
Step 2: Compile the Pipeline
Convert the Python pipeline definition into a YAML workflow file:

# Generate component YAMLs
python src/pipeline_components.py

# Generate pipeline.yaml
python pipeline.py
Step 3: Run the Pipeline
Submit the pipeline to the Kubeflow cluster using the Python SDK:


# Ensure port-forward is running on port 8888
python run_pipeline.py
Step 4: Verify Results
Since the UI graph is optional due to image availability, verify the successful execution and accuracy metrics via logs:


# Check for 'Completed' pods
kubectl get pods -n kubeflow

# View the Evaluation Metrics (MSE/R2)
kubectl logs -n kubeflow final-eval-pod

Continuous Integration (CI)
This project uses GitHub Actions for CI/CD.

Workflow File: .github/workflows/mlops_ci.yml

Triggers: Pushes to the main branch.

Actions:

Sets up Python environment.

Installs dependencies.

Generates component configurations.

Compiles the pipeline to verify syntax integrity.

Check the [Actions Tab] in this repository to see the build status history

---

## MLflow Experiment Tracking

This project integrates **MLflow** for experiment tracking and visualization.

### Run Training with MLflow
Execute the training script that logs metrics, parameters, and visualizations:
```bash
python train_with_mlflow.py
```

### Access MLflow UI
Open your browser to: **http://localhost:5000**

### What MLflow Tracks:
- **Metrics**: MSE, RMSE, MAE, R² Score
- **Parameters**: Model configuration (n_estimators, max_depth, etc.)
- **Artifacts**: 
  - Actual vs Predicted plot
  - Residual plot
  - Feature importance chart
  - Metrics summary visualization
- **Model**: Trained Random Forest model artifact

This provides a professional experiment tracking system for reproducible ML workflows.