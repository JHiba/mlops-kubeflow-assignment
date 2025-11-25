FROM python:3.9-slim

# Install dependencies your components need
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    dvc \
    joblib \
    kfp

# Set working directory (not strictly required for this pipeline)
WORKDIR /app
