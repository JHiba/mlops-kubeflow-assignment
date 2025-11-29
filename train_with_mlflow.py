"""
Train ML Model with MLflow Tracking
This script trains a model and logs everything to MLflow UI
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

print("="*70)
print("🚀 TRAINING MODEL WITH MLFLOW TRACKING")
print("="*70)
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print()

# Set experiment name
experiment_name = "MLOps_Assignment_Task3"
mlflow.set_experiment(experiment_name)
print(f"📊 Experiment: {experiment_name}")
print()

# Start MLflow run
with mlflow.start_run(run_name="Random_Forest_Training") as run:
    print(f"🏃 Run ID: {run.info.run_id}")
    print()
    
    # 1. Load Data
    print("📊 Step 1: Loading dataset...")
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=200, n_features=8, noise=10, random_state=42)
    cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    X_df = pd.DataFrame(X, columns=cols)
    print(f"✅ Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    
    # Log dataset info
    mlflow.log_param("n_samples", X_df.shape[0])
    mlflow.log_param("n_features", X_df.shape[1])
    print()
    
    # 2. Split Data
    print("📊 Step 2: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    print()
    
    # 3. Train Model
    print("🤖 Step 3: Training Random Forest...")
    n_estimators = 100
    max_depth = 10
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # Log model parameters
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    model.fit(X_train, y_train)
    print("✅ Model trained!")
    print()
    
    # 4. Make Predictions
    print("🔮 Step 4: Making predictions...")
    predictions = model.predict(X_test)
    print("✅ Predictions generated!")
    print()
    
    # 5. Calculate Metrics
    print("📈 Step 5: Calculating metrics...")
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Log metrics to MLflow
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    
    print(f"   MSE:  {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f}")
    print()
    
    # 6. Generate and Log Visualizations
    print("🎨 Step 6: Generating visualizations...")
    
    # Create temp directory for plots
    plot_dir = "mlflow_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot 1: Actual vs Predicted
    print("   Creating: Actual vs Predicted...")
    plt.figure(figsize=(10, 7))
    plt.scatter(y_test, predictions, alpha=0.6, edgecolors='k', s=80, c='#2196F3')
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Prediction')
    plt.xlabel('Actual Values', fontsize=13, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=13, fontweight='bold')
    plt.title('Actual vs Predicted Values', fontsize=15, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    textstr = f'R² Score: {r2:.4f}\nRMSE: {rmse:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plot_path = f'{plot_dir}/actual_vs_predicted.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(plot_path)
    print(f"   ✅ Logged to MLflow")
    plt.close()
    
    # Plot 2: Residual Plot
    print("   Creating: Residual Plot...")
    residuals = y_test - predictions
    plt.figure(figsize=(10, 7))
    plt.scatter(predictions, residuals, alpha=0.6, edgecolors='k', s=80, c='#FF9800')
    plt.axhline(y=0, color='r', linestyle='--', lw=3)
    plt.xlabel('Predicted Values', fontsize=13, fontweight='bold')
    plt.ylabel('Residuals', fontsize=13, fontweight='bold')
    plt.title('Residual Plot', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plot_path = f'{plot_dir}/residual_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(plot_path)
    print(f"   ✅ Logged to MLflow")
    plt.close()
    
    # Plot 3: Feature Importance
    print("   Creating: Feature Importance...")
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
    plt.barh(importance_df['feature'], importance_df['importance'], 
            color=colors, edgecolor='black', linewidth=1.5)
    plt.xlabel('Importance Score', fontsize=13, fontweight='bold')
    plt.ylabel('Features', fontsize=13, fontweight='bold')
    plt.title('Feature Importance', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plot_path = f'{plot_dir}/feature_importance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(plot_path)
    print(f"   ✅ Logged to MLflow")
    plt.close()
    
    # Plot 4: Metrics Summary
    print("   Creating: Metrics Summary...")
    plt.figure(figsize=(12, 7))
    metrics_names = ['R² Score', 'RMSE', 'MAE', 'MSE']
    metrics_values = [r2, rmse, mae, mse]
    colors_metrics = ['#4CAF50', '#2196F3', '#9C27B0', '#F44336']
    
    bars = plt.barh(metrics_names, metrics_values, color=colors_metrics, 
                    edgecolor='black', linewidth=2)
    plt.xlabel('Value', fontsize=13, fontweight='bold')
    plt.title('Model Performance Metrics', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    for bar, value in zip(bars, metrics_values):
        plt.text(value + max(metrics_values)*0.02, bar.get_y() + bar.get_height()/2, 
                f'{value:.4f}', va='center', ha='left', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=1.5))
    
    plot_path = f'{plot_dir}/metrics_summary.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(plot_path)
    print(f"   ✅ Logged to MLflow")
    plt.close()
    
    # 7. Log the Model
    print()
    print("💾 Step 7: Logging model to MLflow...")
    mlflow.sklearn.log_model(model, "random_forest_model")
    print("✅ Model logged!")
    print()
    
    print("="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print()
    print(f"🌐 View results in MLflow UI:")
    print(f"   URL: http://localhost:5000")
    print()
    print(f"📊 Experiment: {experiment_name}")
    print(f"🏃 Run ID: {run.info.run_id}")
    print()
    print("What you'll see in MLflow UI:")
    print("  ✓ All metrics (MSE, RMSE, MAE, R²)")
    print("  ✓ All parameters (model config, data info)")
    print("  ✓ 4 visualization graphs")
    print("  ✓ Trained model artifact")
    print()
    print("="*70)
