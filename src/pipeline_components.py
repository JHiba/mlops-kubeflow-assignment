import kfp
from kfp.components import create_component_from_func, InputPath, OutputPath

# 1. Data Extraction
def load_data(output_csv: OutputPath('CSV')):
    import pandas as pd
    from sklearn.datasets import make_regression
    import os
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    print("Generating data (Real Logic)...")
    # Using make_regression to simulate the housing data locally
    X, y = make_regression(n_samples=200, n_features=8, noise=0.1, random_state=42)
    
    cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    df = pd.DataFrame(X, columns=cols)
    df['target'] = y
    
    # Save locally
    df.to_csv(output_csv, index=False)
    print(f"Data generated. Rows: {len(df)}")

# 2. Preprocessing
def preprocess_data(input_csv: InputPath('CSV'), 
                    train_data: OutputPath('CSV'),
                    test_data: OutputPath('CSV')):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import os
    
    os.makedirs(os.path.dirname(train_data), exist_ok=True)
    os.makedirs(os.path.dirname(test_data), exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df.dropna()
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
    
    train_df.to_csv(train_data, index=False)
    test_df.to_csv(test_data, index=False)
    print("Data Preprocessing Complete")

# 3. Training
def train_model(train_data: InputPath('CSV'), 
                model_output: OutputPath('Model')):
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    import os
    
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    
    df = pd.read_csv(train_data)
    X_train = df.drop('target', axis=1)
    y_train = df['target']
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_output)
    print(f"Model Trained.")

# 4. Evaluation
def evaluate_model(test_data: InputPath('CSV'), 
                   model_input: InputPath('Model'),
                   mlpipeline_metrics: OutputPath('Metrics')):
    import pandas as pd
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score
    import json
    import os

    df = pd.read_csv(test_data)
    X_test = df.drop('target', axis=1)
    y_test = df['target']
    
    model = joblib.load(model_input)
    
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        'metrics': [
            {'name': 'MSE', 'numberValue': mse, 'format': 'RAW'},
            {'name': 'R2_Score', 'numberValue': r2, 'format': 'RAW'},
        ]
    }
    
    with open(mlpipeline_metrics, 'w') as f:
        json.dump(metrics, f)
    
    print(f"Evaluation Results -> MSE: {mse}, R2: {r2}")

if __name__ == '__main__':
    # IMPORTANT: This matches the image inside Minikube exactly
    custom_img = 'docker.io/library/mlops-custom:v1'
    
    create_component_from_func(load_data, output_component_file='components/load_data.yaml', base_image=custom_img)
    create_component_from_func(preprocess_data, output_component_file='components/preprocess_data.yaml', base_image=custom_img)
    create_component_from_func(train_model, output_component_file='components/train_model.yaml', base_image=custom_img)
    create_component_from_func(evaluate_model, output_component_file='components/evaluate_model.yaml', base_image=custom_img)
    
    print("SUCCESS: YAML files generated using Custom Image.")