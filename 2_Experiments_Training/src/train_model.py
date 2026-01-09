import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

def main():
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to input data")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for the regressor")
    args = parser.parse_args()

    # Enable MLflow autologging
    # This automatically logs metrics, parameters, and the model
    mlflow.sklearn.autolog()

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)

    # Separate features and target
    # Assuming the target column is named 'target' based on the standard diabetes dataset
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    print(f"Training model with learning_rate: {args.learning_rate}")
    model = GradientBoostingRegressor(learning_rate=args.learning_rate, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    # Explicitly logging metrics if needed (autolog handles this, but explicit logging is good for emphasis)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Save model to outputs folder
    # Azure ML automatically uploads the 'outputs' folder
    os.makedirs("outputs", exist_ok=True)
    mlflow.sklearn.save_model(model, "outputs/model")
    print("Model saved to outputs/model")

if __name__ == "__main__":
    main()
