import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor


def load_data():
    """
    Load the processed dataset used for model training.
    """
    return pd.read_parquet("data/processed/training_data.parquet")


def prepare_features(df):
    """
    Prepare the feature matrix and target variable.

    The target is shifted forward by 24 hours so the model learns
    to forecast next-day solar generation.
    """
    # Work on a copy to avoid modifying the original dataframe
    df = df.copy()

    # Create a 24-hour ahead prediction target
    df["target_t_plus_24h"] = df["solar_generation"].shift(-24)

    # Remove rows with missing values created by shifting
    df = df.dropna().reset_index(drop=True)

    # Exclude non-feature columns
    X = df.drop(columns=["time", "solar_generation", "target_t_plus_24h"])

    # Define target variable
    y = df["target_t_plus_24h"]

    return df, X, y


def train_model(X, y):
    """
    Train a LightGBM regression model and evaluate it
    using Mean Absolute Error (MAE).
    """
    # Keep chronological order when splitting time-series data
    split_index = int(len(X) * 0.8)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Initialize the model with tuned hyperparameters
    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbosity=-1
    )

    # Train the model on historical observations
    model.fit(X_train, y_train)

    # Predict on the holdout test set
    preds = model.predict(X_test)

    # Calculate Mean Absolute Error for evaluation
    mae = mean_absolute_error(y_test, preds)
    print(f"MAE: {mae:.4f}")

    return model


if __name__ == "__main__":
    # Run the full training pipeline
    df = load_data()
    df, X, y = prepare_features(df)

    # Train and evaluate the model
    model = train_model(X, y)

    # Save the trained model for later inference in the dashboard
    joblib.dump(model, "artifacts/lightgbm_model.pkl")
    print("Saved artifacts/lightgbm_model.pkl")