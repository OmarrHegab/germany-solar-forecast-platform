import pandas as pd
from lightgbm import LGBMRegressor
import plotly.graph_objects as go


def load_data():
    """
    Load the processed dataset used for training and evaluation.
    """
    df = pd.read_parquet("data/processed/training_data.parquet")
    return df


def prepare_features(df):
    """
    Prepare the feature matrix and target variable.

    The target is shifted by 24 hours so the model learns to predict
    next-day solar generation.
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()

    # Shift solar generation values to create a 24-hour ahead prediction target
    df["target_t_plus_24h"] = df["solar_generation"].shift(-24)

    # Remove rows with missing target values after shifting
    df = df.dropna().reset_index(drop=True)

    # Features exclude timestamp, current generation, and target column
    X = df.drop(columns=["time", "solar_generation", "target_t_plus_24h"])

    # Target variable for supervised learning
    y = df["target_t_plus_24h"]

    return df, X, y


def train_and_predict(X, y):
    """
    Train a LightGBM regressor and generate predictions
    on the holdout test set.
    """
    # Use a chronological 80/20 split to preserve time order
    split_index = int(len(X) * 0.8)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Initialize the model with a basic configuration
    model = LGBMRegressor(n_estimators=100)

    # Train the model on historical data
    model.fit(X_train, y_train)

    # Generate predictions for the test period
    preds = model.predict(X_test)

    return y_test, preds


def plot_results(y_test, preds):
    """
    Plot actual vs predicted solar generation values
    for visual comparison.
    """
    fig = go.Figure()

    # Plot actual observed values
    fig.add_trace(go.Scatter(
        y=y_test.values[:200],
        mode='lines',
        name='Actual'
    ))

    # Plot predicted values from the model
    fig.add_trace(go.Scatter(
        y=preds[:200],
        mode='lines',
        name='Predicted'
    ))

    fig.update_layout(
        title="Solar Generation Forecast",
        xaxis_title="Time",
        yaxis_title="Generation"
    )

    # Display the interactive chart
    fig.show()


if __name__ == "__main__":
    # Run the full workflow: load data, prepare features,
    # train model, make predictions, and visualize results
    df = load_data()
    df, X, y = prepare_features(df)
    y_test, preds = train_and_predict(X, y)
    plot_results(y_test, preds)