import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import plotly.graph_objects as go


def load_data():
    df = pd.read_parquet("data/processed/training_data.parquet")
    return df


def prepare_features(df):
    df = df.copy()
    df["target_t_plus_24h"] = df["solar_generation"].shift(-24)
    df = df.dropna().reset_index(drop=True)

    X = df.drop(columns=["time", "solar_generation", "target_t_plus_24h"])
    y = df["target_t_plus_24h"]

    return df, X, y


def train_and_predict(X, y):
    split_index = int(len(X) * 0.8)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = LGBMRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return y_test, preds


def plot_results(y_test, preds):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=y_test.values[:200],
        mode='lines',
        name='Actual'
    ))

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

    fig.show()


if __name__ == "__main__":
    df = load_data()
    df, X, y = prepare_features(df)
    y_test, preds = train_and_predict(X, y)
    plot_results(y_test, preds)