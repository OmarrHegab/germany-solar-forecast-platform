import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@st.cache_data
def load_data():
    """
    Load the processed training dataset and create the forecasting target.

    The target is shifted by 24 hours so the model predicts solar generation
    one day ahead. Rows with missing target values at the end of the dataset
    are removed after shifting.
    """
    # Read the preprocessed dataset from disk
    df = pd.read_parquet("data/processed/training_data.parquet")

    # Work on a copy to avoid accidental side effects
    df = df.copy()

    # Create a day-ahead target: current row features -> solar generation 24 hours later
    df["target_t_plus_24h"] = df["solar_generation"].shift(-24)

    # Drop the final rows that no longer have a valid future target
    df = df.dropna().reset_index(drop=True)

    return df


@st.cache_resource
def load_model():
    """
    Load the trained LightGBM model from the artifacts folder.

    Streamlit caches the model so it is only loaded once per session,
    which improves app performance.
    """
    return joblib.load("artifacts/lightgbm_model.pkl")


def prepare_xy(df):
    """
    Split the dataframe into features and target.

    Excludes:
    - 'time' because it is used for display, not model input
    - 'solar_generation' because it is the current value, not the prediction target
    - 'target_t_plus_24h' because that is the label
    """
    feature_cols = [
        c for c in df.columns
        if c not in ["time", "solar_generation", "target_t_plus_24h"]
    ]

    X = df[feature_cols]
    y = df["target_t_plus_24h"]

    return X, y, feature_cols


def make_predictions(df, model):
    """
    Generate predictions on the final 20% of the dataset.

    A simple chronological split is used here instead of random sampling
    because this is time-series style data and we want evaluation to reflect
    future-looking performance more realistically.
    """
    X, y, _ = prepare_xy(df)

    # Use the last 20% of the data as the evaluation window
    split_index = int(len(X) * 0.8)

    X_test = X.iloc[split_index:].copy()
    y_test = y.iloc[split_index:].copy()
    time_test = df["time"].iloc[split_index:].copy()

    # Run model inference on the holdout set
    preds = model.predict(X_test)

    # Combine timestamps, actual values, and predictions into one table
    results = pd.DataFrame({
        "time": time_test,
        "actual": y_test,
        "predicted": preds
    }).reset_index(drop=True)

    return results


def build_plot(results, n_points):
    """
    Build an interactive Plotly chart comparing actual vs predicted values.

    Only the most recent n_points are displayed to keep the chart readable
    and responsive in the Streamlit app.
    """
    # Focus on the latest observations selected by the user
    plot_df = results.tail(n_points)

    fig = go.Figure()

    # Actual observed solar generation
    fig.add_trace(go.Scatter(
        x=plot_df["time"],
        y=plot_df["actual"],
        mode="lines",
        name="Actual"
    ))

    # Model forecast for the same timestamps
    fig.add_trace(go.Scatter(
        x=plot_df["time"],
        y=plot_df["predicted"],
        mode="lines",
        name="Predicted"
    ))

    fig.update_layout(
        title="Day-Ahead Solar Generation Forecast",
        xaxis_title="Time",
        yaxis_title="Generation",
        hovermode="x unified"
    )

    return fig


def main():
    """
    Main entry point for the Streamlit dashboard.

    This function loads the data and model, generates predictions,
    calculates a simple error metric, and renders the interactive UI.
    """
    st.set_page_config(page_title="Solar Forecast Dashboard", layout="wide")

    st.title("Germany Solar Forecast Dashboard")
    st.write("Interactive comparison of actual vs predicted day-ahead solar generation.")

    # Load required resources
    df = load_data()
    model = load_model()

    # Generate evaluation predictions
    results = make_predictions(df, model)

    # Mean Absolute Error gives an easy-to-interpret summary of prediction error
    mae = (results["actual"] - results["predicted"]).abs().mean()

    # Show high-level dashboard metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows in evaluation set", len(results))
    with col2:
        st.metric("MAE", f"{mae:.2f}")

    # Let the user control how much recent history to display
    n_points = st.slider(
        "Number of recent points to display",
        min_value=50,
        max_value=500,
        value=200,
        step=50
    )

    # Render the forecast comparison chart
    fig = build_plot(results, n_points)
    st.plotly_chart(fig, use_container_width=True)

    # Optional table view for users who want to inspect the raw values
    with st.expander("Show recent prediction data"):
        st.dataframe(results.tail(n_points), use_container_width=True)


if __name__ == "__main__":
    main()