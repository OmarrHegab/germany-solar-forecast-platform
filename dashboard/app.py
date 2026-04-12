import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/training_data.parquet")
    df = df.copy()
    df["target_t_plus_24h"] = df["solar_generation"].shift(-24)
    df = df.dropna().reset_index(drop=True)
    return df


@st.cache_resource
def load_model():
    return joblib.load("artifacts/lightgbm_model.pkl")


def prepare_xy(df):
    feature_cols = [c for c in df.columns if c not in ["time", "solar_generation", "target_t_plus_24h"]]
    X = df[feature_cols]
    y = df["target_t_plus_24h"]
    return X, y, feature_cols


def make_predictions(df, model):
    X, y, _ = prepare_xy(df)

    split_index = int(len(X) * 0.8)
    X_test = X.iloc[split_index:].copy()
    y_test = y.iloc[split_index:].copy()
    time_test = df["time"].iloc[split_index:].copy()

    preds = model.predict(X_test)

    results = pd.DataFrame({
        "time": time_test,
        "actual": y_test,
        "predicted": preds
    }).reset_index(drop=True)

    return results


def build_plot(results, n_points):
    plot_df = results.tail(n_points)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df["time"],
        y=plot_df["actual"],
        mode="lines",
        name="Actual"
    ))

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
    st.set_page_config(page_title="Solar Forecast Dashboard", layout="wide")
    st.title("Germany Solar Forecast Dashboard")
    st.write("Interactive comparison of actual vs predicted day-ahead solar generation.")

    df = load_data()
    model = load_model()
    results = make_predictions(df, model)

    mae = (results["actual"] - results["predicted"]).abs().mean()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows in evaluation set", len(results))
    with col2:
        st.metric("MAE", f"{mae:.2f}")

    n_points = st.slider("Number of recent points to display", min_value=50, max_value=500, value=200, step=50)

    fig = build_plot(results, n_points)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show recent prediction data"):
        st.dataframe(results.tail(n_points), use_container_width=True)


if __name__ == "__main__":
    main()