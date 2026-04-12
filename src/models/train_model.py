import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor


def load_data():
    return pd.read_parquet("data/processed/training_data.parquet")


def prepare_features(df):
    df = df.copy()
    df["target_t_plus_24h"] = df["solar_generation"].shift(-24)
    df = df.dropna().reset_index(drop=True)

    X = df.drop(columns=["time", "solar_generation", "target_t_plus_24h"])
    y = df["target_t_plus_24h"]

    return df, X, y


def train_model(X, y):
    split_index = int(len(X) * 0.8)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"MAE: {mae:.4f}")

    return model


if __name__ == "__main__":
    df = load_data()
    df, X, y = prepare_features(df)
    model = train_model(X, y)

    joblib.dump(model, "artifacts/lightgbm_model.pkl")
    print("Saved artifacts/lightgbm_model.pkl")