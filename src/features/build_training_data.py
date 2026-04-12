import pandas as pd


def load_data():
    weather = pd.read_parquet("data/raw/weather.parquet")
    solar = pd.read_parquet("data/raw/solar.parquet")
    return weather, solar


def merge_data(weather, solar):
    df = pd.merge(weather, solar, on="time", how="inner")
    df = df.sort_values("time").reset_index(drop=True)
    return df


def add_time_features(df):
    df = df.copy()

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear

    return df


def save_data(df):
    path = "data/processed/training_data.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {path}")


if __name__ == "__main__":
    weather, solar = load_data()
    df = merge_data(weather, solar)
    df = add_time_features(df)
    save_data(df)

    print(df.head())
    print(df.columns.tolist())