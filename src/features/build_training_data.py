import pandas as pd


def load_data():
    """
    Load raw weather and solar datasets from disk.
    """
    weather = pd.read_parquet("data/raw/weather.parquet")
    solar = pd.read_parquet("data/raw/solar.parquet")

    return weather, solar


def merge_data(weather, solar):
    """
    Merge weather and solar datasets on the timestamp column.

    An inner join ensures that only timestamps present in both
    datasets are kept for training.
    """
    # Merge both datasets using the shared time column
    df = pd.merge(weather, solar, on="time", how="inner")

    # Sort chronologically for time-series consistency
    df = df.sort_values("time").reset_index(drop=True)

    return df


def add_time_features(df):
    """
    Create additional time-based features from the timestamp.

    These features help the model capture daily, weekly,
    and seasonal patterns in solar generation.
    """
    # Work on a copy to avoid modifying the original dataframe
    df = df.copy()

    # Hour of the day (captures daylight cycle)
    df["hour"] = df["time"].dt.hour

    # Day of the week (captures weekday/weekend effects)
    df["day_of_week"] = df["time"].dt.dayofweek

    # Month of the year (captures seasonal variation)
    df["month"] = df["time"].dt.month

    # Day number within the year (captures yearly trends)
    df["day_of_year"] = df["time"].dt.dayofyear

    return df


def save_data(df):
    """
    Save the processed dataset for model training.
    """
    path = "data/processed/training_data.parquet"

    # Save as parquet for efficient storage and loading
    df.to_parquet(path, index=False)

    print(f"Saved {path}")


if __name__ == "__main__":
    # Run preprocessing pipeline step by step
    weather, solar = load_data()

    # Merge datasets into one training table
    df = merge_data(weather, solar)

    # Add engineered time-based features
    df = add_time_features(df)

    # Save the processed dataset
    save_data(df)

    # Print sample output for quick inspection/debugging
    print(df.head())
    print(df.columns.tolist())