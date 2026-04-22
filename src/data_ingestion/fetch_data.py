import requests
import pandas as pd


def fetch_weather():
    """
    Fetch historical hourly weather data from the Open-Meteo archive API.

    The selected features are relevant for solar generation forecasting,
    such as temperature, cloud cover, and solar radiation.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    # API request parameters
    params = {
        "latitude": 51.1657,   # Approximate center of Germany
        "longitude": 10.4515,
        "start_date": "2022-01-01",
        "end_date": "2023-12-31",
        "hourly": [
            "temperature_2m",
            "cloudcover",
            "shortwave_radiation"
        ],
        "timezone": "Europe/Berlin"
    }

    # Send request to the weather API
    response = requests.get(url, params=params)

    # Parse JSON response
    data = response.json()

    # Convert hourly weather data into a dataframe
    df = pd.DataFrame(data["hourly"])

    # Convert timestamp strings to datetime format
    df["time"] = pd.to_datetime(df["time"])

    return df


def fetch_solar_dummy(weather_df):
    """
    Generate a temporary simulated solar generation dataset.

    Since real solar production data is not yet available,
    this uses shortwave radiation as a simple proxy.
    """
    # Create a copy to avoid modifying the original weather dataframe
    df = weather_df.copy()

    # Simulate solar generation based on incoming solar radiation
    df["solar_generation"] = df["shortwave_radiation"] * 0.5

    return df[["time", "solar_generation"]]


def save_data(df, name):
    """
    Save raw data as a parquet file for later processing.
    """
    path = f"data/raw/{name}.parquet"

    # Save in parquet format for efficient storage
    df.to_parquet(path, index=False)

    print(f"Saved {path}")


if __name__ == "__main__":
    # Fetch raw weather data from API
    weather = fetch_weather()

    # Generate placeholder solar data
    solar = fetch_solar_dummy(weather)

    # Save both datasets locally
    save_data(weather, "weather")
    save_data(solar, "solar")