import requests
import pandas as pd


def fetch_weather():
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 51.1657,   # Germany center
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

    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    return df



def fetch_solar_dummy(weather_df):
    # Temporary: simulate solar output
    df = weather_df.copy()
    df["solar_generation"] = df["shortwave_radiation"] * 0.5

    return df[["time", "solar_generation"]]



def save_data(df, name):
    path = f"data/raw/{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {path}")


if __name__ == "__main__":
    weather = fetch_weather()
    solar = fetch_solar_dummy(weather)

    save_data(weather, "weather")
    save_data(solar, "solar")