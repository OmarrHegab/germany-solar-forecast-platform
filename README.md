# 🇩🇪 Germany Solar Forecast Platform

Production-ready machine learning pipeline for **day-ahead solar energy forecasting** using weather data, feature engineering, and a LightGBM model. Includes an interactive dashboard for exploring predictions.

---

## 🚀 Overview

This project simulates a real-world ML system for forecasting solar generation in Germany. It covers the full lifecycle:

* Data ingestion from external APIs
* Feature engineering for time-series forecasting
* Model training and evaluation
* Interactive dashboard for visualization

---

## 🧠 Features

* 📡 Weather data ingestion (Open-Meteo API)
* ⚡ Synthetic solar generation target (based on radiation)
* 🔧 Feature engineering (time-based + weather features)
* 🤖 ML models:

  * Linear Regression (baseline)
  * LightGBM (improved model)
* 📊 Model evaluation with forecasting metrics (MAE)
* 🌐 Interactive dashboard (Streamlit + Plotly)
* 📦 Reproducible project structure

---

## 📁 Project Structure

```
germany-solar-forecast-platform/
│
├── artifacts/
│   └── lightgbm_model.pkl
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── weather.parquet
│   │   └── solar.parquet
│   └── processed/
│       └── training_data.parquet
│
├── reports/
│   └── forecast_plot.png
│
├── src/
│   ├── data_ingestion/
│   │   └── fetch_data.py
│   ├── features/
│   │   └── build_training_data.py
│   └── models/
│       ├── train_model.py
│       └── evaluate_model.py
│
├── run_app.bat
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git https://github.com/OmarrHegab/germany-solar-forecast-platform.git
cd germany-solar-forecast-platform
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1. Fetch data

```bash
python src/data_ingestion/fetch_data.py
```

### 2. Build training dataset

```bash
python src/features/build_training_data.py
```

### 3. Train model

```bash
python src/models/train_model.py
```

### 4. Run dashboard

```bash
python -m streamlit run dashboard/app.py
```

Or simply:

```bash
run_app.bat
```

---

## 📊 Results

| Model             | MAE |
| ----------------- | --- |
| Linear Regression | ~21 |
| LightGBM          | ~16 |

LightGBM improves performance by capturing non-linear relationships in weather and temporal features.

---

## 📈 Dashboard

The Streamlit dashboard provides:

* Interactive visualization of predictions vs actual values
* Zooming and hover functionality
* Adjustable time window
* Real-time model evaluation metrics

![alt text](<Screenshot 2026-04-12 165525.png>)

---

## 🔧 Tech Stack

* Python
* pandas, numpy
* scikit-learn
* LightGBM
* Streamlit
* Plotly

---

## 🚧 Future Improvements

* Replacing synthetic solar data with real data from Fraunhofer ISE
* Adding FastAPI for model serving
* Deploy dashboard online (Streamlit Cloud / AWS)
* Adding automated retraining pipeline
* Improving feature engineering (lag features, rolling windows)

---

## 💡 Motivation

This project demonstrates how to move from a simple ML model to a **production-style system** with data pipelines, evaluation, and user-facing interfaces.

---

## 👤 Author

Omar Hegab: https://github.com/OmarrHegab
