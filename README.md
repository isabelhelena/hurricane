
<h1 align="center"> Hurricane Trajectory & Intensity Forecasting with HURDAT</h1>

<p align="center">
Machine learning & deep learning for short-term hurricane prediction using NOAA HURDAT2 data
</p>

---

## Project Overview

This project applies **machine learning and deep learning** to **short-term hurricane trajectory and intensity prediction** using historical storm data from the **NOAA HURDAT2 dataset**. A **linear regression baseline** is implemented for comparison, and a **Long Short-Term Memory (LSTM)** neural network is used for sequence-based forecasting.

The model predicts the **next time-step**:
- Latitude  
- Longitude  
- Wind Speed  
- Central Pressure  

from the **four most recent storm observations**.

---

## Exploratory Data Analysis (EDA)

<p align="center">
 <img src="https://github.com/user-attachments/assets/fff5d03b-b1b3-48e8-a0b3-69af5d77fb66" width="32%" />

  <img src="https://github.com/user-attachments/assets/00f96057-f4b3-4966-ad85-d707fc4fd0fe" width="32%" />
  <img src="https://github.com/user-attachments/assets/b3e68d6e-cbc1-4ac9-b33e-286252105f5c" width="32%" />

</p>

These figures visualize storm intensity distributions and spatial track behavior across the Atlantic basin.

---

## Project Structure

```text
project-root/
│
├── data/
│   └── hurdat2_full.csv
│
├── figures/
│   ├── wind_hist.png
│   ├── pressure_hist.png
│   └── tracks_scatter.png
│
├── models/
│   └── lstm_best.pt
│
├── outputs/
│   ├── baseline_metrics.json
│   ├── lstm_metrics.json
│   └── lstm_history.json
│
├── src/
│   ├── model_lstm.py
│   └── utils.py
│
├── eda.py
├── train_baseline.py
├── train_lstm.py
└── README.md

```



##  File Descriptions

###  `eda.py`
Performs **exploratory data analysis (EDA)** on the HURDAT2 dataset.
- Generates wind speed distribution histogram  
- Generates central pressure distribution histogram  
- Visualizes hurricane storm tracks (longitude vs latitude)  
- Automatically saves all plots to the `figures/` directory  

---

###  `train_baseline.py`
Implements the **linear regression baseline model** used for comparison.
- Builds 4-step hurricane sequences  
- Flattens each sequence into a feature vector  
- Solves regression using closed-form least squares  
- Evaluates performance using:
  - MAE
  - RMSE
  - R²
  - Haversine distance (km)
- Saves evaluation results to `outputs/baseline_metrics.json`  

---

###  `train_lstm.py`
Handles **training and evaluation of the LSTM deep learning model**.
- Splits data by storm ID into train/validation/test
- Standardizes input features using training-only statistics
- Trains LSTM using Adam optimizer and early stopping
- Evaluates on train, validation, and test datasets
- Saves:
  - Best trained model to `models/lstm_best.pt`
  - Evaluation metrics to `outputs/lstm_metrics.json`
  - Training history to `outputs/lstm_history.json`

---

###  `src/model_lstm.py`
Defines the **PyTorch LSTM architecture** used for forecasting.
- LSTM layer for time-series sequence modeling
- Fully connected output layer for 4-variable prediction
- Outputs next-step:
  - Latitude
  - Longitude
  - Wind speed
  - Pressure

---

###  `src/utils.py`
Contains all **supporting data and evaluation utilities**.
- Sequence construction from time-series
- Storm-wise dataset splitting
- Feature standardization and inverse scaling
- Evaluation metrics (MAE, RMSE, R²)
- Haversine distance calculation
- Raw HURDAT2 `.txt` → `.csv` data conversion
