"""utils/predictor.py – PyTorch LSTM model loader and forecast generator."""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import torch
import torch.nn as nn

MODELS_DIR = 'models'
DATA_PATH  = 'data/air_quality.csv'
SEQ_LEN    = 24
FEATURES   = ['PM2.5', 'PM10', 'NO2', 'CO', 'temperature', 'humidity', 'wind_speed']
DEVICE     = torch.device('cpu')  # CPU only


# ─── LSTM Architecture (must match train_model.py) ────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.bn   = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.drop(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


# ─── Load Model ───────────────────────────────────────────────────────────────
def load_model_and_scaler(pollutant: str = 'PM2.5'):
    safe        = pollutant.replace('.', '_').replace('/', '_')
    model_path  = f"{MODELS_DIR}/{safe}_model.pt"
    scaler_path = f"{MODELS_DIR}/{safe}_scaler.pkl"
    meta_path   = f"{MODELS_DIR}/{safe}_meta.json"

    if not os.path.exists(model_path):
        print(f"[WARN] No trained model found for {pollutant}. Running in demo mode.")
        return None, None

    with open(meta_path) as f:
        meta = json.load(f)

    model = LSTMModel(
        input_size  = len(FEATURES),
        hidden_size = meta.get('hidden_size', 64),
        num_layers  = meta.get('num_layers', 2),
        dropout     = meta.get('dropout', 0.2)
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


# ─── Forecast ─────────────────────────────────────────────────────────────────
def predict_future(model, scaler, pollutant: str, steps: int = 24) -> list:
    if model is None or scaler is None:
        return _demo_forecast(pollutant, steps)

    try:
        df     = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
        recent = df[FEATURES].values[-SEQ_LEN:]
    except Exception:
        return _demo_forecast(pollutant, steps)

    scaled     = scaler.transform(recent)
    target_idx = FEATURES.index(pollutant)
    unit       = _unit(pollutant)

    predictions = []
    seq = scaled.copy()
    now = datetime.now()

    model.eval()
    with torch.no_grad():
        for i in range(steps):
            x = torch.tensor(seq[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
            pred_scaled = model(x).item()

            dummy = seq[-1].copy()
            dummy[target_idx] = pred_scaled
            actual = scaler.inverse_transform(dummy.reshape(1, -1))[0, target_idx]
            actual = max(0.0, float(actual))

            dt  = now + timedelta(hours=i)
            aqi = _value_to_aqi(pollutant, actual)
            predictions.append({
                'timestamp': dt.strftime('%Y-%m-%d %H:%M'),
                'value':    round(actual, 2),
                'aqi':      aqi,
                'category': _aqi_category(aqi),
                'unit':     unit
            })

            new_row = seq[-1].copy()
            new_row[target_idx] = pred_scaled
            seq = np.vstack([seq, new_row])

    return predictions


# ─── Demo fallback ────────────────────────────────────────────────────────────
def _demo_forecast(pollutant: str, steps: int) -> list:
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    base = {'PM2.5': 85, 'PM10': 140, 'NO2': 45, 'CO': 1.2}.get(pollutant, 50)
    unit = _unit(pollutant)
    now  = datetime.now()
    results = []
    value = base
    for i in range(steps):
        dt = now + timedelta(hours=i)
        hour_factor = np.sin(2 * np.pi * dt.hour / 24 - np.pi / 2)
        value = max(0, value + base * 0.05 * hour_factor + np.random.normal(0, base * 0.04))
        aqi   = _value_to_aqi(pollutant, value)
        results.append({
            'timestamp': dt.strftime('%Y-%m-%d %H:%M'),
            'value':    round(float(value), 2),
            'aqi':      aqi,
            'category': _aqi_category(aqi),
            'unit':     unit
        })
    return results


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _unit(pollutant: str) -> str:
    return 'mg/m³' if pollutant == 'CO' else 'µg/m³'

def _value_to_aqi(pollutant: str, value: float) -> int:
    bp = {
        'PM2.5': [(0,30,0,50),(30,60,51,100),(60,90,101,200),(90,120,201,300),(120,250,301,400),(250,500,401,500)],
        'PM10':  [(0,50,0,50),(50,100,51,100),(100,250,101,200),(250,350,201,300),(350,430,301,400),(430,600,401,500)],
        'NO2':   [(0,40,0,50),(40,80,51,100),(80,180,101,200),(180,280,201,300),(280,400,301,400),(400,800,401,500)],
        'CO':    [(0,1,0,50),(1,2,51,100),(2,10,101,200),(10,17,201,300),(17,34,301,400),(34,50,401,500)],
    }
    for clo, chi, ilo, ihi in bp.get(pollutant, []):
        if clo <= value <= chi:
            return int(ilo + (ihi - ilo) * (value - clo) / (chi - clo))
    return 500 if value > 0 else 0

def _aqi_category(aqi: int) -> str:
    if aqi <= 50:   return 'Good'
    if aqi <= 100:  return 'Satisfactory'
    if aqi <= 200:  return 'Moderate'
    if aqi <= 300:  return 'Poor'
    if aqi <= 400:  return 'Very Poor'
    return 'Severe'
