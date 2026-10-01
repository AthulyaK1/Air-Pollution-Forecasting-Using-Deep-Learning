

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


POLLUTANTS  = ['PM2.5', 'PM10', 'NO2', 'CO']
FEATURES    = ['PM2.5', 'PM10', 'NO2', 'CO', 'temperature', 'humidity', 'wind_speed']
SEQ_LEN     = 24
MODELS_DIR  = 'models'
DATA_PATH   = 'data/air_quality.csv'

os.makedirs('data', exist_ok=True)

report_lines = []
def log(msg=''):
    print(msg)
    report_lines.append(msg)



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
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


def build_sequences(data, seq_len, target_col):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len, target_col])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)



log("=" * 60)
log("  MODEL EVALUATION REPORT")
log("=" * 60)
log()

df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])

all_results = []

for pollutant in POLLUTANTS:
    log(f"  Evaluating: {pollutant}")
    log(f"  {'-'*50}")

    safe        = pollutant.replace('.', '_').replace('/', '_')
    model_path  = f"{MODELS_DIR}/{safe}_model.pt"
    scaler_path = f"{MODELS_DIR}/{safe}_scaler.pkl"
    meta_path   = f"{MODELS_DIR}/{safe}_meta.json"

    if not os.path.exists(model_path):
        log(f"  [SKIP] Model not found: {model_path}")
        log()
        continue

    # Load
    with open(meta_path) as f:
        meta = json.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    model = LSTMModel(len(FEATURES), meta['hidden_size'],
                      meta['num_layers'], meta['dropout'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Prepare data
    scaled     = scaler.transform(df[FEATURES].values)
    target_idx = FEATURES.index(pollutant)
    X, y       = build_sequences(scaled, SEQ_LEN, target_idx)

    # Use last 15% as test set
    split = int(len(X) * 0.85)
    X_test = torch.tensor(X[split:], dtype=torch.float32)
    y_test = y[split:]

    # Predict
    with torch.no_grad():
        preds = model(X_test).numpy()

    # Inverse transform
    dummy_true = np.zeros((len(y_test), len(FEATURES)))
    dummy_true[:, target_idx] = y_test
    y_true_actual = scaler.inverse_transform(dummy_true)[:, target_idx]

    dummy_pred = np.zeros((len(preds), len(FEATURES)))
    dummy_pred[:, target_idx] = preds
    y_pred_actual = scaler.inverse_transform(dummy_pred)[:, target_idx]

    # Metrics
    rmse = float(np.sqrt(mean_squared_error(y_true_actual, y_pred_actual)))
    mae  = float(mean_absolute_error(y_true_actual, y_pred_actual))
    r2   = float(r2_score(y_true_actual, y_pred_actual))
    mape = float(np.mean(np.abs((y_true_actual - y_pred_actual) /
                                 np.clip(np.abs(y_true_actual), 1e-5, None))) * 100)

    unit = 'mg/m³' if pollutant == 'CO' else 'µg/m³'

    log(f"  RMSE  : {rmse:.3f} {unit}   (lower is better)")
    log(f"  MAE   : {mae:.3f} {unit}   (lower is better)")
    log(f"  R²    : {r2:.4f}           (closer to 1.0 is better)")
    log(f"  MAPE  : {mape:.2f}%          (lower is better)")
    log()

    # Performance rating
    if r2 >= 0.90:   rating = 'EXCELLENT'
    elif r2 >= 0.80: rating = 'GOOD'
    elif r2 >= 0.70: rating = 'ACCEPTABLE'
    else:            rating = 'NEEDS IMPROVEMENT'
    log(f"  Model Rating : {rating}")
    log()

    all_results.append({
        'Pollutant': pollutant,
        'RMSE': round(rmse, 3),
        'MAE':  round(mae, 3),
        'R2':   round(r2, 4),
        'MAPE': round(mape, 2),
        'Unit': unit,
        'Rating': rating
    })

    # Save predictions sample
    timestamps = df['datetime'].values[SEQ_LEN + split:]
    n = min(len(timestamps), len(y_true_actual))
    sample = pd.DataFrame({
        'datetime':   timestamps[:n],
        'actual':     y_true_actual[:n].round(2),
        'predicted':  y_pred_actual[:n].round(2),
        'error':      (y_true_actual[:n] - y_pred_actual[:n]).round(2)
    })
    sample.to_csv(f'data/{safe}_predictions.csv', index=False)
    log(f"  Predictions saved : data/{safe}_predictions.csv")
    log()



log("=" * 60)
log("  SUMMARY")
log("=" * 60)
log(f"  {'Pollutant':<10} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE':>8}  Rating")
log(f"  {'-'*58}")
for r in all_results:
    log(f"  {r['Pollutant']:<10} {r['RMSE']:>8} {r['MAE']:>8} "
        f"{r['R2']:>8} {r['MAPE']:>7}%  {r['Rating']}")

log()
log("  Metric Guide:")
log("  RMSE  = Root Mean Square Error (penalises large errors)")
log("  MAE   = Mean Absolute Error    (average prediction error)")
log("  R²    = How well model fits    (1.0 = perfect)")
log("  MAPE  = Mean Absolute % Error  (error as percentage)")
log()
log("  Next step: python app.py  ->  http://127.0.0.1:5000")

# Save report
with open('data/evaluation_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
log()
log("  Report saved: data/evaluation_report.txt")
