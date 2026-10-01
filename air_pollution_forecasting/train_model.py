

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


POLLUTANTS     = ['PM2.5', 'PM10', 'NO2', 'CO']
FEATURES       = ['PM2.5', 'PM10', 'NO2', 'CO', 'temperature', 'humidity', 'wind_speed']
SEQ_LEN        = 24
EPOCHS         = 60
BATCH_SIZE     = 32
HIDDEN_SIZE    = 64
NUM_LAYERS     = 2
DROPOUT        = 0.2
LR             = 1e-3
MODELS_DIR     = 'models'
DATA_PATH      = 'data/air_quality.csv'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs('data', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")

def generate_synthetic_dataset(n_days: int = 730) -> pd.DataFrame:
    np.random.seed(42)
    hours = n_days * 24
    t = np.arange(hours)

    seasonal = 20 * np.sin(2 * np.pi * t / (365 * 24))
    diurnal  = 10 * np.sin(2 * np.pi * t / 24 - np.pi / 2)
    trend    = 0.002 * t

    pm25 = np.clip(60 + seasonal + diurnal + trend + np.random.normal(0, 8, hours), 5, 500)
    pm10 = np.clip(1.7 * pm25 + np.random.normal(0, 10, hours), 10, 700)
    no2  = np.clip(30 + 0.3 * pm25 + np.random.normal(0, 5, hours), 5, 200)
    co   = np.clip(0.5 + 0.008 * pm25 + np.random.normal(0, 0.1, hours), 0.1, 10)
    temp = 25 + 10 * np.sin(2 * np.pi * t / (365 * 24)) + np.random.normal(0, 3, hours)
    humi = 60 + 15 * np.sin(2 * np.pi * t / (365 * 24) + np.pi) + np.random.normal(0, 5, hours)
    wind = np.abs(3 + np.random.normal(0, 1.5, hours))

    timestamps = pd.date_range('2023-01-01', periods=hours, freq='h')
    return pd.DataFrame({
        'datetime': timestamps,
        'PM2.5': pm25, 'PM10': pm10, 'NO2': no2, 'CO': co,
        'temperature': temp, 'humidity': humi, 'wind_speed': wind
    })


def load_or_generate_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        print(f"[INFO] Loading dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH, parse_dates=['datetime'])
    print("[INFO] Generating synthetic dataset ...")
    df = generate_synthetic_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"[INFO] Saved to {DATA_PATH}")
    return df



class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_sequences(data: np.ndarray, seq_len: int, target_col: int):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len, target_col])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─── 3. LSTM Model ────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
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



def train_and_save(pollutant: str, df: pd.DataFrame):
    print(f"\n{'='*55}")
    print(f"  Training  ->  {pollutant}")
    print(f"{'='*55}")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES].values)

    target_idx = FEATURES.index(pollutant)
    X, y = build_sequences(scaled, SEQ_LEN, target_idx)

    split    = int(len(X) * 0.85)
    train_ds = TimeSeriesDataset(X[:split], y[:split])
    val_ds   = TimeSeriesDataset(X[split:], y[split:])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = LSTMModel(len(FEATURES), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-5)
    criterion = nn.HuberLoss()

    best_val_loss = float('inf')
    best_state    = None
    patience_cnt  = 0
    PATIENCE      = 10

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            all_preds.extend(model(xb.to(DEVICE)).cpu().numpy())
            all_true.extend(yb.numpy())

    dummy = np.zeros((len(all_true), len(FEATURES)))
    dummy[:, target_idx] = all_true
    y_true_actual = scaler.inverse_transform(dummy)[:, target_idx]
    dummy[:, target_idx] = all_preds
    y_pred_actual = scaler.inverse_transform(dummy)[:, target_idx]

    rmse = float(np.sqrt(mean_squared_error(y_true_actual, y_pred_actual)))
    mae  = float(mean_absolute_error(y_true_actual, y_pred_actual))
    print(f"\n  Result: {pollutant}  ->  RMSE: {rmse:.3f}   MAE: {mae:.3f}")

    
    safe = pollutant.replace('.', '_').replace('/', '_')
    torch.save(model.state_dict(), f"{MODELS_DIR}/{safe}_model.pt")
    with open(f"{MODELS_DIR}/{safe}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    meta = {
        'pollutant': pollutant, 'rmse': round(rmse, 3), 'mae': round(mae, 3),
        'features': FEATURES, 'seq_len': SEQ_LEN, 'target_idx': target_idx,
        'hidden_size': HIDDEN_SIZE, 'num_layers': NUM_LAYERS, 'dropout': DROPOUT
    }
    with open(f"{MODELS_DIR}/{safe}_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved -> models/{safe}_model.pt")
    return rmse, mae



if __name__ == '__main__':
    df = load_or_generate_data()
    results = {}
    for p in POLLUTANTS:
        rmse, mae = train_and_save(p, df)
        results[p] = {'RMSE': rmse, 'MAE': mae}

    print("\n" + "="*55)
    print("  TRAINING COMPLETE")
    print("="*55)
    for p, m in results.items():
        print(f"  {p:6s}  ->  RMSE: {m['RMSE']:.3f}   MAE: {m['MAE']:.3f}")
    print("\nNow run:  python app.py  ->  http://localhost:5000")
