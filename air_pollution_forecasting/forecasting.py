

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import torch
import torch.nn as nn


POLLUTANTS  = ['PM2.5', 'PM10', 'NO2', 'CO']
FEATURES    = ['PM2.5', 'PM10', 'NO2', 'CO', 'temperature', 'humidity', 'wind_speed']
SEQ_LEN     = 24
FORECAST_HOURS = 48       
MODELS_DIR  = 'models'
DATA_PATH   = 'data/air_quality.csv'

os.makedirs('data', exist_ok=True)

report_lines = []
def log(msg=''):
    print(msg)
    report_lines.append(str(msg))



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



def value_to_aqi(pollutant, value):
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

def aqi_category(aqi):
    if aqi <= 50:   return 'Good'
    if aqi <= 100:  return 'Satisfactory'
    if aqi <= 200:  return 'Moderate'
    if aqi <= 300:  return 'Poor'
    if aqi <= 400:  return 'Very Poor'
    return 'Severe'

def health_advice(category):
    advice = {
        'Good':         'Air quality is good. Safe for all activities.',
        'Satisfactory': 'Air quality is acceptable. Sensitive people take care.',
        'Moderate':     'Sensitive groups should limit outdoor activities.',
        'Poor':         'Everyone should reduce prolonged outdoor exertion.',
        'Very Poor':    'Avoid outdoor activities. Wear mask if going outside.',
        'Severe':       'Health emergency. Stay indoors. Keep windows closed.'
    }
    return advice.get(category, 'No data available.')



def load_model(pollutant):
    safe        = pollutant.replace('.', '_').replace('/', '_')
    model_path  = f"{MODELS_DIR}/{safe}_model.pt"
    scaler_path = f"{MODELS_DIR}/{safe}_scaler.pkl"
    meta_path   = f"{MODELS_DIR}/{safe}_meta.json"

    if not os.path.exists(model_path):
        return None, None, None

    with open(meta_path) as f:
        meta = json.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    model = LSTMModel(len(FEATURES), meta['hidden_size'],
                      meta['num_layers'], meta['dropout'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, scaler, meta



def generate_forecast(model, scaler, pollutant, df, hours=48):
    recent     = df[FEATURES].values[-SEQ_LEN:]
    scaled     = scaler.transform(recent)
    target_idx = FEATURES.index(pollutant)
    unit       = 'mg/m³' if pollutant == 'CO' else 'µg/m³'

    predictions = []
    seq = scaled.copy()
    now = datetime.now()

    model.eval()
    with torch.no_grad():
        for i in range(hours):
            x           = torch.tensor(seq[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
            pred_scaled = model(x).item()

            dummy = seq[-1].copy()
            dummy[target_idx] = pred_scaled
            actual = scaler.inverse_transform(dummy.reshape(1, -1))[0, target_idx]
            actual = max(0.0, float(actual))

            dt       = now + timedelta(hours=i)
            aqi      = value_to_aqi(pollutant, actual)
            category = aqi_category(aqi)

            predictions.append({
                'datetime':    dt.strftime('%Y-%m-%d %H:%M'),
                'hour':        i + 1,
                'pollutant':   pollutant,
                'predicted':   round(actual, 2),
                'unit':        unit,
                'AQI':         aqi,
                'category':    category,
                'health_advice': health_advice(category)
            })

            new_row = seq[-1].copy()
            new_row[target_idx] = pred_scaled
            seq = np.vstack([seq, new_row])

    return predictions



log("=" * 60)
log("  AIR POLLUTION FORECASTING MODULE")
log("=" * 60)
log(f"  Generated at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"  Forecast     : Next {FORECAST_HOURS} hours")
log()

df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
log(f"  Dataset loaded : {len(df)} rows")
log(f"  Using last {SEQ_LEN} hours as input seed")
log()

all_forecasts = []

for pollutant in POLLUTANTS:
    log("=" * 60)
    log(f"  FORECAST : {pollutant}")
    log("=" * 60)

    model, scaler, meta = load_model(pollutant)

    if model is None:
        log(f"  [SKIP] Model not found. Run train_model.py first.")
        log()
        continue

    forecasts = generate_forecast(model, scaler, pollutant, df, FORECAST_HOURS)
    all_forecasts.extend(forecasts)

    unit   = forecasts[0]['unit']
    values = [f['predicted'] for f in forecasts]
    aqis   = [f['AQI'] for f in forecasts]
    cats   = [f['category'] for f in forecasts]

    log(f"  {'Hour':<6} {'Time':<18} {'Value':>10} {'AQI':>6}  Category")
    log(f"  {'-'*55}")
    for fc in forecasts[:24]:   # show first 24 hours
        log(f"  {fc['hour']:<6} {fc['datetime']:<18} "
            f"{fc['predicted']:>8.2f} {fc['unit']}  {fc['AQI']:>4}   {fc['category']}")

    log()
    log(f"  24-Hour Summary:")
    log(f"    Min predicted  : {min(values[:24]):.2f} {unit}")
    log(f"    Max predicted  : {max(values[:24]):.2f} {unit}")
    log(f"    Avg predicted  : {np.mean(values[:24]):.2f} {unit}")
    log(f"    Min AQI        : {min(aqis[:24])}")
    log(f"    Max AQI        : {max(aqis[:24])}")
    log(f"    Worst Category : {cats[aqis.index(max(aqis[:24]))]}")
    log()

    # Health advisory
    worst_aqi  = max(aqis[:24])
    worst_cat  = aqi_category(worst_aqi)
    log(f"  Health Advisory:")
    log(f"    {health_advice(worst_cat)}")
    log()



if all_forecasts:
    forecast_df = pd.DataFrame(all_forecasts)
    forecast_df.to_csv('data/forecast_results.csv', index=False)

    log("=" * 60)
    log("  FORECAST COMPLETE")
    log("=" * 60)
    log(f"  Results saved  : data/forecast_results.csv")
    log(f"  Total rows     : {len(forecast_df)}")
    log()

   
    worst_overall = forecast_df.loc[forecast_df['AQI'].idxmax()]
    log(f"  Worst predicted condition:")
    log(f"    Pollutant : {worst_overall['pollutant']}")
    log(f"    Time      : {worst_overall['datetime']}")
    log(f"    Value     : {worst_overall['predicted']} {worst_overall['unit']}")
    log(f"    AQI       : {worst_overall['AQI']}")
    log(f"    Category  : {worst_overall['category']}")
    log(f"    Advice    : {worst_overall['health_advice']}")

log()
log("  Next step: python app.py  ->  http://127.0.0.1:5000")

# Save report
with open('data/forecast_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
log("  Report saved   : data/forecast_report.txt")
