"""utils/data_loader.py – Reads real Delhi AQI data from air_quality.csv"""

import numpy as np
import pandas as pd
from datetime import datetime

DATA_PATH  = 'data/air_quality.csv'
POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'CO']


def get_sample_data() -> dict:
    """Returns current hour's real Delhi AQI readings."""
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
        now  = datetime.now()
        hour = now.hour

        # Get rows matching current hour from dataset
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        hour_data  = df[df['hour'] == hour]

        if len(hour_data) == 0:
            hour_data = df.tail(24)

        # Get latest reading for this hour
        row = hour_data.iloc[-1]

        data = {
            'timestamp': now.strftime('%Y-%m-%d %H:%M'),
            'city': 'Delhi (Anand Vihar)',
            'pollutants': {
                'PM2.5': {
                    'value': round(float(row['PM2.5']), 1),
                    'unit':  'µg/m³'
                },
                'PM10': {
                    'value': round(float(row['PM10']), 1),
                    'unit':  'µg/m³'
                },
                'NO2': {
                    'value': round(float(row['NO2']), 1),
                    'unit':  'µg/m³'
                },
                'CO': {
                    'value': round(float(row['CO']), 2),
                    'unit':  'mg/m³'
                },
            },
            'weather': {
                'temperature': round(float(row['temperature']), 1),
                'humidity':    round(float(row['humidity']), 1),
                'wind_speed':  round(float(row['wind_speed']), 1),
            }
        }
        return data

    except Exception as e:
        print(f"[WARN] Could not read real data: {e}. Using demo values.")
        return _demo_data()


def get_historical_stats(pollutant: str) -> dict:
    """Returns last 7 days (168 hours) of real Delhi data."""
    try:
        df  = pd.read_csv(DATA_PATH, parse_dates=['datetime'])

        # Get last 168 hours
        recent = df[[pollutant, 'datetime']].tail(168)
        values     = recent[pollutant].round(2).tolist()
        timestamps = recent['datetime'].dt.strftime('%Y-%m-%d %H:%M').tolist()

        unit = 'mg/m³' if pollutant == 'CO' else 'µg/m³'

        return {
            'pollutant':  pollutant,
            'unit':       unit,
            'timestamps': timestamps,
            'values':     values,
            'stats': {
                'mean': round(float(np.mean(values)), 2),
                'max':  round(float(np.max(values)), 2),
                'min':  round(float(np.min(values)), 2),
                'std':  round(float(np.std(values)), 2),
            }
        }

    except Exception as e:
        print(f"[WARN] Could not read historical data: {e}. Using demo values.")
        return _demo_historical(pollutant)


# ─── Demo fallback (if CSV not found) ────────────────────────────────────────
def _demo_data() -> dict:
    np.random.seed(int(datetime.now().hour))
    hour    = datetime.now().hour
    diurnal = 1 + 0.4 * np.sin(2 * np.pi * hour / 24 - np.pi / 2)
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'city': 'Delhi (Demo)',
        'pollutants': {
            'PM2.5': {'value': round(float(80  * diurnal + np.random.normal(0, 8)),   1), 'unit': 'µg/m³'},
            'PM10':  {'value': round(float(140 * diurnal + np.random.normal(0, 12)),  1), 'unit': 'µg/m³'},
            'NO2':   {'value': round(float(44  * diurnal + np.random.normal(0, 4)),   1), 'unit': 'µg/m³'},
            'CO':    {'value': round(float(1.2 * diurnal + np.random.normal(0, 0.1)), 2), 'unit': 'mg/m³'},
        },
        'weather': {
            'temperature': round(float(28 + np.random.normal(0, 2)), 1),
            'humidity':    round(float(65 + np.random.normal(0, 5)), 1),
            'wind_speed':  round(abs(float(3  + np.random.normal(0, 1))), 1),
        }
    }


def _demo_historical(pollutant: str) -> dict:
    np.random.seed(42)
    hours  = 168
    t      = np.arange(hours)
    base   = {'PM2.5': 80, 'PM10': 140, 'NO2': 44, 'CO': 1.2}.get(pollutant, 50)
    values = list(np.clip(
        base + base * 0.3 * np.sin(2 * np.pi * t / 24)
        + np.random.normal(0, base * 0.08, hours), 0, None
    ).round(2))
    now        = datetime.now()
    timestamps = [
        (now - pd.Timedelta(hours=hours - 1 - i)).strftime('%Y-%m-%d %H:%M')
        for i in range(hours)
    ]
    unit = 'mg/m³' if pollutant == 'CO' else 'µg/m³'
    return {
        'pollutant':  pollutant,
        'unit':       unit,
        'timestamps': timestamps,
        'values':     values,
        'stats': {
            'mean': round(float(np.mean(values)), 2),
            'max':  round(float(np.max(values)), 2),
            'min':  round(float(np.min(values)), 2),
            'std':  round(float(np.std(values)), 2),
        }
    }
