
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

RAW_DATA_PATH  = 'data/delhi-weather-aqi-2025.csv'
OUTPUT_PATH    = 'data/air_quality.csv'
PROCESSED_PATH = 'data/processed_data.csv'
REPORT_PATH    = 'data/preprocessed_report.txt'

FEATURES   = ['PM2.5', 'PM10', 'NO2', 'CO', 'temperature', 'humidity', 'wind_speed']
POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'CO']

report_lines = []
def log(msg=''):
    print(msg)
    report_lines.append(str(msg))


log("=" * 60)
log("  STEP 1 : LOAD REAL DATASET")
log("=" * 60)

df_raw = pd.read_csv(RAW_DATA_PATH)
log(f"  File     : {RAW_DATA_PATH}")
log(f"  Shape    : {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
log(f"  Columns  : {list(df_raw.columns)}")
log(f"  Locations: {list(df_raw['location'].unique())}")
log()



log("=" * 60)
log("  STEP 2 : SELECT LOCATION")
log("=" * 60)

LOCATION = 'Anand Vihar'
df_raw = df_raw[df_raw['location'] == LOCATION].copy()
df_raw = df_raw.reset_index(drop=True)
log(f"  Selected : {LOCATION}")
log(f"  Rows     : {len(df_raw)}")
log()



log("=" * 60)
log("  COLUMN MAPPING")
log("=" * 60)

df_raw['datetime'] = pd.to_datetime(
    df_raw['date_ist'] + ' ' + df_raw['time_ist'],
    format='%d/%m/%Y %H:%M'
)

df = pd.DataFrame()
df['datetime']    = df_raw['datetime']
df['PM2.5']       = pd.to_numeric(df_raw['pm2_5'],        errors='coerce')
df['PM10']        = pd.to_numeric(df_raw['pm10'],          errors='coerce')
df['NO2']         = pd.to_numeric(df_raw['no2'],           errors='coerce')
df['CO']          = pd.to_numeric(df_raw['co'],            errors='coerce') / 1000
df['temperature'] = pd.to_numeric(df_raw['temp_c'],        errors='coerce')
df['humidity']    = pd.to_numeric(df_raw['humidity'],      errors='coerce')
df['wind_speed']  = pd.to_numeric(df_raw['windspeed_kph'], errors='coerce') / 3.6
df['AQI']         = pd.to_numeric(df_raw['aqi_index'],     errors='coerce')

df = df.sort_values('datetime').reset_index(drop=True)

log(f"  pm2_5         -> PM2.5        (ug/m3)")
log(f"  pm10          -> PM10         (ug/m3)")
log(f"  no2           -> NO2          (ug/m3)")
log(f"  co / 1000     -> CO           (mg/m3)")
log(f"  temp_c        -> temperature  (C)")
log(f"  humidity      -> humidity     (%)")
log(f"  windspeed/3.6 -> wind_speed   (m/s)")
log(f"  Date range    : {df['datetime'].min()} to {df['datetime'].max()}")
log(f"  Total rows    : {len(df)}")
log()



log("=" * 60)
log("  MISSING VALUE ANALYSIS")
log("=" * 60)

missing     = df[FEATURES].isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
log(f"  {'Column':<20} {'Missing':>8} {'Percent':>10}")
log(f"  {'-'*40}")
for col in FEATURES:
    log(f"  {col:<20} {missing[col]:>8} {missing_pct[col]:>9}%")
log(f"\n  Total missing: {missing.sum()}")
log()

log("=" * 60)
log("   HANDLE MISSING VALUES")
log("=" * 60)

df[FEATURES] = df[FEATURES].ffill()
df[FEATURES] = df[FEATURES].bfill()
for col in FEATURES:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

log(f"  Method  : Forward Fill -> Backward Fill -> Mean Fill")
log(f"  Missing after treatment : {df[FEATURES].isnull().sum().sum()}")
log()



log("=" * 60)
log("   REMOVE DUPLICATES")
log("=" * 60)

before = len(df)
df = df.drop_duplicates(subset=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
log(f"  Duplicates removed : {before - len(df)}")
log(f"  Rows remaining     : {len(df)}")
log()



log("=" * 60)
log("   OUTLIER DETECTION & TREATMENT (IQR Method)")
log("=" * 60)

PHYSICAL_LIMITS = {
    'PM2.5':       (0, 500),
    'PM10':        (0, 600),
    'NO2':         (0, 400),
    'CO':          (0, 50),
    'temperature': (-5, 50),
    'humidity':    (0, 100),
    'wind_speed':  (0, 50),
}

log(f"  {'Column':<15} {'Outliers':>10}  Action")
log(f"  {'-'*50}")
for col, (lo, hi) in PHYSICAL_LIMITS.items():
    if col not in df.columns:
        continue
    Q1       = df[col].quantile(0.25)
    Q3       = df[col].quantile(0.75)
    IQR      = Q3 - Q1
    iqr_lo   = Q1 - 3 * IQR
    iqr_hi   = Q3 + 3 * IQR
    final_lo = max(lo, iqr_lo)
    final_hi = min(hi, iqr_hi)
    outliers = ((df[col] < final_lo) | (df[col] > final_hi)).sum()
    df[col]  = df[col].clip(lower=final_lo, upper=final_hi)
    log(f"  {col:<15} {outliers:>10}  Clipped to [{final_lo:.2f}, {final_hi:.2f}]")
log()



log("=" * 60)
log("  SAVE CLEANED DATA")
log("=" * 60)

def aqi_category(aqi):
    if aqi <= 50:   return 'Good'
    if aqi <= 100:  return 'Satisfactory'
    if aqi <= 200:  return 'Moderate'
    if aqi <= 300:  return 'Poor'
    if aqi <= 400:  return 'Very Poor'
    return 'Severe'

df['AQI_Category'] = df['AQI'].apply(aqi_category)
df.to_csv(OUTPUT_PATH, index=False)
log(f"  Saved : {OUTPUT_PATH}")
log(f"  Shape : {df.shape}")
log()



log("=" * 60)
log("   FEATURE ENGINEERING")
log("=" * 60)

df['hour']         = df['datetime'].dt.hour
df['day_of_week']  = df['datetime'].dt.dayofweek
df['month']        = df['datetime'].dt.month
df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
df['is_rush_hour'] = df['hour'].isin([7,8,9,17,18,19]).astype(int)
df['hour_sin']     = np.sin(2 * np.pi * df['hour']  / 24)
df['hour_cos']     = np.cos(2 * np.pi * df['hour']  / 24)
df['month_sin']    = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos']    = np.cos(2 * np.pi * df['month'] / 12)

for col in POLLUTANTS:
    df[f'{col}_roll3']  = df[col].rolling(window=3,  min_periods=1).mean().round(2)
    df[f'{col}_roll6']  = df[col].rolling(window=6,  min_periods=1).mean().round(2)
    df[f'{col}_roll24'] = df[col].rolling(window=24, min_periods=1).mean().round(2)
    df[f'{col}_lag1']   = df[col].shift(1)
    df[f'{col}_lag3']   = df[col].shift(3)
    df[f'{col}_lag24']  = df[col].shift(24)

df = df.dropna().reset_index(drop=True)
log(f"  Added : time features, cyclical encoding, rolling avg, lag features")
log(f"  Shape after feature engineering : {df.shape}")
log()



log("=" * 60)
log("  NORMALISATION (MinMaxScaler -> [0, 1])")
log("=" * 60)

scaler = MinMaxScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

with open('models/preprocessing_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

log(f"  Columns scaled : {FEATURES}")
log(f"  Scaler saved   : models/preprocessing_scaler.pkl")
log()


log("=" * 60)
log("  STEP 11 : DATA SPLIT (70% / 15% / 15%)")
log("=" * 60)

n         = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)
log(f"  Total      : {n:,} rows")
log(f"  Train      : {train_end:,} rows")
log(f"  Validation : {val_end - train_end:,} rows")
log(f"  Test       : {n - val_end:,} rows")
log()



log("=" * 60)
log("  FINAL STATISTICS")
log("=" * 60)

df_clean = pd.read_csv(OUTPUT_PATH)
units = {'PM2.5':'ug/m3','PM10':'ug/m3','NO2':'ug/m3','CO':'mg/m3'}
log(f"\n  {'Pollutant':<10} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}  Unit")
log(f"  {'-'*55}")
for col in POLLUTANTS:
    log(f"  {col:<10} {df_clean[col].min():>8.2f} {df_clean[col].max():>8.2f} "
        f"{df_clean[col].mean():>8.2f} {df_clean[col].std():>8.2f}  {units[col]}")

log()
log(f"  AQI Category Distribution:")
for cat, cnt in df_clean['AQI_Category'].value_counts().items():
    pct = cnt / len(df_clean) * 100
    log(f"    {cat:<15} {cnt:>6,}  ({pct:.1f}%)")
log()

df.to_csv(PROCESSED_PATH, index=False)

log("=" * 60)
log("  PREPROCESSING COMPLETE")
log("=" * 60)
log(f"  Clean data     : {OUTPUT_PATH}")
log(f"  Processed data : {PROCESSED_PATH}")
log(f"  Final shape    : {df.shape[0]:,} rows x {df.shape[1]} columns")
log()
log("  Next step: python train_model.py")

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
log(f"  Report saved   : {REPORT_PATH}")
