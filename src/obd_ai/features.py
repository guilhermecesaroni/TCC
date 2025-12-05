from __future__ import annotations
import os, re
import numpy as np
import pandas as pd

# padrão: timestamp,rpm,speed_kmh,coolant_C,tps_pct,fuel_pct,fuel_rate,engine_load,iat_C,baro_kpa,label
COLUMN_ALIASES = {
    "timestamp": ["timestamp","time","date","datetime","Time","DateTime","logtime"],
    "rpm": ["rpm","RPM","EngineRPM","Engine Speed","engine_rpm","engine_speed","obd.rpm"],
    "speed_kmh": ["speed","Speed","VehicleSpeed","vehicle_speed","spd","obd.speed","speed_kmh","kmh","km/h"],
    "coolant_C": ["coolant","Coolant","Coolant Temperature","Engine Coolant Temperature","obd.coolant","coolant_C","temp","temp_c","ECT"],
    "tps_pct": ["tps","TPS","Throttle Position","Throttle","obd.tps","tps_pct","throttle_pct"],
    "fuel_pct": ["fuel","Fuel","Fuel Level","FuelLevel","obd.fuel","fuel_pct","fuel_level"],
    "fuel_rate": ["fuel_rate","Fuel Rate","FuelRate","maf_lh","lph","L/h","consumption"],
    "engine_load": ["engine_load","Load","Calculated Engine Load","Load_pct","obd.load"],
    "iat_C": ["iat","IAT","Intake Air Temperature","obd.iat","iat_C","intake_temp"],
    "baro_kpa": ["baro","BARO","Barometric Pressure","MAP","map_kpa","baro_kpa","Intake Manifold Pressure"],
    "label": ["label","status","falha","class","target"]
}

NUMERIC_COLS = ["rpm","speed_kmh","coolant_C","tps_pct","fuel_pct",
                "fuel_rate","engine_load","iat_C","baro_kpa"]

# ---------- helpers ----------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c:str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)
    new = {}
    col_lower = {c:c.lower() for c in df.columns}
    for standard, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            for c in df.columns:
                if col_lower[c] == a.lower():
                    new[standard] = c
                    break
            if standard in new:
                break
    # fallback: tenta padrões obvios
    if "timestamp" not in new and "time" in col_lower.values():
        for c in df.columns:
            if col_lower[c] == "time":
                new["timestamp"] = c; break
    df = df.rename(columns={v:k for k,v in new.items()})
    return df

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _uniform_units(df: pd.DataFrame) -> pd.DataFrame:
    # Se speed parece mph (muito > 260), converte pra km/h
    if "speed_kmh" in df.columns:
        sp = df["speed_kmh"].dropna()
        if len(sp) and (sp.quantile(0.99) > 260):
            df["speed_kmh"] = df["speed_kmh"] * 1.60934
    return df

def _basic_clip(df: pd.DataFrame) -> pd.DataFrame:
    clip_ranges = {
        "rpm": (0, 8000),
        "speed_kmh": (0, 300),
        "coolant_C": (-40, 140),
        "tps_pct": (0, 100),
        "fuel_pct": (0, 100),
        "engine_load": (0, 100),
        "fuel_rate": (0, 200),     # L/h
        "iat_C": (-40, 120),
        "baro_kpa": (0, 200),
    }
    for c,(lo,hi) in clip_ranges.items():
        if c in df.columns:
            df[c] = df[c].clip(lower=lo, upper=hi)
    return df

def load_and_standardize_csv(path: str) -> pd.DataFrame:
    # tenta ; ou ,
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=";")
    df = _normalize_columns(df)
    # timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.NaT
    # numéricos
    df = _coerce_numeric(df)
    df = _uniform_units(df)
    df = _basic_clip(df)
    # ordena por tempo se existir
    if df["timestamp"].notna().any():
        df = df.sort_values("timestamp")
    return df

def load_dir_raw_as_df(raw_dir: str) -> pd.DataFrame:
    frames = []
    for fname in os.listdir(raw_dir):
        if not fname.lower().endswith((".csv",".txt")):
            continue
        try:
            fpath = os.path.join(raw_dir, fname)
            df = load_and_standardize_csv(fpath)
            df["__source__"] = fname
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Falha lendo {fname}: {e}")
    if not frames:
        return pd.DataFrame(columns=["timestamp"]+NUMERIC_COLS+["label","__source__"])
    all_df = pd.concat(frames, ignore_index=True, sort=False)
    # remove linhas totalmente vazias
    if NUMERIC_COLS:
        all_df = all_df.dropna(subset=[c for c in NUMERIC_COLS if c in all_df.columns], how="all")
    return all_df

# ---------- janelamento e features ----------
def make_time_windows(df: pd.DataFrame, window_sec: float = 3.0, step_sec: float = 1.0, resample_hz: float = 10.0) -> pd.DataFrame:
    """Gera features agregadas por janela de tempo.
       Se timestamp faltar, assume amostragem uniforme e usa índice."""
    df = df.copy()
    # garante colunas necessárias
    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = np.nan

    if df["timestamp"].notna().any():
        # resample por tempo
        df = df.set_index("timestamp").sort_index()
        # amostragem uniforme
        rule = f"{int(1000/resample_hz)}L"  # ms
        df = df.resample(rule).mean().interpolate(limit=5)
        idx = df.index
        win = int(window_sec*resample_hz)
        step = int(step_sec*resample_hz)
    else:
        # sem timestamp: usa índice
        df = df.reset_index(drop=True)
        idx = df.index
        win = int(window_sec*resample_hz)
        step = int(step_sec*resample_hz)

    feats = []
    start = 0
    n = len(df)
    while start + win <= n:
        seg = df.iloc[start:start+win]
        rowf = {}
        # agregados por coluna
        for c in NUMERIC_COLS:
            s = seg[c]
            rowf[f"{c}_mean"] = s.mean()
            rowf[f"{c}_std"] = s.std(ddof=0)
            rowf[f"{c}_min"] = s.min()
            rowf[f"{c}_max"] = s.max()
            rowf[f"{c}_med"] = s.median()
            rowf[f"{c}_delta"] = s.iloc[-1] - s.iloc[0]
        # label majoritário se houver
        if "label" in seg.columns:
            lab = seg["label"].dropna()
            rowf["label"] = lab.mode().iloc[0] if len(lab) else np.nan
        rowf["t_start"] = idx[start]
        rowf["t_end"] = idx[start+win-1]
        feats.append(rowf)
        start += step

    feat_df = pd.DataFrame(feats)
    return feat_df

def feature_columns():
    cols = []
    for c in NUMERIC_COLS:
        cols += [f"{c}_{agg}" for agg in ("mean","std","min","max","med","delta")]
    return cols
