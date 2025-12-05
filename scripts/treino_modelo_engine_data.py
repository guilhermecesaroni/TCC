import argparse, re, os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

ALIASES = {
    "rpm": [r"\b(engine\s*rpm|rpm)\b"],
    "coolant_temp": [r"\b(coolant\s*temp(erature)?|engine\s*coolant\s*temperature)\b"],
    "lub_oil_temp": [r"\b(lub(\.|ricating)?\s*oil\s*temp(erature)?)\b"],
    "fuel_pressure": [r"\b(fuel\s*pressure)\b"],
    "coolant_pressure": [r"\b(coolant\s*pressure)\b"],
    # opcionais
    "speed": [r"\b(speed|vehicle\s*speed|vss)\b"],
    "tps": [r"\b(throttle|absolute\s*throttle\s*position|tps)\b"],
    "maf": [r"\b(maf|mass\s*air\s*flow|air\s*flow\s*rate)\b"],
    "map": [r"\b(map|intake\s*manifold\s*absolute\s*pressure)\b"],
}

BASES = ["rpm","coolant_temp","lub_oil_temp","fuel_pressure","coolant_pressure","speed","tps","maf","map"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_names = {}
    for c in df.columns:
        c_norm = re.sub(r"\s*\[.*?\]", "", c.strip().lower())
        c_norm = re.sub(r"[^a-z0-9_ ]", " ", c_norm)
        for tgt, pats in ALIASES.items():
            if any(re.search(p, c_norm, re.I) for p in pats):
                new_names[c] = tgt
                break
    return df.rename(columns=new_names)

def build_features(df: pd.DataFrame, window: int):
    cols = [c for c in BASES if c in df.columns]
    if len(cols) < 2:
        raise ValueError(f"Poucas colunas úteis encontradas ({cols}). Mínimo=2.")
    roll = df[cols].rolling(window=window, min_periods=window)
    feats = {}
    for c in cols:
        feats[f"{c}_mean"] = roll[c].mean()
        feats[f"{c}_std"]  = roll[c].std()
    X = pd.DataFrame(feats, index=df.index).dropna().reset_index(drop=True)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="ex.: data/raw/engine_data.csv")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--contamination", type=float, default=0.02)
    ap.add_argument("--n_estimators", type=int, default=256)
    ap.add_argument("--out", default="models/model_obd_iforest.pkl")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = normalize_columns(df)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")

    X = build_features(df, window=args.window)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    clf = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=42,
        n_jobs=-1,
    ).fit(Xs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": clf, "scaler": scaler, "feature_columns": list(X.columns), "window": args.window}
    joblib.dump(payload, args.out)
    print(f"OK: {args.out}")
    print(f"Features: {len(X.columns)} -> {list(X.columns)[:10]}{' ...' if len(X.columns)>10 else ''}")

if __name__ == "__main__":
    main()
