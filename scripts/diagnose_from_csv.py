import argparse, re
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

ALIASES = {
    "rpm": [r"\b(engine\s*rpm|rpm)\b"],
    "coolant_temp": [r"\b(coolant\s*temp(erature)?|engine\s*coolant\s*temperature)\b"],
    "lub_oil_temp": [r"\b(lub(\.|ricating)?\s*oil\s*temp(erature)?)\b"],
    "fuel_pressure": [r"\b(fuel\s*pressure)\b"],
    "coolant_pressure": [r"\b(coolant\s*pressure)\b"],
    # extras comuns
    "speed": [r"\b(speed|vehicle\s*speed|vss)\b"],
    "tps": [r"\b(throttle|absolute\s*throttle\s*position|tps)\b"],
    "maf": [r"\b(maf|mass\s*air\s*flow|air\s*flow\s*rate)\b"],
    "map": [r"\b(map|intake\s*manifold\s*absolute\s*pressure)\b"],
    "label": [r"\b(engine\s*condition|label|target|y)\b"],
}

BASE_COL_ORDER = ["rpm","speed","coolant_temp","tps","maf","map","lub_oil_temp","fuel_pressure","coolant_pressure"]

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

def build_features(df: pd.DataFrame, window: int, feat_cols):
    bases = sorted({c.split("_")[0] for c in feat_cols if c.endswith("_mean")})
    keep = [c for c in bases if c in df.columns]
    roll = df[keep].rolling(window=window, min_periods=window)
    feats = {}
    for c in keep:
        feats[f"{c}_mean"] = roll[c].mean()
        feats[f"{c}_std"]  = roll[c].std()
    X = pd.DataFrame(feats, index=df.index).dropna()
    # alinhar com colunas do modelo
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0.0
    return X[feat_cols].reset_index(drop=True)

def fallback_rules(row):
    notes = []
    if pd.notna(row.get("coolant_temp")) and row["coolant_temp"] > 110:
        notes.append("Superaquecimento provável")
    if pd.notna(row.get("rpm")) and pd.notna(row.get("speed")) and (row["rpm"] > 4500 and row["speed"] < 10):
        notes.append("Incoerência RPM/Velocidade")
    if pd.notna(row.get("tps")) and pd.notna(row.get("rpm")) and (row["tps"] > 95 and row["rpm"] < 900):
        notes.append("Anomalia TPS/Marcha lenta")
    return "; ".join(notes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="ex.: data/raw/engine_data.csv")
    ap.add_argument("--model", default="models/model_obd_iforest.pkl")
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    payload = joblib.load(args.model)
    clf = payload["model"]; scaler = payload["scaler"]
    feat_cols = payload["feature_columns"]; window = args.window or payload["window"]

    df = pd.read_csv(args.csv)
    df = normalize_columns(df)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    X = build_features(df.copy(), window, feat_cols)
    Xs = scaler.transform(X.values)
    pred = clf.predict(Xs)
    score = clf.decision_function(Xs)

    valid_idx = df.index[window-1:]

    # garantir colunas no relatório
    rep_cols = [c for c in BASE_COL_ORDER if c in df.columns]
    if "label" in df.columns:
        rep_cols = rep_cols + ["label"]

    out = df.loc[valid_idx, rep_cols].copy()
    out["score"] = score
    out["pred"] = pred
    out["status"] = out["pred"].map({1:"NORMAL",-1:"ANOMALIA"})
    out["rules"] = [fallback_rules(r) for _, r in df.loc[valid_idx, :].iterrows()]
    out["status_final"] = np.where(out["rules"].astype(str)!="", "ANOMALIA (regras)", out["status"])

    out_path = args.out or (Path(args.csv).with_suffix(".diagnostico.csv"))
    out.to_csv(out_path, index=False)
    print(f"Diagnóstico salvo em: {out_path}")

    if args.plot:
        import matplotlib.pyplot as plt
        t = np.arange(len(out))
        if "rpm" in out.columns:
            plt.plot(t, out["rpm"]/100.0, label="RPM/100")
        if "coolant_temp" in out.columns:
            plt.plot(t, out["coolant_temp"], label="Coolant °C")
        plt.axhline(100, linestyle="--", label="Limite 100°C")
        idx_anom = np.where(out["status_final"].str.contains("ANOMALIA"))[0]
        if len(idx_anom)>0 and "rpm" in out.columns:
            plt.scatter(idx_anom, (out["rpm"]/100.0).iloc[idx_anom], marker="x", label="Anomalias")
        plt.legend(loc="upper left"); plt.xlabel("amostra"); plt.ylabel("valor"); plt.show()

if __name__ == "__main__":
    main()
