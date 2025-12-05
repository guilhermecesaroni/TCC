import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

BASE_FEATURES = ["rpm","speed","coolant_temp","tps","maf","map"]

def build_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    cols = [c for c in BASE_FEATURES if c in df.columns]
    if not cols:
        raise ValueError("Nenhuma coluna esperada encontrada no CSV.")
    roll = df[cols].rolling(window=window, min_periods=window)
    feats = {}
    for c in cols:
        feats[f"{c}_mean"] = roll[c].mean()
        feats[f"{c}_std"]  = roll[c].std()
    X = pd.DataFrame(feats, index=df.index)
    X = X.dropna().reset_index(drop=True)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dados.csv", help="Caminho do CSV de treino")
    ap.add_argument("--window", type=int, default=30, help="Tamanho da janela (amostras)")
    ap.add_argument("--contamination", type=float, default=0.02, help="Taxa de contaminação para IF")
    ap.add_argument("--n_estimators", type=int, default=200, help="N árvores do IsolationForest")
    ap.add_argument("--out", default="model_obd_iforest.pkl", help="Arquivo de saída do modelo")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    # mantém só colunas que interessam, se existirem
    keep = [c for c in BASE_FEATURES if c in df.columns]
    df = df[keep]

    X = build_features(df, window=args.window)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    clf = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(Xs)

    payload = {
        "model": clf,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "base_features_present": keep,
        "window": args.window,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.out)
    print(f"OK: modelo salvo em {args.out}")
    print(f"Cols base presentes: {keep}")
    print(f"Feature columns: {list(X.columns)}")

if __name__ == "__main__":
    main()
