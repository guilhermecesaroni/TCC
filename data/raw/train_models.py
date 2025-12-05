# scripts/train_models.py
# Treino de modelos: IsolationForest (anomalia) + RandomForest (diagnóstico)
import os, json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from obd_ai.features import load_dir_raw_as_df, make_time_windows, feature_columns

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # raiz do projeto
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# -------- parâmetros --------
WINDOW_SEC = 3.0
STEP_SEC = 1.0
RESAMPLE_HZ = 10.0
IFOREST_CONTAM = 0.02
RANDOM_STATE = 42
N_EST = 300

def main():
    print(f"[INFO] Lendo dados de: {RAW_DIR}")
    df = load_dir_raw_as_df(RAW_DIR)
    if df.empty:
        raise SystemExit("[ERRO] Nenhum CSV válido encontrado em data/raw/")

    print(f"[INFO] Total de linhas brutas: {len(df)}")
    feats = make_time_windows(df, window_sec=WINDOW_SEC, step_sec=STEP_SEC, resample_hz=RESAMPLE_HZ)
    Xcols = feature_columns()
    for c in Xcols:
        if c not in feats.columns:
            feats[c] = np.nan
    feats = feats.dropna(subset=Xcols, how="all").fillna(method="ffill").fillna(method="bfill")

    print(f"[INFO] Total de janelas: {len(feats)}")
    # ---------------- IsolationForest (Unsupervised) ----------------
    X_if = feats[Xcols].values.astype(float)
    scaler_if = StandardScaler()
    X_if_sc = scaler_if.fit_transform(X_if)

    iforest = IsolationForest(
        n_estimators=N_EST, contamination=IFOREST_CONTAM,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    iforest.fit(X_if_sc)
    out_iforest = {
        "model": iforest,
        "scaler": scaler_if,
        "feature_names": Xcols,
        "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC,
        "resample_hz": RESAMPLE_HZ,
        "type": "iforest_window"
    }
    if_path = os.path.join(MODELS_DIR, "model_engine_iforest.pkl")
    dump(out_iforest, if_path)
    print(f"[OK] IsolationForest salvo em: {if_path}")

    # ---------------- Classificador de diagnóstico (Supervised) ----------------
    has_label = "label" in feats.columns and feats["label"].notna().any()
    if not has_label:
        print("[WARN] Nenhum rótulo encontrado nas janelas. Pulando classificador.")
        return

    # normaliza rótulos para string simples
    feats["label"] = feats["label"].astype(str).str.strip().str.lower()

    X = feats[Xcols].values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    y = feats["label"].values

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    clf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=RANDOM_STATE
    )
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    print("\n[Relatório de classificação]\n", classification_report(yte, ypred))

    out_diag = {
        "model": clf,
        "scaler": scaler,
        "feature_names": Xcols,
        "classes_": clf.classes_.tolist(),
        "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC,
        "resample_hz": RESAMPLE_HZ,
        "type": "diag_window"
    }
    diag_path = os.path.join(MODELS_DIR, "model_engine_diag.pkl")
    dump(out_diag, diag_path)
    print(f"[OK] Classificador salvo em: {diag_path}")

if __name__ == "__main__":
    main()
