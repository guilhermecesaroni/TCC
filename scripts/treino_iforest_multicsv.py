import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import dump as joblib_dump

# Diretórios base (ajustados ao seu projeto)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Colunas canônicas que o app entende
FEATURE_COLS = [
    "rpm",
    "speed_kmh",
    "coolant_C",
    "tps_pct",
    "fuel_rate",
    "engine_load",
    "iat_C",
    "baro_kpa",
]


def listar_csvs(data_dir):
    files = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".csv"):
            files.append(os.path.join(data_dir, fname))
    return sorted(files)


def padronizar_obd(df, fname):
    """
    Renomeia colunas do dataset OBD-II para os nomes canônicos.
    Se o arquivo não parecer ser do dataset OBD-II, retorna None.
    """
    cols = set(df.columns)

    # chave simples para saber se é um dos CSVs OBD-II
    if "Engine RPM [RPM]" not in cols:
        return None

    rename_map = {
        "Engine RPM [RPM]": "rpm",
        "Vehicle Speed Sensor [km/h]": "speed_kmh",
        "Engine Coolant Temperature [Ã\x82Â°C]": "coolant_C",
        "Absolute Throttle Position [%]": "tps_pct",
        "Air Flow Rate from Mass Flow Sensor [g/s]": "fuel_rate",
        "Accelerator Pedal Position D [%]": "engine_load",
        "Intake Air Temperature [Ã\x82Â°C]": "iat_C",
        "Intake Manifold Absolute Pressure [kPa]": "baro_kpa",
    }

    df2 = df.rename(columns=rename_map)

    missing = [c for c in FEATURE_COLS if c not in df2.columns]
    if missing:
        print(
            f"[AVISO] {os.path.basename(fname)} não possui todas as colunas canônicas {missing}. "
            f"Usaremos apenas as colunas disponíveis."
        )
    return df2


def carregar_todos_csvs(data_dir):
    files = listar_csvs(data_dir)
    if not files:
        raise RuntimeError(f"Nenhum CSV encontrado em {data_dir}")

    print(f"[INFO] Procurando CSVs em {data_dir}")
    for f in files[:10]:
        print("   -", os.path.basename(f))
    if len(files) > 10:
        print(f"   ... (+{len(files) - 10} arquivos)")

    dfs = []
    for path in files:
        fname = os.path.basename(path)
        try:
            print(f"[LENDO] {fname}")
            df = pd.read_csv(path, encoding="latin1", low_memory=False)
        except Exception as e:
            print(f"[ERRO] Falha ao ler {fname}: {e}")
            continue

        df_pad = padronizar_obd(df, fname)
        if df_pad is None:
            # não é do dataset OBD-II, ignoramos (ex.: dados.csv, engine_data.csv, etc.)
            print(f"[PULANDO] {fname} (não parece ser do dataset OBD-II esperado)")
            continue

        dfs.append(df_pad)

    if not dfs:
        raise RuntimeError("Nenhum arquivo válido (OBD-II) foi encontrado para treino.")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Dataset combinado: {df_all.shape[0]} linhas, {df_all.shape[1]} colunas.")
    return df_all


def limpar_e_preparar(df):
    # Seleciona apenas colunas de interesse que existem de fato
    cols_existentes = [c for c in FEATURE_COLS if c in df.columns]
    if not cols_existentes:
        raise RuntimeError("Nenhuma das colunas de FEATURES está presente no DataFrame.")

    print(f"[INFO] Usando as seguintes features para treino: {cols_existentes}")

    X = df[cols_existentes].copy()

    # Converte para float, clampa ranges e preenche NaN com mediana
    for col in cols_existentes:
        X[col] = pd.to_numeric(X[col], errors="coerce")

        if col == "rpm":
            X[col] = X[col].clip(lower=0, upper=8000)
        elif col == "speed_kmh":
            X[col] = X[col].clip(lower=0, upper=250)
        elif col in ("coolant_C", "iat_C"):
            X[col] = X[col].clip(lower=-40, upper=150)
        elif col in ("tps_pct", "engine_load"):
            X[col] = X[col].clip(lower=0, upper=100)
        elif col == "fuel_rate":
            X[col] = X[col].clip(lower=0, upper=300)
        elif col == "baro_kpa":
            X[col] = X[col].clip(lower=50, upper=200)

        med = X[col].median()
        X[col] = X[col].fillna(med)

    # Remove linhas que, por algum motivo, ainda ficaram com NaN
    X = X.dropna()
    print(f"[INFO] Após limpeza, temos {X.shape[0]} linhas para treino.")

    return X, cols_existentes


def treinar_e_salvar(X, feature_names):
    print("[INFO] Normalizando com StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    print("[INFO] Treinando IsolationForest (detecção de anomalias)...")
    iforest = IsolationForest(
        n_estimators=200,
        contamination=0.02,  # ~2% de anomalias esperadas
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(X_scaled)

    y_pred = iforest.predict(X_scaled)
    frac_anom = (y_pred == -1).mean()
    print(f"[INFO] Fração de pontos marcados como anomalia no treino: {frac_anom:.4f}")

    out = {
        "model": iforest,
        "scaler": scaler,
        "feature_names": feature_names,  # <== MUITO IMPORTANTE
    }

    model_path = os.path.join(MODELS_DIR, "model_obd_iforest.pkl")
    joblib_dump(out, model_path)
    print(f"[OK] Modelo salvo em: {model_path}")


def main():
    print("=" * 70)
    print("TREINO IsolationForest a partir do dataset OBD-II (vários CSVs)")
    print("=" * 70)
    print(f"DATA_DIR   = {DATA_DIR}")
    print(f"MODELS_DIR = {MODELS_DIR}")
    print()

    df_all = carregar_todos_csvs(DATA_DIR)
    X, feats = limpar_e_preparar(df_all)
    treinar_e_salvar(X, feats)


if __name__ == "__main__":
    main()
