
import argparse, sys, time, csv
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

try:
    import serial
except Exception:
    serial = None

ALIASES = {
    "rpm": [r"\b(engine\s*rpm|rpm)\b"],
    "speed": [r"\b(speed|vehicle\s*speed|vss)\b"],
    "coolant_temp": [r"\b(coolant\s*temp(erature)?|engine\s*coolant\s*temperature|ect)\b"],
    "tps": [r"\b(throttle|absolute\s*throttle\s*position|tps)\b"],
    "maf": [r"\b(maf|mass\s*air\s*flow|air\s*flow\s*rate)\b"],
    "map": [r"\b(map|intake\s*manifold\s*absolute\s*pressure)\b"],
    "lub_oil_temp": [r"\b(lub(\.|ricating)?\s*oil\s*temp(erature)?)\b"],
    "fuel_pressure": [r"\b(fuel\s*pressure)\b"],
    "coolant_pressure": [r"\b(coolant\s*pressure)\b"],
}

BASE_FEATURES = ["rpm","speed","coolant_temp","tps","maf","map","lub_oil_temp","fuel_pressure","coolant_pressure"]

import re
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

def compute_window_features(buffers: dict):
    feats = {}
    for c, dq in buffers.items():
        if len(dq) == 0:
            continue
        arr = np.array(dq, dtype=float)
        feats[f"{c}_mean"] = float(np.mean(arr))
        feats[f"{c}_std"]  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return feats

def parse_line(line: str):
    parts = line.strip().split(",")
    if len(parts) < 4:
        return None
    try:
        vals = {
            "rpm": float(parts[0]),
            "speed": float(parts[1]),
            "coolant_temp": float(parts[2]),
            "tps": float(parts[3]),
        }
        if len(parts) > 4 and parts[4] != "":
            vals["maf"] = float(parts[4])
        if len(parts) > 5 and parts[5] != "":
            vals["map"] = float(parts[5])
        return vals
    except ValueError:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="COMx ou /dev/ttyACM0. Se omitido, usa --from_csv")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--model", default="models/model_obd_iforest.pkl")
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--log", default="data/raw/dados.csv")
    ap.add_argument("--from_csv", default=None, help="Simular leitura de arquivo CSV (com aliases)")
    args = ap.parse_args()

    payload = joblib.load(args.model)
    clf = payload["model"]; scaler = payload["scaler"]
    feat_cols = payload["feature_columns"]; win_model = payload["window"]
    window = args.window or win_model

    buffers = {c: deque(maxlen=window) for c in BASE_FEATURES}
    time_axis = deque(maxlen=600); rpm_series = deque(maxlen=600); temp_series = deque(maxlen=600)

    plt.ion()
    fig = plt.figure(); ax = fig.add_subplot(111)
    line_rpm, = ax.plot([], [], label="RPM/100")
    line_temp, = ax.plot([], [], label="Coolant °C")
    limit_temp, = ax.plot([], [], linestyle="--", label="Limite 100°C")
    ax.set_xlabel("t (s)"); ax.set_ylabel("valor"); ax.legend(loc="upper left")

    # log
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    new_file = not Path(args.log).exists()
    flog = open(args.log, "a", newline=""); writer = csv.writer(flog)
    if new_file:
        writer.writerow(["timestamp"] + BASE_FEATURES + ["status","score"])

    def handle_sample(vals, t0):
        import time as _time, numpy as _np
        ts = _time.time()
        for c in BASE_FEATURES:
            if c in vals and not _np.isnan(vals[c]):
                buffers[c].append(vals[c])

        feats = compute_window_features({c: buffers[c] for c in buffers if len(buffers[c])>0})
        x = np.array([feats.get(col, 0.0) for col in feat_cols], dtype=float).reshape(1,-1)
        xs = scaler.transform(x)
        pred = clf.predict(xs)[0]; score = clf.decision_function(xs)[0]
        status = "NORMAL" if pred == 1 else "ANOMALIA"

        # simples fallback
        if (vals.get("coolant_temp", 0) > 110):
            status = "ANOMALIA: Superaquecimento provável"
        if (vals.get("rpm",0) > 4500) and (vals.get("speed",0) < 10):
            status = "ANOMALIA: Incoerência RPM/Velocidade"
        if (vals.get("tps",0) > 95) and (vals.get("rpm",0) < 900):
            status = "ANOMALIA: Anomalia TPS/Marcha lenta"

        t_rel = ts - t0
        time_axis.append(t_rel); rpm_series.append(vals.get("rpm",0)/100.0); temp_series.append(vals.get("coolant_temp",0))
        line_rpm.set_data(list(time_axis), list(rpm_series))
        line_temp.set_data(list(time_axis), list(temp_series))
        limit_temp.set_data([time_axis[0] if time_axis else 0, time_axis[-1] if time_axis else 10], [100, 100])
        ax.relim(); ax.autoscale_view(); plt.pause(0.001)

        writer.writerow([int(ts)] + [vals.get(c,"") for c in BASE_FEATURES] + [status, float(score)]); flog.flush()
        print(f"{int(t_rel):4d}s | rpm={int(vals.get('rpm',0))} speed={int(vals.get('speed',0))} temp={int(vals.get('coolant_temp',0))}°C tps={int(vals.get('tps',0))} -> {status} (score={score:.3f})")

    t0 = time.time()
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        df = normalize_columns(df)
        # coerce
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        present = [c for c in BASE_FEATURES if c in df.columns]
        for _, row in df.iterrows():
            vals = {c: float(row[c]) for c in present if pd.notna(row[c])}
            handle_sample(vals, t0); time.sleep(0.1)
    else:
        if args.port is None:
            print("Erro: especifique --port ou --from_csv"); sys.exit(2)
        if serial is None:
            print("pyserial indisponível"); sys.exit(2)
        ser = serial.Serial(args.port, args.baud, timeout=1)
        try:
            while True:
                line = ser.readline().decode(errors="ignore")
                if not line: continue
                vals = parse_line(line)
                if vals is None: continue
                handle_sample(vals, t0)
        except KeyboardInterrupt:
            pass
        finally:
            ser.close(); flog.close()

if __name__ == "__main__":
    main()
