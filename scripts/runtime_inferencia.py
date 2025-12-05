import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import serial  # pyserial
except Exception:
    serial = None

BASE_FEATURES = ["rpm","speed","coolant_temp","tps","maf","map"]

def parse_line(line: str):
    # Espera: rpm,speed,coolant_temp,tps[,maf,map]
    parts = line.strip().split(",")
    vals = {}
    if len(parts) < 4:
        return None
    try:
        vals["rpm"] = float(parts[0])
        vals["speed"] = float(parts[1])
        vals["coolant_temp"] = float(parts[2])
        vals["tps"] = float(parts[3])
        if len(parts) > 4:
            vals["maf"] = float(parts[4]) if parts[4] != "" else np.nan
        if len(parts) > 5:
            vals["map"] = float(parts[5]) if parts[5] != "" else np.nan
        return vals
    except ValueError:
        return None

def compute_window_features(buffers: dict):
    feats = {}
    for c, dq in buffers.items():
        if len(dq) == 0:
            continue
        arr = np.array(dq, dtype=float)
        feats[f"{c}_mean"] = float(np.mean(arr))
        feats[f"{c}_std"]  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3", help="Porta serial (ex: COM3 ou /dev/ttyACM0)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--model", default="model_obd_iforest.pkl")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--log", default="dados.csv", help="arquivo de log (append)")
    ap.add_argument("--from_csv", default=None, help="ler dados de um CSV ao invés de Serial")
    args = ap.parse_args()

    payload = joblib.load(args.model)
    clf = payload["model"]
    scaler = payload["scaler"]
    feat_cols = payload["feature_columns"]
    window = payload.get("window", args.window)

    buffers = {c: deque(maxlen=window) for c in BASE_FEATURES}
    time_axis = deque(maxlen=600)  # ~60s a 10 Hz
    rpm_series = deque(maxlen=600)
    temp_series = deque(maxlen=600)

    # gráfico ao vivo
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line_rpm, = ax.plot([], [], label="RPM/100")
    line_temp, = ax.plot([], [], label="Coolant °C")
    limit_temp, = ax.plot([], [], linestyle="--", label="Limite 100°C")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("valor")
    ax.legend(loc="upper left")

    # log
    log_path = Path(args.log)
    new_file = not log_path.exists()
    flog = open(log_path, "a", newline="")
    writer = csv.writer(flog)
    if new_file:
        writer.writerow(["timestamp","rpm","speed","coolant_temp","tps","maf","map","status","score"])

    def handle_sample(vals, t0):
        ts = time.time()
        for c in BASE_FEATURES:
            if c in vals and not np.isnan(vals[c]):
                buffers[c].append(vals[c])

        feats = compute_window_features({c: buffers[c] for c in buffers if len(buffers[c])>0})


        x = np.array([feats.get(col, 0.0) for col in feat_cols], dtype=float).reshape(1,-1)
        xs = scaler.transform(x)
        pred = clf.predict(xs)[0]
        score = clf.decision_function(xs)[0]

        # fallback do isolation
        status = "NORMAL" if pred == 1 else "ANOMALIA"
        if np.isfinite(vals.get("coolant_temp", np.nan)) and vals.get("coolant_temp", 0) > 110:
            status = "ANOMALIA: Superaquecimento provavel"
        if (vals.get("rpm",0) > 4500) and (vals.get("speed",0) < 10):
            status = "ANOMALIA: Incoerencia RPM/Velocidade"
        if (vals.get("tps",0) > 95) and (vals.get("rpm",0) < 900):
            status = "ANOMALIA: Anomalia TPS/Marcha lenta"

      
        t_rel = ts - t0
        time_axis.append(t_rel)
        rpm_series.append(vals.get("rpm", 0)/100.0)
        temp_series.append(vals.get("coolant_temp", 0))

        # aqui acontece a atualizacao dos graficos
        line_rpm.set_data(list(time_axis), list(rpm_series))
        line_temp.set_data(list(time_axis), list(temp_series))
        limit_temp.set_data([time_axis[0] if time_axis else 0, time_axis[-1] if time_axis else 10], [100, 100])
        ax.relim(); ax.autoscale_view()
        plt.pause(0.001)

        # aqui acontece o log
        writer.writerow([int(ts),
                         vals.get("rpm",""),
                         vals.get("speed",""),
                         vals.get("coolant_temp",""),
                         vals.get("tps",""),
                         vals.get("maf",""),
                         vals.get("map",""),
                         status,
                         float(score)])
        flog.flush()

        # console
        print(f"{int(t_rel):4d}s | "
              f"rpm={int(vals.get('rpm',0))} "
              f"speed={int(vals.get('speed',0))} "
              f"temp={int(vals.get('coolant_temp',0))}"
              f"°C tps={int(vals.get('tps',0))} -> {status} (score={score:.3f})")

    # fonte de dados
    t0 = time.time()
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        # garantir colunas
        cols = [c for c in BASE_FEATURES if c in df.columns]
        for idx, row in df.iterrows():
            vals = {c: float(row[c]) for c in cols}
            handle_sample(vals, t0)
            time.sleep(0.1)
    else:
        if serial is None:
            print("pyserial não disponível. Use --from_csv para simular.")
            sys.exit(1)
        ser = serial.Serial(args.port, args.baud, timeout=1)
        try:
            while True:
                line = ser.readline().decode(errors="ignore")
                if not line:
                    continue
                vals = parse_line(line)
                if vals is None:
                    continue
                handle_sample(vals, t0)
        except KeyboardInterrupt:
            pass
        finally:
            ser.close()
            flog.close()

if __name__ == "__main__":
    main()
