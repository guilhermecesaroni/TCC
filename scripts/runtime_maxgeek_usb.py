import argparse, time, csv, re, sys
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

BASE_ORDER = ["rpm","speed","coolant_temp","tps","maf","map"]

def parse_number(s: str):
    # Extrai primeiro float de uma linha (ex.: "Coolant: 82 C" -> 82.0)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None

def send_cmd(ser, cmd: str, add_newline: bool, quiet=False):
    # Alguns MaxGeek pedem SEM newline. Opção via flag.
    payload = (cmd + ("\r\n" if add_newline else "")).encode("ascii", errors="ignore")
    ser.write(payload)
    ser.flush()
    time.sleep(0.03)  # pequeno intervalo
    # ler buffer disponível
    resp = b""
    t0 = time.time()
    while (time.time() - t0) < 0.15:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            resp += chunk
        else:
            time.sleep(0.01)
    text = resp.decode(errors="ignore")
    if not quiet:
        print(f"[TX] {cmd}  [RX] {text.strip()}")
    return text

def probe_range(ser, start=0, end=200, add_newline=False):
    # Ajuda a descobrir quais RDS têm os sinais desejados no seu firmware
    for idx in range(start, end+1):
        cmd = f"AT+RDS{idx:03d}"
        text = send_cmd(ser, cmd, add_newline)
        time.sleep(0.05)

def parse_map_arg(map_arg: str):
    """
    --map "rpm:AT+RDS101,speed:AT+RDS102,coolant_temp:AT+RDS103,tps:AT+RDS104[,maf:AT+RDS105,map:AT+RDS106]"
    """
    mapping = {}
    if not map_arg:
        return mapping
    parts = [p.strip() for p in map_arg.split(",") if p.strip()]
    for p in parts:
        k, v = p.split(":", 1)
        mapping[k.strip()] = v.strip()
    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="COMx (Windows) ou /dev/ttyUSB0 (Linux)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--model", default="models/model_obd_iforest.pkl")
    ap.add_argument("--window", type=int, default=None, help="override do tamanho de janela (opcional)")
    ap.add_argument("--map", default="", help='ex.: "rpm:AT+RDS101,speed:AT+RDS102,coolant_temp:AT+RDS103,tps:AT+RDS104"')
    ap.add_argument("--interval", type=float, default=0.2, help="intervalo entre polls (s)")
    ap.add_argument("--log", default="data/raw/dados.csv")
    ap.add_argument("--newline", action="store_true", help="anexar \\r\\n ao comando (alguns firmwares aceitam)")
    ap.add_argument("--no-newline", action="store_true", help="(default) enviar sem terminador")
    ap.add_argument("--probe", action="store_true", help="varrer RDS000..RDS200 para descobrir índices")
    args = ap.parse_args()

    if serial is None:
        print("pyserial não instalado. pip install pyserial"); sys.exit(2)

    add_newline = True if args.newline else False
    if args.no_newline:
        add_newline = False

    # carrega modelo treinado pelos seus scripts
    payload = joblib.load(args.model)
    clf = payload["model"]; scaler = payload["scaler"]
    feat_cols = payload["feature_columns"]; win_model = payload["window"]
    window = args.window or win_model

    # buffers para features de janela
    buffers = {k: deque(maxlen=window) for k in BASE_ORDER}
    time_axis = deque(maxlen=600); rpm_series = deque(maxlen=600); temp_series = deque(maxlen=600)

    # gráfico
    plt.ion()
    fig = plt.figure(); ax = fig.add_subplot(111)
    line_rpm, = ax.plot([], [], label="RPM/100")
    line_temp, = ax.plot([], [], label="Coolant °C")
    limit_temp, = ax.plot([], [], linestyle="--", label="Limite 100°C")
    ax.set_xlabel("t (s)"); ax.set_ylabel("valor"); ax.legend(loc="upper left")

    # log CSV
    log_path = Path(args.log); log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    flog = open(log_path, "a", newline=""); writer = csv.writer(flog)
    if new_file:
        writer.writerow(["timestamp"] + BASE_ORDER + ["status","score"])

    # mapeamento de comandos
    cmd_map = parse_map_arg(args.map)
    if not cmd_map:
        print("ATENÇÃO: forneça --map com seus índices RDS para rpm/speed/coolant_temp/tps.")
        print("Exemplo: --map \"rpm:AT+RDS101,speed:AT+RDS102,coolant_temp:AT+RDS103,tps:AT+RDS104\"")
        # seguimos, mas não haverá leituras úteis

    print(f"Abrindo {args.port} @ {args.baud}  (newline={'ON' if add_newline else 'OFF'}) ...")
    ser = serial.Serial(args.port, args.baud, timeout=0.1)

    # teste básico de vida: peça VIN para ver se responde (se suportado)
    try:
        _ = send_cmd(ser, "AT+RVIN", add_newline, quiet=False)
    except Exception:
        pass

    if args.probe:
        print("== PROBING RDS000..RDS200 ==")
        probe_range(ser, 0, 200, add_newline)
        print("== FIM DO PROBE ==")
        ser.close(); sys.exit(0)

    t0 = time.time()
    try:
        while True:
            vals = {}
            for key in ["rpm","speed","coolant_temp","tps","maf","map"]:
                cmd = cmd_map.get(key)
                if not cmd:
                    continue
                text = send_cmd(ser, cmd, add_newline, quiet=True)
                val = parse_number(text)
                if val is not None:
                    vals[key] = float(val)

            # atualizaa buffers
            ts = time.time()
            for k in buffers:
                if k in vals and vals[k] is not None:
                    buffers[k].append(vals[k])

            # features de janela -> scaler -> IF
            feats = {}
            for c, dq in buffers.items():
                if len(dq) > 0:
                    arr = np.asarray(dq, dtype=float)
                    feats[f"{c}_mean"] = float(np.mean(arr))
                    feats[f"{c}_std"]  = float(np.std(arr, ddof=1)) if len(arr)>1 else 0.0

            x = np.array([feats.get(col, 0.0) for col in feat_cols], dtype=float).reshape(1,-1)
            xs = scaler.transform(x)
            pred = clf.predict(xs)[0]; score = clf.decision_function(xs)[0]
            status = "NORMAL" if pred == 1 else "ANOMALIA"

            # fallbacks
            ect = vals.get("coolant_temp", np.nan)
            rpm = vals.get("rpm", np.nan)
            spd = vals.get("speed", np.nan)
            tps = vals.get("tps", np.nan)

            if np.isfinite(ect) and ect > 110:
                status = "ANOMALIA: Superaquecimento provável"
            if np.isfinite(rpm) and np.isfinite(spd) and (rpm > 4500 and spd < 10):
                status = "ANOMALIA: Incoerência RPM/Velocidade"
            if np.isfinite(tps) and np.isfinite(rpm) and (tps > 95 and rpm < 900):
                status = "ANOMALIA: Anomalia TPS/Marcha lenta"

            # séries p/ gráfico
            t_rel = ts - t0
            time_axis.append(t_rel)
            rpm_series.append((rpm if np.isfinite(rpm) else 0.0)/100.0)
            temp_series.append((ect if np.isfinite(ect) else 0.0))
            line_rpm.set_data(list(time_axis), list(rpm_series))
            line_temp.set_data(list(time_axis), list(temp_series))
            limit_temp.set_data(
                [time_axis[0] if time_axis else 0, time_axis[-1] if time_axis else 10],
                [100, 100]
            )
            ax.relim(); ax.autoscale_view(); plt.pause(0.001)

            # log
            row = [int(ts)] + [vals.get(k,"") for k in BASE_ORDER] + [status, float(score)]
            writer.writerow(row); flog.flush()

            # console compacto
            print(f"{int(t_rel):4d}s | rpm={int(rpm) if np.isfinite(rpm) else '-'} "
                  f"speed={int(spd) if np.isfinite(spd) else '-'} "
                  f"temp={int(ect) if np.isfinite(ect) else '-'}°C "
                  f"tps={int(tps) if np.isfinite(tps) else '-'} -> {status} (score={score:.3f})")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        pass
    finally:
        try: ser.close()
        except: pass
        try: flog.close()
        except: pass

if __name__ == "__main__":
    main()
