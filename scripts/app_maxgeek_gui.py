import threading, time, re, sys, queue, csv, os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None

try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    StandardScaler = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None
    list_ports = None

DEFAULT_BAUD = 115200
DEFAULT_CRLF = True  # MaxGeek costuma exigir CRLF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_PATHS = [
    os.path.join(BASE_DIR, "model_iforest.pkl"),
    os.path.join(BASE_DIR, "model_engine_iforest.pkl"),
    os.path.join(BASE_DIR, "model_engine_data_iforest.pkl"),

    os.path.join(BASE_DIR, "..", "models", "model_iforest.pkl"),
    os.path.join(BASE_DIR, "..", "models", "model_engine_iforest.pkl"),
    os.path.join(BASE_DIR, "..", "models", "model_engine_data_iforest.pkl"),
]


DEFAULT_DIAG_MODEL_PATHS = [
    "model_engine_diag.pkl",
    os.path.join(os.getcwd(), "model_engine_diag.pkl"),
    os.path.join(os.path.dirname(os.getcwd()), "models", "model_engine_diag.pkl"),
]


RDS_IDX = {
    "rpm":   93,
    "speed": 154,
    "cool":  125,
    "fuel":  178,
}

EXTRAS = {
    "fuel_rate":   170,   # L/h
    "tps_pct":     173,   # %
    "engine_load": 174,   # %
    "iat_C":       175,   # °C
    "baro_kpa":    187,   # kPa
}

CSV_HEADER = [
    "timestamp","rpm","speed_kmh","coolant_C","tps_pct","fuel_pct",
    "fuel_rate","engine_load","iat_C","baro_kpa","status"
]

#Utils
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")

def clamp(v, lo, hi):
    if v is None:
        return None
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

def parse_val(txt):
    if not txt:
        return None
    part = txt.split(":",1)[-1] if ":" in txt else txt
    for u in ("rpm","RPM","km/h","KM/H","kph","KPH","°C"," C","%","kPa",
              "bar","psi","V","L/h","l/h","L","l"):
        part = part.replace(u,"")
    part = part.replace(",", ".").strip()
    m = _NUM.findall(part)
    try:
        return float(m[-1]) if m else None
    except:
        return None

#Serial MaxGeek
class MaxGeekSerial:
    def __init__(self, port, baud=DEFAULT_BAUD, crlf=DEFAULT_CRLF,
                 hard_timeout=0.25, quiet_ms=12, retries=1):
        self.port = port
        self.baud = baud
        self.crlf = crlf
        self.hard_timeout = float(hard_timeout)
        self.quiet_ms = int(quiet_ms)
        self.retries = int(retries)
        self.ser = None

    def open(self):
        if serial is None:
            raise RuntimeError("pyserial não instalado. pip install pyserial")
        self.close()
        # timeout=0.0 => não-bloqueante (baixa latência)
        self.ser = serial.Serial(
            port=self.port, baudrate=self.baud,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE, timeout=0.0, write_timeout=None
        )

    def close(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except:
            pass
        self.ser = None

    def _read_until_quiet(self):
        buf = bytearray()
        t_start = time.monotonic()
        last_byte_t = t_start
        quiet_s = self.quiet_ms / 1000.0
        while True:
            n = self.ser.in_waiting if self.ser else 0
            if n:
                chunk = self.ser.read(n)
                if chunk:
                    buf.extend(chunk)
                    last_byte_t = time.monotonic()
                    continue
            now = time.monotonic()
            if (now - last_byte_t) >= quiet_s:
                break
            if (now - t_start) >= self.hard_timeout:
                break
            time.sleep(0)  # cede CPU
        return bytes(buf).decode(errors="ignore").strip()

    def _send_cmd_instant(self, cmd):
        if not (self.ser and self.ser.is_open):
            return ""
        suffix = "\r\n" if self.crlf else "\n"
        payload = (cmd + suffix).encode("ascii", errors="ignore")
        try:
            self.ser.reset_input_buffer()
        except:
            pass
        self.ser.write(payload)
        self.ser.flush()
        # aguarda chegar algum byte
        t0 = time.monotonic()
        while (self.ser.in_waiting == 0) and (time.monotonic() - t0 < self.hard_timeout):
            time.sleep(0)
        return self._read_until_quiet()

    def read_rds(self, idx):
        cmd = f"AT+RDS{idx:03d}"
        val = None
        for _ in range(max(1, self.retries)):
            v = parse_val(self._send_cmd_instant(cmd))
            if v is not None:
                val = v
                break
        return val

#Thread de leitura
class ReaderThread(threading.Thread):

    def __init__(self, mg: MaxGeekSerial, interval=0.12,
                 stop_event=None, queue_out=None):
        super().__init__(daemon=True)
        self.mg = mg
        self.interval = max(0.05, float(interval))   # ~8–20 FPS alvo
        self.stop_event = stop_event or threading.Event()
        self.q = queue_out or queue.Queue()

    def run(self):
        try:
            self.mg.open()
        except Exception as e:
            self.q.put({"error": f"Erro abrindo {self.mg.port}: {e}"})
            return

        while not self.stop_event.is_set():
            t_start = time.time()
            row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

            # básicos
            row["rpm"]        = self.mg.read_rds(RDS_IDX["rpm"])
            row["speed_kmh"]  = self.mg.read_rds(RDS_IDX["speed"])
            row["coolant_C"]  = self.mg.read_rds(RDS_IDX["cool"])
            row["fuel_pct"]   = self.mg.read_rds(RDS_IDX["fuel"])

            # extras
            row["fuel_rate"]   = self.mg.read_rds(EXTRAS["fuel_rate"])
            row["tps_pct"]     = self.mg.read_rds(EXTRAS["tps_pct"])
            row["engine_load"] = self.mg.read_rds(EXTRAS["engine_load"])
            row["iat_C"]       = self.mg.read_rds(EXTRAS["iat_C"])
            row["baro_kpa"]    = self.mg.read_rds(EXTRAS["baro_kpa"])

            self.q.put(row)

            dt = time.time() - t_start
            wait = self.interval - dt
            if wait > 0:
                time.sleep(wait)

        try:
            self.mg.close()
        except:
            pass

# IA
class SimpleIARuntime:
    def __init__(self, status_cb=None):
        self.model = None
        self.scaler = None
        self.feature_names = None  # <-- lista de colunas usadas no treino
        self.status_cb = status_cb

    def _notify(self, msg):
        if self.status_cb:
            self.status_cb(msg)

    def auto_load_if_exists(self):
        if joblib_load is None or np is None:
            self._notify("IA: joblib/scikit-learn/numpy não instalados (usando fallback).")
            return False
        for p in DEFAULT_MODEL_PATHS:
            try:
                if p and os.path.isfile(p):
                    self.load_model(p)
                    self._notify(f"IA: modelo carregado ({os.path.basename(p)}).")
                    return True
            except Exception as e:
                self._notify(f"IA: falha ao carregar {p}: {e}")
        self._notify("IA: nenhum modelo .pkl encontrado (usando fallback).")
        return False

    def load_model(self, path):
        if joblib_load is None or np is None:
            raise RuntimeError("Instale numpy, joblib e scikit-learn para carregar o modelo.")
        obj = joblib_load(path)

        # aceita tanto dict (novo formato) quanto modelo simples (antigo)
        if isinstance(obj, dict):
            self.model = obj.get("model")
            self.scaler = obj.get("scaler")
            self.feature_names = obj.get("feature_names", None)
        else:
            self.model = obj
            self.scaler = None
            self.feature_names = None

    def _clean(self, v):
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    def _build_feature_vector(self, row):
        #modo novo: usa feature_names do .pkl
        if self.feature_names:
            feats = []
            for name in self.feature_names:
                val = self._clean(row.get(name))

                if name == "rpm":
                    val = clamp(val, 0, 8000)
                elif name == "speed_kmh":
                    val = clamp(val, 0, 250)
                elif name in ("coolant_C", "iat_C"):
                    val = clamp(val, -40, 150)
                elif name in ("tps_pct", "engine_load"):
                    val = clamp(val, 0, 100)
                elif name == "fuel_rate":
                    val = clamp(val, 0, 300)
                elif name == "fuel_pct":
                    val = clamp(val, 0, 100)
                elif name == "baro_kpa":
                    if val in (None, 0.0, 0):
                        val = 101.3
                    val = clamp(val, 50, 200)

                feats.append(0.0 if val is None else float(val))

            return np.array([feats], dtype=float)

        #modo antigo
        feats = [
            self._clean(row.get("rpm")),
            self._clean(row.get("speed_kmh")),
            self._clean(row.get("coolant_C")),
            self._clean(row.get("tps_pct")),
            self._clean(row.get("fuel_pct")),
            self._clean(row.get("fuel_rate")),
            self._clean(row.get("engine_load")),
            self._clean(row.get("iat_C")),
            self._clean(row.get("baro_kpa")),
        ]

        feats[0] = clamp(feats[0], 0, 8000)
        feats[1] = clamp(feats[1], 0, 250)
        feats[2] = clamp(feats[2], -40, 130)
        feats[3] = clamp(feats[3], 0, 100)
        feats[4] = clamp(feats[4], 0, 100)

        if feats[8] in (None, 0.0, 0):
            feats[8] = 101.3

        return np.array([[0.0 if v is None else float(v) for v in feats]], dtype=float)



    def predict_status(self, row):

        c    = self._clean(row.get("coolant_C"))     # temperatura
        r    = self._clean(row.get("rpm"))           # rotação
        s    = self._clean(row.get("speed_kmh"))     # velocidade
        tps  = self._clean(row.get("tps_pct"))       # posição borboleta
        load = self._clean(row.get("engine_load"))   # carga do motor

        c    = c    if c    is not None else -1
        r    = r    if r    is not None else -1
        s    = s    if s    is not None else -1
        tps  = tps  if tps  is not None else -1
        load = load if load is not None else -1


        if c >= 115:
            return "ANOMALIA"
        if r > 7800:
            return "ANOMALIA"
        if s >= 60 and r >= 0 and r < 400:
            return "ANOMALIA"

        if s >= 60 and (tps >= 15 or load >= 20) and r < 800:
            return "ANOMALIA"

        # tenta usar o IsolationForest
        status_iforest = None
        if self.model is not None and np is not None:
            try:
                X = self._build_feature_vector(row)
                if self.scaler is not None:
                    X = self.scaler.transform(X)
                pred = int(self.model.predict(X)[0])
                status_iforest = "NORMAL" if pred == 1 else "ANOMALIA"
                print(f"[IA] IsolationForest utilizado: {status_iforest}")
            except Exception as e:
                self._notify(f"IA: erro no predict ({e}). Usando apenas regras.")
                status_iforest = None

        iat = self._clean(row.get("iat_C"))
        iat = iat if iat is not None else -1

        if status_iforest == "ANOMALIA":
            critico = (
                    (c >= 100) or
                    (r >= 6500) or
                    (s >= 180) or
                    ((load >= 95) and (tps >= 60)) or
                    ((iat >= 90) and (load >= 80))
            )
            return "ANOMALIA" if critico else "NORMAL"

        return "NORMAL"



    def predict_causa(self, row):

        return None, None




# ===== Explicação textual do diagnóstico =====
def explicar_diagnostico(row, status):
    if status == "NORMAL":
        return "Sem anomalias aparentes de acordo com o modelo."

    msgs = []

    rpm   = row.get("rpm") or 0
    v     = row.get("speed_kmh") or 0
    temp  = row.get("coolant_C") or 0
    tps   = row.get("tps_pct") or 0
    load  = row.get("engine_load") or 0
    iat   = row.get("iat_C") or 0
    fuel  = row.get("fuel_pct") or 0

    if iat >= 90:
        msgs.append(
            "Temperatura do ar de admissão (IAT) muito alta. "
            "Em condição real pode indicar heat-soak, falha no sensor IAT, ou operação sob calor extremo. "
            "Se estiver alto com velocidade constante, suspeitar da leitura/simulador."
        )

    if load >= 95 and rpm < 6500 and v < 180:
        msgs.append(
            "Carga do motor em 100% fora de regime extremo. "
            "Pode ser pico de aceleração, mas se persistente é forte indicativo de leitura fora do padrão "
            "(sensor/escala do simulador) ou dado raro no conjunto de treino."
        )

    if temp == 0 and rpm > 600:
        msgs.append(
            "Incoerência: coolant em 0°C com motor em funcionamento. "
            "Isso aponta para leitura inválida/índice RDS incorreto/frame misturado."
        )

    baro_raw = row.get("baro_kpa")

    baro = baro_raw if baro_raw not in (None, 0.0) else None

    if temp >= 115:
        msgs.append(
            "Possível superaquecimento do motor (temperatura do líquido de arrefecimento muito alta). "
            "Verificar ventoinha, válvula termostática, bomba d'água e nível do fluido."
        )

    if rpm > 4000 and v < 5:
        msgs.append(
            "Motor em alta rotação com veículo parado ou quase parado. "
            "Em condição real, pode indicar problema de embreagem ou uso incorreto de marcha."
        )

    if load > 80 and temp >= 110:
        msgs.append(
            "Alta carga do motor combinada com temperatura elevada. "
            "Risco de superaquecimento sob carga intensa."
        )

    if rpm >= 7800:
        if v >= 10:
            msgs.append(
                "RPM no limite (regime de redline/limitador) com o veículo em movimento. "
                "Em condição real, indica uso extremo (redução agressiva/alta carga) e risco de desgaste. "
                "Se ocorrer sem comando (TPS baixo), pode indicar inconsistência de leitura/sensor."
            )
        else:
            msgs.append(
                "RPM no limite com o veículo parado/quase parado. "
                "Em condição real, pode indicar aceleração em vazio ou patinagem de embreagem."
            )

    if v > 200 and 0 <= tps < 5:
        msgs.append(
            "Velocidade muito alta com TPS quase fechado. "
            "Pode ser freio-motor/desaceleração, mas se persistir é forte sinal de inconsistência de leitura "
            "(sensor TPS/velocidade) ou ruído no dado."
        )

    if tps > 70 and v < 10 and rpm > 2000:
        msgs.append(
            "Aceleração alta com baixa velocidade. "
            "Em veículo real, pode indicar excesso de carga ou problema na transmissão."
        )

    if fuel > 0 and fuel < 10:
        msgs.append(
            "Nível de combustível muito baixo. "
            "Em uso real, pode causar falhas de alimentação e apagões intermitentes."
        )

    if rpm >= 7800 and v >= 5:
        msgs.append(
            "Rotação extremamente alta (próximo/igual ao corte). "
            "Em condição real, pode indicar marcha muito reduzida, patinação (embreagem/roda) "
            "ou leitura inválida do RPM."
        )

    if v >= 60 and rpm >= 0 and rpm < 400:
        msgs.append(
            "Incoerência entre velocidade e RPM (velocidade alta com RPM muito baixo). "
            "Isso costuma ser leitura incompleta/frame misturado ou índice RDS incorreto."
        )

    if not msgs:
        msgs.append(
            "Comportamento fora da faixa aprendida pelo modelo, mas sem um padrão claro "
            "para um diagnóstico específico. Verificar contexto e demais sinais."
        )

    return " | ".join(msgs)


# GUI UI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        # estilo geral
        try:
            self.call("tk", "scaling", 1.3)
        except Exception:
            pass

        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure("Title.TLabel", font=("Segoe UI", 11, "bold"))
        self.style.configure("BigValue.TLabel", font=("Segoe UI", 18, "bold"))
        self.style.configure("DashValue.TLabel", font=("Segoe UI", 16, "bold"))
        self.style.configure("SmallValue.TLabel", font=("Segoe UI", 11))
        self.style.configure("Card.TFrame", relief="ridge", borderwidth=1)

        self.title("MaxGeek USB — Diagnóstico Inteligente")
        self.geometry("1000x620")
        self.minsize(1000, 620)

        # estado
        self.reader_thread = None
        self.reader_stop = threading.Event()
        self.reader_queue = queue.Queue()
        self.mg = None
        self.status = tk.StringVar(value="Pronto.")

        # IA
        self.ia = SimpleIARuntime(status_cb=self._set_status)

        self.csv_file = None
        self.csv_writer = None
        self.logging_enabled = False
        self.live_win = None
        self.ai_win = None

        # frame fundido (para IA e Live)
        self._last_frame = None
        self._pending_live_row = None

        # cache LIVE para cor de “stale”
        self._last_ok = {}
        self._last_ts = {}
        self._stale_sec = 2.0

        # topo (conexão)
        top = ttk.LabelFrame(self, text="Conexão")
        top.pack(fill="x", padx=14, pady=10)

        ttk.Label(top, text="Porta:", style="Title.TLabel")\
            .grid(row=0, column=0, padx=8, pady=8, sticky="e")
        self.port_cmb = ttk.Combobox(
            top, width=18,
            values=(self._list_ports() if list_ports else []),
            state="readonly"
        )
        if self.port_cmb["values"]:
            self.port_cmb.current(0)
        self.port_cmb.grid(row=0, column=1, padx=8, pady=8, sticky="w")

        ttk.Label(top, text="Baud:", style="Title.TLabel")\
            .grid(row=0, column=2, padx=8, pady=8, sticky="e")
        self.baud_cmb = ttk.Combobox(top, width=10, values=[115200,230400], state="readonly")
        self.baud_cmb.set(DEFAULT_BAUD)
        self.baud_cmb.grid(row=0, column=3, padx=8, pady=8, sticky="w")

        self.crlf_var = tk.BooleanVar(value=DEFAULT_CRLF)
        ttk.Checkbutton(top, text="CRLF", variable=self.crlf_var)\
            .grid(row=0, column=4, padx=8, pady=8, sticky="w")

        # menu principal
        menu = ttk.LabelFrame(self, text="Menu")
        menu.pack(fill="x", padx=14, pady=10)

        ttk.Button(menu, text="Analisar possíveis Erros com I.A",
                   command=self.open_ai_window, width=50)\
            .grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ttk.Button(menu, text="Analisar painéis (tempo real)",
                   command=self.open_live_window, width=50)\
            .grid(row=1, column=0, padx=10, pady=10, sticky="w")
        ttk.Button(menu, text="Exportar dados (CSV)",
                   command=self.export_csv_dialog, width=50)\
            .grid(row=2, column=0, padx=10, pady=10, sticky="w")
        ttk.Button(menu, text="Sair", command=self.safe_exit, width=50)\
            .grid(row=3, column=0, padx=10, pady=10, sticky="w")

        # status bar
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w")\
            .pack(fill="x", padx=14, pady=(0,12))

        # timers
        self.after(150, self._pump_queue)
        self.after(80, self._ui_tick)

        self.ia.auto_load_if_exists()

    def _set_status(self, msg):
        self.after(0, lambda: self.status.set(msg))

    def _list_ports(self):
        return [p.device for p in list_ports.comports()] if list_ports else []

    #leitura
    def _start_reader(self):
        if self.reader_thread and self.reader_thread.is_alive():
            return True
        port = (self.port_cmb.get() or "").strip()
        if not port:
            messagebox.showerror("Erro", "Selecione a porta serial.")
            return False
        baud = int(self.baud_cmb.get())
        crlf = self.crlf_var.get()
        self.reader_stop.clear()
        self.mg = MaxGeekSerial(
            port=port, baud=baud, crlf=crlf,
            hard_timeout=0.25, quiet_ms=12, retries=1
        )
        self.reader_thread = ReaderThread(
            self.mg, interval=0.12,
            stop_event=self.reader_stop, queue_out=self.reader_queue
        )
        self.reader_thread.start()
        self.status.set(f"Lendo {port} @ {baud} (CRLF={crlf})…")
        return True

    def _stop_reader(self):
        self.reader_stop.set()
        if self.reader_thread:
            self.reader_thread.join(timeout=1.5)
        self.reader_thread = None
        try:
            if self.mg:
                self.mg.close()
        except:
            pass
        self.mg = None

    def _pump_queue(self):
        try:
            latest = None
            while True:
                item = self.reader_queue.get_nowait()
                if "error" in item:
                    self.status.set(item["error"])
                    continue
                latest = self._sanitize_row(item)
        except queue.Empty:
            pass

        if latest:
            if self._last_frame is None:
                fused = dict(latest)
            else:
                fused = dict(self._last_frame)
                fused["timestamp"] = latest.get("timestamp", fused.get("timestamp"))
                for key in ["rpm","speed_kmh","coolant_C","tps_pct","fuel_pct",
                            "fuel_rate","engine_load","iat_C","baro_kpa"]:
                    v = latest.get(key, None)
                    if v is not None:
                        fused[key] = v

            self._last_frame = fused

            self._pending_live_row = fused
            if self.ai_win:
                self.ai_win_update(fused)


            if self.logging_enabled and self.csv_writer:
                out = [
                    fused.get("timestamp"), fused.get("rpm"), fused.get("speed_kmh"), fused.get("coolant_C"),
                    fused.get("tps_pct"), fused.get("fuel_pct"), fused.get("fuel_rate"),
                    fused.get("engine_load"), fused.get("iat_C"), fused.get("baro_kpa"),
                    fused.get("status","")
                ]
                self.csv_writer.writerow(out)

        self.after(120, self._pump_queue)

    def _sanitize_row(self, d):
        row = dict(d)

        # clamps básicos
        row["rpm"] = clamp(row.get("rpm"), 0, 8000)
        row["speed_kmh"] = clamp(row.get("speed_kmh"), 0, 250)
        row["coolant_C"] = clamp(row.get("coolant_C"), 0, 130)
        row["tps_pct"] = clamp(row.get("tps_pct"), 0, 100)
        row["fuel_pct"] = clamp(row.get("fuel_pct"), 0, 100)
        row["fuel_rate"] = clamp(row.get("fuel_rate"), 0, 300)
        row["engine_load"] = clamp(row.get("engine_load"), 0, 100)
        row["iat_C"] = clamp(row.get("iat_C"), -40, 150)

        # PATCH 1A: baro 0.0 do MaxGeek = "desconhecido" (não deixa isso poluir o modelo)
        baro = row.get("baro_kpa")
        if baro in (None, 0, 0.0):
            row["baro_kpa"] = None
        else:
            row["baro_kpa"] = clamp(baro, 50, 200)

        rpm = row.get("rpm")
        cool = row.get("coolant_C")
        if cool is not None and rpm is not None:
            if cool == 0 and rpm > 600:
                row["coolant_C"] = None

        # PATCH opcional: IAT absurdo também pode ser ruído (se quiser só “suavizar”)
        iat = row.get("iat_C")
        if iat is not None and rpm is not None:
            if iat == 0 and rpm > 600:
                row["iat_C"] = None

        spd = row.get("speed_kmh")
        if rpm is not None and spd is not None:
            if spd > 5 and rpm < 400:
                row["rpm"] = None

        return row

    def open_live_window(self):
        if self.live_win and tk.Toplevel.winfo_exists(self.live_win):
            self.live_win.lift()
            return
        self.live_win = tk.Toplevel(self)
        self.live_win.title("Tempo real — Painel do veículo")
        self.live_win.geometry("680x520")
        self.live_win.resizable(False, False)

        self.value_labels = {}  # key -> (label_widget, stringvar, unit)

        # Painel principal (RPM / Velocidade)
        main_panel = ttk.LabelFrame(self.live_win, text="Painel principal")
        main_panel.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # RPM
        rpm_frame = ttk.Frame(main_panel, padding=6, style="Card.TFrame")
        rpm_frame.grid(row=0, column=0, padx=8, pady=4, sticky="nsew")
        ttk.Label(rpm_frame, text="RPM", style="Title.TLabel")\
            .pack(anchor="w")
        rpm_var = tk.StringVar(value="—")
        rpm_lbl = ttk.Label(rpm_frame, textvariable=rpm_var, style="DashValue.TLabel")
        rpm_lbl.pack(anchor="center", pady=2)
        self.value_labels["rpm"] = (rpm_lbl, rpm_var, " rpm")

        # Velocidade
        spd_frame = ttk.Frame(main_panel, padding=6, style="Card.TFrame")
        spd_frame.grid(row=0, column=1, padx=8, pady=4, sticky="nsew")
        ttk.Label(spd_frame, text="Velocidade", style="Title.TLabel")\
            .pack(anchor="w")
        spd_var = tk.StringVar(value="—")
        spd_lbl = ttk.Label(spd_frame, textvariable=spd_var, style="DashValue.TLabel")
        spd_lbl.pack(anchor="center", pady=2)
        self.value_labels["speed_kmh"] = (spd_lbl, spd_var, " km/h")

        # Outros parâmetros em cards menores
        grid_panel = ttk.LabelFrame(self.live_win, text="Parâmetros do motor")
        grid_panel.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        def add_card(r, c, name, key, unit=""):
            f = ttk.Frame(grid_panel, padding=4, style="Card.TFrame")
            f.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
            ttk.Label(f, text=name, style="Title.TLabel").pack(anchor="w")
            var = tk.StringVar(value="—")
            lbl = ttk.Label(f, textvariable=var, style="SmallValue.TLabel")
            lbl.pack(anchor="center", pady=2)
            self.value_labels[key] = (lbl, var, unit)

        add_card(0, 0, "Temperatura (Coolant)", "coolant_C", " °C")
        add_card(0, 1, "Combustível", "fuel_pct", " %")
        add_card(0, 2, "TPS", "tps_pct", " %")
        add_card(1, 0, "Taxa de combustível", "fuel_rate", " L/h")
        add_card(1, 1, "Carga do motor", "engine_load", " %")
        add_card(1, 2, "IAT", "iat_C", " °C")
        add_card(2, 0, "Barométrica", "baro_kpa", " kPa")

        btns = ttk.Frame(self.live_win)
        btns.grid(row=2, column=0, columnspan=2, pady=12)
        ttk.Button(btns, text="Iniciar", command=self._start_reader)\
            .grid(row=0, column=0, padx=8)
        ttk.Button(btns, text="Parar", command=self._stop_reader)\
            .grid(row=0, column=1, padx=8)

        self._start_reader()

    def _apply_row_to_cache(self, row):
        now = time.time()
        for key in ["rpm","speed_kmh","coolant_C","fuel_pct","tps_pct",
                    "fuel_rate","engine_load","iat_C","baro_kpa"]:
            if key in row and row[key] is not None:
                self._last_ok[key] = row[key]
                self._last_ts[key] = now

    def _render_live_from_cache(self):
        now = time.time()
        for key, (lbl, var, unit) in self.value_labels.items():
            val = self._last_ok.get(key, None)
            var.set("—" if val is None else (f"{val:.2f}{unit}" if isinstance(val,float) else f"{val}{unit}"))
            ts = self._last_ts.get(key, 0)
            fresh = (now - ts) <= self._stale_sec
            try:
                lbl.configure(foreground=("black" if fresh else "#B36B00"))
            except Exception:
                pass

    def _ui_tick(self):
        if self._pending_live_row:
            self._apply_row_to_cache(self._pending_live_row)
            self._pending_live_row = None
        if self.live_win:
            self._render_live_from_cache()
        self.after(80, self._ui_tick)

    # -------- IA (janela mais bonita) ----------
    def open_ai_window(self):
        if self.ai_win and tk.Toplevel.winfo_exists(self.ai_win):
            self.ai_win.lift()
            return

        self.ai_win = tk.Toplevel(self)
        self.ai_win.title("IA — Detecção de Anomalias do Motor")
        # janela maior e redimensionável
        self.ai_win.geometry("1000x620")
        self.ai_win.resizable(True, True)

        # topo: modelo + status grande
        top = ttk.Frame(self.ai_win)
        top.pack(fill="x", padx=10, pady=(8, 4))

        ttk.Label(top, text="Modelo:", style="Title.TLabel")\
            .grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.model_path_var = tk.StringVar(value="(usando fallback de regras)")
        ttk.Label(top, textvariable=self.model_path_var, width=60)\
            .grid(row=0, column=1, sticky="w", padx=4, pady=4)

        self.status_var = tk.StringVar(value="—")
        self.status_label = ttk.Label(
            top, textvariable=self.status_var,
            font=("Segoe UI", 18, "bold")
        )
        self.status_label.grid(row=1, column=0, columnspan=2, pady=(6, 4))

        # meio: cards de sinais + mini-gráfico
        mid = ttk.Frame(self.ai_win)
        mid.pack(fill="x", padx=10, pady=4)

        # cards de sinais
        cards = ttk.LabelFrame(mid, text="Resumo dos sinais atuais")
        cards.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.ai_val_vars = {}  # key -> (StringVar, unit)

        def add_ai_card(r, c, title, key, unit=""):
            f = ttk.Frame(cards, padding=4, style="Card.TFrame")
            f.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
            ttk.Label(f, text=title, style="Title.TLabel").pack(anchor="w")
            var = tk.StringVar(value="—")
            ttk.Label(f, textvariable=var, style="BigValue.TLabel")\
                .pack(anchor="center", pady=2)
            self.ai_val_vars[key] = (var, unit)

        add_ai_card(0, 0, "RPM", "rpm", " rpm")
        add_ai_card(0, 1, "Velocidade", "speed_kmh", " km/h")
        add_ai_card(1, 0, "Temperatura coolant", "coolant_C", " °C")
        add_ai_card(1, 1, "Carga do motor", "engine_load", " %")
        add_ai_card(2, 0, "TPS", "tps_pct", " %")
        add_ai_card(2, 1, "Combustível", "fuel_pct", " %")

        # mini gráfico (barras horizontais)
        gframe = ttk.LabelFrame(mid, text="Indicadores visuais")
        gframe.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.ai_canvas = tk.Canvas(
            gframe,
            width=380,  # era 320
            height=230,  # era 200
            bg="#111111",
            highlightthickness=0
        )
        self.ai_canvas.pack(fill="both", expand=True, padx=4, pady=4)

        # botões
        btns = ttk.Frame(self.ai_win)
        btns.pack(fill="x", padx=10, pady=(4, 4))
        ttk.Button(btns, text="Carregar Modelo (*.pkl)", command=self._load_model)\
            .pack(side="left", padx=4)
        ttk.Button(btns, text="Iniciar", command=self._start_reader)\
            .pack(side="left", padx=4)
        ttk.Button(btns, text="Parar", command=self._stop_reader)\
            .pack(side="left", padx=4)

        # texto detalhado — MAIOR + SCROLLBAR
        text_frame = ttk.LabelFrame(self.ai_win, text="diagnosticoo")
        text_frame.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        self.last_row_text = tk.Text(
            text_frame,
            height=18,      # mais alto
            width=110,      # mais largo
            wrap="word"     # quebra palavras certinho
        )
        self.last_row_text.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        scroll_y = ttk.Scrollbar(text_frame, orient="vertical", command=self.last_row_text.yview)
        scroll_y.pack(side="right", fill="y")
        self.last_row_text.configure(yscrollcommand=scroll_y.set)

        self.last_row_text.configure(state="disabled")

        # refletir auto-load na UI, se existir arquivo
        for p in DEFAULT_MODEL_PATHS:
            if p and os.path.isfile(p):
                self.model_path_var.set(os.path.basename(p))
                break

        self._start_reader()

    def _load_model(self):
        path = filedialog.askopenfilename(
            title="Selecione o modelo (IsolationForest)",
            filetypes=[("Pickle/Joblib","*.pkl *.joblib *.sav"),("Todos","*.*")]
        )
        if not path:
            return
        try:
            self.ia.load_model(path)
            self.model_path_var.set(os.path.basename(path))
            messagebox.showinfo("OK", "Modelo carregado.")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar modelo: {e}")

    def _update_ai_canvas(self, row, status):
        if not hasattr(self, "ai_canvas") or not self.ai_canvas:
            return
        c = self.ai_canvas
        c.delete("all")

        bg = "#111111"
        c.configure(bg=bg)

        # escolha de cor por status
        if status == "NORMAL":
            status_color = "#00CC66"
        elif status == "ANOMALIA":
            status_color = "#FF3333"
        else:
            status_color = "#CCCC33"

        c.create_text(10, 10, anchor="nw",
                      text=f"Status: {status}",
                      fill=status_color,
                      font=("Segoe UI", 11, "bold"))

        # barras simples
        bars = [
            ("RPM", row.get("rpm") or 0, 0, 8000),
            ("Velocidade", row.get("speed_kmh") or 0, 0, 250),
            ("Coolant", row.get("coolant_C") or 0, 0, 130),
        ]

        # usa largura real do canvas e deixa espaço para o número à direita
        canvas_w = max(int(c.winfo_width()), 380)
        x0 = 20
        x1_max = canvas_w - 80  # sobra ~80px para o valor numérico

        y = 70  # posição inicial das barras
        bar_h = 24  # altura da barra
        gap = 26  # espaço vertical entre barras
        label_gap = 10  # distância entre label e barra

        for name, val, vmin, vmax in bars:
            try:
                v = float(val)
            except Exception:
                v = 0.0

            # normaliza valor 0–1
            frac = 0.0
            if vmax > vmin:
                frac = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
            x1 = x0 + frac * (x1_max - x0)

            # rótulo ACIMA da barra
            label_y = y - label_gap
            c.create_text(
                x0, label_y,
                anchor="sw",
                text=name,
                fill="#DDDDDD",
                font=("Segoe UI", 9, "bold")
            )

            # trilho
            c.create_rectangle(
                x0, y, x1_max, y + bar_h,
                outline="#444444",
                fill="#222222"
            )

            # barra preenchida
            c.create_rectangle(
                x0, y, x1, y + bar_h,
                outline="",
                fill=status_color if status == "ANOMALIA" else "#3478f6"
            )

            # valor numérico à direita
            c.create_text(
                x1_max + 6, y + bar_h / 2,
                text=f"{v:.1f}",
                anchor="w",
                fill="#FFFFFF",
                font=("Segoe UI", 9)
            )

            # próxima barra
            y += bar_h + gap

    def ai_win_update(self, row):
        # frame já fundido, com todos os sinais disponíveis
        status = self.ia.predict_status(row)
        row["status"] = status

        # texto base (regras explicativas)
        diag_base = explicar_diagnostico(row, status)

        # tenta descobrir a causa provável com o segundo modelo (se existir)
        causa_label, causa_proba = self.ia.predict_causa(row)

        if status == "ANOMALIA" and causa_label not in (None, "", "—"):
            if causa_proba is not None:
                causa_txt = f"Causa provável: {causa_label}  (confiança ~ {causa_proba * 100:.0f}%)"
            else:
                causa_txt = f"Causa provável: {causa_label}"
        else:
            causa_txt = None

        # status grande + cor
        self.status_var.set(f"{status}  •  {row.get('timestamp','')}")
        if status == "NORMAL":
            fg = "#00AA55"
        elif status == "ANOMALIA":
            fg = "#CC0000"
        else:
            fg = "#666666"
        try:
            self.status_label.configure(foreground=fg)
        except Exception:
            pass

        for key, (var, unit) in self.ai_val_vars.items():
            val = row.get(key)
            if isinstance(val, (int, float)):
                var.set(f"{val:.1f}{unit}")
            elif val is not None:
                var.set(f"{val}{unit}")
            else:
                var.set("—")

        self._update_ai_canvas(row, status)

        self.last_row_text.configure(state="normal")
        self.last_row_text.delete("1.0", "end")

        self.last_row_text.insert("end", "dados\n")
        ordered = ["timestamp", "rpm", "speed_kmh", "coolant_C", "tps_pct", "fuel_pct",
                   "fuel_rate", "engine_load", "iat_C", "baro_kpa"]
        for k in ordered:
            val = row.get(k)
            self.last_row_text.insert("end", f"{k}: {val}\n")

        self.last_row_text.insert("end", "\n=== Resultado da IA ===\n")
        self.last_row_text.insert("end", f"Status: {status}\n")
        if causa_txt:
            self.last_row_text.insert("end", causa_txt + "\n")
        self.last_row_text.insert("end", f"Diagnóstico detalhado: {diag_base}\n")

        self.last_row_text.configure(state="disabled")


    # -------- CSV
    def export_csv_dialog(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV","*.csv")],
                                            title="Salvar CSV")
        if not path:
            return
        try:
            f = open(path, "w", newline="", encoding="utf-8")
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            self.csv_file = f
            self.csv_writer = w
            self.logging_enabled = True
            self.status.set(f"Gravando CSV em: {path}")
            self._start_reader()
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível abrir CSV: {e}")

    # -------- sair
    def safe_exit(self):
        try:
            self.logging_enabled = False
            if self.csv_file:
                self.csv_file.close()
        except:
            pass
        try:
            self._stop_reader()
        except:
            pass
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
