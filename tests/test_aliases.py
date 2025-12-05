from src.obd_ai.io_serial import normalize_header

def test_normalize_header():
    cols = ["Engine RPM [rpm]", "Vehicle speed sensor [km/h]", "Engine coolant temperature [C]","Absolute throttle position [%]"]
    m = normalize_header(cols)
    assert set(["rpm","speed","coolant_temp","tps"]).issubset(set(m.keys()))
