def apply_rules(rpm, speed, coolant_temp, tps):
    msgs = []
    if coolant_temp is not None and coolant_temp > 110:
        msgs.append("Superaquecimento provável")
    if rpm is not None and speed is not None and rpm > 4500 and speed < 10:
        msgs.append("Incoerência RPM/Velocidade")
    if tps is not None and rpm is not None and tps > 95 and rpm < 900:
        msgs.append("Anomalia TPS/Marcha lenta")
    return "; ".join(msgs)
