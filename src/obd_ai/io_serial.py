import re, time, csv
from typing import Dict, Optional, Iterable

ALIASES = {
    "rpm": [r"\b(rpm|engine\s*rpm|engine_rpm)\b"],
    "speed": [r"\b(speed|vehicle\s*speed|vehicle\s*speed\s*sensor|vss)\b"],
    "coolant_temp": [r"\b(coolant|engine\s*coolant\s*temperature|ect|coolant_temp)\b"],
    "tps": [r"\b(throttle|absolute\s*throttle\s*position|tps)\b"],
    "maf": [r"\b(maf|air\s*flow\s*rate|mass\s*air\s*flow)\b"],
    "map": [r"\b(map|intake\s*manifold\s*absolute\s*pressure)\b"],
}

def normalize_header(cols: Iterable[str]) -> Dict[str,str]:
    m = {}
    for tgt, pats in ALIASES.items():
        for c in cols:
            c_norm = re.sub(r"\s*\[.*?\]", "", c.strip().lower())
            c_norm = re.sub(r"[^a-z0-9_ ]", " ", c_norm)
            if any(re.search(p, c_norm, re.I) for p in pats):
                m[tgt] = c
                break
    return m
