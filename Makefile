VENV=.venv/Scripts

init:
	python -m venv .venv && $(VENV)/pip install -U pip && $(VENV)/pip install -r requirements.txt

train:
	$(VENV)/python scripts/treino_modelo.py --csv data/raw/dados.csv --out models/model_obd_iforest.pkl

diagnose:
	$(VENV)/python scripts/diagnose_from_csv.py --csv data/raw/dados.csv --model models/model_obd_iforest.pkl --plot

run:
	$(VENV)/python scripts/runtime_inferencia.py --port COM3 --baud 115200 --model models/model_obd_iforest.pkl
