python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\pip.exe install -r requirements.txt
Write-Host "OK. Ambiente criado. Próximos:" -ForegroundColor Green
Write-Host "1) Treino:    .\.venv\Scripts\python.exe scripts\treino_modelo.py --csv data\raw\dados.csv --out models\model_obd_iforest.pkl"
Write-Host "2) Diagnóstico: .\.venv\Scripts\python.exe scripts\diagnose_from_csv.py --csv data\raw\dados.csv --model models\model_obd_iforest.pkl --plot"
Write-Host "3) Runtime:   .\.venv\Scripts\python.exe scripts\runtime_inferencia.py --port COM3 --baud 115200 --model models\model_obd_iforest.pkl"
