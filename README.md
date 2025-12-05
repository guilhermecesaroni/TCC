# TCC

## 1) Pré-requisitos
- Windows 10/11
- Python 3.10+

## 2) Instalar dependências
py -m pip install --upgrade pip
py -m pip install numpy pandas scikit-learn joblib

## 3) Treinar o modelo (escolha um)
py -m pip install --upgrade pip
py -m pip install numpy pandas scikit-learn joblib

Treino com vários CSVs em data/raw/:

py scripts\treino_iforest_multicsv.py

## 4) Treino com um CSV específico:
py scripts\treino_modelo.py --csv dados.csv --out models\model_obd_iforest.pkl

## 5) Iniciar
Precisa ter um simulador MaxGeek conectado ao PC, selecionar a porta e executar o comando py scripts\app_maxgeek_gui.py

