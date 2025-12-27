import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score
from datetime import datetime
import numpy as np
import random

# Utilise le runtime léger (pas besoin de tensorflow complet)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    # fallback si tflite-runtime non dispo et que tensorflow est installé
    import tensorflow as tf
    tflite = tf.lite

print("📂 Chargement du dataset...")
file_path = os.path.join("/home/elias/PROJECT/AICryptoPredictor/data", "btcusd_1-min_data.csv")
btc = pd.read_csv(file_path)

print("📂 Traitement du dataset...")
btc['Timestamp'] = pd.to_datetime(btc['Timestamp'], unit='s')
btc.set_index('Timestamp', inplace=True)

btc_daily = btc.resample('1D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

btc_daily['Return'] = btc_daily['Close'].pct_change()
btc_daily['MA7'] = btc_daily['Close'].rolling(7).mean()
btc_daily['MA30'] = btc_daily['Close'].rolling(30).mean()
btc_daily['Volatility'] = btc_daily['Return'].rolling(7).std()

delta = btc_daily['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
btc_daily['RSI'] = 100 - (100 / (1 + rs))

ema12 = btc_daily['Close'].ewm(span=12, adjust=False).mean()
ema26 = btc_daily['Close'].ewm(span=26, adjust=False).mean()
btc_daily['MACD'] = ema12 - ema26
btc_daily['Signal'] = btc_daily['MACD'].ewm(span=9, adjust=False).mean()

ma20 = btc_daily['Close'].rolling(20).mean()
std20 = btc_daily['Close'].rolling(20).std()
btc_daily['Bollinger_Upper'] = ma20 + 2*std20
btc_daily['Bollinger_Lower'] = ma20 - 2*std20

POURCENT_MARGE = 1.0075
btc_daily["Target"] = (
    ((btc_daily["High"].shift(-1) > (btc_daily["Close"] * POURCENT_MARGE)) |
     (btc_daily["High"].shift(-2) > (btc_daily["Close"] * POURCENT_MARGE)))
).astype(int)

btc_daily.dropna(inplace=True)

split_date = '2023-05-01'
train = btc_daily[btc_daily.index < split_date]
test = btc_daily[btc_daily.index >= split_date]

# Features EXACTEMENT dans le même ordre qu’au training (14 colonnes)
feature_cols = [
    "Open", "High", "Low", "Close", "Volume",
    "Return", "MA7", "MA30", "Volatility",
    "RSI", "MACD", "Signal",
    "Bollinger_Upper", "Bollinger_Lower"
]

X_train = train[feature_cols]
X_test = test[feature_cols]
y_train = train["Target"]
y_test = test["Target"]

# Validation pour éventuels besoins, mais l’inférence ne s’en sert pas
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

SEED = 73
np.random.seed(SEED)
random.seed(SEED)

# Chargement du modèle TFLite
model_path = os.path.join("/home/elias/PROJECT/AICryptoPredictor/notebooks", "model.tflite")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Fichier TFLite introuvable: {model_path}. Convertis ton modèle Keras en .tflite sur une autre machine, puis copie-le ici.")

print("🧠 Chargement du modèle TFLite...")
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Vérifie forme d’entrée attendue (doit être [None, 14])
expected_input_shape = input_details[0]["shape"]
if expected_input_shape[-1] != len(feature_cols):
    raise ValueError(f"Le modèle TFLite attend {expected_input_shape[-1]} features, mais on en fournit {len(feature_cols)}.")

# Préparation des données au bon dtype
input_dtype = input_details[0]["dtype"]  # souvent float32
X_test_np = X_test.astype(np.float32).values
# Ajuste la forme si le modèle attend une batch dimension
if len(expected_input_shape) == 2:
    # [batch, features] -> OK
    pass
else:
    # Si le modèle est différent, adapter ici
    X_test_np = X_test_np.reshape((-1,) + tuple(expected_input_shape[1:]))

print("⚡ Inférence TFLite...")
# Inférence par batch pour éviter de saturer de vieilles machines
batch_size = 256
y_pred_proba_list = []

for i in range(0, len(X_test_np), batch_size):
    batch = X_test_np[i:i+batch_size]
    # Définir le tenseur d'entrée
    interpreter.set_tensor(input_details[0]["index"], batch)
    interpreter.invoke()
    # Récupérer la sortie
    out = interpreter.get_tensor(output_details[0]["index"])
    y_pred_proba_list.append(out)

y_pred_proba = np.vstack(y_pred_proba_list)
threshold = 0.789
y_pred = (y_pred_proba > threshold).astype(int)

print("📊 Évaluation...")
report = classification_report(y_test, y_pred, digits=4)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

final_prediction = int(y_pred[-1][0])
output_file = f"/home/elias/PROJECT/AICryptoPredictor/results/prediction_{datetime.now().strftime('%Y%m%d_%H')}.txt"
last_date = btc_daily.index[-1].strftime("%Y-%m-%d")

os.makedirs("/home/elias/PROJECT/AICryptoPredictor/results", exist_ok=True)
with open(output_file, "w") as f:
    f.write(f"Dernière date du DataSet: {last_date}\n\n")
    f.write("Resultat de la pipeline 7 (TFLite)\n\n")
    f.write(f"Augmentation prévue de: {POURCENT_MARGE*100}% \n\n")
    f.write(f"Prediction finale (0 ou 1): {final_prediction}\n")
    f.write(f"Prix de cloture du {last_date} est à {btc_daily['Close'].iloc[-1]}\n")
    f.write(f"Prix du High du lendemain est estimé à {btc_daily['Close'].iloc[-1] * POURCENT_MARGE}\n\n")
    f.write("Rapport de classification:\n")
    f.write(report + "\n")
    f.write(f"Precision classe 1: {precision}\n")
    f.write(f"Recall classe 1: {recall}\n")
print(f"✅ Résultats sauvegardés dans {output_file}")

signal_file = "/home/elias/PROJECT/AICryptoPredictor/Output/signal.txt"
os.makedirs("/home/elias/PROJECT/AICryptoPredictor/Output", exist_ok=True)
with open(signal_file, "w") as f:
    f.write(str(final_prediction))
    f.write("\n")
    f.write(str(btc_daily['Close'].iloc[-1] * POURCENT_MARGE))
print(f"✅ Signal sauvegardé dans {signal_file}")