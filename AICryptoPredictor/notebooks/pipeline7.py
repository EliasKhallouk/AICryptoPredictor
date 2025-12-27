import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, precision_score, recall_score
from keras import Input, Model
from keras.layers import Dense, BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from datetime import datetime
import tensorflow as tf
import numpy as np
import random

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
btc_daily = btc_daily.dropna()

split_date = '2023-05-01'
train = btc_daily[btc_daily.index < split_date]
test = btc_daily[btc_daily.index >= split_date]

X_train = train.drop(columns=["Target"])
X_test = test.drop(columns=["Target"])
y_train = train["Target"]
y_test = test["Target"]

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

SEED = 73
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

inputs = Input(shape=(14,))
x = Dense(256, activation='relu')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.88),
              loss='binary_crossentropy', 
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

print("⚡ Entraînement du modèle...")
history = model.fit(
    X_train_split, y_train_split,
    epochs=100, batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    class_weight={0: 1, 1: 2}
)

print("📊 Évaluation...")
y_pred_proba = model.predict(X_test)
threshold = 0.789
y_pred = (y_pred_proba > threshold).astype(int)

report = classification_report(y_test, y_pred, digits=4)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Convertir le modèle en TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Sauvegarder le modèle TensorFlow Lite
tflite_model_file = "/home/elias/PROJECT/AICryptoPredictor/results/model.tflite"
with open(tflite_model_file, "wb") as f:
    f.write(tflite_model)

print(f"✅ Modèle TensorFlow Lite sauvegardé dans {tflite_model_file}")