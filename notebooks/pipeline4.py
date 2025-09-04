from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, precision_score, recall_score
from keras import Input, Model
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from datetime import datetime
import tensorflow as tf
import numpy as np
import random

print("📂 Chargement du dataset...")
file_path = os.path.join("/home/elias/PROJECT/AICryptoPredictor/data", "btcusd_1-min_data.csv")
btc = pd.read_csv(file_path)



print("📂 Traitement du dataset...")
# Convertir la colonne "Date" en datetime
btc['Timestamp'] = pd.to_datetime(btc['Timestamp'], unit='s')
btc.set_index('Timestamp', inplace=True)

# Agréger par jour : prix moyen, min, max et volume total
btc_daily = btc.resample('1D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Créer des features simples
btc_daily['Return'] = btc_daily['Close'].pct_change()       # Rendement quotidien
btc_daily['MA7'] = btc_daily['Close'].rolling(7).mean()     # Moyenne mobile 7 jours
btc_daily['MA30'] = btc_daily['Close'].rolling(30).mean()   # Moyenne mobile 30 jours
btc_daily['Volatility'] = btc_daily['Return'].rolling(7).std()  # Volatilité sur 7 jours

# RSI (14 jours)
delta = btc_daily['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
btc_daily['RSI'] = 100 - (100 / (1 + rs))

# MACD
ema12 = btc_daily['Close'].ewm(span=12, adjust=False).mean()
ema26 = btc_daily['Close'].ewm(span=26, adjust=False).mean()
btc_daily['MACD'] = ema12 - ema26
btc_daily['Signal'] = btc_daily['MACD'].ewm(span=9, adjust=False).mean()

# Bollinger Bands
ma20 = btc_daily['Close'].rolling(20).mean()
std20 = btc_daily['Close'].rolling(20).std()
btc_daily['Bollinger_Upper'] = ma20 + 2*std20
btc_daily['Bollinger_Lower'] = ma20 - 2*std20


POURCENT_MARGE = 1.0075
# Créer la colonne Target : 1 si le prix du high de demain est supérieur de 0.5% à l'ouverture d'aujourd'hui, sinon 0
btc_daily["Target"] = (
    ((btc_daily["High"].shift(-1) > (btc_daily["Close"] * POURCENT_MARGE)) |
     (btc_daily["High"].shift(-2) > (btc_daily["Close"] * POURCENT_MARGE)))
).astype(int)
btc_daily = btc_daily.dropna() # Retirer la dernière ligne car elle n'a pas de valeur pour Target

# Supprimer les lignes avec NaN (les premières lignes des rolling)
btc_daily.dropna(inplace=True)

# Split train/test
split_date = '2023-05-01'
train = btc_daily[btc_daily.index < split_date]
test = btc_daily[btc_daily.index >= split_date]


# Variables explicatives (toutes les colonnes sauf Target)
X_train = train.drop(columns=["Target"])
X_test = test.drop(columns=["Target"])

# Variable cible (ce qu’on veut prédire : Target)
y_train = train["Target"]
y_test = test["Target"]


# Création d'un jeu de validation à partir de X_train et y_train
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Préparation des séquences
# -----------------------------
def create_sequences(X, y, n_steps=30):
    Xs, ys = [], []
    for i in range(len(X) - n_steps):
        Xs.append(X[i:i+n_steps])
        ys.append(y[i+n_steps])
    return np.array(Xs), np.array(ys)

n_steps = 300  # fenêtre de 30 jours
X = btc_daily.drop(columns=["Target"]).values
y = btc_daily["Target"].values

X_seq, y_seq = create_sequences(X, y, n_steps)

# Split train/test
split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# -----------------------------
# CNN 1D Model
# -----------------------------

model = Sequential([
    Conv1D(filters=128, kernel_size=5, activation='relu', padding='same', input_shape=(n_steps, X_train.shape[2])),
    BatchNormalization(),
    Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),  # Réduction du dropout pour ne pas trop perdre d'informations
    
    Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),
    Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    
    Conv1D(filters=512, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),
    Conv1D(filters=512, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')  # 6 classes pour la classification
])

"""model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])
"""

from keras.optimizers import SGD
model.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.9),
              loss='binary_crossentropy', 
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# -----------------------------
# Entraînement
# -----------------------------
print("⚡ Entraînement du CNN...")
history = model.fit(
    X_train, y_train,
    epochs=50, batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# -----------------------------
# Évaluation
# -----------------------------
print("📊 Évaluation...")
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.85).astype(int)

print(classification_report(y_test, y_pred, digits=4))
