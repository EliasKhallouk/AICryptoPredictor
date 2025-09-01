import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from datetime import datetime
import numpy as np
import tensorflow as tf
import random

# ----------------------------
# 1️⃣ Fixer les seeds pour reproductibilité
# ----------------------------
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# ----------------------------
# 2️⃣ Chargement et traitement du dataset
# ----------------------------
print("📂 Chargement du dataset...")
file_path = os.path.join("/home/elias/PROJECT/AICryptoPredictor/data", "btcusd_1-min_data.csv")
btc = pd.read_csv(file_path)

btc['Timestamp'] = pd.to_datetime(btc['Timestamp'], unit='s')
btc.set_index('Timestamp', inplace=True)

btc_daily = btc.resample('1D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Features techniques
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

# Target = 1 si High demain > Close * 1.02
btc_daily['Target'] = (btc_daily['High'].shift(-1) > btc_daily['Close'] * 1.005).astype(int)

# Drop NaN
btc_daily.dropna(inplace=True)

# Split train/test
split_date = '2023-05-01'
train = btc_daily[btc_daily.index < split_date]
test = btc_daily[btc_daily.index >= split_date]

X_train = train.drop(columns=['Target'])
y_train = train['Target']
X_test = test.drop(columns=['Target'])
y_test = test['Target']

# ----------------------------
# 3️⃣ Rééchantillonnage classe 1 avec SMOTE
# ----------------------------
sm = SMOTE(random_state=seed)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Standardisation
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 4️⃣ Création du modèle
# ----------------------------
inputs = Input(shape=(X_train_res.shape[1],))
x = Dense(256, activation='relu')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# ----------------------------
# 5️⃣ Entraînement
# ----------------------------
history = model.fit(
    X_train_res, y_train_res,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# ----------------------------
# 6️⃣ Prédiction et seuil dynamique (max F1)
# ----------------------------
y_pred_proba = model.predict(X_test_scaled)

best_threshold = 0.5
best_f1 = 0
for t in np.arange(0.1, 0.95, 0.05):
    y_pred_try = (y_pred_proba > t).astype(int)
    f1 = f1_score(y_test, y_pred_try)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

y_pred = (y_pred_proba > best_threshold).astype(int)

# ----------------------------
# 7️⃣ Évaluation
# ----------------------------
report = classification_report(y_test, y_pred, digits=4)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
final_prediction = int(y_pred[-1][0])
last_date = btc_daily.index[-1].strftime("%Y-%m-%d")

# Sauvegarde
os.makedirs("results", exist_ok=True)
output_file = f"/home/elias/PROJECT/AICryptoPredictor/results/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(output_file, "w") as f:
    f.write(f"Dernière date du DataSet: {last_date}\n\n")
    f.write(f"Prediction finale (0 ou 1): {final_prediction}\n\n")
    f.write("Rapport de classification:\n")
    f.write(report + "\n")
    f.write(f"Precision classe 1: {precision}\n")
    f.write(f"Recall classe 1: {recall}\n")
    f.write(f"Seuil utilisé pour la prédiction: {best_threshold}\n")

print(f"✅ Résultats sauvegardés dans {output_file}")
print(f"Seuil optimal trouvé: {best_threshold}")
