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
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

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

"""
eps = 1e-9

# === 1) Volatilité / Range ===
# True Range components
prev_close = btc_daily['Close'].shift(1)
tr1 = btc_daily['High'] - btc_daily['Low']
tr2 = (btc_daily['High'] - prev_close).abs()
tr3 = (btc_daily['Low'] - prev_close).abs()
btc_daily['TrueRange'] = np.max(np.vstack([tr1.values, tr2.values, tr3.values]), axis=0)
btc_daily['ATR14'] = btc_daily['TrueRange'].rolling(14).mean()

# Parkinson HL volatility (annualisée optionnelle, ici brute rolling)
btc_daily['Parkinson20'] = (np.log(btc_daily['High'] / (btc_daily['Low'] + eps))**2).rolling(20).mean()

# Garman-Klass (approx daily)
hl = np.log(btc_daily['High'] / (btc_daily['Low'] + eps))**2
co = np.log(btc_daily['Close'] / (btc_daily['Open'] + eps))**2
btc_daily['GarmanKlass20'] = (0.5*hl - (2*np.log(2)-1)*co).rolling(20).mean()

# Position et range relatifs
rng = (btc_daily['High'] - btc_daily['Low']).replace(0, eps)
btc_daily['Position_in_range'] = (btc_daily['Close'] - btc_daily['Low']) / rng
btc_daily['Range_pct'] = (btc_daily['High'] - btc_daily['Low']) / (btc_daily['Close'] + eps)

# === 2) Structure de bougie ===
body = (btc_daily['Close'] - btc_daily['Open']).abs()
btc_daily['Body_pct'] = body / rng
btc_daily['Upper_wick_pct'] = (btc_daily['High'] - btc_daily['Close']).clip(lower=0) / (rng + eps)
btc_daily['Lower_wick_pct'] = (btc_daily['Close'] - btc_daily['Low']).clip(lower=0) / (rng + eps)

# === 3) Momentum multi-horizons ===
btc_daily['ROC5'] = btc_daily['Close'].pct_change(5)
btc_daily['ROC10'] = btc_daily['Close'].pct_change(10)
btc_daily['Zscore_Return_20'] = (
    (btc_daily['Return'] - btc_daily['Return'].rolling(20).mean()) /
    (btc_daily['Return'].rolling(20).std() + eps)
)
btc_daily['Slope_MA7'] = btc_daily['MA7'] - btc_daily['MA7'].shift(1)

# === 4) Oscillateurs complémentaires ===
# Stochastique %K/%D (14,3)
low14 = btc_daily['Low'].rolling(14).min()
high14 = btc_daily['High'].rolling(14).max()
btc_daily['Stoch_K'] = 100 * (btc_daily['Close'] - low14) / ((high14 - low14) + eps)
btc_daily['Stoch_D'] = btc_daily['Stoch_K'].rolling(3).mean()

# Williams %R (14)
btc_daily['WilliamsR'] = -100 * (high14 - btc_daily['Close']) / ((high14 - low14) + eps)

# MACD histogram
btc_daily['MACD_hist'] = btc_daily['MACD'] - btc_daily['Signal']

# === 5) Volume / Flux ===
btc_daily['Volume_z20'] = (
    (btc_daily['Volume'] - btc_daily['Volume'].rolling(20).mean()) /
    (btc_daily['Volume'].rolling(20).std() + eps)
)

# OBV
obv = [0.0]
for i in range(1, len(btc_daily)):
    if btc_daily['Close'].iloc[i] > btc_daily['Close'].iloc[i-1]:
        obv.append(obv[-1] + btc_daily['Volume'].iloc[i])
    elif btc_daily['Close'].iloc[i] < btc_daily['Close'].iloc[i-1]:
        obv.append(obv[-1] - btc_daily['Volume'].iloc[i])
    else:
        obv.append(obv[-1])
btc_daily['OBV'] = obv
btc_daily['OBV_roc5'] = btc_daily['OBV'].pct_change(5)

# Chaikin Money Flow (CMF 20)
mfm = ((btc_daily['Close'] - btc_daily['Low']) - (btc_daily['High'] - btc_daily['Close'])) / (rng + eps)
mfv = mfm * btc_daily['Volume']
btc_daily['CMF20'] = mfv.rolling(20).sum() / (btc_daily['Volume'].rolling(20).sum() + eps)
"""
"""
eps = 1e-9
rng = (btc_daily['High'] - btc_daily['Low']).replace(0, eps)
body = (btc_daily['Close'] - btc_daily['Open']).abs()
low14 = btc_daily['Low'].rolling(14).min()
high14 = btc_daily['High'].rolling(14).max()

prev_close = btc_daily['Close'].shift(1)
tr1 = btc_daily['High'] - btc_daily['Low']
tr2 = (btc_daily['High'] - prev_close).abs()
tr3 = (btc_daily['Low'] - prev_close).abs()
btc_daily['TrueRange'] = np.max(np.vstack([tr1.values, tr2.values, tr3.values]), axis=0)
btc_daily['ATR14'] = btc_daily['TrueRange'].rolling(14).mean()

btc_daily['MACD_hist'] = btc_daily['MACD'] - btc_daily['Signal']

btc_daily['Volume_z20'] = (
    (btc_daily['Volume'] - btc_daily['Volume'].rolling(20).mean()) /
    (btc_daily['Volume'].rolling(20).std() + eps)
)
"""
# Supprimer les lignes avec NaN (les premières lignes des rolling)
btc_daily.dropna(inplace=True)

# ---------------------------
# ↪️ INSÈRE ICI : LSTM (entre préparation des features et l'évaluation)
# ---------------------------
from sklearn.preprocessing import StandardScaler
from keras.losses import BinaryFocalCrossentropy



# Paramètres
WINDOW_SIZE = 14          # nombre de jours pour la séquence (tu peux ajuster)
BATCH_SIZE = 32
EPOCHS = 100
THRESHOLD = 0.789         # seuil de classification (tu avais 0.789)
SPLIT_DATE = '2023-05-01' # si tu veux utiliser un split temporel fixe

# 1) Split train/test temporel (garantit qu'on n'utilise pas le futur)
# Si tu as déjà un split ailleurs, ajuste cette partie en conséquence.
train = btc_daily[btc_daily.index < SPLIT_DATE]
test = btc_daily[btc_daily.index >= SPLIT_DATE]

X_train_df = train.drop(columns=["Target"])
X_test_df  = test.drop(columns=["Target"])
y_train_ser = train["Target"]
y_test_ser  = test["Target"]

# 2) Normalisation (fit uniquement sur le train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled  = scaler.transform(X_test_df)

# 3) Fonction pour créer des séquences (X_seq, y_seq)
def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:(i + window)])
        # label ciblé = la cible juste après la fenêtre (prévoir le futur)
        ys.append(y.iloc[i + window])
    return np.array(Xs), np.array(ys)

# 4) Créer séquences pour train et test
X_train_seq, y_train_seq = create_sequences(pd.DataFrame(X_train_df), y_train_ser.reset_index(drop=True), WINDOW_SIZE)
X_test_seq,  y_test_seq  = create_sequences(pd.DataFrame(X_test_df),  y_test_ser.reset_index(drop=True),  WINDOW_SIZE)

# 5) Si nécessaire : remettre la dimension features (timesteps, features)
# Ici X_train_seq shape = (samples, window, n_features) (n_features = nb de colonnes de X)
print(f"Sequences shapes -> X_train: {X_train_seq.shape}, y_train: {y_train_seq.shape}, X_test: {X_test_seq.shape}, y_test: {y_test_seq.shape}")

# 6) Construire le modèle LSTM
tf.keras.backend.clear_session()

inputs = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))  # (window, n_features)
x = LSTM(64, return_sequences=True)(inputs)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = LSTM(64, return_sequences=True)(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = LSTM(32)(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inputs, outputs=outputs)

# 7) Compilation avec focal loss (tensorflow-addons)
#loss_fn = SigmoidFocalCrossEntropy(from_logits=False, reduction=tf.keras.losses.Reduction.AUTO)
loss=BinaryFocalCrossentropy()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=loss,
              metrics=["accuracy"])

# 8) Callbacks et class_weight (mettre plus de poids sur la classe 0 = "baisse")
early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
class_weight = {0: 1.0, 1: 4.0}  # renforce apprentissage de la classe minoritaire (baisse)

# 9) Entraînement
print("⚡ Entraînement LSTM...")
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_test_seq, y_test_seq),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    class_weight=class_weight,
    verbose=2
)

# 10) Remplacer l'appel plus loin : on prédit sur X_test_seq (séquences)
# (Le reste de ta pipeline d'évaluation utilise `y_test` et `model.predict(X_test)` ;
# ici on remplace par:)
y_pred_proba = model.predict(X_test_seq)
y_pred = (y_pred_proba > THRESHOLD).astype(int)

# Pour la suite, remplacer y_test par y_test_seq si besoin (alignement temporel)
# On réassigne pour que la suite du script fonctionne tel quel
X_test = X_test_seq
y_test = y_test_seq
# ---------------------------
# Fin du bloc LSTM
# ---------------------------



# 5. Évaluation
print("📊 Évaluation...")
y_pred_proba = model.predict(X_test)
threshold = 0.789  # seuil fixe pour la classification
y_pred = (y_pred_proba > threshold).astype(int)

report = classification_report(y_test, y_pred, digits=4)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# 6. Sauvegarder prédiction finale (0 ou 1) + log
final_prediction = int(y_pred[-1][0])  # dernière valeur prédite
output_file = f"/home/elias/PROJECT/AICryptoPredictor/results/prediction_{datetime.now().strftime('%Y%m%d_%H')}.txt"
last_date = btc_daily.index[-1].strftime("%Y-%m-%d")

os.makedirs("results", exist_ok=True)
with open(output_file, "w") as f:
    f.write(f"Dernière date du DataSet: {last_date}\n\n")
    f.write("Resultat de la pipeline 6 \n\n")
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
with open(signal_file, "w") as f:
    f.write(str(final_prediction))
    f.write("\n")
    f.write(str(btc_daily['Close'].iloc[-1] * POURCENT_MARGE))


