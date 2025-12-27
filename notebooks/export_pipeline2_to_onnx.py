# filepath: export_pipeline2_to_onnx.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

import tf2onnx

def build_and_train_model(X_train, y_train, X_val, y_val, seed=73):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    input_dim = X_train.shape[1]
    inputs = Input(shape=(input_dim,))

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

    model.fit(
        X_train, y_train,
        epochs=100, batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        class_weight={0: 1, 1: 2}
    )
    return model

def create_features_and_target(btc_daily, margin=1.0075):
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
    btc_daily['Bollinger_Upper'] = ma20 + 2 * std20
    btc_daily['Bollinger_Lower'] = ma20 - 2 * std20

    btc_daily["Target"] = (
        ((btc_daily["High"].shift(-1) > (btc_daily["Close"] * margin)) |
         (btc_daily["High"].shift(-2) > (btc_daily["Close"] * margin)))
    ).astype(int)

    eps = 1e-9
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

    btc_daily.dropna(inplace=True)
    return btc_daily

def load_and_prepare_data(csv_path, split_date='2023-05-01'):
    btc = pd.read_csv(csv_path)
    btc['Timestamp'] = pd.to_datetime(btc['Timestamp'], unit='s')
    btc.set_index('Timestamp', inplace=True)

    btc_daily = btc.resample('1D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    btc_daily = create_features_and_target(btc_daily)

    train = btc_daily[btc_daily.index < split_date]
    test = btc_daily[btc_daily.index >= split_date]

    X_train_full = train.drop(columns=["Target"])
    y_train_full = train["Target"]
    X_test = test.drop(columns=["Target"])
    y_test = test["Target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val, X_test, y_test, btc_daily

def export_to_onnx(model, sample_input, onnx_path):
    onnx_model, _ = tf2onnx.convert.from_keras(
        model, input_signature=[tf.TensorSpec(sample_input.shape, tf.float32)],
        opset=13
    )
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

def main():
    data_path = "/home/elias/PROJECT/AICryptoPredictor/data/btcusd_1-min_data.csv"
    onnx_output = "/home/elias/PROJECT/AICryptoPredictor/results/pipeline2_model.onnx"
    os.makedirs(os.path.dirname(onnx_output), exist_ok=True)

    X_train, X_val, y_train, y_val, X_test, y_test, btc_daily = load_and_prepare_data(data_path)

    model = build_and_train_model(X_train, y_train, X_val, y_val)

    # Evaluate and produce a prediction file similar to pipeline2.py
    y_pred_proba = model.predict(X_test)
    threshold = 0.789
    y_pred = (y_pred_proba > threshold).astype(int)

    report = classification_report(y_test, y_pred, digits=4)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    POURCENT_MARGE = 1.0075
    final_prediction = int(y_pred[-1][0])
    last_date = btc_daily.index[-1].strftime("%Y-%m-%d")
    output_file = f"/home/elias/PROJECT/AICryptoPredictor/results/prediction_{datetime.now().strftime('%Y%m%d_%H')}.txt"
    with open(output_file, "w") as f:
        f.write(f"Dernière date du DataSet: {last_date}\n\n")
        f.write("Resultat de la pipeline 2 (ONNX export)\n\n")
        f.write(f"Augmentation prévue de: {POURCENT_MARGE*100}% \n\n")
        f.write(f"Prediction finale (0 ou 1): {final_prediction}\n")
        f.write(f"Prix de cloture du {last_date} est à {btc_daily['Close'].iloc[-1]}\n")
        f.write(f"Prix du High du lendemain est estimé à {btc_daily['Close'].iloc[-1] * POURCENT_MARGE}\n\n")
        f.write("Rapport de classification:\n")
        f.write(report + "\n")
        f.write(f"Precision classe 1: {precision}\n")
        f.write(f"Recall classe 1: {recall}\n")

    # Export trained Keras model to ONNX
    sample_input = tf.convert_to_tensor(X_test.iloc[:1].astype(np.float32).values)
    export_to_onnx(model, sample_input, onnx_output)

    # Also write signal file as in original script
    signal_file = "/home/elias/PROJECT/AICryptoPredictor/Output/signal.txt"
    os.makedirs(os.path.dirname(signal_file), exist_ok=True)
    with open(signal_file, "w") as f:
        f.write(str(final_prediction))
        f.write("\n")
        f.write(str(btc_daily['Close'].iloc[-1] * POURCENT_MARGE))

if __name__ == "__main__":
    main()