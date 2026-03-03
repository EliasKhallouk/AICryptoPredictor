import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import onnxruntime as ort

# Optionnel pour le rapport de classification
try:
    from sklearn.metrics import classification_report, precision_score, recall_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

DATA_CSV = "./data/btcusd_1-min_data.csv"
SIGNAL_OUT = "/home/elias/PROJECT/AICryptoPredictor/Output/signal.txt"
RESULTS_DIR = "/home/elias/PROJECT/AICryptoPredictor/results"
POURCENT_MARGE = 1.0075
THRESHOLD = 0.789
SPLIT_DATE = "2023-05-01"  # aligner avec pipeline2.py

def build_features_and_target(df):
    required = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante dans le CSV: {c}")

    # Timestamp -> datetime
    if not np.issubdtype(df["Timestamp"].dtype, np.datetime64):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s", errors="coerce")
        if not np.issubdtype(df["Timestamp"].dtype, np.datetime64):
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df.set_index("Timestamp").sort_index()

    # Journalier comme pipeline2
    daily = df.resample("1D").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    # Features (14 au total, comme pipeline2)
    daily["Return"] = daily["Close"].pct_change()
    daily["MA7"] = daily["Close"].rolling(7).mean()
    daily["MA30"] = daily["Close"].rolling(30).mean()
    daily["Volatility"] = daily["Return"].rolling(7).std()

    delta = daily["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    daily["RSI"] = 100 - (100 / (1 + rs))

    ema12 = daily["Close"].ewm(span=12, adjust=False).mean()
    ema26 = daily["Close"].ewm(span=26, adjust=False).mean()
    daily["MACD"] = ema12 - ema26
    daily["Signal"] = daily["MACD"].ewm(span=9, adjust=False).mean()

    ma20 = daily["Close"].rolling(20).mean()
    std20 = daily["Close"].rolling(20).std()
    daily["Bollinger_Upper"] = ma20 + 2 * std20
    daily["Bollinger_Lower"] = ma20 - 2 * std20

    # === Extras identiques à l'export ===
    eps = 1e-9
    prev_close = daily["Close"].shift(1)
    tr1 = daily["High"] - daily["Low"]
    tr2 = (daily["High"] - prev_close).abs()
    tr3 = (daily["Low"] - prev_close).abs()
    daily["TrueRange"] = np.max(np.vstack([tr1.values, tr2.values, tr3.values]), axis=0)
    daily["ATR14"] = daily["TrueRange"].rolling(14).mean()
    daily["MACD_hist"] = daily["MACD"] - daily["Signal"]
    daily["Volume_z20"] = (
        (daily["Volume"] - daily["Volume"].rolling(20).mean()) /
        (daily["Volume"].rolling(20).std() + eps)
    )

    # Target comme pipeline2/export
    target = (
        (daily["High"].shift(-1) > (daily["Close"] * POURCENT_MARGE)) |
        (daily["High"].shift(-2) > (daily["Close"] * POURCENT_MARGE))
    ).astype(int)

    # Colonnes EXACTES utilisées à l'entraînement ONNX (ordre identique)
    cols_export = [
        "Open", "High", "Low", "Close", "Volume",
        "Return", "MA7", "MA30", "Volatility", "RSI",
        "MACD", "Signal", "Bollinger_Upper", "Bollinger_Lower",
        "TrueRange", "ATR14", "MACD_hist", "Volume_z20"
    ]

    daily = daily.dropna()
    X = daily[cols_export]
    y = target.reindex(X.index)

    return X, y, daily

def ensure_features_count(df_features, expected_feat):
    # Avec l'alignement ci-dessus, cette fonction devient une sécurité pour ONNX
    if expected_feat is not None:
        if df_features.shape[1] > expected_feat:
            df_features = df_features.iloc[:, :expected_feat]
        elif df_features.shape[1] < expected_feat:
            # si l'ONNX attend moins de colonnes (ex: 14), tronquer la liste aux 14 premières
            df_features = df_features.iloc[:, :expected_feat]
    return df_features

def run_inference(sess, input_name, X_np, expected_batch):
    # Si le modèle impose batch=1, boucle sur les lignes
    if expected_batch == 1 and X_np.shape[0] > 1:
        outs = []
        for i in range(X_np.shape[0]):
            o = sess.run(None, {input_name: X_np[i:i+1]})[0]
            outs.append(o)
        return np.vstack(outs)
    return sess.run(None, {input_name: X_np})[0]

def main():
    model_path = "./results/pipeline2_model.onnx"
    input_name_pref = "input"

    if not os.path.exists(model_path):
        print(f"Model non trouvé: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(DATA_CSV):
        print(f"CSV non trouvé: {DATA_CSV}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(SIGNAL_OUT), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    X_all, y_all, daily = build_features_and_target(df)
    if X_all.empty:
        print("Pas de lignes valides après calcul des features.", file=sys.stderr)
        sys.exit(1)

    # Session ONNX
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Détection du nom d'entrée et formes
    model_input = sess.get_inputs()[0]
    input_name = model_input.name if input_name_pref not in [inp.name for inp in sess.get_inputs()] else input_name_pref
    expected_shape = model_input.shape  # exemple: [1, 18] ou [None, 14]
    expected_batch = expected_shape[0] if isinstance(expected_shape[0], int) else None
    expected_feat = expected_shape[1] if isinstance(expected_shape[1], int) else None

    # Adapter le nombre de features (principalement tronquer si nécessaire)
    X_all = ensure_features_count(X_all, expected_feat)

    # Split identique à pipeline2/export pour rapport
    X_test = X_all[X_all.index >= SPLIT_DATE]
    y_test = y_all[y_all.index >= SPLIT_DATE]
    # fallback si vide
    if X_test.empty or y_test.empty:
        X_test = X_all
        y_test = y_all

    # Numpy (pour tout et pour test)
    X_np_all = X_all.to_numpy(dtype=np.float32)
    X_np_test = X_test.to_numpy(dtype=np.float32)

    # Probabilités pour le TEST (pour rapport comparable à pipeline2.py)
    probs_test = run_inference(sess, input_name, X_np_test, expected_batch).astype(np.float32).reshape(-1)
    preds_test = (probs_test > THRESHOLD).astype(int)

    # Dernière ligne pour le signal et le fichier prediction (inchangé)
    last_row = X_np_all[-1:].astype(np.float32)
    prob_last = float(run_inference(sess, input_name, last_row, expected_batch).reshape(-1)[0])
    pred_last = int(prob_last > THRESHOLD)

    # Dernières infos (format identique à pipeline2.py)
    last_date = X_all.index[-1].strftime("%Y-%m-%d")
    last_close = float(daily.loc[X_all.index[-1], "Close"])
    next_high_est = last_close * POURCENT_MARGE

    # Écrire le signal (identique au format de pipeline2.py)
    with open(SIGNAL_OUT, "w") as f:
        f.write(str(pred_last))
        f.write("\n")
        f.write(str(next_high_est))

    # Rapport de classification aligné sur le split
    if SKLEARN_OK and len(y_test) == len(preds_test) and len(y_test) > 0:
        report = classification_report(y_test, preds_test, digits=4)
        precision = precision_score(y_test, preds_test, pos_label=1, zero_division=0)
        recall = recall_score(y_test, preds_test, pos_label=1, zero_division=0)
    else:
        report = "sklearn non disponible ou tailles incohérentes.\n"
        precision = 0.0
        recall = 0.0

    ts = datetime.now().strftime("%Y%m%d_%H")
    output_file = os.path.join(RESULTS_DIR, f"prediction_{ts}.txt")
    with open(output_file, "w") as f:
        f.write(f"Dernière date du DataSet: {last_date}\n\n")
        f.write("Resultat de la pipeline 2 \n\n")
        f.write(f"Augmentation prévue de: {POURCENT_MARGE*100}% \n\n")
        f.write(f"Prediction finale (0 ou 1): {pred_last}\n")
        f.write(f"Prix de cloture du {last_date} est à {last_close}\n")
        f.write(f"Prix du High du lendemain est estimé à {next_high_est}\n\n")
        f.write("Rapport de classification:\n")
        f.write(report + "\n")
        f.write(f"Precision classe 1: {precision}\n")
        f.write(f"Recall classe 1: {recall}\n")

    print(f"Signal écrit: {SIGNAL_OUT}")
    print(f"Rapport écrit: {output_file}")

if __name__ == "__main__":
    main()