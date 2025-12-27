import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import onnxruntime as ort

# Optionnel pour le rapport de classification
try:
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

DATA_CSV = "./data/btcusdt_1-min_data.csv"
SIGNAL_OUT = "./Output/signal.txt"
RESULTS_DIR = "./results"

POURCENT_MARGE = 1.0075

def build_features_and_target(df):
    required = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante dans le CSV: {c}")

    # Timestamp -> datetime (unix seconds ou ISO)
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

    # Features (14 au total)
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

    # Target comme pipeline2
    target = (
        (daily["High"].shift(-1) > (daily["Close"] * POURCENT_MARGE)) |
        (daily["High"].shift(-2) > (daily["Close"] * POURCENT_MARGE))
    ).astype(int)

    cols_14 = [
        "Open", "High", "Low", "Close", "Volume",
        "Return", "MA7", "MA30", "Volatility", "RSI",
        "MACD", "Signal", "Bollinger_Upper", "Bollinger_Lower"
    ]
    X = daily[cols_14].dropna()

    # Aligner y sur X et retirer les NaN dus aux shifts
    y = target.reindex(X.index).dropna()
    X = X.loc[y.index]

    return X, y

def ensure_features_count(df_features, expected_feat):
    # Ajouts cohérents si le modèle attend >14 (ex: 18)
    if df_features.shape[1] < expected_feat:
        close = df_features["Close"]
        open_ = df_features["Open"]
        eps = 1e-9

        if "MACD" in df_features.columns and "Signal" in df_features.columns and "MACD_hist" not in df_features.columns:
            df_features["MACD_hist"] = df_features["MACD"] - df_features["Signal"]

        rng = (df_features["High"] - df_features["Low"]).replace(0, eps)
        if "Range_pct" not in df_features.columns:
            df_features["Range_pct"] = (df_features["High"] - df_features["Low"]) / (close + eps)
        if "Position_in_range" not in df_features.columns:
            df_features["Position_in_range"] = (close - df_features["Low"]) / (rng + eps)

        body = (close - open_).abs()
        if "Body_pct" not in df_features.columns:
            df_features["Body_pct"] = body / (rng + eps)

        df_features = df_features.dropna()

    if df_features.shape[1] > expected_feat:
        df_features = df_features.iloc[:, :expected_feat]

    return df_features

def run_inference(sess, input_name, X_np, expected_batch):
    # Si le modèle impose batch=1, on boucle
    if expected_batch == 1 and X_np.shape[0] > 1:
        outs = []
        for i in range(X_np.shape[0]):
            o = sess.run(None, {input_name: X_np[i:i+1]})[0]
            outs.append(o)
        return np.vstack(outs)
    return sess.run(None, {input_name: X_np})[0]

def main():
    ap = argparse.ArgumentParser(description="Run ONNX pipeline (fixed CSV path) and write outputs")
    ap.add_argument("--model", default="./results/pipeline2_model.onnx", help="Chemin du fichier ONNX")
    ap.add_argument("--input_name", default="input", help="Nom du tensor d'entrée")
    ap.add_argument("--threshold", type=float, default=0.789, help="Seuil de classification")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"Model non trouvé: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(DATA_CSV):
        print(f"CSV non trouvé: {DATA_CSV}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(SIGNAL_OUT), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    X_all, y_all = build_features_and_target(df)

    if X_all.empty:
        print("Pas de lignes valides après calcul des features.", file=sys.stderr)
        sys.exit(1)

    # Session ONNX
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    # Détection du nom d'entrée et formes
    model_input = sess.get_inputs()[0]
    if args.input_name != model_input.name:
        print(f"Info: input_name '{args.input_name}' introuvable. Utilisation de '{model_input.name}'.")
        args.input_name = model_input.name

    expected_shape = model_input.shape  # ex: [1, 18] ou [None, 14]
    expected_batch = expected_shape[0] if isinstance(expected_shape[0], int) else None
    expected_feat = expected_shape[1] if isinstance(expected_shape[1], int) else None

    # Adapter le nombre de features si nécessaire
    if expected_feat is not None:
        X_all = ensure_features_count(X_all, expected_feat)

    # X pour rapport (aligné à y_all)
    X_report = X_all.loc[y_all.index]
    X_report_np = X_report.to_numpy(dtype=np.float32)

    # X pour le signal (dernière ligne dispo)
    last_row = X_all.tail(1).to_numpy(dtype=np.float32)

    # Ajuster colonnes (pad/tronc) si besoin
    def adjust_features(x):
        if expected_feat is not None:
            if x.shape[1] < expected_feat:
                pad = np.zeros((x.shape[0], expected_feat - x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)
            elif x.shape[1] > expected_feat:
                x = x[:, :expected_feat]
        return x

    X_report_np = adjust_features(X_report_np)
    last_row = adjust_features(last_row)

    # Inference
    probs_report = run_inference(sess, args.input_name, X_report_np, expected_batch)
    probs_report = probs_report.astype(np.float32).reshape(-1)
    preds_report = (probs_report > args.threshold).astype(int)

    probs_last = run_inference(sess, args.input_name, last_row, expected_batch)
    prob_last = float(np.array(probs_last).reshape(-1)[0])
    pred_last = int(prob_last > args.threshold)
    last_date = str(X_all.index[-1])

    # Ecrire le signal
    with open(SIGNAL_OUT, "w", encoding="utf-8") as f:
        f.write(f"date={last_date}\n")
        f.write(f"prob={prob_last:.6f}\n")
        f.write(f"threshold={args.threshold:.6f}\n")
        f.write(f"signal={pred_last}\n")

    # Rapport de classification (si sklearn dispo et y_all non vide)
    if SKLEARN_OK and len(y_all) > 0:
        y_true = y_all.to_numpy(dtype=int)
        acc = accuracy_score(y_true, preds_report)
        prec1 = precision_score(y_true, preds_report, pos_label=1, zero_division=0)
        rec1 = recall_score(y_true, preds_report, pos_label=1, zero_division=0)
        report = classification_report(y_true, preds_report, digits=4)

        ts = datetime.now().strftime("%Y%m%d_%H")
        report_path = os.path.join(RESULTS_DIR, f"prediction_{ts}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Dernière date du DataSet: {last_date}\n\n")
            f.write("Resultat de la pipeline 2\n\n")
            f.write(f"Probabilité dernière ligne: {prob_last:.6f}\n")
            f.write(f"Prediction finale (0 ou 1): {pred_last}\n\n")
            f.write("Rapport de classification:\n")
            f.write(report)
            f.write("\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Precision classe 1: {prec1:.6f}\n")
            f.write(f"Recall classe 1: {rec1:.6f}\n")
    else:
        print("Info: sklearn non disponible, rapport de classification non généré.")

if __name__ == "__main__":
    main()