import argparse
import os
import sys
import numpy as np
import pandas as pd
import onnxruntime as ort

def build_features(df):
    # Attendu: colonnes de base
    required = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante dans le CSV: {c}")

    # Convertir Timestamp en datetime si nécessaire
    if not np.issubdtype(df["Timestamp"].dtype, np.datetime64):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s', errors='ignore')
        # Si la conversion ci-dessus échoue (ex: déjà en datetime), on retente sans unit
        if not np.issubdtype(df["Timestamp"].dtype, np.datetime64):
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
    df = df.set_index("Timestamp").sort_index()

    # Resample journalier (comme pipeline2)
    daily = df.resample('1D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # 8 features dérivées pour compléter à 14 colonnes au total
    daily["Return"] = daily["Close"].pct_change()
    daily["MA7"] = daily["Close"].rolling(7).mean()
    daily["MA30"] = daily["Close"].rolling(30).mean()
    daily["Volatility"] = daily["Return"].rolling(7).std()

    # RSI 14 jours
    delta = daily["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    daily["RSI"] = 100 - (100 / (1 + rs))

    # MACD + Signal
    ema12 = daily["Close"].ewm(span=12, adjust=False).mean()
    ema26 = daily["Close"].ewm(span=26, adjust=False).mean()
    daily["MACD"] = ema12 - ema26
    daily["Signal"] = daily["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands 20
    ma20 = daily["Close"].rolling(20).mean()
    std20 = daily["Close"].rolling(20).std()
    daily["Bollinger_Upper"] = ma20 + 2 * std20
    daily["Bollinger_Lower"] = ma20 - 2 * std20

    # Conserver exactement 14 colonnes dans l’ordre attendu par le modèle
    cols_14 = [
        "Open", "High", "Low", "Close", "Volume",
        "Return", "MA7", "MA30", "Volatility", "RSI",
        "MACD", "Signal", "Bollinger_Upper", "Bollinger_Lower"
    ]
    daily = daily[cols_14].dropna()
    return daily

def main():
    ap = argparse.ArgumentParser(description="Run ONNX inference from CSV with 6 base columns and 14 computed features")
    ap.add_argument("--model", default="/home/elias/PROJECT/AICryptoPredictor/results/pipeline2.onnx",
                    help="Chemin du fichier ONNX")
    ap.add_argument("--csv", required=True,
                    help="Chemin du CSV (columns: Timestamp, Open, High, Low, Close, Volume)")
    ap.add_argument("--input_name", default="input",
                    help="Nom du tensor d'entrée dans le modèle ONNX (par défaut: input)")
    ap.add_argument("--last_only", action="store_true",
                    help="N'inférer que la dernière ligne (sinon infère tout le batch)")
    ap.add_argument("--threshold", type=float, default=0.789,
                    help="Seuil de classification binaire")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"Model non trouvé: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.csv):
        print(f"CSV non trouvé: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    features = build_features(df)

    if features.empty:
        print("Pas de lignes valides après calcul des features (rolling entraîne des NaN). Fournis plus de données.", file=sys.stderr)
        sys.exit(1)

    if args.last_only:
        x = features.tail(1).to_numpy(dtype=np.float32)
        idx = [features.index[-1]]
    else:
        x = features.to_numpy(dtype=np.float32)
        idx = list(features.index)

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    outputs = sess.run(None, {args.input_name: x})
    prob = np.array(outputs[0], dtype=np.float32).reshape(-1)

    print("Dates:", [str(i) for i in idx])
    print("Probabilités:", prob.tolist())

    preds = (prob > args.threshold).astype(int)
    print("Prédictions (seuil {:.3f}):".format(args.threshold), preds.tolist())

if __name__ == "__main__":
    main()