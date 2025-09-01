import os
import subprocess
from datetime import datetime

# Nom du dataset Kaggle
DATASET = "mczielinski/bitcoin-historical-data"
LOCAL_PATH = "./PROJECT/AICryptoPredictor/data"

def update_dataset():
    print(f"[{datetime.now()}] Téléchargement du dataset...")
    # Télécharge uniquement si il y a une nouvelle version
    subprocess.run([
        "/home/elias/anaconda3/envs/crypto_predictor/bin/kaggle", "datasets", "download", "-d", DATASET,
        "-p", LOCAL_PATH, "--unzip", "--force"
    ])
    print("✅ Dataset mis à jour dans /data")

if __name__ == "__main__":
    update_dataset()
