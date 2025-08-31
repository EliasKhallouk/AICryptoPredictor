import os
import subprocess
from datetime import datetime

# Nom du dataset Kaggle
DATASET = "mczielinski/bitcoin-historical-data"
LOCAL_PATH = "./PROJECT/AICryptoPredictor/data/btcusd_1-min_data.csv"

def update_dataset():
    print(f"[{datetime.now()}] Téléchargement du dataset...")
    # Télécharge uniquement si il y a une nouvelle version
    subprocess.run([
        "kaggle", "datasets", "download", "-d", DATASET,
        "-p", "data", "--unzip", "--force"
    ])
    print("✅ Dataset mis à jour dans /data")

if __name__ == "__main__":
    update_dataset()
