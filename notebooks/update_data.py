"""import os
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
"""

import os
import subprocess
from datetime import datetime

# Nom du dataset Kaggle
DATASET = "mczielinski/bitcoin-historical-data"
LOCAL_PATH = "./PROJECT/AICryptoPredictor/data"

def update_dataset():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # suffixe unique basé sur la date/heure
    dataset_folder = os.path.join(LOCAL_PATH, f"bitcoin_dataset_{timestamp}")
    
    os.makedirs(dataset_folder, exist_ok=True)
    
    print(f"[{datetime.now()}] Téléchargement du dataset...")
    subprocess.run([
        "/home/elias/anaconda3/envs/crypto_predictor/bin/kaggle", "datasets", "download", "-d", DATASET,
        "-p", dataset_folder, "--unzip"
    ])
    print(f"✅ Nouveau dataset téléchargé dans {dataset_folder}")

if __name__ == "__main__":
    update_dataset()
