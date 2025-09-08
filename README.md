# AICryptoPredictor
**Auteur :** Elias Khallouk

**Date de début :** 30/08/2025  
**Dernière mise à jour :** 08/09/2025  

## Description du projet

**AICryptoPredictor** est un projet d’IA appliqué au trading de cryptomonnaies (BTC/USDT).  
L’objectif est de prédire si le prix du Bitcoin augmentera d’au moins **+0.75% dans les 1 ou 2 prochains jours**, et d’automatiser un trade d’achat/vente basé sur cette prédiction.  

Le projet se compose de plusieurs parties :  
- **Collecte de données** depuis un dataset via Kaggle.
- **Pipeline de machine learning** (préprocessing, features techniques, réseau de neurones).  
- **AutoTrader** : script de trading automatisé connecté à l’API Binance.  
- **AutoMail** : notification par email des résultats et du statut du bot. 
- **Exécution planifiée avec crontab** pour mettre à jour les données, entraîner le modèle, générer une prédiction régulière et envoyer le rapport par mail.  

---



## ⚙️ Installation

### 1. Cloner le projet
```bash
git clone https://github.com/EliasKhallouk/AICryptoPredictor.git
cd AICryptoPredictor
```

### 2. Créer un environnement conda
```bash
conda create -n crypto_predictor python=3.10
conda activate crypto_predictor
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```



## 📂 Structure du projet
```bash
AICryptoPredictor/
│
├── data/                  # Données brutes et datasets
├── results/               # Résultats des prédictions et logs
├── notebooks/             # Scripts principaux
│   ├── update_data.py     # Télécharge/Met à jour le dataset
│   ├── pipeline2.py       # Entraîne et évalue le modèle
│   ├── AutoTrader.py      # Lance le trade auto via Binance API
│   ├── AutoMail.py        # Envoi de mails des résultats
│
├── .env                   # Clés API Binance et config (NE PAS PARTAGER)
├── requirements.txt
├── README.md
```



## 🤖 Utilisation

### 1. Configuration .env
À la racine du projet, créer un fichier .env :
```bash
API_KEY=ta_cle_api_binance
API_SECRET=ton_secret_binance
EMAIL_PASSWORD = mot_de_passe_application_Gmail 
EMAIL_ENV = "email_envoyeur@email.com"
EMAIL_DEST = "email_destinataire@email.com"
```

### 2. Mettre à jour les données
```bash
cd ~
python ./PROJECT/AICryptoPredictor/notebooks/update_data.py
```

### 3. Lancer la pipeline IA
```bash 
python notebooks/pipeline2.py
```
Cela génère un fichier results/prediction_YYYYMMDD_HH.txt contenant :
- Dernière date utilisée
- Prédiction finale (0 = pas d’achat / 1 = signal d’achat)
- Rapport de classification

Cela génère/modifie un fichier Output/signal.txt contenant :
- Prédiction finale (0 = pas d’achat / 1 = signal d’achat)

### 4. Lancer le trader
```bash
python notebooks/AutoTrader.py
```
Cela lit la dernière prédiction et exécute une action.
Le solde est aussi enregistré dans le fichier results/prediction_YYYYMMDD_HH.txt.

### 5. Envoi d'email
```bash
python notebooks/AutoMail.py
```



## 📅 Automatisation avec Crontab
Pour exécuter automatiquement les scripts, ajouter la config suivante avec :
```bash
crontab -e
```

Exemple :
```bash
15 2 * * * /home/elias/anaconda3/envs/crypto_predictor/bin/python /home/elias/PROJECT/AICryptoPredictor/notebooks/update_data.py >> /home/elias/PROJECT/AICryptoPredictor/Crontab/cron_update.log 2>&1

17 2 * * * /home/elias/anaconda3/envs/crypto_predictor/bin/python /home/elias/PROJECT/AICryptoPredictor/notebooks/pipeline2.py >> /home/elias/PROJECT/AICryptoPredictor/Crontab/cron_pipeline2.log 2>&1

26 2 * * * /home/elias/anaconda3/envs/crypto_predictor/bin/python /home/elias/PROJECT/AICryptoPredictor/notebooks/AutoTrader2.py >> /home/elias/PROJECT/AICryptoPredictor/Crontab/cron_trader2.log 2>&1

28 2 * * * /home/elias/anaconda3/envs/crypto_predictor/bin/python /home/elias/PROJECT/AICryptoPredictor/notebooks/AutoMail.py >> /home/elias/PROJECT/AICryptoPredictor/Crontab/cron_mail.log 2>&1

```

Ici, chaque jour à 2h du matin :

- 02h15 → mise à jour dataset

- 02h17 → entraînement + prédiction IA

- 02h26 → exécution d’un trade (selon le signal)

- 02h28 → envoi d’un email récapitulatif


⚠️ Attention : il faut changer tout les chemins absolue qui sont dans mon code et les remplacer par les votres