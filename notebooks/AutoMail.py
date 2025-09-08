import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# 📧 Paramètres de l'email
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)

SMTP_SERVER = "smtp.gmail.com" 
SMTP_PORT = 587
EMAIL_ADDRESS = os.getenv("EMAIL_ENV")
EMAIL_PASSWORD =  os.getenv("EMAIL_PASSWORD") 

#TO_ADDRESSES = [os.getenv("EMAIL_ENV"), os.getenv("EMAIL_DEST")]
TO_ADDRESSES = [os.getenv("EMAIL_ENV")]

# 📄 Fichier de résultat généré par ton modèle
result_file = f"/home/elias/PROJECT/AICryptoPredictor/results/prediction_{datetime.now().strftime('%Y%m%d_%H')}.txt"

# Lire le contenu du fichier de résultat
with open(result_file, "r") as f:
    result_content = f.read()

# Créer le message
msg = MIMEMultipart()
msg['From'] = EMAIL_ADDRESS
#msg['To'] = TO_ADDRESS
msg['To'] = ", ".join(TO_ADDRESSES)
msg['Subject'] = "Résultats de prédiction BTC"
msg.attach(MIMEText(result_content, 'plain'))

# Envoyer l'email
try:
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()  # sécurise la connexion
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    server.send_message(msg)
    server.quit()
    print("✅ Email envoyé avec succès !")
except Exception as e:
    print(f"❌ Erreur lors de l'envoi de l'email : {e}")
