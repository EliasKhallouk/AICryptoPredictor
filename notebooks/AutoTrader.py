from binance.client import Client
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

#CONNEXION A BINANCE
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)

"""API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

client = Client(API_KEY, API_SECRET)"""

API_KEY_TEST = "wXklSWGHk3qvzrrXUcMzqvJNhNT589qYkkFfwq83z92GtT3PGwsyPR7gopIkgvqU"
API_SECRET_TEST = "DrBnrFW51syCtBwFZWZdCicdC7T7Ri06s6NiYUUcxJa7xBr7dOZzA6SCDhnfc0bZ"

client = Client(API_KEY_TEST, API_SECRET_TEST, testnet=True)

# RÉCUPÉRER LE SOLDE DU COMPTE
account_info = client.get_account()

def Afficher_solde():
    f.write("\n=== SOLDES DU COMPTE ===\n")
    """for balance in account_info['balances']:
        asset = balance['asset']
        free = float(balance['free'])
        locked = float(balance['locked'])
        if free > 0 or locked > 0:
            f.write(f"{asset} | Disponible: {free} | Bloqué: {locked}\n")"""

    # Exemple : solde uniquement en USDT et BTC
    usdt_balance = client.get_asset_balance(asset='USDT')
    btc_balance = client.get_asset_balance(asset='BTC')

    f.write(f"USDT: {usdt_balance}\n")
    f.write(f"BTC : {btc_balance}\n")
    f.write("-" * 30)


def Lecture_signal(info="buy_signal"):
    signal_file = "/home/elias/PROJECT/AICryptoPredictor/results/signal.txt"
    with open(signal_file, "r") as f:
        lines = f.readlines()
        if len(lines) < 2:
            f.write("❌ signal.txt ne contient pas de prix de vente sur la 2ème ligne.\n")
            return
        if info == "sell_price":
            return float(lines[1].strip())
        return float(lines[0].strip())


def Buy_all_BTC():
    if Lecture_signal() == 1:
        f.write("🚀 Signal d'achat détecté !\n")

        # Récupérer solde USDT disponible
        usdt_balance = client.get_asset_balance(asset='USDT')
        usdt_free = float(usdt_balance['free'])

        if usdt_free < 10:  # Minimum requis pour trader
            f.write(f"⚠️ Solde insuffisant en USDT ({usdt_free}). Pas d'achat.\n")
            return

        try:
            # Récupérer le prix du BTC en USDT
            ticker = client.get_symbol_ticker(symbol="BTCUSDT")
            btc_price = float(ticker['price'])

            # Calculer la quantité de BTC achetable
            qty = usdt_free / btc_price

            # Adapter quantité au pas minimum autorisé (par ex. 0.00001 BTC)
            qty = round(qty, 5)

            # Passer l'ordre d'achat
            order = client.order_market_buy(
                symbol="BTCUSDT",
                quantity=qty
            )

            f.write(f"✅ Ordre d'achat exécuté: {order}\n")

        except Exception as e:
            f.write(f"❌ Erreur lors de l'achat: {e}\n")

    else:
        f.write("❌ Pas d'achat, rester en position.\n")


def Sell_all_BTC():
    """Vend tout le solde BTC disponible contre USDT en LIMIT au prix indiqué dans signal.txt"""
    try:
        # Lire le prix de vente depuis signal.txt (2ème ligne)
        sell_price = Lecture_signal("sell_price")
        f.write(f"📈 Prix de vente cible: {sell_price} USDT\n")
        if not sell_price or sell_price <= 0:
            f.write("❌ Prix de vente invalide dans signal.txt.\n")
            return
        f.write("🚀 Tentative de vente de tout le BTC disponible...\n")
    
        # Récupérer solde BTC disponible
        btc_balance = client.get_asset_balance(asset='BTC')
        btc_free = float(btc_balance['free'])

        if btc_free < 0.00001:  # seuil minimum de trading
            f.write(f"⚠️ Solde BTC insuffisant ({btc_free}). Pas de vente.\n")
            return

        # Adapter au pas minimal (ex: 0.00001 BTC)
        qty = round(btc_free, 5)

        # Passer l'ordre LIMIT
        order = client.order_limit_sell(
            symbol="BTCUSDT",
            quantity=qty,
            price=str(sell_price)  # Binance attend une string pour le prix
        )

        f.write(f"✅ Ordre LIMIT de vente BTC exécuté: {order}\n")

    except Exception as e:
        f.write(f"❌ Erreur lors de la vente LIMIT BTC: {e}\n")


output_file = f"/home/elias/PROJECT/AICryptoPredictor/results/prediction_{datetime.now().strftime('%Y%m%d_%H')}.txt"
with open(output_file, "a") as f:  # "a" = append (ajouter à la fin du fichier)
    Afficher_solde()
    f.write("\n")
    Buy_all_BTC()
    f.write("\n")
    Sell_all_BTC()
    f.write("\n")
    Afficher_solde()