from binance.client import Client
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timedelta

# ==============================
# 🔑 CONNEXION BINANCE
# ==============================
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)

API_KEY_TEST = "wXklSWGHk3qvzrrXUcMzqvJNhNT589qYkkFfwq83z92GtT3PGwsyPR7gopIkgvqU"
API_SECRET_TEST = "DrBnrFW51syCtBwFZWZdCicdC7T7Ri06s6NiYUUcxJa7xBr7dOZzA6SCDhnfc0bZ"
client = Client(API_KEY_TEST, API_SECRET_TEST, testnet=True)


# ==============================
# 🔧 PARAMÈTRES STRATÉGIE
# ==============================
PAIR = "BTCUSDT"
TARGET_PROFIT = 0.0075      # +0.75%
STOP_LOSS_LIMIT = -0.017    # -1.7%
STOP_LOSS_MARKET = -0.02    # -2%
MIN_TRADE_USDT = 10
SIGNAL_FILE = "/home/elias/PROJECT/AICryptoPredictor/Output/signal.txt"


# ==============================
# 📂 LIRE LE SIGNAL
# ==============================
def lire_signal():
    with open(SIGNAL_FILE, "r") as f:
        lignes = f.readlines()
    signal = int(lignes[0].strip())     # 0 ou 1
    prix_signal = float(lignes[1].strip()) if len(lignes) > 1 else None
    return signal, prix_signal


# ==============================
# 💰 SOLDE
# ==============================
def get_balance(asset):
    balance = client.get_asset_balance(asset=asset)
    return float(balance['free'])


# ==============================
# 🛒 ACHAT MARKET
# ==============================
def buy_btc():
    usdt = get_balance("USDT")
    if usdt < MIN_TRADE_USDT:
        print(f"⚠️ Pas assez d'USDT ({usdt}) pour acheter.")
        return None
    
    ticker = client.get_symbol_ticker(symbol=PAIR)
    price = float(ticker['price'])
    qty = (usdt - 0.5) / price # Laisser 0.5 USDT pour que l'arondis ne dépasse pas le solde
    qty = round(qty, 5)

    if qty <= 0:
            print("⚠️ Quantité calculée invalide.")
            return None

    order = client.order_market_buy(symbol=PAIR, quantity=qty)
    prix_achat = float(order['fills'][0]['price'])

    print(f"✅ Achat BTC exécuté : {order}")
    rapport("Achat", prix=prix_achat, qty=qty, ordre=order)
    return prix_achat


# ==============================
# 💰 VENTE LIMIT (+0.75%)
# ==============================
def place_sell_limit(prix_achat, prix_signal=None):
    """
    Place un ordre de vente LIMIT.
    - Si prix_signal est fourni par l'IA, on l'utilise comme cible de vente.
    - Sinon, on vend à prix_achat * 1.0075 (+0.75 %).
    """
    try:
        target_price = float(prix_signal) if prix_signal else prix_achat * 1.0075

        # Arrondi du prix et quantité pour respecter Binance
        target_price = round(target_price, 2)   # ETH/BTC → 2 décimales suffisent
        btc_balance = client.get_asset_balance(asset="BTC")
        btc_free = float(btc_balance['free'])
        qty = round(btc_free * 0.999, 5)  # vend ~99.9% du solde

        

        if qty < 0.00001:
            print("⚠️ Pas assez de BTC pour vendre.")
            return None

        order = client.order_limit_sell(
            symbol=PAIR,
            quantity=qty,
            price=str(target_price)
        )

        print(f"✅ Ordre de vente LIMIT placé à {target_price} USDT pour {qty} BTC")
        rapport("Vente LIMIT", prix=target_price, qty=qty, ordre=order)
        return order

    except Exception as e:
        print(f"❌ Erreur lors du placement de l'ordre de vente LIMIT : {e}")
        return None


# ==============================
# ⛔ STOP LOSS DOUBLE (Stop-Limit & Stop-Market)
# ==============================
def place_stop_loss(entry_price):
    """
    Place 2 ordres stop-loss :
    - Stop-Limit à -1.7 %
    - Stop-Market à -2 %
    Si l’un s’exécute, l’autre est annulé.
    """
    btc_free = get_balance("BTC")
    qty = round(btc_free * 0.999, 5)  # vend 99.9% du solde

    if qty < 0.00001:
        print("⚠️ Pas assez de BTC pour placer un stop-loss.")
        return None

    # Prix stop-limit (-1.7 %)
    stop_limit_price = round(entry_price * (1 + STOP_LOSS_LIMIT ), 2)
    limit_price      = round(stop_limit_price * 0.999, 2)  # légèrement plus bas pour l’exécution

    # Prix stop-market (-2 %)
    stop_market_price = round(entry_price * (1 + STOP_LOSS_MARKET ), 2)

    try:
        # Stop-Limit
        stop_limit_order = client.create_order(
            symbol=PAIR,
            side="SELL",
            type="STOP_LOSS_LIMIT",
            timeInForce="GTC",
            quantity=qty,
            price=str(limit_price),
            stopPrice=str(stop_limit_price)
        )

        # Stop-Market
        stop_market_order = client.create_order(
            symbol=PAIR,
            side="SELL",
            type="STOP_MARKET",
            stopPrice=str(stop_market_price),
            quantity=qty
        )

        print(f"🛑 Stop-Limit placé à {stop_limit_price} USDT (limit {limit_price})")
        print(f"🛑 Stop-Market placé à {stop_market_price} USDT")

        # Rapport
        rapport("Stop-Limit", prix=stop_limit_price, qty=qty, ordre=stop_limit_order)
        rapport("Stop-Market", prix=stop_market_price, qty=qty, ordre=stop_market_order)

        return {
            "stop_limit_id": stop_limit_order["orderId"],
            "stop_market_id": stop_market_order["orderId"]
        }

    except Exception as e:
        print(f"❌ Erreur placement stop-loss : {e}")
        return None


# ==============================
# 🔍 SURVEILLER LES STOPS
# ==============================
def surveiller_stops(stop_orders):
    """
    Vérifie si un des stops est exécuté.
    Si oui, annule l’autre.
    """
    if not stop_orders:
        return

    try:
        orders = client.get_open_orders(symbol=PAIR)
        open_ids = [o["orderId"] for o in orders]

        # Si un des stops n’est plus dans les ordres ouverts → il a été exécuté
        if stop_orders["stop_limit_id"] not in open_ids:
            print("✅ Stop-Limit exécuté, annulation du Stop-Market...")
            try:
                client.cancel_order(symbol=PAIR, orderId=stop_orders["stop_market_id"])
            except:
                pass

        elif stop_orders["stop_market_id"] not in open_ids:
            print("✅ Stop-Market exécuté, annulation du Stop-Limit...")
            try:
                client.cancel_order(symbol=PAIR, orderId=stop_orders["stop_limit_id"])
            except:
                pass

    except Exception as e:
        print(f"⚠️ Erreur surveillance stops : {e}")


# ==============================
# 📌 ORDRES OUVERTS
# ==============================
def afficher_ordres_ouverts():
    orders = client.get_open_orders(symbol=PAIR)
    if not orders:
        print("✅ Aucun ordre en attente.")
    else:
        print("📋 Ordres en attente :")
        for o in orders:
            print(f" - {o['side']} {o['origQty']} BTC @ {o['price']} (ID: {o['orderId']})")


# ==============================
# ❌ ANNULER TOUS LES ORDRES
# ==============================
def cancel_all_orders():
    try:
        orders = client.get_open_orders(symbol=PAIR)
        if not orders:
            print("✅ Aucun ordre à annuler.")
            return
        for o in orders:
            client.cancel_order(symbol=PAIR, orderId=o['orderId'])
            print(f"🗑️ Ordre annulé : {o['side']} {o['origQty']} @ {o.get('price','MKT')}")
        rapport("Annulation des ordres", ordre={"nb_ordres": len(orders)})
    except Exception as e:
        print(f"⚠️ Erreur annulation ordres : {e}")

# ==============================
# 🔴 VENTE IMMÉDIATE (MARKET)
# ==============================
def sell_all_market():
    btc_balance = client.get_asset_balance(asset="BTC")
    btc_free = float(btc_balance['free'])
    qty = round(btc_free * 0.999, 5)  # vend ~99.9% pour éviter erreur Binance

    if qty < 0.00001:
        print("⚠️ Pas assez de BTC pour vendre.")
        return None

    order = client.order_market_sell(symbol=PAIR, quantity=qty)
    prix_vente = float(order['fills'][0]['price'])
    print(f"✅ Vente immédiate {qty} BTC @ {prix_vente} USDT")
    rapport("Vente Market", prix=prix_vente, qty=qty, ordre=order)
    return prix_vente


# ==============================
# 🧠 STRATÉGIE
# ==============================
def strategie():
    signal, prix_signal = lire_signal()
    print(f"\n📊 Signal IA = {signal}, prix cible = {prix_signal}")

    if signal == 1:  # Prédiction hausse
        print("🚀 Signal d'achat détecté.")
        prix_achat = buy_btc()
        if prix_achat:
            place_sell_limit(prix_achat, prix_signal)
            stop_orders = place_stop_loss(prix_achat)
            surveiller_stops(stop_orders)

    else:
        print("❌ Signal baisse détecté : on ferme la position.")
        cancel_all_orders()
        sell_all_market()

# ==============================
# 📂 RAPPORT
# ==============================

def rapport(action, prix=None, qty=None, ordre=None):
    """
    Sauvegarde un rapport dans un fichier texte horodaté.
    """
    # Nom de fichier = horodaté à l'heure
    output_file = f"/home/elias/PROJECT/AICryptoPredictor/results/prediction_{datetime.now().strftime('%Y%m%d_%H')}.txt"

    with open(output_file, "a") as f:
        f.write(f"\n--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"Action : {action}\n")
        if prix:
            f.write(f"Prix : {prix}\n")
        if qty:
            f.write(f"Quantité : {qty}\n")
        if ordre:
            f.write(f"Ordre ID : {ordre.get('orderId', 'N/A')}\n")
            f.write(f"Status : {ordre.get('status', 'N/A')}\n")

        # 🔹 Solde actuel
        solde = client.get_account()
        f.write("\n📊 Solde actuel :\n")
        for asset in solde['balances']:
            free = float(asset['free'])
            locked = float(asset['locked'])
            if free > 0 or locked > 0:
                f.write(f"  {asset['asset']} : {free} libres / {locked} bloqués\n")

        # 🔹 Ordres ouverts
        try:
            open_orders = client.get_open_orders(symbol=PAIR)
            f.write("\n📌 Ordres ouverts :\n")
            if len(open_orders) == 0:
                f.write("  Aucun ordre ouvert.\n")
            else:
                for o in open_orders:
                    f.write(f"  ID {o['orderId']} | {o['side']} {o['origQty']} {o['symbol']} "
                            f"à {o.get('price', 'MKT')} | statut : {o['status']}\n")
        except Exception as e:
            f.write(f"\n⚠️ Impossible de récupérer les ordres ouverts : {e}\n")

        f.write("\n")


# ==============================
# ▶️ EXECUTION
# ==============================
if __name__ == "__main__":
    rapport("Rapport final")
    strategie()
    #afficher_ordres_ouverts()
    rapport("Rapport final")
    print("\n✅ Fin du script de AutoTrader2.")    
