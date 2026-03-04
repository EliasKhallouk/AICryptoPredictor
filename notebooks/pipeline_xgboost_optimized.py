"""
Pipeline OPTIMISÉE de Prédiction BTC
====================================
Objectif: MAXIMUM PRÉCISION + MEILLEUR GAIN
- Target simplifié: +2.5% en 48h
- Features enrichies: lags, rolling stats, trends
- XGBoost hyperoptimisé
- Threshold calibré: 0.65
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_score, 
    recall_score,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION OPTIMISÉE
# =============================================================================

THRESHOLD = 0.65  # Seuil optimisé
TARGET_GAIN = 0.025  # +2.5% (gain significatif)
TARGET_HORIZON = 24  # 48 heures
RANDOM_STATE = 42

# =============================================================================
# 1. CHARGEMENT
# =============================================================================

def load_data():
    """Charge le dataset BTC 1-minute"""
    print("📂 Chargement du dataset...")
    file_path = os.path.join("/home/elias/PROJECT/AICryptoPredictor/data", "btcusd_1-min_data.csv")
    btc = pd.read_csv(file_path)
    
    btc['Timestamp'] = pd.to_datetime(btc['Timestamp'], unit='s')
    btc.set_index('Timestamp', inplace=True)
    
    print(f"✅ Données chargées : {len(btc)} lignes (1-minute)")
    return btc


# =============================================================================
# 2. PREPROCESSING
# =============================================================================

def preprocess_data(btc):
    """Resample en horaire"""
    print("📊 Resampling horaire (1H)...")
    
    btc_hourly = btc.resample('1H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    btc_hourly = btc_hourly.dropna()
    
    print(f"✅ Resampling terminé : {len(btc_hourly)} heures")
    return btc_hourly


# =============================================================================
# 3. FEATURE ENGINEERING OPTIMISÉ
# =============================================================================

def create_features(df):
    """Features avancées: indicateurs + lags + rolling stats + trends"""
    print("🔧 Feature engineering optimisé...")
    
    btc = df.copy()
    eps = 1e-9
    
    # --- BASICS ---
    btc['Return'] = btc['Close'].pct_change()
    btc['Log_Return'] = np.log(btc['Close'] / (btc['Close'].shift(1) + eps))
    btc['Dist_High'] = (btc['Close'] - btc['High']) / (btc['Close'] + eps)
    btc['Dist_Low'] = (btc['Close'] - btc['Low']) / (btc['Close'] + eps)
    btc['Intraday_Range'] = (btc['High'] - btc['Low']) / (btc['Close'] + eps)
    
    # --- MOVING AVERAGES ---
    btc['EMA_20'] = btc['Close'].ewm(span=20, adjust=False).mean()
    btc['EMA_50'] = btc['Close'].ewm(span=50, adjust=False).mean()
    btc['EMA_200'] = btc['Close'].ewm(span=200, adjust=False).mean()
    btc['Price_to_EMA20'] = (btc['Close'] - btc['EMA_20']) / (btc['EMA_20'] + eps)
    btc['Price_to_EMA50'] = (btc['Close'] - btc['EMA_50']) / (btc['EMA_50'] + eps)
    btc['EMA_Cross'] = btc['EMA_20'] - btc['EMA_50']
    
    # --- RSI ---
    for period in [7, 14]:
        delta = btc['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / (loss + eps)
        btc[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    # --- MACD ---
    ema12 = btc['Close'].ewm(span=12, adjust=False).mean()
    ema26 = btc['Close'].ewm(span=26, adjust=False).mean()
    btc['MACD'] = ema12 - ema26
    btc['MACD_Signal'] = btc['MACD'].ewm(span=9, adjust=False).mean()
    btc['MACD_Hist'] = btc['MACD'] - btc['MACD_Signal']
    
    # --- BOLLINGER BANDS ---
    ma20 = btc['Close'].rolling(20).mean()
    std20 = btc['Close'].rolling(20).std()
    btc['BB_Upper'] = ma20 + 2*std20
    btc['BB_Lower'] = ma20 - 2*std20
    btc['BB_Position'] = (btc['Close'] - btc['BB_Lower']) / ((btc['BB_Upper'] - btc['BB_Lower']) + eps)
    btc['BB_Width'] = (btc['BB_Upper'] - btc['BB_Lower']) / (ma20 + eps)
    
    # --- VOLATILITY ---
    btc['Volatility_7'] = btc['Return'].rolling(7).std()
    btc['Volatility_14'] = btc['Return'].rolling(14).std()
    btc['Volatility_30'] = btc['Return'].rolling(30).std()
    
    # ATR
    prev_close = btc['Close'].shift(1)
    tr1 = btc['High'] - btc['Low']
    tr2 = (btc['High'] - prev_close).abs()
    tr3 = (btc['Low'] - prev_close).abs()
    btc['TR'] = np.max(np.vstack([tr1.values, tr2.values, tr3.values]), axis=0)
    btc['ATR_14'] = btc['TR'].rolling(14).mean()
    btc['ATR_Pct'] = btc['ATR_14'] / (btc['Close'] + eps)
    
    # --- MOMENTUM ---
    btc['ROC_5'] = btc['Close'].pct_change(5)
    btc['ROC_10'] = btc['Close'].pct_change(10)
    btc['ROC_20'] = btc['Close'].pct_change(20)
    btc['Momentum_Z_20'] = (
        (btc['Return'] - btc['Return'].rolling(20).mean()) / 
        (btc['Return'].rolling(20).std() + eps)
    )
    
    # --- STOCHASTIC ---
    low14 = btc['Low'].rolling(14).min()
    high14 = btc['High'].rolling(14).max()
    btc['Stoch_K'] = 100 * (btc['Close'] - low14) / ((high14 - low14) + eps)
    btc['Stoch_D'] = btc['Stoch_K'].rolling(3).mean()
    btc['Williams_R'] = -100 * (high14 - btc['Close']) / ((high14 - low14) + eps)
    
    # --- VOLUME ---
    btc['Volume_MA_20'] = btc['Volume'].rolling(20).mean()
    btc['Volume_Ratio'] = btc['Volume'] / (btc['Volume_MA_20'] + eps)
    btc['Volume_Z_20'] = (
        (btc['Volume'] - btc['Volume'].rolling(20).mean()) / 
        (btc['Volume'].rolling(20).std() + eps)
    )
    
    # OBV
    obv = [0.0]
    for i in range(1, len(btc)):
        if btc['Close'].iloc[i] > btc['Close'].iloc[i-1]:
            obv.append(obv[-1] + btc['Volume'].iloc[i])
        elif btc['Close'].iloc[i] < btc['Close'].iloc[i-1]:
            obv.append(obv[-1] - btc['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    btc['OBV'] = obv
    btc['OBV_ROC_5'] = btc['OBV'].pct_change(5)
    
    # --- CANDLESTICK ---
    rng = (btc['High'] - btc['Low']).replace(0, eps)
    body = (btc['Close'] - btc['Open']).abs()
    btc['Body_Pct'] = body / rng
    btc['Upper_Shadow'] = (btc['High'] - btc[['Close', 'Open']].max(axis=1)) / rng
    btc['Lower_Shadow'] = (btc[['Close', 'Open']].min(axis=1) - btc['Low']) / rng
    
    # --- LAGS (valeurs passées) ---
    for lag in [1, 2, 3, 6, 12, 24]:
        btc[f'Close_Lag_{lag}'] = btc['Close'].shift(lag)
        btc[f'Return_Lag_{lag}'] = btc['Return'].shift(lag)
    
    # --- ROLLING STATISTICS ---
    for window in [6, 12, 24, 48]:
        btc[f'Return_Mean_{window}'] = btc['Return'].rolling(window).mean()
        btc[f'Return_Std_{window}'] = btc['Return'].rolling(window).std()
        btc[f'Volume_Mean_{window}'] = btc['Volume'].rolling(window).mean()
    
    # --- TRENDS ---
    btc['Trend_Slope_12'] = btc['Close'].rolling(12).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 12 else np.nan, raw=False
    )
    btc['Trend_Slope_24'] = btc['Close'].rolling(24).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 24 else np.nan, raw=False
    )
    
    # --- SUPPORT/RESISTANCE ---
    btc['HH_24'] = btc['High'].rolling(24).max()
    btc['LL_24'] = btc['Low'].rolling(24).min()
    btc['Dist_HH'] = (btc['Close'] - btc['HH_24']) / (btc['HH_24'] + eps)
    btc['Dist_LL'] = (btc['Close'] - btc['LL_24']) / (btc['LL_24'] + eps)
    
    # Nettoyage
    btc = btc.replace([np.inf, -np.inf], np.nan)
    btc = btc.dropna()
    
    print(f"✅ {btc.shape[1]} features créées, {len(btc)} lignes")
    return btc


# =============================================================================
# 4. TARGET SIMPLIFIÉ
# =============================================================================

def create_target(df, gain_threshold=TARGET_GAIN, horizon=TARGET_HORIZON):
    """
    Target simplifié pour max précision:
    1 si High atteint +2.5% dans les N heures suivantes
    0 sinon
    
    IMPORTANT: Les dernières lignes n'ont PAS de target (on ne connaît pas le futur)
    mais on les garde pour pouvoir prédire sur les données les plus récentes
    """
    print("🎯 Création du target optimisé...")
    
    btc = df.copy()
    
    # Prix maximum futur sur N heures
    btc['Future_Max'] = btc['High'].shift(-1).rolling(horizon).max()
    
    # Target binaire (sera NaN pour les dernières lignes)
    btc['Target'] = (btc['Future_Max'] >= btc['Close'] * (1 + gain_threshold)).astype(float)
    
    # Supprimer la colonne temporaire
    btc = btc.drop(columns=['Future_Max'])
    
    # Compter les targets valides (non-NaN)
    valid_targets = btc['Target'].notna()
    pos = btc.loc[valid_targets, 'Target'].sum()
    total = valid_targets.sum()
    
    print(f"✅ Target: {pos:.0f} positifs / {total:.0f} ({pos/total*100:.2f}%)")
    print(f"   Gain cible: +{gain_threshold*100}% sur {horizon}h")
    print(f"   ⚠️  Dernières {horizon}h sans target (pour prédiction future)")
    
    return btc


# =============================================================================
# 5. SPLIT
# =============================================================================

def temporal_split(df, split_date='2023-05-01', val_date='2024-01-01'):
    """Split temporel SANS FUITE: train/validation/test séparés"""
    print(f"✂️ Split temporel: Train < {split_date} | Validation [{split_date}, {val_date}) | Test >= {val_date}")
    
    # Split en 3 périodes distinctes
    train = df[df.index < split_date].copy()
    validation = df[(df.index >= split_date) & (df.index < val_date)].copy()
    test = df[df.index >= val_date].copy()
    
    feature_cols = [col for col in df.columns if col != 'Target']
    
    # TRAIN
    X_train = train[feature_cols]
    y_train = train['Target']
    train_valid = y_train.notna()
    X_train = X_train[train_valid]
    y_train = y_train[train_valid]
    
    # VALIDATION (pour early stopping, SANS FUITE)
    X_val = validation[feature_cols]
    y_val = validation['Target']
    val_valid = y_val.notna()
    X_val = X_val[val_valid]
    y_val = y_val[val_valid]
    
    # TEST (garde les lignes sans target pour prédiction future)
    X_test = test[feature_cols]
    y_test = test['Target']
    test_valid = y_test.notna()
    
    print(f"✅ Train: {len(X_train)} ({y_train.mean()*100:.2f}% positifs)")
    print(f"✅ Validation: {len(X_val)} ({y_val.mean()*100:.2f}% positifs)")
    print(f"✅ Test: {test_valid.sum():.0f} avec target + {(~test_valid).sum():.0f} sans target")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# 6. XGBOOST OPTIMISÉ
# =============================================================================

def train_xgboost(X_train, y_train, X_val, y_val):
    """XGBoost optimisé SANS FUITE - utilise validation set séparé"""
    print("🚀 Entraînement XGBoost SANS FUITE...")
    
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos
    
    print(f"   Imbalance ratio: {scale:.2f}")
    
    # HYPERPARAMÈTRES OPTIMISÉS
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'max_depth': 8,
        'learning_rate': 0.03,
        'n_estimators': 500,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'colsample_bylevel': 0.85,
        'scale_pos_weight': scale * 1.2,
        'min_child_weight': 5,
        'gamma': 0.2,
        'reg_alpha': 0.3,
        'reg_lambda': 2.0,
        'max_delta_step': 1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'tree_method': 'hist',
        'early_stopping_rounds': 50
    }
    
    model = xgb.XGBClassifier(**params)
    
    # ✅ CORRECTION CRITIQUE: eval_set utilise VALIDATION (pas test!)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    print(f"✅ Modèle entraîné SANS FUITE (early stopping sur validation set)")
    return model


# =============================================================================
# 7. PRÉDICTION
# =============================================================================

def predict_with_threshold(model, X_test, threshold=THRESHOLD):
    """Prédiction avec threshold optimisé"""
    print(f"🔮 Prédiction (threshold={threshold})")
    
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > threshold).astype(int)
    
    print(f"✅ {y_pred.sum()} signaux positifs / {len(y_pred)}")
    return y_pred, y_proba


# =============================================================================
# 8. ÉVALUATION
# =============================================================================

def evaluate_model(y_test, y_pred, y_proba):
    """Évaluation - filtre les NaN dans y_test"""
    print("\n" + "="*70)
    print("📊 ÉVALUATION")
    print("="*70)
    
    # Filtrer les lignes avec target valide
    valid_mask = y_test.notna()
    y_test_valid = y_test[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    y_proba_valid = y_proba[valid_mask]
    
    print(f"\n📌 Évaluation sur {len(y_test_valid)} samples avec target")
    
    print("\n📋 Classification Report:")
    report = classification_report(y_test_valid, y_pred_valid, digits=4)
    print(report)
    
    precision = precision_score(y_test_valid, y_pred_valid, zero_division=0)
    recall = recall_score(y_test_valid, y_pred_valid, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test_valid, y_proba_valid)
        print(f"🎯 AUC-ROC: {auc:.4f}")
    except:
        auc = None
    
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🎯 Recall: {recall:.4f}")
    
    cm = confusion_matrix(y_test_valid, y_pred_valid)
    print(f"\n📊 Confusion Matrix:")
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    
    return {
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': report
    }


def plot_diagnostics(model, X_train, y_proba, metrics):
    """Visualisations"""
    print("\n📈 Génération visualisations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Feature Importance
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    axes[0, 0].barh(importance_df['feature'], importance_df['importance'])
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].set_title('Top 20 Features')
    axes[0, 0].invert_yaxis()
    
    # Distribution probabilités
    axes[0, 1].hist(y_proba, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=THRESHOLD, color='red', linestyle='--', label=f'Threshold={THRESHOLD}')
    axes[0, 1].set_xlabel('Probabilité')
    axes[0, 1].set_ylabel('Fréquence')
    axes[0, 1].set_title('Distribution des Probabilités')
    axes[0, 1].legend()
    
    # Confusion matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    axes[1, 0].set_ylabel('Vrai')
    axes[1, 0].set_xlabel('Prédit')
    axes[1, 0].set_title('Matrice de Confusion')
    
    # Métriques
    axes[1, 1].axis('off')
    auc_text = f"{metrics['auc']:.4f}" if metrics['auc'] else 'N/A'
    text = (
        "MÉTRIQUES CLÉS\n"
        "=" * 25 + "\n\n"
        f"Precision:  {metrics['precision']:.4f}\n"
        f"Recall:     {metrics['recall']:.4f}\n"
        f"AUC-ROC:    {auc_text}\n\n"
        f"Threshold:  {THRESHOLD}\n\n"
        f"TP: {cm[1,1]:5d}\n"
        f"FP: {cm[0,1]:5d}\n"
        f"TN: {cm[0,0]:5d}\n"
        f"FN: {cm[1,0]:5d}\n"
    )
    axes[1, 1].text(0.1, 0.5, text, fontsize=12, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('/home/elias/PROJECT/AICryptoPredictor/results/xgboost_optimized.png', dpi=150)
    print("✅ Graphiques: results/xgboost_optimized.png")


# =============================================================================
# 9. SAUVEGARDE
# =============================================================================

def save_prediction(y_pred, y_proba, btc_hourly, metrics):
    """Sauvegarde résultats"""
    print("\n💾 Sauvegarde...")
    
    final_prediction = int(y_pred[-1])
    final_probability = float(y_proba[-1])
    last_date = btc_hourly.index[-1].strftime("%Y-%m-%d %H:%M")
    last_close = btc_hourly['Close'].iloc[-1]
    target_price = last_close * (1 + TARGET_GAIN)
    
    output_file = f"/home/elias/PROJECT/AICryptoPredictor/results/prediction_{datetime.now().strftime('%Y%m%d_%H')}.txt"
    
    os.makedirs("/home/elias/PROJECT/AICryptoPredictor/results", exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("PRÉDICTION OPTIMISÉE BTC - HAUTE PRÉCISION\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"📅 Date: {last_date}\n")
        f.write(f"💰 Prix actuel: ${last_close:.2f}\n")
        f.write(f"🎯 Prix cible (+{TARGET_GAIN*100}%): ${target_price:.2f}\n")
        f.write(f"⏱️  Horizon: {TARGET_HORIZON}h\n\n")
        
        f.write("="*70 + "\n")
        f.write("SIGNAL FINAL\n")
        f.write("="*70 + "\n")
        f.write(f"Prédiction: {final_prediction}\n")
        f.write(f"Probabilité: {final_probability:.4f}\n")
        f.write(f"Threshold: {THRESHOLD}\n\n")
        
        f.write("="*70 + "\n")
        f.write("PERFORMANCES\n")
        f.write("="*70 + "\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        if metrics['auc']:
            f.write(f"AUC-ROC: {metrics['auc']:.4f}\n")
        
        cm = metrics['confusion_matrix']
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}\n")
        f.write(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}\n\n")
        
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT DÉTAILLÉ\n")
        f.write("="*70 + "\n")
        f.write(metrics['classification_report'] + "\n")
    
    print(f"✅ Détails: {output_file}")
    
    # Signal simple
    signal_file = "/home/elias/PROJECT/AICryptoPredictor/Output/signal.txt"
    with open(signal_file, "w") as f:
        f.write(str(final_prediction))
        f.write("\n")
        f.write(str(target_price))
    
    print(f"✅ Signal: {signal_file}")


# =============================================================================
# 10. PIPELINE PRINCIPALE
# =============================================================================

def main():
    """Pipeline complète optimisée"""
    print("\n" + "="*70)
    print("🚀 PIPELINE OPTIMISÉE - MAXIMUM PRÉCISION")
    print("="*70 + "\n")
    
    btc = load_data()
    btc_hourly = preprocess_data(btc)
    btc_hourly = create_features(btc_hourly)
    
    # Garder une copie AVANT le target pour avoir les données les plus récentes
    btc_hourly_full = btc_hourly.copy()
    
    btc_hourly = create_target(btc_hourly)
    
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(btc_hourly)
    
    model = train_xgboost(X_train, y_train, X_val, y_val)
    y_pred, y_proba = predict_with_threshold(model, X_test)
    
    metrics = evaluate_model(y_test, y_pred, y_proba)
    plot_diagnostics(model, X_train, y_proba, metrics)
    
    # Utiliser btc_hourly_full pour avoir la dernière date disponible
    save_prediction(y_pred, y_proba, btc_hourly_full, metrics)
    
    print("\n" + "="*70)
    print("✅ PIPELINE TERMINÉE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
