import os
import subprocess
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics import classification_report, precision_score, recall_score
from keras import Input, Model
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# 2. Charger dataset
print("📂 Chargement du dataset...")
df = pd.read_csv("../data/btcusd_1-min_data.csv")

# TODO : préparation features + split
X = df.drop("target", axis=1).values
y = df["target"].values

from sklearn.model_selection import train_test_split
X_train_split, X_test, y_train_split, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_split, y_train_split, test_size=0.2, shuffle=False)

# 3. Définition du modèle
inputs = Input(shape=(X.shape[1],))
x = Dense(256, activation='relu')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 4. Entraînement
print("⚡ Entraînement du modèle...")
history = model.fit(
    X_train_split, y_train_split,
    epochs=100, batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    class_weight={0: 1, 1: 2}
)

# 5. Évaluation
print("📊 Évaluation...")
y_pred_proba = model.predict(X_test)
threshold = 0.9
y_pred = (y_pred_proba > threshold).astype(int)

report = classification_report(y_test, y_pred, digits=4)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# 6. Sauvegarder prédiction finale (0 ou 1) + log
final_prediction = int(y_pred[-1][0])  # dernière valeur prédite
output_file = f"./results/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

os.makedirs("results", exist_ok=True)
with open(output_file, "w") as f:
    f.write(f"Prediction finale (0 ou 1): {final_prediction}\n\n")
    f.write("Rapport de classification:\n")
    f.write(report + "\n")
    f.write(f"Precision classe 1: {precision}\n")
    f.write(f"Recall classe 1: {recall}\n")

print(f"✅ Résultats sauvegardés dans {output_file}")
