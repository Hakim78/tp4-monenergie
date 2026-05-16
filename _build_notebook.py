"""Genere le notebook Kaggle all-in-one a partir des 4 scripts phase*.py.

Usage : python _build_notebook.py
Sortie : ../kaggle/tp4_monenergie_all_phases.ipynb

Ce script construit un seul notebook qui :
- Lance les 4 phases du TP4 sequentiellement
- Sauve tous les modeles + scalers + figures + results.md dans /kaggle/working/
- Liberer la RAM (del + gc) entre les phases pour eviter l'OOM
"""

import json
import uuid
from pathlib import Path


def md(text):
    """Cellule markdown."""
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": text,
    }


def code(text):
    """Cellule code."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": text,
    }


cells = []

# --- INTRO ---
cells.append(md("""# TP4 - MonEnergie - All Phases en un notebook

Pipeline complet RNN/LSTM sur la consommation electrique francaise.
- **Phase 1+2** : LSTM forecast 24h (Household Power Consumption, Sceaux FR)
- **Phase 3** : Benchmark GRU vs LSTM sur memes donnees
- **Phase 4+5** : Bi-LSTM sentiment FR (Allocine 200k avis)
- **Phase 6** : LSTM multiclasse classification profil consommateur 4 classes

## Pre-requis Kaggle

1. Compte verifie par telephone (sinon Internet OFF par defaut)
2. Settings -> Internet **ON** (necessaire pour Allocine via HuggingFace)
3. Accelerator **GPU T4 x2** (gratuit)
4. Add Data -> `uciml/electric-power-consumption-data-set`

## Apres execution

Tous les artefacts sont dans `/kaggle/working/` :
- 4 modeles `.keras` + 3 metadata `.joblib`
- `results.md` rempli automatiquement
- Figures PNG (loss, forecasts, confusion matrix)

Apres `Run All` : cliquer **Save Version** pour zipper et telecharger."""))

# --- SETUP ---
cells.append(md("## Setup : imports, seeds, dependencies"))

cells.append(code("""# Installation des dependances qui ne sont pas sur Kaggle par defaut
# keras-preprocessing : Keras 3 a supprime keras.preprocessing.text.Tokenizer
# on utilise le package standalone (drop-in replacement, compatible Keras 3)
import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "datasets", "joblib", "keras-preprocessing"
])"""))

cells.append(code("""import os
import gc
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import keras
from keras import layers
# Imports compatibles Keras 3 :
# - Tokenizer : retire de Keras 3, on utilise le package legacy keras_preprocessing
# - pad_sequences : disponible dans keras.utils (Keras 3)
from keras_preprocessing.text import Tokenizer
from keras.utils import pad_sequences

SEED = 42
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(exist_ok=True)

# Conteneur des metriques pour generer results.md a la fin
METRICS = {}
print(f"Keras : {keras.__version__}")
print(f"Output dir : {OUTPUT_DIR}")"""))

# --- PHASE 1 ---
cells.append(md("""## Phase 1 : Household Power Consumption + sliding window

Dataset UCI : 2M+ mesures minute-par-minute, foyer francais Sceaux 2006-2010.
On aggrege en horaire (~35k lignes) puis on fait une sliding window 7 jours -> next 24h."""))

cells.append(code("""# Chemin du dataset attache via Add Data
CSV_PATH = "/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt"
if not Path(CSV_PATH).exists():
    # Fallback : nom alternatif parfois
    candidates = list(Path("/kaggle/input").rglob("household_power_consumption*"))
    if candidates:
        CSV_PATH = str(candidates[0])
    else:
        raise FileNotFoundError("Dataset non attache. Add Data -> 'electric power consumption data set' (uciml)")

print(f"CSV : {CSV_PATH}")
df = pd.read_csv(CSV_PATH, sep=";", na_values="?", low_memory=False)
df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
df = df.set_index("timestamp").sort_index()

for c in ["Global_active_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print(f"Shape brut : {df.shape}")"""))

cells.append(code("""# Aggregation horaire + features cycliques
hourly = pd.DataFrame({
    "active_kw": df["Global_active_power"].resample("h").mean(),
    "sub1_wh": df["Sub_metering_1"].resample("h").sum(),
    "sub2_wh": df["Sub_metering_2"].resample("h").sum(),
    "sub3_wh": df["Sub_metering_3"].resample("h").sum(),
}).dropna()

hourly["hour_sin"] = np.sin(2 * np.pi * hourly.index.hour / 24)
hourly["hour_cos"] = np.cos(2 * np.pi * hourly.index.hour / 24)
hourly["is_weekend"] = (hourly.index.dayofweek >= 5).astype(int)

# Visualisation : conso journaliere moyenne
plt.figure(figsize=(12, 4))
hourly["active_kw"].resample("D").mean().plot()
plt.title("Consommation electrique journaliere moyenne - Sceaux FR")
plt.ylabel("Puissance active (kW)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_phase1_conso_journaliere.png", dpi=100)
plt.show()

print(f"Apres aggregation horaire : {hourly.shape}")
print(hourly.head())"""))

cells.append(code("""# Features + target + split temporel 80/20
FEATURES = ["active_kw", "sub1_wh", "sub2_wh", "sub3_wh", "hour_sin", "hour_cos", "is_weekend"]
TARGET = "active_kw"

split_idx = int(len(hourly) * 0.8)
train_df = hourly.iloc[:split_idx]
test_df = hourly.iloc[split_idx:]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
train_X_scaled = scaler_X.fit_transform(train_df[FEATURES].values)
test_X_scaled = scaler_X.transform(test_df[FEATURES].values)
train_y_scaled = scaler_y.fit_transform(train_df[[TARGET]].values)
test_y_scaled = scaler_y.transform(test_df[[TARGET]].values)

WINDOW_HOURS = 168   # 7 jours
HORIZON_HOURS = 24

def create_sequences(X, y, window, horizon):
    X_seq, y_seq = [], []
    for i in range(len(X) - window - horizon + 1):
        X_seq.append(X[i:i + window])
        y_seq.append(y[i + window:i + window + horizon].mean())
    return np.array(X_seq), np.array(y_seq)

X_train_ts, y_train_ts = create_sequences(train_X_scaled, train_y_scaled, WINDOW_HOURS, HORIZON_HOURS)
X_test_ts, y_test_ts = create_sequences(test_X_scaled, test_y_scaled, WINDOW_HOURS, HORIZON_HOURS)

print(f"X_train : {X_train_ts.shape}")
print(f"X_test  : {X_test_ts.shape}")"""))

# --- PHASE 2 ---
cells.append(md("## Phase 2 : LSTM forecast 24h"))

cells.append(code("""model_lstm = keras.Sequential([
    layers.Input(shape=(WINDOW_HOURS, len(FEATURES))),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1),
], name="energie_lstm")
model_lstm.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
model_lstm.summary()"""))

cells.append(code("""early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

t0 = time.time()
history_lstm = model_lstm.fit(
    X_train_ts, y_train_ts,
    epochs=30, batch_size=128,
    validation_split=0.1,
    callbacks=[early], verbose=1,
)
lstm_duration = time.time() - t0"""))

cells.append(code("""# Evaluation
train_pred = model_lstm.predict(X_train_ts, verbose=0)
test_pred = model_lstm.predict(X_test_ts, verbose=0)
train_pred_inv = scaler_y.inverse_transform(train_pred)
test_pred_inv = scaler_y.inverse_transform(test_pred)
y_train_inv = scaler_y.inverse_transform(y_train_ts.reshape(-1, 1))
y_test_inv = scaler_y.inverse_transform(y_test_ts.reshape(-1, 1))

rmse_train = float(np.sqrt(np.mean((train_pred_inv - y_train_inv) ** 2)))
rmse_test = float(np.sqrt(np.mean((test_pred_inv - y_test_inv) ** 2)))
mae_test = float(np.mean(np.abs(test_pred_inv - y_test_inv)))

METRICS["phase2"] = {
    "rmse_train": rmse_train,
    "rmse_test": rmse_test,
    "mae_test": mae_test,
    "params": int(model_lstm.count_params()),
    "epochs_run": len(history_lstm.history["loss"]),
    "duration_s": round(lstm_duration, 1),
    "val_loss_final": float(history_lstm.history["val_loss"][-1]),
}

print(f"RMSE train : {rmse_train:.3f} kW")
print(f"RMSE test  : {rmse_test:.3f} kW")
print(f"MAE test   : {mae_test:.3f} kW")
print(f"Conso moy ref : {train_df[TARGET].mean():.2f} kW")"""))

cells.append(code("""# Figure : courbes de loss
plt.figure(figsize=(10, 4))
plt.plot(history_lstm.history["loss"], label="train loss")
plt.plot(history_lstm.history["val_loss"], label="val loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Phase 2 : Convergence LSTM forecast")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_phase2_loss.png", dpi=100)
plt.show()

# Figure : prediction vs reel
plt.figure(figsize=(13, 4))
plt.plot(y_test_inv[:200], label="Reel", linewidth=1.3, color="black")
plt.plot(test_pred_inv[:200], label="Predit", linewidth=1.3, color="tab:orange")
plt.xlabel("Heure (test set)")
plt.ylabel("Conso moy. 24h (kW)")
plt.title(f"Phase 2 : LSTM forecast - RMSE {rmse_test:.2f} kW")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_phase2_forecast.png", dpi=100)
plt.show()"""))

cells.append(code("""# Sauvegarde modele + scalers
model_lstm.save(OUTPUT_DIR / "energie_lstm.keras")
joblib.dump(
    {"scaler_X": scaler_X, "scaler_y": scaler_y, "features": FEATURES},
    OUTPUT_DIR / "energie_scalers.joblib",
)
print("Sauves : energie_lstm.keras + energie_scalers.joblib")"""))

# --- PHASE 3 ---
cells.append(md("""## Phase 3 : GRU vs LSTM benchmark

Memes donnees, meme pipeline. On evalue le gain de parametres GRU (~25% en moins)."""))

cells.append(code("""def build_and_train(rnn_class, label):
    keras.utils.set_random_seed(SEED)
    m = keras.Sequential([
        layers.Input(shape=(WINDOW_HOURS, len(FEATURES))),
        rnn_class(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ], name=label.lower())
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)
    t0 = time.time()
    h = m.fit(X_train_ts, y_train_ts, epochs=30, batch_size=128, validation_split=0.1, callbacks=[es], verbose=0)
    dur = time.time() - t0

    pred = m.predict(X_test_ts, verbose=0)
    pred_inv = scaler_y.inverse_transform(pred)
    rmse = float(np.sqrt(np.mean((pred_inv - y_test_inv) ** 2)))

    return {
        "model": m, "history": h, "rmse_test": rmse,
        "params": int(m.count_params()),
        "epochs_run": len(h.history["loss"]),
        "duration_s": round(dur, 1),
        "duration_per_epoch": round(dur / max(len(h.history["loss"]), 1), 3),
        "val_loss_final": float(h.history["val_loss"][-1]),
        "label": label,
    }

print("Entrainement LSTM (Phase 3 baseline)...")
lstm_res = build_and_train(layers.LSTM, "LSTM")
print(f"  LSTM : RMSE {lstm_res['rmse_test']:.3f} kW, {lstm_res['params']:,} params, {lstm_res['duration_s']}s")

print("Entrainement GRU...")
gru_res = build_and_train(layers.GRU, "GRU")
print(f"  GRU  : RMSE {gru_res['rmse_test']:.3f} kW, {gru_res['params']:,} params, {gru_res['duration_s']}s")"""))

cells.append(code("""METRICS["phase3"] = {
    "lstm": lstm_res,
    "gru": gru_res,
}

# Figure comparative
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, res in zip(axes, [lstm_res, gru_res]):
    ax.plot(res["history"].history["loss"], label="train")
    ax.plot(res["history"].history["val_loss"], label="val")
    ax.set_title(f"{res['label']} - RMSE {res['rmse_test']:.3f} kW, {res['params']:,} params")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_phase3_gru_vs_lstm.png", dpi=100)
plt.show()

# Sauvegarde du GRU (utile pour portage mobile - moins de RAM)
gru_res["model"].save(OUTPUT_DIR / "energie_gru.keras")
print("Sauve : energie_gru.keras")"""))

# --- CLEANUP ---
cells.append(md("## Cleanup memoire avant la phase NLP"))

cells.append(code("""# Libere la RAM avant de charger Allocine (200k reviews + Embedding 20k x 128)
del model_lstm, history_lstm, lstm_res, gru_res
del X_train_ts, y_train_ts, X_test_ts, y_test_ts
del train_pred, test_pred, train_pred_inv, test_pred_inv
del train_X_scaled, test_X_scaled, train_y_scaled, test_y_scaled
keras.backend.clear_session()
gc.collect()
print("Memoire liberee, prets pour la phase NLP")"""))

# --- PHASE 4+5 ---
cells.append(md("""## Phase 4+5 : Bi-LSTM sentiment FR (Allocine)

200k avis de films francais via HuggingFace `tblard/allocine`.
Le modele entraine ici sert ensuite a analyser les avis Trustpilot fournisseurs energie."""))

cells.append(code("""from datasets import load_dataset

VOCAB_SIZE = 20000
MAX_LEN = 200
EMBED_DIM = 128

# Telechargement Allocine (~50 MB) - necessite Internet ON
dataset = load_dataset("tblard/allocine")
print(dataset)

train_texts = dataset["train"]["review"]
train_labels_nlp = np.array(dataset["train"]["label"])
test_texts = dataset["test"]["review"]
test_labels_nlp = np.array(dataset["test"]["label"])

print(f"Train : {len(train_texts)}, distribution : {np.bincount(train_labels_nlp)}")
print(f"Test  : {len(test_texts)}, distribution : {np.bincount(test_labels_nlp)}")"""))

cells.append(code("""# Tokenization Keras
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

X_train_nlp = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN, padding="pre", truncating="pre")
X_test_nlp = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=MAX_LEN, padding="pre", truncating="pre")

print(f"X_train_nlp : {X_train_nlp.shape}")
print(f"X_test_nlp  : {X_test_nlp.shape}")"""))

cells.append(code("""# Architecture Bi-LSTM
model_sentiment = keras.Sequential([
    layers.Input(shape=(MAX_LEN,)),
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
], name="allocine_bilstm")
model_sentiment.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
model_sentiment.summary()"""))

cells.append(code("""es_nlp = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1)

t0 = time.time()
history_nlp = model_sentiment.fit(
    X_train_nlp, train_labels_nlp,
    epochs=8, batch_size=128,
    validation_split=0.1, callbacks=[es_nlp], verbose=1,
)
nlp_duration = time.time() - t0"""))

cells.append(code("""# Evaluation
loss_test, acc_test = model_sentiment.evaluate(X_test_nlp, test_labels_nlp, verbose=0)
y_pred_proba = model_sentiment.predict(X_test_nlp, verbose=0).ravel()
y_pred = (y_pred_proba >= 0.5).astype(int)
f1 = f1_score(test_labels_nlp, y_pred)

METRICS["phase5"] = {
    "accuracy_test": float(acc_test),
    "f1_test": float(f1),
    "params": int(model_sentiment.count_params()),
    "duration_s": round(nlp_duration, 1),
    "classification_report": classification_report(test_labels_nlp, y_pred, target_names=["Negatif", "Positif"]),
}

print(f"Accuracy test : {acc_test:.4f}")
print(f"F1-score test : {f1:.4f}")
print(classification_report(test_labels_nlp, y_pred, target_names=["Negatif", "Positif"]))"""))

cells.append(code("""# Test transferabilite sur faux avis fournisseur energie
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="pre", truncating="pre")
    proba = float(model_sentiment.predict(pad, verbose=0)[0, 0])
    return ("Positif" if proba >= 0.5 else "Negatif"), proba

avis = [
    "EDF nous a augmente la facture sans prevenir, service client introuvable, scandaleux",
    "Tres satisfait du conseiller TotalEnergies, reactif et prix bon",
    "Encore 4h de coupure, j'en peux plus de ce fournisseur catastrophique",
    "Installation Linky par Engie nickel, technicien rapide et pro",
]
for a in avis:
    label, p = predict_sentiment(a)
    print(f"  [{label} {p:.2f}] {a[:70]}{'...' if len(a) > 70 else ''}")"""))

cells.append(code("""# Sauvegarde
model_sentiment.save(OUTPUT_DIR / "allocine_sentiment.keras")
joblib.dump(
    {"tokenizer": tokenizer, "max_len": MAX_LEN, "vocab_size": VOCAB_SIZE},
    OUTPUT_DIR / "allocine_tokenizer.joblib",
)

# Courbes de loss
plt.figure(figsize=(10, 4))
plt.plot(history_nlp.history["accuracy"], label="train acc")
plt.plot(history_nlp.history["val_accuracy"], label="val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Phase 5 : Allocine Bi-LSTM - acc test {acc_test:.3f}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_phase5_allocine_accuracy.png", dpi=100)
plt.show()

print("Sauves : allocine_sentiment.keras + allocine_tokenizer.joblib")"""))

# --- CLEANUP NLP ---
cells.append(md("## Cleanup memoire avant la phase 6"))

cells.append(code("""del model_sentiment, history_nlp, X_train_nlp, X_test_nlp
del train_texts, test_texts, y_pred_proba, y_pred
keras.backend.clear_session()
gc.collect()
print("RAM liberee")"""))

# --- PHASE 6 ---
cells.append(md("""## Phase 6 : classification profil temporel (4 classes)

**Reformulation pedagogique** : au lieu de predire la conso FUTURE (tache
sous-determinee sans features meteo), on classe le **profil temporel observe**
sur la fenetre de 7 jours.

Le label est derive du **ratio peak/off-peak** sur la fenetre :
- Peak hours (18h-21h) : pic vespertine (cuisson + TV + eclairage)
- Off-peak (1h-6h) : nuit (cumulus, basse activite)

Le ratio = conso_peak_moyenne / conso_offpeak_moyenne. Plus le ratio est haut,
plus le foyer concentre sa conso en heures pleines.

On decoupe en 4 quartiles :
- 0 = **Nocturne** (off-peak dominant) -> reco tarif Heures Creuses
- 1 = **Equilibre nuit** -> reco tarif Base
- 2 = **Equilibre soir** -> reco tarif Tempo
- 3 = **Vespertine** (peak dominant) -> reco tarif Heures Pleines

Cette tache est :
- Bien posee (le label est calcule depuis le pattern temporel observe)
- Commercialement pertinente (recommandation tarif dynamique)
- LSTM-adaptee (necessite d'analyser le rythme circadien, pas juste un niveau)"""))

cells.append(code("""PROFILS = ["Nocturne", "Equilibre nuit", "Equilibre soir", "Vespertine"]

PEAK_HOURS = {18, 19, 20}     # heures pleines vespertines typiques
OFFPEAK_HOURS = {1, 2, 3, 4, 5}  # creux nocturne profond

# On reutilise hourly (du debut du notebook)
# Pour chaque fenetre, on calcule le ratio peak/offpeak SUR LA FENETRE
def compute_peak_ratio(hours_idx, values, window_start, window_end):
    \"\"\"Ratio mean(peak) / mean(offpeak) sur [window_start, window_end).\"\"\"
    sub_hours = hours_idx[window_start:window_end]
    sub_values = values[window_start:window_end]
    peak_mask = np.isin(sub_hours, list(PEAK_HOURS))
    off_mask = np.isin(sub_hours, list(OFFPEAK_HOURS))
    peak_mean = sub_values[peak_mask].mean() if peak_mask.any() else 0
    off_mean = sub_values[off_mask].mean() if off_mask.any() else 1e-6
    return float(peak_mean / (off_mean + 1e-6))

hours_idx = hourly.index.hour.values
active_values = hourly["active_kw"].values

# Calcul du ratio pour chaque fenetre possible
ratios = []
for i in range(len(hourly) - WINDOW_HOURS):
    r = compute_peak_ratio(hours_idx, active_values, i, i + WINDOW_HOURS)
    ratios.append(r)
ratios = np.array(ratios)

# Discretisation en quartiles
q = np.quantile(ratios, [0.25, 0.5, 0.75])
labels = np.digitize(ratios, q)   # 0/1/2/3 selon la position dans les quartiles

print(f"Bornes ratios peak/offpeak : {q.round(2).tolist()}")
print(f"Distribution profils : {np.bincount(labels).tolist()}")
print(f"  {PROFILS[0]:18s} : {(labels==0).sum():>5d} fenetres")
print(f"  {PROFILS[1]:18s} : {(labels==1).sum():>5d} fenetres")
print(f"  {PROFILS[2]:18s} : {(labels==2).sum():>5d} fenetres")
print(f"  {PROFILS[3]:18s} : {(labels==3).sum():>5d} fenetres")"""))

cells.append(code("""# Construction des sequences : X (window x features), y (label)
# Split temporel 80/20 (jamais de shuffle)
n_total = len(ratios)
split_idx_clf = int(n_total * 0.8)

# Re-extraction des features normalisees sur la totalite, puis split
hourly_features = hourly[FEATURES].values
scaler_X_c = MinMaxScaler()
features_scaled = scaler_X_c.fit_transform(hourly_features)

X_seq = np.array([
    features_scaled[i:i + WINDOW_HOURS]
    for i in range(len(hourly) - WINDOW_HOURS)
])
y_seq = labels

X_train_clf = X_seq[:split_idx_clf]
y_train_clf = y_seq[:split_idx_clf]
X_test_clf = X_seq[split_idx_clf:]
y_test_clf = y_seq[split_idx_clf:]

print(f"X_train_clf : {X_train_clf.shape}, distribution train : {np.bincount(y_train_clf).tolist()}")
print(f"X_test_clf  : {X_test_clf.shape}, distribution test  : {np.bincount(y_test_clf).tolist()}")"""))

cells.append(code("""model_profil = keras.Sequential([
    layers.Input(shape=(WINDOW_HOURS, len(FEATURES))),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.4),
    layers.LSTM(32),
    layers.Dropout(0.4),
    layers.Dense(32, activation="relu"),
    layers.Dense(4, activation="softmax"),
], name="profil_consommateur")

# LR initial plus bas (5e-4) : moins d'oscillation de val_accuracy
# au debut du training
model_profil.compile(
    optimizer=keras.optimizers.Adam(5e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model_profil.summary()

# Patience tres elevee : sur ce dataset la val_acc oscille beaucoup au debut.
# Si on coupe trop tot on perd les ameliorations apres warmup.
es_clf = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1
)
# ReduceLROnPlateau plus reactif : adapte le LR des que val_loss stagne
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1
)

t0 = time.time()
history_clf = model_profil.fit(
    X_train_clf, y_train_clf,
    epochs=60, batch_size=128,
    validation_split=0.1,
    callbacks=[es_clf, reduce_lr],
    verbose=1,
)
clf_duration = time.time() - t0"""))

cells.append(code("""# Evaluation
loss_clf, acc_clf = model_profil.evaluate(X_test_clf, y_test_clf, verbose=0)
y_pred_proba_clf = model_profil.predict(X_test_clf, verbose=0)
y_pred_clf = np.argmax(y_pred_proba_clf, axis=1)
f1_macro = f1_score(y_test_clf, y_pred_clf, average="macro")

METRICS["phase6"] = {
    "accuracy_test": float(acc_clf),
    "f1_macro": float(f1_macro),
    "params": int(model_profil.count_params()),
    "duration_s": round(clf_duration, 1),
    "task": "Classification profil temporel (ratio peak/offpeak sur 7 jours)",
    "quartiles_ratio": q.round(2).tolist(),
    "classification_report": classification_report(y_test_clf, y_pred_clf, target_names=PROFILS),
}

print(f"Accuracy : {acc_clf:.4f}")
print(f"F1 macro : {f1_macro:.4f}")
print(classification_report(y_test_clf, y_pred_clf, target_names=PROFILS))"""))

cells.append(code("""# Matrice de confusion
cm = confusion_matrix(y_test_clf, y_pred_clf)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(PROFILS, rotation=20)
ax.set_yticklabels(PROFILS)
ax.set_xlabel("Predit")
ax.set_ylabel("Reel")
ax.set_title(f"Phase 6 : profil consommateur - acc {acc_clf:.3f}")
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_phase6_confusion.png", dpi=100)
plt.show()

# Sauvegarde
model_profil.save(OUTPUT_DIR / "profil_conso.keras")
joblib.dump(
    {
        "scaler_X": scaler_X_c,
        "features": FEATURES,
        "quartiles_ratio": q.tolist(),
        "peak_hours": list(PEAK_HOURS),
        "offpeak_hours": list(OFFPEAK_HOURS),
        "profils": PROFILS,
        "window_hours": WINDOW_HOURS,
        "task": "peak_offpeak_ratio_classification",
    },
    OUTPUT_DIR / "profil_conso_meta.joblib",
)
print("Sauves : profil_conso.keras + profil_conso_meta.joblib")"""))

# --- FINAL ---
cells.append(md("## Generation automatique de results.md"))

cells.append(code("""def fmt(value, fmt_str=".4f"):
    if value is None:
        return "..."
    try:
        return f"{value:{fmt_str}}"
    except (ValueError, TypeError):
        return str(value)

p2 = METRICS.get("phase2", {})
p3_lstm = METRICS.get("phase3", {}).get("lstm", {})
p3_gru = METRICS.get("phase3", {}).get("gru", {})
p5 = METRICS.get("phase5", {})
p6 = METRICS.get("phase6", {})

results_md = f\"\"\"# Resultats TP4 - MonEnergie (auto-genere)

## Phase 2 : LSTM forecast 24h

| Metrique | Valeur |
|---|---|
| RMSE train (kW) | {fmt(p2.get('rmse_train'), '.3f')} |
| RMSE test (kW) | {fmt(p2.get('rmse_test'), '.3f')} |
| MAE test (kW) | {fmt(p2.get('mae_test'), '.3f')} |
| Parametres | {p2.get('params', '...'):,} |
| Epochs effectives | {p2.get('epochs_run', '...')} |
| Duree (s, T4) | {fmt(p2.get('duration_s'), '.1f')} |
| val_loss finale | {fmt(p2.get('val_loss_final'), '.6f')} |

## Phase 3 : LSTM vs GRU

| Metrique | LSTM | GRU |
|---|---|---|
| RMSE test (kW) | {fmt(p3_lstm.get('rmse_test'), '.3f')} | {fmt(p3_gru.get('rmse_test'), '.3f')} |
| Parametres | {p3_lstm.get('params', '...'):,} | {p3_gru.get('params', '...'):,} |
| Epochs | {p3_lstm.get('epochs_run', '...')} | {p3_gru.get('epochs_run', '...')} |
| Duree totale (s) | {fmt(p3_lstm.get('duration_s'), '.1f')} | {fmt(p3_gru.get('duration_s'), '.1f')} |
| Duree par epoch (s) | {fmt(p3_lstm.get('duration_per_epoch'), '.3f')} | {fmt(p3_gru.get('duration_per_epoch'), '.3f')} |
| val_loss finale | {fmt(p3_lstm.get('val_loss_final'), '.5f')} | {fmt(p3_gru.get('val_loss_final'), '.5f')} |

## Phase 5 : Bi-LSTM Allocine sentiment FR

| Metrique | Valeur |
|---|---|
| Accuracy test | {fmt(p5.get('accuracy_test'), '.4f')} |
| F1-score test | {fmt(p5.get('f1_test'), '.4f')} |
| Parametres | {p5.get('params', '...'):,} |
| Duree (s) | {fmt(p5.get('duration_s'), '.1f')} |

Rapport de classification :
```
{p5.get('classification_report', '...')}
```

## Phase 6 : LSTM multiclasse - profil temporel (peak/off-peak)

Tache : classer le foyer selon son ratio de conso heures pleines (18-21h) /
heures creuses (1-6h) observe sur 7 jours. Reco tarif dynamique en sortie.

| Metrique | Valeur |
|---|---|
| Accuracy test | {fmt(p6.get('accuracy_test'), '.4f')} |
| F1 macro | {fmt(p6.get('f1_macro'), '.4f')} |
| Parametres | {p6.get('params', '...'):,} |
| Duree (s) | {fmt(p6.get('duration_s'), '.1f')} |
| Bornes quartiles ratio | {p6.get('quartiles_ratio', '...')} |

Rapport par profil :
```
{p6.get('classification_report', '...')}
```
\"\"\"

(OUTPUT_DIR / "results.md").write_text(results_md, encoding="utf-8")
print(results_md)"""))

cells.append(md("""## Outputs finaux

Tous les artefacts dans `/kaggle/working/` :

| Fichier | Phase |
|---|---|
| `energie_lstm.keras` | 2 |
| `energie_scalers.joblib` | 2 |
| `energie_gru.keras` | 3 |
| `allocine_sentiment.keras` | 5 |
| `allocine_tokenizer.joblib` | 5 |
| `profil_conso.keras` | 6 |
| `profil_conso_meta.joblib` | 6 |
| `results.md` | toutes |
| `fig_phase1_conso_journaliere.png` | 1 |
| `fig_phase2_loss.png` | 2 |
| `fig_phase2_forecast.png` | 2 |
| `fig_phase3_gru_vs_lstm.png` | 3 |
| `fig_phase5_allocine_accuracy.png` | 5 |
| `fig_phase6_confusion.png` | 6 |

Cliquer **Save Version** en haut a droite pour zipper et telecharger."""))

cells.append(code("""# Verification finale : liste les fichiers produits
import os
print("Fichiers dans /kaggle/working/ :")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size_kb = os.path.getsize(OUTPUT_DIR / f) / 1024
    print(f"  {f:45s} {size_kb:>8.1f} KB")"""))

# --- ORGANISATION FIGURES ---
cells.append(md("""## Organisation des figures en sous-dossiers par phase

On range les PNG dans `figures/phase1_eda/`, `figures/phase2_lstm/`, etc.
pour un repo GitHub clean."""))

cells.append(code("""import shutil
from pathlib import Path

FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PHASE_FOLDERS = {
    "fig_phase1": "phase1_eda",
    "fig_phase2": "phase2_lstm",
    "fig_phase3": "phase3_gru_vs_lstm",
    "fig_phase5": "phase5_allocine",
    "fig_phase6": "phase6_profil",
}

moved = []
for fname in [f.name for f in OUTPUT_DIR.glob("fig_*.png")]:
    for prefix, subfolder in PHASE_FOLDERS.items():
        if fname.startswith(prefix):
            target_dir = FIGURES_DIR / subfolder
            target_dir.mkdir(exist_ok=True)
            shutil.move(str(OUTPUT_DIR / fname), str(target_dir / fname))
            moved.append((fname, subfolder))
            break

print(f"{len(moved)} figures rangees :")
for fname, subfolder in moved:
    print(f"  figures/{subfolder}/{fname}")

print("\\nArborescence finale :")
for p in sorted(FIGURES_DIR.rglob("*")):
    if p.is_file():
        size_kb = p.stat().st_size / 1024
        print(f"  {p.relative_to(OUTPUT_DIR)}  ({size_kb:.1f} KB)")"""))

# --- PUSH GITHUB ---
cells.append(md("""## Push automatique vers GitHub (optionnel)

Pousse les modeles, figures et `results.md` directement sur ton repo GitHub
depuis Kaggle, sans repasser par ton ordinateur.

### Prerequis (a faire 1 seule fois)

1. **Generer un Personal Access Token GitHub** :
   - GitHub.com -> Settings -> Developer settings -> Personal access tokens -> **Tokens (classic)**
   - **Generate new token (classic)**
   - Nom : `kaggle-tp4`
   - Expiration : 90 jours (raisonnable)
   - Scope : cocher **`repo`** uniquement (pas plus, pour la securite)
   - **Generate token** -> copier le `ghp_...`

2. **Ajouter le token dans Kaggle Secrets** :
   - Sur ce notebook Kaggle : panneau droit -> **Add-ons** -> **Secrets**
   - **Add a new secret**
   - Label : `GITHUB_TOKEN`
   - Value : coller le `ghp_...`
   - **Attach** au notebook

3. **Adapter GITHUB_USER et REPO_NAME** dans la cellule ci-dessous si necessaire.

### Securite

- Le token n'est jamais visible dans le notebook (Kaggle Secrets le masque)
- Il est limite au scope `repo` (pas d'acces a tes autres comptes)
- Tu peux le revoquer a tout moment depuis GitHub Settings"""))

cells.append(code("""# Configuration
GITHUB_USER = "Hakim78"
REPO_NAME = "tp4-monenergie"
COMMIT_MESSAGE = "auto: kaggle run all phases - models + figures + results"

# --- Push ---
import subprocess
import shutil
import os

try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception as e:
    print(f"Kaggle Secret 'GITHUB_TOKEN' introuvable : {e}")
    print("\\nLis la section ci-dessus pour configurer le token. Push annule.")
else:
    repo_url = f"https://{GITHUB_USER}:{token}@github.com/{GITHUB_USER}/{REPO_NAME}.git"
    repo_path = Path("/kaggle/working/_repo")

    # Configure git identite
    subprocess.run(["git", "config", "--global", "user.email", "kaggle-bot@local"], check=True)
    subprocess.run(["git", "config", "--global", "user.name", "Kaggle Bot (TP4)"], check=True)

    # Clone (shallow pour aller vite) ou pull si deja la
    if repo_path.exists():
        shutil.rmtree(repo_path)
    print(f"Clone {REPO_NAME}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
        check=True, capture_output=True, text=True,
    )

    # Copie tous les outputs dans le repo
    for item in OUTPUT_DIR.iterdir():
        if item.name in {"_repo", "__pycache__"}:
            continue
        dest = repo_path / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        elif item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)

    # Commit + push
    os.chdir(repo_path)
    subprocess.run(["git", "add", "-A"], check=True)

    status = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
    if not status.stdout.strip():
        print("Rien a commiter (aucun changement par rapport au repo distant).")
    else:
        print(f"Fichiers a commiter :\\n{status.stdout}")
        subprocess.run(["git", "commit", "-m", COMMIT_MESSAGE], check=True)
        result = subprocess.run(
            ["git", "push", "origin", "main"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"\\nPush vers https://github.com/{GITHUB_USER}/{REPO_NAME} OK")
        else:
            print(f"Erreur push : {result.stderr}")

    os.chdir("/kaggle/working")"""))


# --- ECRITURE DU NOTEBOOK ---
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path(__file__).resolve().parent.parent / "kaggle" / "tp4_monenergie_all_phases.ipynb"
output_path.parent.mkdir(exist_ok=True)
output_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")

size_kb = output_path.stat().st_size / 1024
print(f"Notebook genere : {output_path}")
print(f"Taille : {size_kb:.1f} KB")
print(f"Cellules : {len(cells)}")
