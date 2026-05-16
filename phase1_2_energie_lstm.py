"""TP4 - Phases 1 et 2 : forecast de consommation electrique

Dataset : Household Power Consumption (UCI), 2M+ mesures minute-par-minute
d'un menage francais a Sceaux (16/12/2006 -> 26/11/2010).

Phase 1 : chargement + agregation en consommation HORAIRE + sliding window
          7 jours d'historique -> 24h cibles.
Phase 2 : LSTM multivarie + entrainement + RMSE + courbes + sauvegarde.

A coller cellule par cellule (separateurs `# %%`) dans un notebook Kaggle
avec l'accelerateur GPU T4 x2.

Kaggle : Add Data -> chercher 'electric-power-consumption-data-set' (uciml).
"""

# %% [markdown]
# # Phase 1 : Household Power Consumption (Sceaux FR)
#
# Le dataset contient les colonnes :
# - Date, Time
# - Global_active_power (kW)
# - Global_reactive_power (kVAR)
# - Voltage (V)
# - Global_intensity (A)
# - Sub_metering_1, _2, _3 (Wh, cuisine / laverie / chauffage-clim)
#
# Volume brut : 2M+ lignes minute-par-minute. On agrege en HORAIRE
# (~35k lignes) pour rendre le training tractable et plus pertinent
# commercialement (la facture est mensuelle, le pilotage horaire suffit).

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import layers

SEED = 42
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

# %%
# Sur Kaggle, le fichier est dans /kaggle/input/electric-power-consumption-data-set/
# Hors Kaggle : ajuster CSV_PATH.
import os

if os.path.exists("/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt"):
    CSV_PATH = "/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt"
else:
    # Fallback local si tu as telecharge le dataset
    CSV_PATH = "data/household_power_consumption.txt"

# Le fichier utilise des ; comme separateur et '?' pour les valeurs manquantes
df = pd.read_csv(
    CSV_PATH,
    sep=";",
    na_values="?",
    low_memory=False,
)
print(f"Shape brut : {df.shape}")
print(df.head())

# %%
# Construction d'un timestamp UTC unique
df["timestamp"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S",
)
df = df.set_index("timestamp").sort_index()

# Conversion en float (les colonnes sont parfois en object a cause des '?')
num_cols = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print(f"NaN par colonne :\n{df[num_cols].isna().sum()}")

# %%
# Agregation en consommation HORAIRE
# Global_active_power est en kW (puissance) ; en agregeant on prend la moyenne.
# Sub_metering est en Wh (energie consommee pendant la minute), on prend la somme.
hourly = pd.DataFrame(
    {
        "active_kw": df["Global_active_power"].resample("h").mean(),
        "reactive_kvar": df["Global_reactive_power"].resample("h").mean(),
        "voltage": df["Voltage"].resample("h").mean(),
        "intensity": df["Global_intensity"].resample("h").mean(),
        "sub1_wh": df["Sub_metering_1"].resample("h").sum(),
        "sub2_wh": df["Sub_metering_2"].resample("h").sum(),
        "sub3_wh": df["Sub_metering_3"].resample("h").sum(),
    }
)

# Drop les heures avec NaN (rares apres aggregation)
hourly = hourly.dropna()
print(f"Apres aggregation horaire : {hourly.shape}")
print(hourly.describe().round(2))

# %%
# Visualisation : conso journaliere moyenne pour voir la saisonnalite
plt.figure(figsize=(12, 4))
hourly["active_kw"].resample("D").mean().plot()
plt.title("Consommation electrique journaliere moyenne - Sceaux FR (2006-2010)")
plt.ylabel("Puissance active (kW)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Pattern attendu : pic en hiver (chauffage), baisse en ete.
# Cycles hebdomadaires visibles aussi.

# %%
# Features additionnelles : heure de la journee, jour de la semaine
# Ces features cycliques aident massivement le LSTM a apprendre les habitudes.
hourly["hour"] = hourly.index.hour
hourly["dayofweek"] = hourly.index.dayofweek
hourly["is_weekend"] = (hourly["dayofweek"] >= 5).astype(int)

# Encodage cyclique des heures (sin/cos) pour eviter la discontinuite 23h->0h
hourly["hour_sin"] = np.sin(2 * np.pi * hourly["hour"] / 24)
hourly["hour_cos"] = np.cos(2 * np.pi * hourly["hour"] / 24)

FEATURES = [
    "active_kw",
    "sub1_wh",
    "sub2_wh",
    "sub3_wh",
    "hour_sin",
    "hour_cos",
    "is_weekend",
]
TARGET = "active_kw"

print(f"Features utilisees : {FEATURES}")
print(f"Target : {TARGET}")

# %%
# Split temporel : 80% pour train, 20% pour test. JAMAIS de shuffle.
split_idx = int(len(hourly) * 0.8)
train_df = hourly.iloc[:split_idx]
test_df = hourly.iloc[split_idx:]
print(f"Train : {len(train_df)} heures ({train_df.index[0]} -> {train_df.index[-1]})")
print(f"Test  : {len(test_df)} heures ({test_df.index[0]} -> {test_df.index[-1]})")

# %%
# Normalisation MinMax. Fit UNIQUEMENT sur train (pas de fuite de donnees).
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

train_X_scaled = scaler_X.fit_transform(train_df[FEATURES].values)
test_X_scaled = scaler_X.transform(test_df[FEATURES].values)

train_y_scaled = scaler_y.fit_transform(train_df[[TARGET]].values)
test_y_scaled = scaler_y.transform(test_df[[TARGET]].values)

# %%
# Sliding window : 7 jours d'historique (168h) -> next 24h (forecast)
# C'est le sweet spot pour un forecast journalier domestique.
WINDOW_HOURS = 24 * 7   # 168
HORIZON_HOURS = 24      # prediction sur 24h


def create_sequences(X: np.ndarray, y: np.ndarray, window: int, horizon: int):
    """Construit des paires (sequence_input, target_24h)."""
    X_seq, y_seq = [], []
    for i in range(len(X) - window - horizon + 1):
        X_seq.append(X[i : i + window])
        # Cible : moyenne de la conso sur les 24h qui suivent.
        # On predit donc le NIVEAU moyen de demain (utile pour ajuster chauffage).
        y_seq.append(y[i + window : i + window + horizon].mean())
    return np.array(X_seq), np.array(y_seq)


X_train, y_train = create_sequences(train_X_scaled, train_y_scaled, WINDOW_HOURS, HORIZON_HOURS)
X_test, y_test = create_sequences(test_X_scaled, test_y_scaled, WINDOW_HOURS, HORIZON_HOURS)

print(f"X_train : {X_train.shape}")  # (N_train, 168, 7)
print(f"y_train : {y_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_test  : {y_test.shape}")

# %% [markdown]
# ## Quality gate Phase 1
#
# - Happy path : X_train ~ (27000, 168, 7), conso journaliere visible avec
#   pic hivernal sur la courbe.
# - Edge case : reduire WINDOW_HOURS a 24 (1 seul jour d'historique) -> moins
#   de contexte saisonnier, RMSE attendu plus mauvais.
# - Adversarial : oublier de fit le scaler sur train uniquement -> fuite,
#   le RMSE test sera artificiellement bas.
#
# Commit suggere : `feat: phase1 sliding window household power consumption`

# %% [markdown]
# # Phase 2 : LSTM forecast conso

# %%
model = keras.Sequential(
    [
        layers.Input(shape=(WINDOW_HOURS, len(FEATURES))),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),  # prediction de la moyenne horaire des 24 prochaines heures
    ],
    name="energie_lstm",
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"],
)
model.summary()

# %%
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1,
)

# %%
# Predictions denormalisees pour avoir des kW interpretables
train_pred = model.predict(X_train, verbose=0)
test_pred = model.predict(X_test, verbose=0)

train_pred_inv = scaler_y.inverse_transform(train_pred)
test_pred_inv = scaler_y.inverse_transform(test_pred)
y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))

rmse_train = float(np.sqrt(np.mean((train_pred_inv - y_train_inv) ** 2)))
rmse_test = float(np.sqrt(np.mean((test_pred_inv - y_test_inv) ** 2)))
mae_test = float(np.mean(np.abs(test_pred_inv - y_test_inv)))

print(f"\nRMSE train : {rmse_train:.3f} kW")
print(f"RMSE test  : {rmse_test:.3f} kW")
print(f"MAE  test  : {mae_test:.3f} kW")
print(
    f"Pour reference, conso moyenne du menage : "
    f"{train_df[TARGET].mean():.2f} kW"
)

# %%
# Courbes de convergence
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.xlabel("Epoch")
plt.ylabel("MSE (normalisee)")
plt.title("Convergence LSTM energie")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Visualisation : forecast vs reel sur 200 premieres heures du test set
plt.figure(figsize=(13, 4))
plt.plot(y_test_inv[:200], label="Reel", linewidth=1.4, color="black")
plt.plot(test_pred_inv[:200], label="Predit", linewidth=1.4, color="tab:orange")
plt.xlabel("Heure (test set)")
plt.ylabel("Conso moyenne 24h (kW)")
plt.title(f"Forecast conso - RMSE test {rmse_test:.2f} kW")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Sauvegarde du modele et du scaler pour la WebApp
import joblib

model.save("energie_lstm.keras")

# joblib pour conserver les scalers (necessaires en inference)
joblib.dump(
    {"scaler_X": scaler_X, "scaler_y": scaler_y, "features": FEATURES},
    "energie_scalers.joblib",
)

print("Modele sauve : energie_lstm.keras")
print("Scalers sauves : energie_scalers.joblib")

print("\nA reporter dans results.md phase 2 :")
print(f"  RMSE train : {rmse_train:.3f} kW")
print(f"  RMSE test  : {rmse_test:.3f} kW")
print(f"  MAE  test  : {mae_test:.3f} kW")
print(f"  Parametres : {model.count_params():,}")
print(f"  Epochs effectives : {len(history.history['loss'])}")
print(f"  val_loss finale : {history.history['val_loss'][-1]:.6f}")

# %% [markdown]
# ## Quality gate Phase 2
#
# - Happy path : RMSE test < 0.4 kW (conso moyenne ~1.1 kW). La courbe
#   predite suit les pics journaliers et hebdomadaires.
# - Edge case : retirer les features cycliques (hour_sin, hour_cos) ->
#   le modele perd la notion d'heure de la journee, RMSE degrade nettement.
# - Adversarial : entrainer sur l'ete et evaluer sur l'hiver -> RMSE explose
#   car le pic chauffage n'a jamais ete vu pendant le training.
#   Lecon : il faut au moins une annee complete dans le train pour capturer
#   la saisonnalite annuelle.
#
# Commit suggere : `feat: phase2 lstm forecast conso electrique 24h`
