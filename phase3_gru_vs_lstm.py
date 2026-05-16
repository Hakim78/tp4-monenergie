"""TP4 - Phase 3 : benchmark GRU vs LSTM sur la conso energie

Memes donnees que phase 1+2 (Household Power Consumption). On compare deux
architectures equivalentes pour decider de la version a deployer en prod.
Argument cle pour le SaaS : GRU est 25-30% plus leger, donc moins de RAM /
plus rapide pour les inferences ou pour un futur portage mobile.

A coller cellule par cellule dans un notebook Kaggle, accelerateur GPU T4 x2.
"""

# %% [markdown]
# # Phase 3 : LSTM vs GRU sur la conso electrique
#
# Memes hyperparametres, memes donnees, seule l'architecture change.
# Objectif : remplir le tableau comparatif de results.md.

# %%
import os
import time
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
# Repreparation du dataset (identique a phase 1, encapsule pour repetabilite)
if os.path.exists("/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt"):
    CSV_PATH = "/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt"
else:
    CSV_PATH = "data/household_power_consumption.txt"

df = pd.read_csv(CSV_PATH, sep=";", na_values="?", low_memory=False)
df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
df = df.set_index("timestamp").sort_index()

num_cols = [
    "Global_active_power",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

hourly = pd.DataFrame(
    {
        "active_kw": df["Global_active_power"].resample("h").mean(),
        "sub1_wh": df["Sub_metering_1"].resample("h").sum(),
        "sub2_wh": df["Sub_metering_2"].resample("h").sum(),
        "sub3_wh": df["Sub_metering_3"].resample("h").sum(),
    }
).dropna()

hourly["hour_sin"] = np.sin(2 * np.pi * hourly.index.hour / 24)
hourly["hour_cos"] = np.cos(2 * np.pi * hourly.index.hour / 24)
hourly["is_weekend"] = (hourly.index.dayofweek >= 5).astype(int)

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


def create_sequences(X, y, window=168, horizon=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - window - horizon + 1):
        X_seq.append(X[i : i + window])
        y_seq.append(y[i + window : i + window + horizon].mean())
    return np.array(X_seq), np.array(y_seq)


WINDOW_HOURS = 168
HORIZON_HOURS = 24
X_train, y_train = create_sequences(train_X_scaled, train_y_scaled, WINDOW_HOURS, HORIZON_HOURS)
X_test, y_test = create_sequences(test_X_scaled, test_y_scaled, WINDOW_HOURS, HORIZON_HOURS)

print(f"X_train : {X_train.shape}, X_test : {X_test.shape}")

# %%
def build_and_train(rnn_layer_class, units: int = 64, epochs: int = 30, label: str = "") -> dict:
    """Construit un modele avec le layer RNN demande, l'entraine, mesure les metriques."""
    keras.utils.set_random_seed(SEED)

    model = keras.Sequential(
        [
            layers.Input(shape=(WINDOW_HOURS, len(FEATURES))),
            rnn_layer_class(units),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ],
        name=label.lower(),
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=0,
    )

    t0 = time.time()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )
    duration = time.time() - t0

    test_pred = model.predict(X_test, verbose=0)
    test_pred_inv = scaler_y.inverse_transform(test_pred)
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    rmse_test = float(np.sqrt(np.mean((test_pred_inv - y_test_inv) ** 2)))

    epochs_run = len(history.history["loss"])
    return {
        "label": label,
        "model": model,
        "history": history,
        "rmse_test": rmse_test,
        "params": model.count_params(),
        "epochs_run": epochs_run,
        "duration_s": duration,
        "duration_per_epoch": duration / max(epochs_run, 1),
        "val_loss_final": history.history["val_loss"][-1],
    }


# %%
print("Entrainement LSTM...")
lstm_res = build_and_train(layers.LSTM, units=64, epochs=30, label="LSTM")
print("Entrainement GRU...")
gru_res = build_and_train(layers.GRU, units=64, epochs=30, label="GRU")


def format_row(res: dict) -> str:
    return (
        f"{res['label']:5} | "
        f"RMSE {res['rmse_test']:5.3f} kW | "
        f"params {res['params']:>7,} | "
        f"epochs {res['epochs_run']:2d} | "
        f"total {res['duration_s']:6.1f}s | "
        f"per-epoch {res['duration_per_epoch']:.2f}s | "
        f"val_loss {res['val_loss_final']:.5f}"
    )


print()
print(format_row(lstm_res))
print(format_row(gru_res))

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, res in zip(axes, [lstm_res, gru_res]):
    ax.plot(res["history"].history["loss"], label="train")
    ax.plot(res["history"].history["val_loss"], label="val")
    ax.set_title(f"{res['label']} - RMSE test {res['rmse_test']:.3f} kW")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Sauvegarde du modele GRU (utile pour le portage mobile - moins de RAM)
gru_res["model"].save("energie_gru.keras")
print("Modele GRU sauve : energie_gru.keras")

# %% [markdown]
# ## A reporter dans results.md (Phase 3)
#
# | Metrique               | LSTM | GRU |
# |------------------------|------|-----|
# | RMSE test (kW)         | ...  | ... |
# | Nombre de parametres   | ...  | ... |
# | Epochs effectives      | ...  | ... |
# | Training time total (s)| ...  | ... |
# | Duree par epoch (s)    | ...  | ... |
# | val_loss finale        | ...  | ... |
#
# **Argument commercial** : GRU ~25% plus leger en parametres, donc :
# - Moins de RAM en inference -> meilleure containerisation
# - Plus rapide a charger sur app mobile (Phase 8 piste future React Native)
# - RMSE proche -> pas de sacrifice qualite
#
# ## Quality gate Phase 3
#
# - Happy path : tableau rempli, GRU moins de params, RMSE comparable.
# - Edge case : doubler epochs pour GRU -> recupere-t-il la perfo LSTM ?
# - Adversarial : tester les deux sur bruit blanc (np.random.randn) -> RMSE
#   doit etre mauvais pour les deux. Si "bon" RMSE -> overfit.
#
# Commit suggere : `feat: phase3 gru vs lstm comparison conso energie`
