"""TP4 - Phase 6 : classification de profil consommateur (multiclasse)

Sur le meme dataset Household Power Consumption, on construit une tache de
classification multiclasse :

    Entree : sequence de 7 jours de conso (168h, multivariate)
    Sortie : profil du menage parmi 4 classes

Les labels sont DERIVES automatiquement a partir de la conso quotidienne
moyenne du menage sur la fenetre, segmentee par quartiles :
    0 = Econome      (Q1, faible conso)
    1 = Standard     (Q2)
    2 = Energivore   (Q3)
    3 = Critique     (Q4, tres haute conso)

Cas d'usage produit : sur l'app MonEnergie, des qu'on a 7 jours de courbe,
on classe le menage en un coup pour orienter les reco (isolation, fournisseur,
gestes simples).

A coller cellule par cellule dans un notebook Kaggle, GPU T4 x2.
"""

# %% [markdown]
# # Phase 6 : profil consommateur, 4 classes

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import layers

SEED = 42
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

PROFILS = ["Econome", "Standard", "Energivore", "Critique"]

# %%
# Repreparation Household Power Consumption (identique aux phases precedentes)
if os.path.exists("/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt"):
    CSV_PATH = "/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt"
else:
    CSV_PATH = "data/household_power_consumption.txt"

df = pd.read_csv(CSV_PATH, sep=";", na_values="?", low_memory=False)
df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
df = df.set_index("timestamp").sort_index()

for c in ["Global_active_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
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

# %%
# Construction des sequences (sliding window 7 jours)
WINDOW_HOURS = 168

# On ajoute la cible : kWh total consomme pendant les 24h SUIVANTES
# Cette consommation future definit le profil energetique du menage.
hourly["future_24h_kwh"] = hourly["active_kw"].shift(-24).rolling(24).sum()
hourly = hourly.dropna()

print(f"Dataset apres derivation cible : {len(hourly)} heures")
print(
    f"future_24h_kwh : moyenne {hourly['future_24h_kwh'].mean():.1f} kWh, "
    f"min {hourly['future_24h_kwh'].min():.1f}, max {hourly['future_24h_kwh'].max():.1f}"
)

# %%
# Discretisation en 4 classes via QUARTILES (q=4)
# Cela garantit que les 4 classes sont equilibrees, important pour le training.
hourly["profil"] = pd.qcut(hourly["future_24h_kwh"], q=4, labels=[0, 1, 2, 3])
hourly["profil"] = hourly["profil"].astype(int)

# Bornes des quartiles pour la documentation / Streamlit
q = hourly["future_24h_kwh"].quantile([0.25, 0.5, 0.75]).values
print(f"\nBornes des quartiles (kWh/24h) : {q.round(1).tolist()}")
print(f"  Econome    : conso < {q[0]:.1f}")
print(f"  Standard   : {q[0]:.1f} <= conso < {q[1]:.1f}")
print(f"  Energivore : {q[1]:.1f} <= conso < {q[2]:.1f}")
print(f"  Critique   : conso >= {q[2]:.1f}")

print(f"\nDistribution profils : {hourly['profil'].value_counts().sort_index().tolist()}")

# %%
# Split temporel 80/20 + normalisation X uniquement sur train
split_idx = int(len(hourly) * 0.8)
train_df = hourly.iloc[:split_idx]
test_df = hourly.iloc[split_idx:]

scaler_X = MinMaxScaler()
train_X_scaled = scaler_X.fit_transform(train_df[FEATURES].values)
test_X_scaled = scaler_X.transform(test_df[FEATURES].values)


def create_classification_sequences(X, labels, window=WINDOW_HOURS):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i : i + window])
        y_seq.append(labels[i + window])  # profil au pas suivant la fenetre
    return np.array(X_seq), np.array(y_seq)


X_train, y_train = create_classification_sequences(
    train_X_scaled, train_df["profil"].values
)
X_test, y_test = create_classification_sequences(
    test_X_scaled, test_df["profil"].values
)

print(f"\nX_train : {X_train.shape}, y_train : {y_train.shape}")
print(f"Distribution train : {np.bincount(y_train)}")
print(f"Distribution test  : {np.bincount(y_test)}")

# %%
# Modele : LSTM -> Dropout -> Dense softmax 4 classes
model_profil = keras.Sequential(
    [
        layers.Input(shape=(WINDOW_HOURS, len(FEATURES))),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(4, activation="softmax"),
    ],
    name="profil_consommateur",
)

# sparse_categorical_crossentropy : labels entiers 0/1/2/3, pas one-hot
model_profil.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model_profil.summary()

# %%
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1,
)

history_profil = model_profil.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1,
)

# %%
loss_p, acc_p = model_profil.evaluate(X_test, y_test, verbose=0)
print(f"\nAccuracy test : {acc_p:.4f}")

y_pred_proba = model_profil.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
f1_macro = f1_score(y_test, y_pred, average="macro")
print(f"F1 macro : {f1_macro:.4f}")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=PROFILS))

# %%
# Matrice de confusion visuelle
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(PROFILS, rotation=20)
ax.set_yticklabels(PROFILS)
ax.set_xlabel("Predit")
ax.set_ylabel("Reel")
ax.set_title("Matrice de confusion - profils consommateur")
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

# %%
# Sauvegarde pour la WebApp
import joblib

model_profil.save("profil_conso.keras")
joblib.dump(
    {
        "scaler_X": scaler_X,
        "features": FEATURES,
        "quartiles": q.tolist(),
        "profils": PROFILS,
        "window_hours": WINDOW_HOURS,
    },
    "profil_conso_meta.joblib",
)
print("Modele sauve : profil_conso.keras")
print("Metadata sauves : profil_conso_meta.joblib")

print("\nA reporter dans results.md phase 6 :")
print(f"  Accuracy test : {acc_p:.4f}")
print(f"  F1 macro      : {f1_macro:.4f}")
print(f"  Parametres    : {model_profil.count_params():,}")

# %% [markdown]
# ## Quality gate Phase 6
#
# - Happy path : accuracy > 0.85, profils equilibres (qcut garantit la
#   distribution), confusion principalement entre classes adjacentes
#   (ex Econome confondu avec Standard, peu de saut a Critique).
# - Edge case : utiliser une fenetre de 24h seulement (1 jour) au lieu de
#   168h. L'accuracy chute car on rate les variations week-end / semaine.
# - Adversarial : un menage avec un boiler electrique geant (consommation
#   atypique permanente) sera systematiquement classe Critique meme s'il fait
#   des efforts. Solution future : intervalle de confiance + reco
#   contextuelle.
#
# ## Cas d'usage produit
#
# Sur l'app MonEnergie, des qu'un nouvel utilisateur a uploade ou connecte
# son compteur Linky pour 7 jours, on classe son profil. Le profil oriente
# ensuite :
# - Econome -> reco "garde tes habitudes, valide le DPE"
# - Standard -> reco "petits gestes pour gagner 10-15%"
# - Energivore -> reco "isolation + audit thermique payant peut-etre rentable"
# - Critique -> reco "audit obligatoire + comparateur fournisseur en avant"
#
# Le modele s'enrichit ensuite avec les vraies donnees Enedis Open Data
# (compteurs Linky des utilisateurs consentants).
#
# Commit suggere : `feat: phase6 classification profil consommateur 4 classes`
