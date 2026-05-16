"""TP4 - Phases 4 et 5 : sentiment FR via Bi-LSTM (Allocine)

Dataset Allocine (HuggingFace 'tblard/allocine'), 200k avis de films en
francais binaires (positif/negatif). On entraine un Bi-LSTM qui pourra
ensuite servir a analyser les avis Trustpilot des fournisseurs d'energie
(EDF / Engie / TotalEnergies / autres) -> module "Quel fournisseur choisir ?"
de la WebApp MonEnergie.

Phase 4 : tokenization + padding + Embedding.
Phase 5 : Bi-LSTM + entrainement + F1 + sauvegarde du modele.

A coller cellule par cellule dans un notebook Kaggle, GPU T4 x2.

Sur Kaggle, premiere cellule a executer :
    !pip install -q datasets
"""

# %% [markdown]
# # Phase 4 : Allocine - tokenization + Embedding

# %%
import numpy as np
import keras
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, f1_score
from datasets import load_dataset

SEED = 42
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

VOCAB_SIZE = 20000
MAX_LEN = 200
EMBED_DIM = 128

# %%
# Telechargement Allocine (HF, ~50 MB). Splits : train, validation, test.
dataset = load_dataset("tblard/allocine")
print(dataset)
print(f"\nExemple :\n{dataset['train'][0]}")

train_texts = dataset["train"]["review"]
train_labels = np.array(dataset["train"]["label"])
test_texts = dataset["test"]["review"]
test_labels = np.array(dataset["test"]["label"])

print(f"\nTrain : {len(train_texts)} avis, distribution : {np.bincount(train_labels)}")
print(f"Test  : {len(test_texts)} avis, distribution : {np.bincount(test_labels)}")

# %%
# Tokenizer Keras : on construit notre vocabulaire a partir du corpus train.
# Pas de pre-tokenizer comme Allocine n'est pas pre-tokenise (contrairement a IMDB).
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

X_train_seq = tokenizer.texts_to_sequences(train_texts)
X_test_seq = tokenizer.texts_to_sequences(test_texts)

# Stats de longueur sur train
seq_lens = [len(s) for s in X_train_seq]
print(f"Longueur sequences train : min={min(seq_lens)}, mediane={np.median(seq_lens):.0f}, max={max(seq_lens)}")

# %%
# Padding 'pre' : place les zeros au DEBUT
# -> le LSTM termine sur les vrais tokens, le hidden state final encode du signal utile
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="pre", truncating="pre")
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="pre", truncating="pre")

print(f"X_train_pad : {X_train_pad.shape}")
print(f"X_test_pad  : {X_test_pad.shape}")

# %% [markdown]
# ## Quality gate Phase 4
#
# - Happy path : shapes (160000, 200) train, mediane des longueurs ~80 tokens.
# - Edge case : num_words=100 -> 99% OOV, modele incapable d'apprendre.
# - Adversarial : un avis de 1000 mots tronque a 200 -> on perd la conclusion
#   de l'avis. Le padding 'pre' garde au moins les premiers tokens.
#
# Commit suggere : `feat: phase4 allocine tokenization embedding`

# %% [markdown]
# # Phase 5 : Bi-LSTM sentiment FR

# %%
model_sentiment = keras.Sequential(
    [
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="allocine_bilstm",
)
model_sentiment.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model_sentiment.summary()

# %%
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1,
)

history_sentiment = model_sentiment.fit(
    X_train_pad,
    train_labels,
    epochs=8,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1,
)

# %%
loss_test, acc_test = model_sentiment.evaluate(X_test_pad, test_labels, verbose=0)
print(f"\nAccuracy test : {acc_test:.4f}")

y_pred_proba = model_sentiment.predict(X_test_pad, verbose=0).ravel()
y_pred = (y_pred_proba >= 0.5).astype(int)
f1 = f1_score(test_labels, y_pred)
print(f"F1-score test : {f1:.4f}")

print("\nRapport de classification :")
print(classification_report(test_labels, y_pred, target_names=["Negatif", "Positif"]))

# %%
# Test sur des avis Trustpilot factices de fournisseurs d'energie
# Demontre la transferabilite du modele entraine sur Allocine
def predict_sentiment(text: str) -> tuple[str, float]:
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="pre", truncating="pre")
    proba = float(model_sentiment.predict(pad, verbose=0)[0, 0])
    label = "Positif" if proba >= 0.5 else "Negatif"
    return label, proba


avis_simules = [
    "EDF nous a augmente notre facture sans prevenir et le service client est introuvable, scandaleux",
    "Tres satisfait de mon changement chez TotalEnergies, le conseiller a ete reactif et le prix est bon",
    "Engie m'a installe le compteur Linky sans probleme, technicien rapide et professionnel",
    "Encore une panne d'electricite pendant 4 heures, j'en peux plus de ce fournisseur catastrophique",
    "Apres 6 mois chez ce fournisseur alternatif aucune mauvaise surprise, je recommande",
]
print("\nTest transferabilite sur faux avis fournisseur energie :")
for avis in avis_simules:
    label, proba = predict_sentiment(avis)
    print(f"  [{label} {proba:.2f}] {avis[:80]}{'...' if len(avis) > 80 else ''}")

# %%
import joblib

model_sentiment.save("allocine_sentiment.keras")
joblib.dump(
    {"tokenizer": tokenizer, "max_len": MAX_LEN, "vocab_size": VOCAB_SIZE},
    "allocine_tokenizer.joblib",
)
print("\nModele sauve : allocine_sentiment.keras")
print("Tokenizer sauve : allocine_tokenizer.joblib")

print("\nA reporter dans results.md phase 5 :")
print(f"  Accuracy test : {acc_test:.4f}")
print(f"  F1-score test : {f1:.4f}")
print(f"  Parametres : {model_sentiment.count_params():,}")

# %% [markdown]
# ## Quality gate Phase 5
#
# - Happy path : accuracy > 0.92 sur Allocine (FR sentiment relativement simple).
# - Edge case : avis avec ironie ("genial vraiment, du grand art ce film...
#   pour s'endormir") -> souvent classe positif (le modele ne capte pas l'ironie).
# - Adversarial : avis melangeant compliments et critiques sur le meme produit
#   ("le service client est top mais les prix sont fous") -> resultat aleatoire.
#
# ## Usage business
#
# Le modele entraine ici peut etre rebrandé "module fournisseur" dans l'app
# MonEnergie : on scrape (avec moderation) les avis Trustpilot d'EDF, Engie,
# TotalEnergies, etc. On donne a chaque fournisseur un score sentiment moyen
# + extraction de thematiques (relevant ce qui est positif vs negatif).
#
# Le modele Allocine est transferable au domaine energie car :
# - Vocabulaire francais standard (pas medical, pas juridique)
# - Sentiment binaire generique (positif/negatif independant du domaine)
# - 200k samples = robustesse contre les biais corpus
#
# Commit suggere : `feat: phase5 allocine bidirectional lstm french sentiment`
