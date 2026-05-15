# WebApp MonÉnergie (Phase 7 + 8)

Streamlit qui consomme les trois modèles entrainés en TP4 :
- `energie_lstm.keras` (Phase 1+2) — forecast conso 24h
- `profil_conso.keras` (Phase 6) — classification profil consommateur
- `allocine_sentiment.keras` (Phase 4+5) — sentiment FR pour avis fournisseur

## Prérequis : fichiers modèles

Les 3 fichiers `.keras` + leurs scalers `.joblib` doivent être présents soit à la racine du repo, soit dans `webapp/` :

```
TP4/tp/
├── energie_lstm.keras            <- phase 2
├── energie_scalers.joblib        <- phase 2
├── profil_conso.keras            <- phase 6
├── profil_conso_meta.joblib      <- phase 6
├── allocine_sentiment.keras      <- phase 5
├── allocine_tokenizer.joblib     <- phase 5
└── webapp/
    └── app.py
```

Tous ces fichiers sont produits par les scripts `phase*.py` quand tu les lances sur Kaggle. Tu les télécharges depuis `/kaggle/working/` à la fin de chaque notebook.

## Phase 7 — Lancer en local

Si tu as Python + Streamlit installés (~50 Mo) :

```bash
cd TP4/tp
pip install -r requirements.txt
streamlit run webapp/app.py
```

L'app s'ouvre sur http://localhost:8501.

Si pas de stockage local : passe directement à la **Piste A** ci-dessous.

## Phase 8 — Piste A : Streamlit Cloud (recommandé)

Streamlit Cloud héberge gratuitement les apps Streamlit (limite : 1 Go RAM, repos publics).

### Étapes

1. **Pousser sur un repo GitHub public**. Vérifie que les 6 fichiers (modèles `.keras` + scalers `.joblib`) sont commités. Taille totale ~50-100 Mo, sous la limite Git 100 Mo par fichier.

2. **Se connecter sur https://share.streamlit.io** avec ton compte GitHub.

3. **Cliquer "New app"** :
   - Repository : `Hakim78/tp4-rnn-lstm` (ou ton nom)
   - Branch : `main`
   - Main file path : `TP4/tp/webapp/app.py`

4. **Deploy**. Build ~3-5 min (installation `requirements.txt` + premier lancement).

5. URL publique : `https://<app-name>.streamlit.app`

### Limites Streamlit Cloud (compte gratuit)

- Repos publics uniquement
- Mise en veille après 7 jours d'inactivité (réveil ~30s à la première visite)
- 1 Go RAM (largement suffisant pour 3 modèles légers ~5-15 Mo chacun)
- Pas de GPU (inference CPU < 100 ms par prédiction)

### Si un fichier `.keras` > 100 Mo

Cas peu probable vu la taille des modèles. Si ça arrive :
- Activer **Git LFS** : `git lfs track "*.keras"` puis `git add .gitattributes`
- Ou héberger le modèle sur **Hugging Face Hub** (privé OK) et le télécharger au démarrage via `huggingface_hub.hf_hub_download()` dans une fonction cachée par `@st.cache_resource`.

## Phase 8 — Piste B : Logging d'inférence

[log_inference.py](log_inference.py) écrit chaque prédiction dans `inference_log.csv` :

```
timestamp_iso,input_preview,prediction,confidence_pct
2026-05-15T16:42:13+00:00,"Le service client EDF est top",Positif,92.10
2026-05-15T16:43:01+00:00,"Encore une coupure de courant 4h",Negatif,87.34
```

Pour activer : aucun changement requis. `app.py` importe `log_inference` automatiquement s'il est dispo, sinon il continue sans logger.

Pour désactiver : supprimer ou renommer `log_inference.py`.

**Limitation Cloud** : sur Streamlit Cloud, le filesystem est éphémère (effacé au redeploy). Pour persister les logs :
- Bucket S3 / Supabase Storage / Vercel KV
- Webhook vers un Google Sheet via Apps Script
- Vraie DB (Postgres Neon free tier, Turso, etc.)

## Phase 8 — Piste C : Comparaison LSTM vs GRU dans l'app

Ajouter un onglet "Architecture" qui charge `energie_lstm.keras` ET `energie_gru.keras` et fait passer la même courbe dans les deux, affichant les deux forecasts côte à côte avec les temps d'inférence.

Implémentation rapide : copier la fonction `render_forecast_tab()`, dupliquer pour GRU, ajouter `st.columns(2)` pour le rendu côte à côte.

## Phase 8 — Piste D : LSTM stacké pour le forecast

Reprendre `phase1_2_energie_lstm.py` avec 2 couches :

```python
model = keras.Sequential([
    layers.Input(shape=(168, 7)),
    layers.LSTM(64, return_sequences=True),  # IMPORTANT pour stacker
    layers.Dropout(0.2),
    layers.LSTM(32),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1),
])
```

Sur ~27k sequences train, ça vaut le coup d'essayer. Comparer le RMSE avec la version 1 couche.

## Cas d'usage commercial

Une fois la WebApp en ligne sur Streamlit Cloud :

1. Partager l'URL aux 5-10 premiers prospects (early access)
2. Logger chaque interaction (Piste B) pour comprendre les patterns
3. Récolter les emails (champ optionnel ou Calendly)
4. Itérer : raffiner les profils, ajouter le module DPE, intégrer Linky API

Le tier payant peut commencer à 4,99€/mois pour :
- Historique sur 12 mois (vs 1 mois en free)
- Reco personnalisées par ChatGPT API
- Comparateur fournisseur en temps réel
- Export PDF du diagnostic
