# TP4 - RNN, LSTM, Séries temporelles, Déploiement

Livrables de la journée J4 du parcours MIA4 Deep Learning.

## Contenu

| Fichier | Phases | Description |
|---|---|---|
| [phase1_2_airline_lstm.py](phase1_2_airline_lstm.py) | 1 + 2 | Airline Passengers : sliding window + LSTM training + RMSE + plot |
| [phase3_gru_vs_lstm.py](phase3_gru_vs_lstm.py) | 3 | Benchmark cross-architecture GRU vs LSTM sur le même dataset |
| [phase4_5_imdb_bilstm.py](phase4_5_imdb_bilstm.py) | 4 + 5 | IMDB sentiment : tokenization + Embedding + Bidirectional LSTM |
| [phase6_ag_news_multiclass.py](phase6_ag_news_multiclass.py) | 6 | AG News 4 classes : LSTM + softmax + multiclasse |
| [webapp/app.py](webapp/app.py) | 7 | Streamlit WebApp pour l'inférence IMDB |
| [webapp/log_inference.py](webapp/log_inference.py) | 8 | Logging CSV piste B |
| [results.md](results.md) | 3 | Tableau comparatif LSTM vs GRU à remplir |

## Format des fichiers

Les fichiers `.py` utilisent la syntaxe **VSCode interactive** (`# %%` pour séparer les cellules). Trois façons de les exécuter :

1. **VSCode / Cursor** : ouvrir le fichier → cliquer "Run Cell" au-dessus de chaque `# %%`
2. **Jupyter classique** : `jupytext --to ipynb phase1_2_airline_lstm.py` puis ouvrir l'`.ipynb`
3. **Kaggle** : copier-coller chaque cellule dans un notebook Kaggle (recommandé si stockage local plein)

## Workflow Kaggle recommandé

Vu que les datasets sont petits (Airline 144 points, IMDB 50k reviews, AG News 120k titres), un seul kernel Kaggle suffit pour chaque section :

1. Créer un notebook Kaggle vide
2. Activer le GPU si dispo (P100 / T4 gratuit)
3. Coller les cellules du fichier `.py` une par une
4. À la fin, télécharger les modèles `.keras` depuis `/kaggle/working/`
5. Pousser sur HuggingFace Hub privé pour la WebApp

## Commits par phase

Conformément aux règles de notation, **un commit par phase** :

```bash
git add phase1_2_airline_lstm.py
git commit -m "feat: phase1 sliding window airline passengers"

git add phase1_2_airline_lstm.py airline_lstm.keras
git commit -m "feat: phase2 lstm airline passengers training"

git add phase3_gru_vs_lstm.py results.md
git commit -m "feat: phase3 gru vs lstm comparison airline passengers"

git add phase4_5_imdb_bilstm.py
git commit -m "feat: phase4 imdb tokenization embedding"

git add phase4_5_imdb_bilstm.py imdb_sentiment.keras
git commit -m "feat: phase5 imdb bidirectional lstm training"

git add phase6_ag_news_multiclass.py
git commit -m "feat: phase6 ag_news multiclass lstm"

git add webapp/app.py imdb_sentiment.keras
git commit -m "feat: phase7 streamlit webapp imdb inference"

git add webapp/log_inference.py webapp/README.md
git commit -m "feat: phase8 inference logging csv + deploy doc"
```

## Déploiement Streamlit Cloud (Phase 8 Piste A)

Voir [webapp/README.md](webapp/README.md) pour le pas-à-pas. Résumé :

1. Pousser ce repo public sur GitHub avec `app.py` + `imdb_sentiment.keras`
2. Aller sur https://share.streamlit.io → "New app"
3. Sélectionner le repo, le branch, le fichier `webapp/app.py`
4. Build automatique, URL publique partageable en 2 min

## Dépendances

Voir [requirements.txt](requirements.txt). Sur Kaggle, tout est pré-installé sauf `datasets` (pour AG News) :

```python
!pip install datasets
```
