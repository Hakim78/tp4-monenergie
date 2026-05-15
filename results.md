# Résultats TP4

À remplir au fur et à mesure des phases.

## Phase 2 : LSTM Airline Passengers

| Métrique | Valeur |
|---|---|
| RMSE train | ... |
| RMSE test | ... |
| Nombre de paramètres | ... |
| Epochs effectives (EarlyStopping) | ... |
| Durée totale entraînement (s) | ... |
| val_loss finale | ... |

**Observation** : ...

## Phase 3 : Comparaison LSTM vs GRU (Airline Passengers)

| Métrique | LSTM | GRU |
|---|---|---|
| RMSE test (passagers) | ... | ... |
| Training time total (s) | ... | ... |
| Durée par epoch (s) | ... | ... |
| Nombre de paramètres | ... | ... |
| val_loss finale | ... | ... |

**Conclusion** : ...

## Phase 5 : IMDB Bidirectional LSTM

| Métrique | Valeur |
|---|---|
| Accuracy test | ... |
| F1-score test | ... |
| Précision macro | ... |
| Rappel macro | ... |
| Nombre de paramètres | ... |

**Rapport de classification** (à coller depuis la sortie du script) :

```
              precision    recall  f1-score   support

     Négatif       ...       ...       ...       ...
     Positif       ...       ...       ...       ...
```

## Phase 6 : AG News Multiclasse

| Métrique | Valeur |
|---|---|
| Accuracy test | ... |
| F1 macro test | ... |

**Rapport par classe** :

```
              precision    recall  f1-score   support

       World       ...       ...       ...       ...
      Sports       ...       ...       ...       ...
    Business       ...       ...       ...       ...
    Sci/Tech       ...       ...       ...       ...
```

## Phase 7 : WebApp Streamlit

- [ ] App lance correctement avec `streamlit run webapp/app.py`
- [ ] Inférence sur une review test renvoie un sentiment cohérent
- [ ] Score de confiance affiché
- [ ] Gestion input vide
- [ ] Latence < 100 ms après cache

## Phase 8 : Améliorations

Piste(s) choisie(s) : ...

- [ ] A - Streamlit Cloud (URL : ...)
- [ ] B - Logging CSV (`inference_log.csv` exemple ci-joint)
- [ ] C - Comparaison LSTM vs GRU dans la WebApp
- [ ] D - LSTM stacké 2 couches
