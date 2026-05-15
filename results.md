# Resultats TP4 - MonEnergie (auto-genere)

## Phase 2 : LSTM forecast 24h

| Metrique | Valeur |
|---|---|
| RMSE train (kW) | 0.302 |
| RMSE test (kW) | 0.241 |
| MAE test (kW) | 0.176 |
| Parametres | 20,545 |
| Epochs effectives | 7 |
| Duree (s, T4) | 18.9 |
| val_loss finale | 0.001945 |

## Phase 3 : LSTM vs GRU

| Metrique | LSTM | GRU |
|---|---|---|
| RMSE test (kW) | 0.241 | 0.236 |
| Parametres | 20,545 | 16,129 |
| Epochs | 7 | 8 |
| Duree totale (s) | 15.8 | 16.5 |
| Duree par epoch (s) | 2.258 | 2.059 |
| val_loss finale | 0.00196 | 0.00194 |

## Phase 5 : Bi-LSTM Allocine sentiment FR

| Metrique | Valeur |
|---|---|
| Accuracy test | 0.9383 |
| F1-score test | 0.9353 |
| Parametres | 2,658,945 |
| Duree (s) | 121.5 |

Rapport de classification :
```
              precision    recall  f1-score   support

     Negatif       0.94      0.95      0.94     10408
     Positif       0.94      0.93      0.94      9592

    accuracy                           0.94     20000
   macro avg       0.94      0.94      0.94     20000
weighted avg       0.94      0.94      0.94     20000

```

## Phase 6 : LSTM multiclasse - profil consommateur

| Metrique | Valeur |
|---|---|
| Accuracy test | 0.4042 |
| F1 macro | 0.3721 |
| Parametres | 20,644 |
| Duree (s) | 11.2 |
| Bornes quartiles (kWh/24h) | [19.7, 25.9, 31.7] |

Rapport par profil :
```
              precision    recall  f1-score   support

     Econome       0.47      0.83      0.60      1721
    Standard       0.42      0.18      0.25      2084
  Energivore       0.56      0.20      0.29      1973
    Critique       0.24      0.58      0.34       879

    accuracy                           0.40      6657
   macro avg       0.43      0.45      0.37      6657
weighted avg       0.45      0.40      0.37      6657

```
