# Resultats TP4 - MonEnergie (auto-genere)

## Phase 2 : LSTM forecast 24h

| Metrique | Valeur |
|---|---|
| RMSE train (kW) | 0.302 |
| RMSE test (kW) | 0.241 |
| MAE test (kW) | 0.176 |
| Parametres | 20,545 |
| Epochs effectives | 7 |
| Duree (s, T4) | 15.7 |
| val_loss finale | 0.001963 |

## Phase 3 : LSTM vs GRU

| Metrique | LSTM | GRU |
|---|---|---|
| RMSE test (kW) | 0.241 | 0.236 |
| Parametres | 20,545 | 16,129 |
| Epochs | 7 | 8 |
| Duree totale (s) | 15.0 | 16.2 |
| Duree par epoch (s) | 2.150 | 2.024 |
| val_loss finale | 0.00195 | 0.00194 |

## Phase 5 : Bi-LSTM Allocine sentiment FR

| Metrique | Valeur |
|---|---|
| Accuracy test | 0.9334 |
| F1-score test | 0.9300 |
| Parametres | 2,658,945 |
| Duree (s) | 98.2 |

Rapport de classification :
```
              precision    recall  f1-score   support

     Negatif       0.93      0.94      0.94     10408
     Positif       0.94      0.92      0.93      9592

    accuracy                           0.93     20000
   macro avg       0.93      0.93      0.93     20000
weighted avg       0.93      0.93      0.93     20000

```

## Phase 6 : LSTM multiclasse - profil temporel (peak/off-peak)

Tache : classer le foyer selon son ratio de conso heures pleines (18-21h) /
heures creuses (1-6h) observe sur 7 jours. Reco tarif dynamique en sortie.

| Metrique | Valeur |
|---|---|
| Accuracy test | 0.5228 |
| F1 macro | 0.4705 |
| Parametres | 32,036 |
| Duree (s) | 38.7 |
| Bornes quartiles ratio | [2.55, 3.6, 4.66] |

Rapport par profil :
```
                precision    recall  f1-score   support

      Nocturne       0.87      0.61      0.72      2640
Equilibre nuit       0.52      0.55      0.54      2219
Equilibre soir       0.34      0.21      0.26      1314
    Vespertine       0.25      0.72      0.37       627

      accuracy                           0.52      6800
     macro avg       0.50      0.52      0.47      6800
  weighted avg       0.60      0.52      0.54      6800

```
