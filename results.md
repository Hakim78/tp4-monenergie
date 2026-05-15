# Resultats TP4 - MonEnergie (auto-genere)

## Phase 2 : LSTM forecast 24h

| Metrique | Valeur |
|---|---|
| RMSE train (kW) | 0.302 |
| RMSE test (kW) | 0.241 |
| MAE test (kW) | 0.176 |
| Parametres | 20,545 |
| Epochs effectives | 7 |
| Duree (s, T4) | 16.7 |
| val_loss finale | 0.001945 |

## Phase 3 : LSTM vs GRU

| Metrique | LSTM | GRU |
|---|---|---|
| RMSE test (kW) | 0.241 | 0.236 |
| Parametres | 20,545 | 16,129 |
| Epochs | 7 | 8 |
| Duree totale (s) | 15.8 | 17.3 |
| Duree par epoch (s) | 2.259 | 2.160 |
| val_loss finale | 0.00196 | 0.00194 |

## Phase 5 : Bi-LSTM Allocine sentiment FR

| Metrique | Valeur |
|---|---|
| Accuracy test | 0.9394 |
| F1-score test | 0.9370 |
| Parametres | 2,658,945 |
| Duree (s) | 122.3 |

Rapport de classification :
```
              precision    recall  f1-score   support

     Negatif       0.94      0.94      0.94     10408
     Positif       0.93      0.94      0.94      9592

    accuracy                           0.94     20000
   macro avg       0.94      0.94      0.94     20000
weighted avg       0.94      0.94      0.94     20000

```

## Phase 6 : LSTM multiclasse - profil temporel (peak/off-peak)

Tache : classer le foyer selon son ratio de conso heures pleines (18-21h) /
heures creuses (1-6h) observe sur 7 jours. Reco tarif dynamique en sortie.

| Metrique | Valeur |
|---|---|
| Accuracy test | 0.5329 |
| F1 macro | 0.4928 |
| Parametres | 32,036 |
| Duree (s) | 71.3 |
| Bornes quartiles ratio | [2.55, 3.6, 4.66] |

Rapport par profil :
```
                precision    recall  f1-score   support

      Nocturne       1.00      0.50      0.67      2640
Equilibre nuit       0.50      0.67      0.57      2219
Equilibre soir       0.42      0.36      0.39      1314
    Vespertine       0.26      0.54      0.35       627

      accuracy                           0.53      6800
     macro avg       0.54      0.52      0.49      6800
  weighted avg       0.65      0.53      0.55      6800

```
