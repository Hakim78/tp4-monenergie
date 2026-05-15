# Résultats TP4 - MonÉnergie

À remplir au fur et à mesure des phases sur Kaggle.

## Phase 2 : LSTM forecast consommation 24h (Household Power Consumption)

Dataset : Household Power Consumption (UCI, Sceaux FR, 2006-2010).
Features : `active_kw`, `sub1_wh`, `sub2_wh`, `sub3_wh`, `hour_sin`, `hour_cos`, `is_weekend`.
Window : 168h (7 jours), Horizon : 24h (moyenne).

| Métrique | Valeur |
|---|---|
| RMSE train (kW) | ... |
| RMSE test (kW) | ... |
| MAE test (kW) | ... |
| Conso moyenne du ménage (kW) | ... |
| Nombre de paramètres | ... |
| Epochs effectives (EarlyStopping) | ... |
| Durée totale (s, GPU T4) | ... |
| val_loss finale | ... |

**Observation** : ...

## Phase 3 : Comparaison LSTM vs GRU sur conso

| Métrique | LSTM | GRU |
|---|---|---|
| RMSE test (kW) | ... | ... |
| Nombre de paramètres | ... | ... |
| Epochs effectives | ... | ... |
| Training time total (s) | ... | ... |
| Durée par epoch (s) | ... | ... |
| val_loss finale | ... | ... |

**Verdict pour le déploiement** : ...

## Phase 5 : Bi-LSTM Allociné sentiment FR

Dataset : Allociné Reviews (HF `tblard/allocine`, 200k avis FR).

| Métrique | Valeur |
|---|---|
| Accuracy test | ... |
| F1-score test | ... |
| Précision macro | ... |
| Rappel macro | ... |
| Nombre de paramètres | ... |

**Rapport de classification** :

```
              precision    recall  f1-score   support

     Negatif       ...       ...       ...       ...
     Positif       ...       ...       ...       ...
```

**Transférabilité aux avis fournisseurs énergie** : à valider sur quelques avis manuels (voir cellule de test du script phase4_5).

## Phase 6 : LSTM multiclasse - profil consommateur

Labels : 4 classes dérivées des quartiles de consommation 24h future.

| Métrique | Valeur |
|---|---|
| Accuracy test | ... |
| F1 macro test | ... |
| Bornes quartiles (kWh/24h) | ... |
| Nombre de paramètres | ... |

**Rapport par profil** :

```
              precision    recall  f1-score   support

    Econome        ...       ...       ...       ...
    Standard       ...       ...       ...       ...
    Energivore     ...       ...       ...       ...
    Critique       ...       ...       ...       ...
```

**Observation matrice confusion** : ...

## Phase 7 : Streamlit MonÉnergie

- [ ] App lance correctement avec `streamlit run webapp/app.py`
- [ ] Onglet Forecast : upload CSV 168 lignes → forecast 24h cohérent
- [ ] Onglet Profil : classification + recommandation contextuelle
- [ ] Onglet Sentiment : avis fournisseur → positif/négatif + confiance
- [ ] Sidebar avec context produit + sources données

## Phase 8 : Améliorations

- [ ] Piste A — Déploiement Streamlit Cloud : URL = `https://...streamlit.app`
- [ ] Piste B — Logging CSV (`inference_log.csv` capturé)
- [ ] Piste C — Comparaison LSTM vs GRU directement dans l'app
- [ ] Piste D — LSTM stacké 2 couches sur forecast conso (test overfit)

## Notes business

- Cible primaire : 15M propriétaires de maison FR
- Pricing cible : 4,99-9,99€/mois (B2C)
- ROI utilisateur estimé : -200 à -500€/an de facture
- Open data utilisés (toutes Etalab 2.0) :
  - Enedis Open Data : conso régionale/communale
  - RTE Open Data : production/conso nationale
  - GRDF Open Data : conso gaz par maille
  - ADEME ObsDPE : 18M DPE A-G
- Concurrence directe (à différencier) : Hello Watt, Selectra
