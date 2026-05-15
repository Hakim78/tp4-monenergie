# TP4 - MonÉnergie : forecast et analyse de la consommation française

Sujet libre TP4, fil rouge énergie : application qui aide les ménages français
à prévoir leur consommation, comprendre leur profil énergétique et choisir un
fournisseur grâce à l'analyse d'avis. Modèles RNN/LSTM + déploiement Streamlit.

## Vision commerciale

| Élément | Valeur |
|---|---|
| Cible primaire | 15M propriétaires de maison FR + 20M locataires |
| Niche FR | Crise énergétique post-2022, marché sensibilisé |
| Pricing cible | 4,99-9,99€/mois (B2C) - 49€/mois (B2B artisans/syndics) |
| ROI utilisateur | -200 à -500€/an de facture pour 60-120€/an d'abonnement |
| Concurrence | Hello Watt (gratuit ad), Selectra (commission), aucun DL grand public |

## Pipeline pédagogique (8 phases)

| Phase | Dataset | Modèle | Sortie business |
|---|---|---|---|
| 1+2 | Household Power Consumption (UCI Sceaux FR, 2M lignes) | LSTM forecast 24h | Prédiction conso de demain pour adapter chauffage |
| 3 | Idem | GRU vs LSTM benchmark | Choix archi pour déploiement mobile (GRU plus léger) |
| 4+5 | Allociné Reviews (HF, 200k avis FR) | Bi-LSTM sentiment FR | Modèle générique transférable aux avis Trustpilot fournisseurs |
| 6 | Household Power agrégé | LSTM multiclass | Classifie le profil énergétique d'un ménage (4 classes) |
| 7 | Modèles précédents | Streamlit | App MonÉnergie : upload courbe → forecast + profil |
| 8 | - | Logging + déploiement | Streamlit Cloud + audit CSV des inférences |

## Datasets

| Fichier source | Téléchargement Kaggle | Volume |
|---|---|---|
| `household_power_consumption.txt` | Kaggle dataset `uciml/electric-power-consumption-data-set` | 130 MB compressé |
| `allocine` | HuggingFace `tblard/allocine` (auto via `datasets`) | 200k reviews |

## Workflow Kaggle (stockage local plein)

Pour chaque script `phase*.py` :

1. **Nouveau notebook Kaggle**, accélérateur GPU T4 x2 (gratuit) pour phases 4-6
2. **Add Data** (panneau de droite) → chercher `electric-power-consumption-data-set` pour les phases conso
3. **Coller les cellules `# %%` une par une**
4. Lancer **Run All**
5. Quand le script affiche "Modele sauve : *.keras", téléchargez depuis `/kaggle/working/`
6. Place le `.keras` dans `TP4/tp/` localement
7. Commit + push

## Commits par phase

```bash
git add phase1_2_energie_lstm.py
git commit -m "feat: phase1 sliding window household power consumption"

git add energie_lstm.keras results.md
git commit -m "feat: phase2 lstm forecast conso electrique 24h"

git add phase3_gru_vs_lstm.py energie_gru.keras
git commit -m "feat: phase3 gru vs lstm comparison conso energie"

git add phase4_5_allocine_bilstm.py
git commit -m "feat: phase4 allocine tokenization embedding"

git add phase4_5_allocine_bilstm.py allocine_sentiment.keras
git commit -m "feat: phase5 allocine bidirectional lstm french sentiment"

git add phase6_profil_consommateur.py profil_conso.keras
git commit -m "feat: phase6 classification profil consommateur 4 classes"

git add webapp/app.py energie_lstm.keras profil_conso.keras
git commit -m "feat: phase7 streamlit MonEnergie webapp"

git add webapp/log_inference.py webapp/README.md
git commit -m "feat: phase8 inference logging csv + streamlit cloud doc"
```

## Déploiement WebApp (Phase 7-8)

Streamlit Cloud (gratuit) : voir [webapp/README.md](webapp/README.md).
Sans installation locale : push GitHub → connect Streamlit Cloud → URL publique.

## Datasets / open data utilisables ensuite pour la prod

| Source | Donnée | Licence |
|---|---|---|
| Enedis Open Data | Conso régionale/communale temps réel | Etalab 2.0 |
| RTE Open Data | Production/conso FR nationale | Etalab 2.0 |
| GRDF Open Data | Conso gaz par maille | Etalab 2.0 |
| ADEME ObsDPE | 18M DPE avec étiquette A-G | Etalab 2.0 |
| INSEE | Revenus, composition ménages | Etalab 2.0 |
| Trustpilot | Avis EDF/Engie/TotalEnergies | Scrapping (avec ToS prudent) |

Tout est gratuit et ouvert. Aucune licence commerciale à négocier.
