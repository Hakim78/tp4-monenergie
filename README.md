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

## Exploration data science — résultats visuels

Toutes les figures sont générées automatiquement par le notebook all-in-one
[`notebooks/tp4_monenergie_all_phases.ipynb`](notebooks/tp4_monenergie_all_phases.ipynb)
exécuté sur Kaggle. Les PNG sont organisés dans `figures/<phase>/` et auto-poussés
sur GitHub après chaque run.

### Phase 1 · Exploration EDA — série temporelle Sceaux FR

#### 1.1 Évolution journalière sur 4 ans

![Consommation journalière moyenne](figures/phase1_eda/fig_phase1_conso_journaliere.png)

La série Household Power Consumption (Sceaux, Île-de-France, 2006-2010) agrégée en consommation journalière moyenne révèle 3 signaux structurels :
- **Saisonnalité annuelle marquée** : pic hivernal vers 2 kW/jour (chauffage électrique + chauffe-eau), creux estival vers 0,5 kW/jour
- **Cycles hebdomadaires** : différentiel week-end/semaine visible dans la haute fréquence
- **Régime non-stationnaire** : moyenne et variance évoluent dans le temps → un LSTM (mémoire longue) est mieux adapté qu'un ARMA classique

**Implication produit** : la consommation d'un foyer FR est **fortement structurée temporellement** → forecast 24h faisable avec haute précision sans features externes (la conso passée porte 80% du signal).

#### 1.2 Distribution statistique de la consommation horaire

![Distribution active_kw](figures/phase1_eda/fig_phase1_distribution.png)

Histogramme sur 35k heures (4 ans). Skewness positive marquée : la majorité du temps un ménage consomme < 1 kW, mais quelques heures de pic (chauffage matin/soir) tirent la moyenne au-dessus de la médiane. Cette asymétrie justifie la **normalisation MinMax avant LSTM** (pas StandardScaler qui suppose une distribution gaussienne).

#### 1.3 Pattern circadien moyen

![Pattern horaire](figures/phase1_eda/fig_phase1_hourly_pattern.png)

Conso moyenne par heure de la journée sur 4 ans, avec band ±1 écart-type. **Deux pics nets** :
- **7h** : matin (douche + petit-déjeuner)
- **20h** : soir (cuisson + éclairage + TV)

Le creux nocturne (2h-5h) est très stable (faible σ), ce qui confirme que **le pattern temporel porte une signature de profil**. Cette régularité est ce que la Phase 6 exploite pour classifier les foyers en 4 profils.

### Phase 2 · LSTM forecast — convergence et diagnostic

#### 2.1 Courbes de convergence

![Courbes de loss LSTM](figures/phase2_lstm/fig_phase2_loss.png)

`train_loss` (MSE) descend en 2-3 epochs vers ~0.002 et se stabilise. La `val_loss` suit jusqu'à epoch 5 puis plateau sans divergence → pas d'overfit sur ce dataset (~27k séquences train). EarlyStopping (patience=5) coupe à epoch 7 en restaurant les meilleurs poids.

#### 2.2 Prédictions vs vérité terrain

![Forecast vs réalité](figures/phase2_lstm/fig_phase2_forecast.png)

Sur 200 heures de test set, la courbe prédite (orange) suit fidèlement la vérité terrain (noir) avec **RMSE test = 0,241 kW**. Sur une moyenne ménagère ~1 kW, cela correspond à **~24% d'erreur relative** — excellent vu la complexité du signal.

#### 2.3 Distribution des résidus

![Histogramme des résidus](figures/phase2_lstm/fig_phase2_residuals.png)

Distribution des erreurs `pred - reel` quasi-gaussienne centrée sur 0 → **pas de biais systématique** du modèle. L'écart-type ~0,24 kW est cohérent avec le RMSE. Cette propriété (résidus iid normaux) est attendue d'un bon modèle de régression et permet de construire des intervalles de confiance autour des prédictions futures.

#### 2.4 Scatter plot prédiction vs réalité

![Scatter pred vs true](figures/phase2_lstm/fig_phase2_scatter_pred_true.png)

Le nuage de points s'aligne sur la diagonale y=x avec une dispersion contrôlée. **Pas d'effet de saturation** aux extrêmes (le modèle prédit aussi bien les pics que les creux). C'est un excellent diagnostic visuel : un modèle qui sous-estime systématiquement les hautes valeurs (effet de saturation classique des MSE) aurait un nuage qui s'aplatit en haut. Ici ce n'est pas le cas.

### Phase 3 · LSTM vs GRU — benchmark cross-architecture

![GRU vs LSTM comparison](figures/phase3_gru_vs_lstm/fig_phase3_gru_vs_lstm.png)

**Verdict architecture** :

| Métrique | LSTM | GRU | Δ |
|---|---|---|---|
| Paramètres | 20 545 | 16 129 | **-21%** |
| RMSE test (kW) | 0,241 | 0,236 | équivalent |
| Durée / epoch (s, T4) | 2,15 | 2,02 | **-6%** |

Convergence quasi-identique sur les 7 epochs effectifs. **GRU choisi pour la prod mobile** : 21% moins de paramètres = empreinte ONNX plus légère + inference plus rapide sur device.

### Phase 5 · Bi-LSTM sentiment FR — Allociné 200k avis

#### 5.1 Distribution des longueurs de reviews

![Distribution longueurs reviews](figures/phase5_allocine/fig_phase5_review_lengths.png)

La médiane des reviews Allociné se situe autour de 80 tokens. La queue à droite (reviews > 500 tokens) représente <5% du corpus. Notre **padding `maxlen=200`** capture donc l'essentiel sans gonfler artificiellement les séquences courtes ni perdre l'information sur les longues. La stratégie `padding="pre"` place les zéros au début pour que le hidden state final du LSTM voie les vrais tokens.

#### 5.2 Convergence de l'entraînement

![Accuracy training](figures/phase5_allocine/fig_phase5_allocine_accuracy.png)

`train_acc` atteint 96% en 4 epochs, `val_acc` plateau à 93% dès l'epoch 1. L'écart train/val est sain (<3 points), pas d'overfit. EarlyStopping coupe à epoch 4 en restaurant les meilleurs poids.

#### 5.3 Matrice de confusion sentiment

![Confusion sentiment binary](figures/phase5_allocine/fig_phase5_confusion.png)

Les vrais positifs (Positif → Positif) et vrais négatifs (Négatif → Négatif) dominent largement la diagonale. Les faux positifs et faux négatifs sont équilibrés (~6-7% chacun), ce qui indique **un classifieur bien calibré** sans biais vers l'une des classes. Le seuil de décision 0,5 est approprié.

**Résultats finaux test set (20 000 avis)** :
- Accuracy : **93,3%** · F1-score : **0,93**
- Précision/rappel équilibrés sur les deux classes (Négatif / Positif)

**Transférabilité** : modèle entraîné sur avis de films FR, mais le vocabulaire générique (positif/négatif) se transfère bien aux avis Trustpilot des fournisseurs d'énergie (testé sur 4 avis simulés EDF/Engie/TotalEnergies dans le notebook).

### Phase 6 · LSTM multiclasse — profil temporel peak/off-peak

#### 6.1 Distribution du ratio peak/off-peak

![Distribution ratio](figures/phase6_profil/fig_phase6_ratio_distribution.png)

Distribution du ratio `peak(18-21h) / off-peak(1-6h)` sur toutes les fenêtres de 7 jours du dataset. Le ratio est continu et asymétrique (skew positif). Les **3 lignes verticales rouges/oranges/bleues** marquent les quartiles Q25/Q50/Q75, qui définissent les 4 classes de profil (Nocturne / Équilibre nuit / Équilibre soir / Vespertine). Cette discrétisation **garantit l'équilibre des classes en training**, mais le split temporel introduit un déséquilibre sur le test set.

#### 6.2 Matrice de confusion des profils

![Confusion 4 profils](figures/phase6_profil/fig_phase6_confusion.png)

**Tâche** : classer chaque fenêtre de 7 jours selon son ratio peak/off-peak en 4 quartiles.

**Performance** :
- Accuracy globale : 0,52 (vs baseline aléatoire 0,25)
- F1 macro : 0,47
- Le modèle excelle sur les **classes extrêmes** (Nocturne precision 0,87) et a plus de mal sur les classes intermédiaires

**Lecture matrice** : les erreurs sont **majoritairement entre classes adjacentes** (Nocturne ↔ Équilibre nuit) ce qui est attendu. Aucune confusion entre classes opposées (Nocturne ↔ Vespertine) → le modèle a appris l'axe principal du ratio temporel.

**Roadmap V2** : enrichir le signal avec features météo (température extérieure depuis Météo France API) pour réduire l'ambiguïté entre profils intermédiaires.

## Reproductibilité

| Paramètre | Valeur |
|---|---|
| Seed numpy / Keras | `42` (fixé via `np.random.seed(42)` + `keras.utils.set_random_seed(42)`) |
| Python | 3.11 (Streamlit Cloud) / 3.10 (Kaggle) |
| TensorFlow | `2.17.0` |
| Keras | `>=3.5,<3.6` (Keras 3 stable) |
| Hardware training | Kaggle Tesla T4 x2 (gratuit) |
| Durée totale Run All | ~25-30 min (5 phases) |
| Dataset principal | `uciml/electric-power-consumption-data-set` (Kaggle, MD5 inchangé depuis 2017) |
| Dataset NLP | `tblard/allocine` (Hugging Face, version stable) |

### Comment reproduire exactement les résultats

1. Cloner le repo : `git clone https://github.com/Hakim78/tp4-monenergie.git`
2. Sur Kaggle : New Notebook → Settings → GPU T4 x2 + Internet ON + Add Data `electric-power-consumption-data-set`
3. File → Import Notebook → upload `notebooks/tp4_monenergie_all_phases.ipynb`
4. **Run All** → ~28 min sur T4
5. Les artefacts (modèles + figures + `results.md`) sont auto-poussés sur GitHub via le secret `GITHUB_TOKEN` (cellule finale)

### Note sur la variance des résultats

Avec le seed `42` fixé, les résultats devraient être **bit-exacts reproductibles** sur même hardware (T4 Kaggle). Sur un GPU différent (V100, A100), des micro-différences de l'ordre de 1e-4 sont possibles à cause des kernels CUDA non-déterministes (matmul, conv). Le RMSE de la Phase 2 reste dans la fenêtre 0.23-0.25 kW et l'accuracy Allociné dans 0.92-0.94.

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

## Roadmap data : V1 → V2 → V3

Le TP4 est un **proof-of-concept** sur un seul foyer (UCI Sceaux, données publiques). Voici comment la stack data évolue ensuite vers un produit commercial réel.

### V1 — Aujourd'hui (TP4, training)

| Source | Pourquoi | Licence |
|---|---|---|
| UCI Sceaux Household Power | Seul dataset open avec courbe minute-par-minute individuelle française sur 4 ans | CC0 |
| Allociné Reviews (HF) | Bi-LSTM sentiment FR transférable aux avis Trustpilot fournisseurs | Public |
| Météo France API (optionnel) | Features température pour le forecast | Etalab 2.0 |

→ Démontre que la pipeline LSTM marche. Pas encore commercialisable.

### V2 — MVP commercial (3-6 mois)

Le pivot clé : passer de "1 foyer historique" à "N utilisateurs réels".

| Source | Usage | Comment |
|---|---|---|
| **API Conso&Henry / Linky** | Récupérer les courbes individuelles des utilisateurs consentants | OAuth via [Datahub Enedis](https://datahub-enedis.fr/data-connect/documentation/). 30s d'auth, données minute-par-minute sur 3 ans |
| **Enedis Open Data** | Benchmarks "votre conso vs voisins" type T3 électrique | Conso annuelle par IRIS, profilage par type de logement |
| **Météo France** | Features DJU/température en features du modèle | API gratuite, station la plus proche du code postal |
| **ADEME ObsDPE** | Cross-référencer le DPE du logement | 18M DPE A-G en open data |

→ Stade commercial : 4,99€/mois B2C, mise en avant "connectez votre Linky, gagnez 200€/an".

### V3 — Scale (12-24 mois)

| Source | Usage |
|---|---|
| **GRDF Open Data** | Étendre au gaz (~11M ménages chauffés gaz FR) |
| **GreenSpark API** | Intégration capteurs IoT (panneaux solaires, batteries domestiques) |
| **Trustpilot scraping (modéré)** | Sentiment fournisseurs en temps réel (module "Quel fournisseur choisir") |
| **INSEE Filosofi** | Croiser revenus de l'IRIS pour adapter les recos (un foyer modeste vs aisé n'ont pas les mêmes leviers) |

→ Stade scale : 9,99€/mois B2C + 49€/mois B2B (syndics, bailleurs sociaux, artisans).

## Aspects légaux et RGPD

- Les courbes Linky restent la **propriété de l'utilisateur**. Tu accèdes via OAuth avec consentement explicit (révocable).
- Aucune courbe individuelle Enedis n'est publique (le RGPD interdit). Tout ce qui est en Open Data est **agrégé** par IRIS/commune/profil.
- L'agrégat Enedis Open Data permet du **benchmarking comparatif** sans toucher aux données individuelles d'autres ménages.
- **DPO obligatoire** au-delà de 250 utilisateurs actifs (peut être externalisé ~3000€/an).

## Pourquoi cette stack rend MonÉnergie défendable

1. **API Linky = barrière à l'entrée**. Hello Watt et Selectra l'utilisent, mais aucun ne fait du Deep Learning par-dessus pour personnaliser le forecast (ils font du SQL simple).
2. **Données 100% françaises** = avantage GDPR + AI Act EU (hébergement EU obligatoire = moat vs concurrents US/CN).
3. **Triple modèle (forecast + profil + sentiment)** = l'app couvre 3 use cases sur 1 acquisition utilisateur. ARPU augmenté.
4. **Open data Etalab 2.0** = pas de licence à payer, **0€ de COGS** sur les benchmarks et features externes.
