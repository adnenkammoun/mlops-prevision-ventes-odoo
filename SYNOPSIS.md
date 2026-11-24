# SYNOPSIS — Plateforme MLOps de prévision des ventes à partir de données ERP Odoo

**Module :** MLOps & Industrialisation de l'Intelligence Artificielle — 2026-2027
**Niveau :** Bac+5 Data Science & IA — TEK-UP
**Équipe :** 1 membre — Adnen Kammoun (responsabilité polyvalente / pilotage global)

---

## 1. Contexte métier

Un distributeur de fournitures scolaires et de bureau en Tunisie vend plusieurs centaines de références
à des librairies, grandes surfaces, grossistes, entreprises et écoles. Les ventes, gérées dans l'ERP Odoo,
sont très saisonnières : forte hausse à la rentrée scolaire (juillet–septembre), pic des agendas en fin d'année,
baisse pendant le Ramadan.

Les achats sont aujourd'hui décidés surtout à l'expérience, ce qui provoque :
- des **ruptures de stock** (livraisons bloquées, ventes perdues) ;
- du **surstock** (trésorerie et espace immobilisés).

## 2. Problème et objectif

**Prédire, pour chaque produit, la quantité vendue la semaine suivante (S+1)**, afin d'aider le service achats
à planifier le réapprovisionnement.

**Décision assistée :** si la demande prévue dépasse le stock disponible + stock de sécurité, le système génère
une alerte de rupture et propose une quantité à commander, validée par le responsable achats.

## 3. Formulation ML

| Élément | Valeur |
|---|---|
| Type | Régression |
| Cible | `cible_qte_s1` — quantité vendue du produit la semaine suivante |
| Granularité | 1 ligne = 1 produit × 1 semaine |
| Découpage | Temporel : entraînement 2024–2025, test 2026 |
| Baseline | Prévision naïve : ventes de la semaine précédente (`lag_1`) |

## 4. Dataset

- **Source :** dataset synthétique généré par `src/generate_fake_odoo.py`, reproduisant la structure exacte
  d'un export de ventes Odoo 19 (commandes + lignes de commande). Le pipeline accepte un vrai export Odoo sans modification.
- **Données brutes :** `data/raw/ventes_odoo.xlsx` — 70 663 lignes de commande, 320 produits, 220 clients, 2024–2026.
- **Dataset ML :** `data/processed/demande_hebdo.csv` — **46 574 lignes × 18 colonnes**, produit par `src/prepare_dataset.py`.

| Type | Variables |
|---|---|
| Numériques (14) | `qte`, `prix_unitaire`, `remise`, `nb_clients`, `mois`, `semaine_annee`, `rentree`, `lag_1`, `lag_2`, `lag_4`, `lag_52`, `moy_4`, `moy_12`, `cible_qte_s1` |
| Catégorielles (3) | `produit` (anonymisé), `categorie` (12), `marque` (10) |
| Date (1) | `semaine` |

**Qualité :** pas de doublon produit–semaine ; valeurs manquantes uniquement dans `lag_52` et `moy_12`
(historique insuffisant en début de série) ; cible très asymétrique avec ~55 % de semaines à zéro (demande intermittente).

## 5. Métriques visées

- **ML :** MAE et RMSE au moins 20 % meilleures que la baseline naïve ; WAPE ≤ 35 % ; R² ≥ 0,75.
- **Métier :** détecter ≥ 85 % des produits à risque de rupture en période de rentrée.
- **Technique :** latence API < 200 ms ; ré-entraînement hebdomadaire automatique.

## 6. Architecture MLOps

```
Export Odoo (xlsx) ─► DVC (versioning, remote DagsHub)
        │
        ▼
 prepare (nettoyage, anonymisation, features) ─► train ─► evaluate
        │                                          │
        │                                   MLflow Tracking + Model Registry (alias "champion")
        ▼                                          │
 Prefect 3.x (flow hebdomadaire, CRON + événement) ▼
        │                                FastAPI (/predict, /predict/batch, /health, /model-info, /metrics, /reload)
        │                                          │
        ▼                                    Docker (multi-stage, non-root) ─► GHCR
 Evidently (KS, Chi², PSI) ─► Prometheus ─► Grafana (alertes)
                                                   ▲
 GitHub Actions : ruff + pytest (couverture ≥ 75 %) + CML + build Docker + Trivy
```

| Brique | Outil |
|---|---|
| Versioning des données | DVC (remote DagsHub) |
| Tracking & Registry | MLflow |
| Modèles comparés | Baseline naïve, Ridge, Random Forest, XGBoost, LightGBM |
| Serving | FastAPI + Pydantic + Swagger |
| Conteneurisation | Docker multi-stage, utilisateur non-root, Docker Compose |
| CI/CD | GitHub Actions, Pytest, CML |
| Orchestration | Prefect 3.x |
| Monitoring | Evidently, Prometheus (OpenMetrics), Grafana |

## 7. Data drift et feedback loop

- **Scénario :** hausse des prix fournisseurs, nouveaux produits/marques, décalage du pic de rentrée, changement du comportement des gros clients.
- **Détection :** KS (numériques), Chi² (catégorielles), PSI.
- **Seuils :** PSI ≥ 0,20 sur une variable clé ou sur les prédictions ; ≥ 30 % des variables en dérive ; hausse de la MAE > 15 %.
- **Simulation :** partitionnement temporel 2024–2025 vs 2026 + injection contrôlée (prix +15 %, quantités −25 % sur une catégorie, nouvelles marques, décalage de la rentrée).
- **Feedback loop :** détection ► alerte ► ré-entraînement automatique (Prefect) ► comparaison challenger / champion ► promotion dans MLflow si gain ≥ 5 % ► rechargement à chaud de l'API.

## 8. Déploiement

Docker Compose. Nouveau modèle : rechargement à chaud via `/reload` sans redémarrage. Nouvelle version du code :
démarrage du nouveau conteneur, vérification de `/health`, arrêt de l'ancien. Rollback par réaffectation de l'alias
`champion` à la version précédente dans MLflow.

## 9. Planning prévisionnel

| Semaine | Jalon |
|---|---|
| S1 | Cadrage, dataset, dépôt GitHub, DVC, EDA |
| S2 | Entraînement baseline + modèles, MLflow Tracking & Registry |
| S3 | API FastAPI + tests Pytest + Dockerfile |
| S4 | CI/CD GitHub Actions + CML |
| S5 | Orchestration Prefect |
| S6 | Monitoring Evidently + Prometheus + Grafana, simulation du drift |
| S7 | Finalisation, documentation, soutenance |

## 10. Risques identifiés

| Risque | Solution envisagée |
|---|---|
| Dataset synthétique | Pipeline conçu pour un vrai export Odoo |
| Demande intermittente (beaucoup de zéros) | Transformation log(1+x), métriques adaptées (MAE, WAPE) |
| Charge de travail pour 1 personne | Priorisation : DVC, MLflow, API, Docker, CI d'abord ; Prefect et monitoring ensuite |
| Ressources de calcul limitées | Modèles légers (arbres boostés), entraînement local |
