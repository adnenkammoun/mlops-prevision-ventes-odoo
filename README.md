# Plateforme MLOps de prévision des ventes à partir de données ERP Odoo

Projet de fin de module MLOps — TEK-UP 2026-2027. Voir [SYNOPSIS.md](SYNOPSIS.md).

## Démarrage rapide
```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
dvc repro            # génère les données brutes puis le dataset ML
pytest               # lance les tests
```

## Structure
```
src/generate_fake_odoo.py   génération d'un export Odoo synthétique
src/prepare_dataset.py      nettoyage, anonymisation, feature engineering, cible S+1
data/raw/                   données brutes (DVC)
data/processed/             dataset ML (DVC)
notebooks/                  EDA
tests/                      tests Pytest
dvc.yaml / params.yaml      pipeline de données
```
