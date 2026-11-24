"""
Projet MLOps - Préparation du dataset à partir d'un export Excel Odoo (Saphir).

Entrée  : data/raw/ventes_odoo.xlsx  (export Odoo : Ventes > Commandes > Exporter)
Sorties : data/interim/ventes_anonymes.csv   (une ligne = une ligne de commande, anonymisée)
          data/processed/demande_hebdo.csv   (dataset ML : demande par produit et par semaine)

Usage :
    python prepare_dataset.py --input data/raw/ventes_odoo.xlsx
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

SALT = os.environ.get("ANON_SALT", "change-me")  # à garder hors du dépôt Git

# Noms des colonnes tels qu'ils sortent de l'export Odoo (en français).
# Si un nom diffère dans ton fichier, corrige-le seulement ici.
COLS = {
    "Date de la commande": "date",
    "Client": "client",
    "Vendeur": "commercial",
    "Lignes de commande/Article": "produit",
    "Lignes de commande/Article/Catégorie de produits": "categorie",
    "Lignes de commande/Article/Marque": "marque",
    "Lignes de commande/Quantité": "qte",
    "Lignes de commande/Prix unitaire": "prix_unitaire",
    "Lignes de commande/Remise (%)": "remise",
    "Lignes de commande/Sous-total": "sous_total",
}
ORDER_LEVEL = ["date", "client", "commercial"]  # vides sur les lignes 2, 3... d'une commande


def anon(prefix, value):
    if pd.isna(value) or value == "":
        return None
    h = hashlib.sha256(f"{SALT}:{prefix}:{value}".encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


def load_and_clean(path):
    df = pd.read_excel(path)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        print("Colonnes absentes (ignorées) :", missing)
    df = df.rename(columns={k: v for k, v in COLS.items() if k in df.columns})

    # L'export Odoo n'écrit les infos de la commande que sur la 1re ligne -> on recopie vers le bas
    for c in ORDER_LEVEL:
        if c in df.columns:
            df[c] = df[c].ffill()

    df = df.dropna(subset=["produit", "qte"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["qte"] > 0]

    # Anonymisation
    df["client"] = df["client"].map(lambda v: anon("CLI", v))
    if "commercial" in df.columns:
        df["commercial"] = df["commercial"].map(lambda v: anon("COM", v))
    df["produit"] = df["produit"].map(lambda v: anon("PRD", v))
    return df


def build_weekly(df, min_weeks=10):
    df["semaine"] = df["date"].dt.to_period("W-SUN").dt.start_time  # semaine du lundi au dimanche
    agg = {"qte": "sum", "prix_unitaire": "mean", "client": "nunique"}
    if "remise" in df.columns:
        agg["remise"] = "mean"
    w = df.groupby(["produit", "semaine"]).agg(agg).rename(columns={"client": "nb_clients"}).reset_index()

    # Compléter les semaines sans vente par 0 (important pour la prévision)
    all_weeks = pd.date_range(w["semaine"].min(), w["semaine"].max(), freq="W-MON")
    idx = pd.MultiIndex.from_product([w["produit"].unique(), all_weeks], names=["produit", "semaine"])
    w = w.set_index(["produit", "semaine"]).reindex(idx).reset_index()
    w["qte"] = w["qte"].fillna(0)
    w["nb_clients"] = w["nb_clients"].fillna(0)
    w["prix_unitaire"] = w.groupby("produit")["prix_unitaire"].transform(lambda s: s.ffill().bfill())
    if "remise" in w.columns:
        w["remise"] = w["remise"].fillna(0)

    # Infos produit (catégorie, marque)
    info_cols = [c for c in ["categorie", "marque"] if c in df.columns]
    if info_cols:
        info = df.groupby("produit")[info_cols].first().reset_index()
        w = w.merge(info, on="produit", how="left")

    # Variables de calendrier (saisonnalité rentrée scolaire)
    w["mois"] = w["semaine"].dt.month
    w["semaine_annee"] = w["semaine"].dt.isocalendar().week.astype(int)
    w["rentree"] = w["mois"].isin([8, 9]).astype(int)

    # Lags et moyennes mobiles (uniquement le passé -> pas de fuite de données)
    w = w.sort_values(["produit", "semaine"])
    g = w.groupby("produit")["qte"]
    for k in [1, 2, 4, 52]:
        w[f"lag_{k}"] = g.shift(k)
    w["moy_4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    w["moy_12"] = g.transform(lambda s: s.shift(1).rolling(12).mean())

    # Cible : quantité de la semaine suivante
    w["cible_qte_s1"] = g.shift(-1)

    # Garder les produits avec assez d'historique
    actifs = w.groupby("produit")["qte"].apply(lambda s: (s > 0).sum())
    w = w[w["produit"].isin(actifs[actifs >= min_weeks].index)]
    return w.dropna(subset=["lag_4", "moy_4", "cible_qte_s1"])


def main(path):
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    df = load_and_clean(path)
    df.to_csv("data/interim/ventes_anonymes.csv", index=False)

    w = build_weekly(df)
    w.to_csv("data/processed/demande_hebdo.csv", index=False)

    print(f"Lignes de vente : {len(df)} | Produits : {df['produit'].nunique()} | "
          f"Période : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Dataset ML : {w.shape[0]} lignes x {w.shape[1]} colonnes, "
          f"{w['produit'].nunique()} produits retenus")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/ventes_odoo.xlsx")
    main(ap.parse_args().input)
