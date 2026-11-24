"""
Génère un export Odoo FICTIF (ventes d'un distributeur de fournitures scolaires et de bureau en Tunisie),
au même format que : Ventes > Commandes > Exporter (XLSX).

- 3 ans d'historique (01/2024 -> 11/2026), dimanches et jours fériés exclus
- Saisonnalité forte de la rentrée scolaire (juillet-septembre), effet Ramadan, hausse des prix chaque année
- Types de clients : librairie, grande surface, grossiste, entreprise, école
- Certains clients arrêtent d'acheter (churn) -> utile pour une 2e problématique ML

Usage : python generate_fake_odoo.py  -> data/raw/ventes_odoo.xlsx
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
START, END = "2024-01-02", "2026-11-20"

# ---------------- Catalogue ----------------
CATEGORIES = {
    # catégorie: (saison "scolaire" ou "bureau", prix min, prix max, qté moyenne par ligne)
    "Scolaire / Cahiers": ("scolaire", 0.8, 6.0, 60),
    "Scolaire / Cartables & Sacs": ("scolaire", 25.0, 120.0, 6),
    "Scolaire / Trousses": ("scolaire", 5.0, 25.0, 12),
    "Scolaire / Coloriage & Peinture": ("scolaire", 2.0, 18.0, 20),
    "Scolaire / Pâte à modeler": ("scolaire", 1.5, 12.0, 24),
    "Écriture / Stylos": ("mixte", 0.4, 8.0, 80),
    "Écriture / Crayons & Gommes": ("mixte", 0.3, 4.0, 90),
    "Écriture / Marqueurs & Surligneurs": ("mixte", 1.2, 9.0, 36),
    "Bureau / Ramettes papier": ("bureau", 11.0, 22.0, 25),
    "Bureau / Classement": ("bureau", 2.0, 15.0, 20),
    "Bureau / Petit matériel": ("bureau", 1.5, 35.0, 15),
    "Bureau / Agendas & Calendriers": ("fin_annee", 6.0, 40.0, 10),
}
BRANDS = ["Atlas", "Carthago", "Jasmin", "Medina", "Oasis", "Sahara", "Zitouna", "Nova", "Kidoo", "Prisma"]
N_PRODUCTS = 320

products = []
for i in range(N_PRODUCTS):
    cat = rng.choice(list(CATEGORIES))
    saison, pmin, pmax, qmoy = CATEGORIES[cat]
    brand = rng.choice(BRANDS)
    products.append({
        "nom": f"[ART-{1000 + i}] {cat.split(' / ')[1].rstrip('s')} {brand} réf.{rng.integers(10, 99)}",
        "categorie": cat, "marque": brand, "saison": saison,
        "prix": round(float(rng.uniform(pmin, pmax)), 3),
        "qmoy": qmoy,
        "popularite": float(rng.pareto(1.5) + 0.2),          # quelques best-sellers, beaucoup de petits articles
        "lancement": pd.Timestamp(START) + pd.Timedelta(days=int(rng.choice([0] * 8 + [200, 400, 600]))),
    })
prod = pd.DataFrame(products)

# ---------------- Clients ----------------
TYPES = {
    # type: (part, nb lignes moyen, multiplicateur qté, remise min/max %, fréquence relative)
    "Librairie": (0.50, 5, 1.0, (0, 5), 1.0),
    "Grande surface": (0.08, 12, 4.0, (5, 12), 3.0),
    "Grossiste": (0.07, 9, 6.0, (10, 18), 2.5),
    "Entreprise": (0.20, 4, 0.8, (0, 8), 0.6),
    "École": (0.15, 6, 1.5, (3, 10), 0.4),
}
VILLES = ["Tunis", "Ben Arous", "Ariana", "Manouba", "Nabeul", "Sousse", "Sfax", "Monastir", "Bizerte", "Kairouan"]
N_CLIENTS = 220
clients = []
for i in range(N_CLIENTS):
    t = rng.choice(list(TYPES), p=[v[0] for v in TYPES.values()])
    debut = pd.Timestamp(START) + pd.Timedelta(days=int(rng.choice([0] * 6 + list(range(30, 800, 60)))))
    fin = pd.Timestamp(END) + pd.Timedelta(days=1)
    if rng.random() < 0.18:                                   # ~18 % de clients perdus
        fin = debut + pd.Timedelta(days=int(rng.integers(150, 900)))
    clients.append({"nom": f"{t} {rng.choice(['El', 'Dar', 'Al', 'Les', 'Sté'])} "
                           f"{rng.choice(['Nour', 'Amal', 'Wafa', 'Najah', 'Ilm', 'Fajr', 'Riadh', 'Salem', 'Yasmine', 'Hana'])}"
                           f" {rng.choice(VILLES)} {i:03d}",
                    "type": t, "debut": debut, "fin": fin,
                    "poids": TYPES[t][4] * float(rng.lognormal(0, 0.5))})
cli = pd.DataFrame(clients)

COMMERCIAUX = [f"Commercial {n}" for n in ["Ahmed B.", "Sami K.", "Mouna T.", "Walid H.", "Rania M.", "Karim S.", "Nadia F."]]
cli["commercial"] = rng.choice(COMMERCIAUX, size=len(cli))

# ---------------- Saisonnalité ----------------
MOIS_GLOBAL = {1: .95, 2: .85, 3: .85, 4: .85, 5: .75, 6: .8, 7: 1.4, 8: 2.4, 9: 2.1, 10: 1.1, 11: .95, 12: 1.05}
MOIS_SAISON = {
    "scolaire": {1: .6, 2: .5, 3: .5, 4: .5, 5: .4, 6: .6, 7: 1.8, 8: 3.2, 9: 2.6, 10: .9, 11: .6, 12: .6},
    "mixte": {m: 1.0 for m in range(1, 13)} | {8: 1.8, 9: 1.7, 7: 1.3},
    "bureau": {m: 1.0 for m in range(1, 13)} | {1: 1.3, 9: 1.2, 8: .8},
    "fin_annee": {m: .3 for m in range(1, 13)} | {11: 2.5, 12: 3.5, 1: 1.5},
}
RAMADAN = [("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-30"), ("2026-02-18", "2026-03-19")]
FERIES = {"2024-01-01", "2024-03-20", "2024-04-09", "2024-04-10", "2024-05-01", "2024-07-25", "2024-08-13",
          "2025-01-01", "2025-03-20", "2025-03-31", "2025-04-01", "2025-05-01", "2025-07-25", "2025-08-13",
          "2026-01-01", "2026-03-20", "2026-05-01", "2026-05-27", "2026-07-25", "2026-08-13"}


def is_ramadan(d):
    return any(pd.Timestamp(a) <= d <= pd.Timestamp(b) for a, b in RAMADAN)


# ---------------- Génération ----------------
rows, order_no = [], 1
for d in pd.date_range(START, END, freq="D"):
    if d.dayofweek == 6 or d.strftime("%Y-%m-%d") in FERIES:
        continue
    tendance = 1 + 0.08 * (d - pd.Timestamp(START)).days / 365          # croissance ~8 %/an
    f = MOIS_GLOBAL[d.month] * tendance * (0.7 if is_ramadan(d) else 1) * (0.6 if d.dayofweek == 5 else 1)
    n_orders = rng.poisson(9 * f)
    actifs = cli[(cli["debut"] <= d) & (cli["fin"] > d)]
    if actifs.empty or n_orders == 0:
        continue
    inflation = 1.06 ** (d.year - 2024)                                  # hausse des prix chaque janvier
    pdispo = prod[prod["lancement"] <= d]
    w_prod = pdispo["popularite"] * pdispo["saison"].map(lambda s: MOIS_SAISON[s][d.month])
    w_prod = (w_prod / w_prod.sum()).values

    for _ in range(n_orders):
        c = actifs.sample(1, weights="poids", random_state=int(rng.integers(1e9))).iloc[0]
        _, nlig, qmult, (rmin, rmax), _ = TYPES[c["type"]]
        n_lines = max(1, min(len(pdispo), rng.poisson(nlig * (1.3 if d.month in (8, 9) else 1))))
        idx = rng.choice(len(pdispo), size=n_lines, replace=False, p=w_prod)
        heure = pd.Timedelta(hours=int(rng.integers(8, 17)), minutes=int(rng.integers(0, 60)))
        remise_client = float(rng.integers(rmin, rmax + 1))
        for j, k in enumerate(idx):
            p = pdispo.iloc[k]
            qty = max(1, int(rng.poisson(p["qmoy"] * qmult * rng.uniform(0.4, 1.6))))
            promo = 5.0 if (d.month == 8 and rng.random() < 0.15) else 0.0
            remise = min(remise_client + promo, 25.0)
            pu = round(p["prix"] * inflation, 3)
            rows.append({
                "Référence de la commande": f"S{order_no:05d}" if j == 0 else None,
                "Date de la commande": (d + heure) if j == 0 else None,
                "Client": c["nom"] if j == 0 else None,
                "Vendeur": c["commercial"] if j == 0 else None,
                "Lignes de commande/Article": p["nom"],
                "Lignes de commande/Article/Catégorie de produits": p["categorie"],
                "Lignes de commande/Article/Marque": p["marque"],
                "Lignes de commande/Quantité": qty,
                "Lignes de commande/Prix unitaire": pu,
                "Lignes de commande/Remise (%)": remise,
                "Lignes de commande/Sous-total": round(qty * pu * (1 - remise / 100), 3),
            })
        order_no += 1

df = pd.DataFrame(rows)
out = Path("data/raw")
out.mkdir(parents=True, exist_ok=True)
df.to_excel(out / "ventes_odoo.xlsx", index=False)
print(f"{order_no - 1} commandes | {len(df)} lignes | {prod.shape[0]} articles | {cli.shape[0]} clients")
print("Fichier :", (out / "ventes_odoo.xlsx").resolve())
