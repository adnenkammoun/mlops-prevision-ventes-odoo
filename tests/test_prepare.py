import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare_dataset as p  # noqa: E402


def _fake_lines():
    rows = []
    for i, d in enumerate(pd.date_range("2024-01-01", periods=120, freq="3D")):
        rows.append({"date": d, "client": f"C{i % 5}", "produit": "A", "qte": 10 + i % 7,
                     "prix_unitaire": 2.0, "remise": 0.0, "categorie": "Stylos"})
    return pd.DataFrame(rows)


def test_anon_stable_et_non_reversible():
    a = p.anon("CLI", "Librairie X")
    assert a == p.anon("CLI", "Librairie X")
    assert a.startswith("CLI_") and "Librairie" not in a


def test_cible_est_qte_semaine_suivante():
    w = p.build_weekly(_fake_lines(), min_weeks=5)
    w = w.sort_values("semaine").reset_index(drop=True)
    assert len(w) > 0
    assert w.loc[0, "cible_qte_s1"] == w.loc[1, "qte"]


def test_pas_de_fuite_lag1():
    w = p.build_weekly(_fake_lines(), min_weeks=5).sort_values("semaine").reset_index(drop=True)
    assert w.loc[1, "lag_1"] == w.loc[0, "qte"]
