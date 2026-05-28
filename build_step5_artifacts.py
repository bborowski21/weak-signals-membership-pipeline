
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).parent
PARENT_DIR = BASE_DIR.parent

PHASES = {
    "1": {
        "data_csv": "wos_qc_phase1_2000_2015.csv",
        "output_dir": "output_phase1",
        "label": "Phase 1 (2000-2015)",
        "year_min": 2000,
        "year_max": 2015,
    },
    "2": {
        "data_csv": "wos_qc_phase2_2016_2025.csv",
        "output_dir": "output_phase2",
        "label": "Phase 2 (2016-2025)",
        "year_min": 2016,
        "year_max": 2025,
    },
}


def usage_and_exit() -> None:
    print(__doc__)
    sys.exit(1)


def parse_phase_arg() -> str:
    if len(sys.argv) < 2:
        usage_and_exit()
    arg = sys.argv[1].strip().lower().replace("phase", "").replace("p", "").strip()
    if arg not in PHASES:
        print(f"Unbekanntes Phasen-Argument: {sys.argv[1]!r}")
        usage_and_exit()
    return arg


def load_and_filter_df(data_path: Path, year_min: int, year_max: int) -> pd.DataFrame:
    print(f"Lade {data_path.name} ...")
    df = pd.read_csv(data_path)
    n0 = len(df)

    df["text"] = df["Title"].fillna("") + ". " + df["Abstract"].fillna("")
    df["text_clean"] = df["text"].apply(
        lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s\-]", " ", t.lower())).strip()
        if isinstance(t, str) else ""
    )

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[(df["Year"] >= year_min) & (df["Year"] <= year_max)].copy()
    n_year = len(df)

    df = df[df["text_clean"].str.split().str.len() >= 10].reset_index(drop=True)
    print(f"  {n0} → {n_year} → {len(df)} (Year-Filter [{year_min},{year_max}] "
          f"+ Längen-Filter ≥ 10 Wörter)")
    return df


def load_model_results(output_dir: Path) -> dict:
    path = output_dir / "model_results.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def build_step1_artifact(output_dir: Path, data_path: Path,
                          year_min: int, year_max: int) -> None:
    df = load_and_filter_df(data_path, year_min, year_max)

    canon_columns = {
        "Publication Year":      "Year",
        "Author Full Names":     "Authors",
        "Affiliations":          "Affiliations",
        "Addresses":             "Addresses",
        "Source Title":          "Source title",
        "Document Type":         "Document Type",
        "WoS Categories":        "WoS Categories",
        "Author Keywords":       "Author Keywords",
        "Keywords Plus":         "Keywords Plus",
        "Times Cited, WoS Core": "Times Cited",
        "ORCIDs":                "ORCIDs",
        "Funding Orgs":          "Funding Orgs",
        "RTW":                   "RTW",
        "CTW":                   "CTW",
    }
    added = []
    for src, dst in canon_columns.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
            added.append(f"{src!r}→{dst!r}")
    if added:
        print(f"  Kanonische Alias-Spalten ergänzt: {', '.join(added)}")

    model = load_model_results(output_dir)
    labels = np.asarray(model["labels"])
    embeddings_sbert = np.asarray(model["embeddings_sbert"])

    n_df, n_lab, n_emb = len(df), len(labels), len(embeddings_sbert)
    if not (n_df == n_lab == n_emb):
        print(f"FEHLER: Längen-Mismatch — df={n_df}, labels={n_lab}, "
              f"embeddings={n_emb}")
        print("  Filterlogik in step01 hat sich vermutlich geändert; "
              "build_step5_artifacts.py muss aktualisiert werden.")
        sys.exit(3)

    artifact = {
        "df": df,
        "labels": labels,
        "embeddings_sbert": embeddings_sbert,
    }
    out_path = output_dir / "step1_artifacts.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"  → {out_path.name}: df={n_df}, labels={n_lab}, "
          f"embeddings_sbert={embeddings_sbert.shape}")


def verify_membership_artifacts(output_dir: Path) -> None:
    indicator_df = pd.read_csv(
        output_dir / "indicators_16.csv", index_col=0)
    dim_scores = pd.read_csv(
        output_dir / "dimension_scores.csv", index_col=0)
    memberships = pd.read_csv(
        output_dir / "signal_memberships.csv", index_col=0)

    required_memb_cols = {"m_ws", "m_trend", "m_ec", "m_latent", "margin"}
    if not required_memb_cols.issubset(set(memberships.columns)):
        print(f"FEHLER: signal_memberships.csv unvollständig. "
              f"Vorhandene Spalten: {list(memberships.columns)}; "
              f"erwartet: {sorted(required_memb_cols)}")
        sys.exit(4)

    print(f"  indicators_16.csv     : {indicator_df.shape}")
    print(f"  dimension_scores.csv  : {dim_scores.shape}")
    print(f"  signal_memberships.csv: {memberships.shape}  "
          f"(Spalten: {list(memberships.columns)})")


def main() -> None:
    phase_key = parse_phase_arg()
    cfg = PHASES[phase_key]
    data_path = PARENT_DIR / cfg["data_csv"]
    output_dir = BASE_DIR / cfg["output_dir"]

    print("=" * 70)
    print(f"STEP-5-ARTEFAKT-BUILDER — {cfg['label']} (Pipeline V2)")
    print("=" * 70)
    print(f"  Eingabe-CSV : {data_path}")
    print(f"  Output-Dir  : {output_dir}")
    print()

    required = [
        "model_results.pkl",
        "indicators_16.csv",
        "dimension_scores.csv",
        "signal_memberships.csv",
    ]
    missing = [f for f in required if not (output_dir / f).exists()]
    if missing:
        print(f"FEHLER: Voraussetzungen in {output_dir} unvollständig.")
        print(f"  Fehlend: {', '.join(missing)}")
        sys.exit(2)
    if not data_path.exists():
        print(f"FEHLER: Eingabe-CSV nicht gefunden: {data_path}")
        sys.exit(2)

    print("--- step1_artifacts.pkl ---")
    build_step1_artifact(output_dir, data_path, cfg["year_min"], cfg["year_max"])

    print("\n--- Validierung Membership-Artefakte (kein step2_artifacts.pkl in V2) ---")
    verify_membership_artifacts(output_dir)

    print()
    print("=" * 70)
    print(f"FERTIG: Step-5-Artefakte für {cfg['label']} (Pipeline V2)")
    print(f"  Ergebnisse in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
