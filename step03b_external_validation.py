"""
SCHRITT 3b: Externe Validierung der Indikator-Konstrukte (RTW/CTW)
====================================================================

Zweck
-----
Prüft die Konstruktvalidität der intern berechneten Indikatoren gegen die
paper-level von Clarivate vorberechneten Topic-Breitenmaße:

  - RTW (Reference Topic Width)
        Streuung der Citation-Topics, denen die *Referenzen* eines Papers
        zugeordnet sind. Operativ: Maß für die *intellektuelle Vielfalt
        der Inputs* eines Papers.
  - CTW (Citation Topic Width)
        Streuung der Citation-Topics, denen die *Zitationen* eines Papers
        zugeordnet sind. Operativ: Maß für die *intellektuelle Vielfalt
        der Outputs* eines Papers.

Aggregation auf Topic-Ebene erfolgt via Median der paper-level Werte
(robust gegenüber Outliern in den stark rechtsschiefen RTW/CTW-Verteilungen).

Validierungsmuster (MTMM-Logik, Campbell & Fiske 1959)
------------------------------------------------------
  Konvergent erwartet (ρ ≥ EXTVAL_CONVERGENT_MIN):
    - RTW vs. EO2 disciplinary_entropy        (beide: Inputs-Diversität)
    - CTW vs. WP3 field_breadth                (beide: Outputs-Diversität)

  Diskriminant erwartet (|ρ| ≤ EXTVAL_DISCRIMINANT_MAX):
    - RTW/CTW vs. DI3 geographic_concentration  (Inhaltsvs. geographische Diffusion)
    - RTW/CTW vs. EP1 temporal_novelty           (Inhaltsvs. temporale Lage)

Output
------
  external_validation_paper.csv      Paper-level RTW/CTW + Topic-Zuordnung
  external_validation_topic.csv      Topic-Median RTW/CTW + Indikator-Joinpunkt
  external_validation_corr.csv       Spearman-ρ-Matrix (RTW/CTW × Indikatoren)
  external_validation_mtmm.csv       MTMM-Diagnostik: Konvergenz/Diskriminanz
  external_validation_report.md      Konsolidierter Bericht (Anhangsmaterial)

Literatur
---------
  Campbell, D. T., & Fiske, D. W. (1959). Convergent and discriminant
    validation by the multitrait-multimethod matrix. Psychological
    Bulletin, 56(2), 81–105.
  Cronbach, L. J., & Meehl, P. E. (1955). Construct validity in
    psychological tests. Psychological Bulletin, 52(4), 281–302.
  Klavans, R., & Boyack, K. W. (2017). Which type of citation analysis
    generates the most accurate taxonomy of scientific and technical
    knowledge? JASIST, 68(4), 984–998.
  Waltman, L., & van Eck, N. J. (2012). A new methodology for constructing
    a publication-level classification system of science. JASIST, 63(12),
    2378–2392.

Autor: Ben Borowski
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from config import (
        OUTPUT_DIR, DATA_PATH,
        EXTERNAL_VALIDATION_AGGREGATION,
        EXTVAL_CONVERGENT_MIN, EXTVAL_DISCRIMINANT_MAX,
    )
except ImportError:
    OUTPUT_DIR = Path(__file__).parent / "output"
    DATA_PATH = None
    EXTERNAL_VALIDATION_AGGREGATION = "median"
    EXTVAL_CONVERGENT_MIN = 0.50
    EXTVAL_DISCRIMINANT_MAX = 0.30


# =============================================================================
# ERWARTUNGSPROFIL: KONVERGENZ/DISKRIMINANZ
# =============================================================================
# Indikatoren werden anhand des theoretisch erwarteten Zusammenhangs zu
# RTW/CTW als „convergent", „discriminant" oder „neutral" markiert.
# Die Beurteilung erfolgt im Bericht; die Tabelle ist die operative Referenz.

EXPECTED_PROFILE = {
    "RTW": {
        # konvergent: RTW = Inputs-Diversität ↔ EO2 = thematische Streuung
        "disciplinary_entropy":     "convergent",
        # diskriminant erwartet
        "geographic_concentration": "discriminant",
        "author_concentration":     "discriminant",
        "temporal_novelty":         "discriminant",
        # neutral / sekundär
        "field_breadth":            "neutral",
        "keyword_volatility":       "neutral",
    },
    "CTW": {
        # konvergent: CTW = Outputs-Diversität ↔ WP3 = Streuung Keywords Plus
        "field_breadth":            "convergent",
        # diskriminant erwartet
        "geographic_concentration": "discriminant",
        "author_concentration":     "discriminant",
        "temporal_novelty":         "discriminant",
        # neutral / sekundär
        "disciplinary_entropy":     "neutral",
        "review_absence":           "neutral",
    },
}


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def _load_paper_level_rtw_ctw(data_path: Path) -> pd.DataFrame:
    """Lädt RTW/CTW + UID/Year aus der Pipeline-Eingangs-CSV."""
    df = pd.read_csv(data_path)
    keep = [c for c in ["UID", "Title", "Year", "RTW", "CTW"] if c in df.columns]
    if "RTW" not in keep or "CTW" not in keep:
        raise ValueError(
            "Eingangs-CSV enthält weder RTW noch CTW — externe Validierung "
            "nicht möglich. Bitte prepare_kati_data.py mit aktueller "
            "WoS-Categories-Tranche neu ausführen."
        )
    df = df[keep].copy()
    df["RTW"] = pd.to_numeric(df["RTW"], errors="coerce")
    df["CTW"] = pd.to_numeric(df["CTW"], errors="coerce")
    return df


def _load_topic_assignments(output_dir: Path) -> pd.DataFrame:
    """Liest die Paper-Topic-Zuordnung aus topic_assignments.csv."""
    path = output_dir / "topic_assignments.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"topic_assignments.csv nicht gefunden in {output_dir}. "
            "Schritt 1 (step01_topic_modeling.py) muss vorher gelaufen sein."
        )
    return pd.read_csv(path)


def _load_indicator_matrix(output_dir: Path) -> pd.DataFrame:
    """Liest die topic-level Indikatoren aus indicators_16.csv."""
    path = output_dir / "indicators_16.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"indicators_16.csv nicht gefunden in {output_dir}. "
            "Schritt 2 (step02_indicators.py) muss vorher gelaufen sein."
        )
    return pd.read_csv(path, index_col="topic")


def _aggregate_by_topic(paper_df: pd.DataFrame,
                         agg: str = "median") -> pd.DataFrame:
    """Aggregiert RTW/CTW pro Topic via Median (default) oder Mean.

    Topic = -1 (HDBSCAN-Noise) wird ausgeklammert.
    """
    grp = paper_df[paper_df["topic"] >= 0].groupby("topic")
    if agg == "mean":
        topic_df = grp[["RTW", "CTW"]].mean()
    else:
        topic_df = grp[["RTW", "CTW"]].median()
    topic_df["n_papers_with_rtw"] = grp["RTW"].apply(lambda s: s.notna().sum())
    topic_df["n_papers_with_ctw"] = grp["CTW"].apply(lambda s: s.notna().sum())
    topic_df["n_papers_total"]    = grp.size()
    return topic_df


def _spearman_corr(s1: pd.Series, s2: pd.Series) -> tuple[float, float, int]:
    """Spearman-ρ + p-Wert + n; gemeinsame nicht-NaN-Auswertung."""
    df = pd.concat([s1, s2], axis=1).dropna()
    if len(df) < 5:
        return float("nan"), float("nan"), len(df)
    rho, p = spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return float(rho), float(p), int(len(df))


def _classify_correlation(rho: float, expected: str) -> str:
    """Kategorisiert ein einzelnes ρ relativ zur Erwartung.

    Rückgabewerte:
      - "OK"               → Erwartung erfüllt
      - "FAIL_CONVERGENT"  → ρ unter Konvergenzschwelle
      - "FAIL_DISCRIMINANT"→ |ρ| über Diskriminanzgrenze
      - "INFO"             → neutrale Erwartung (deskriptiv)
    """
    if np.isnan(rho):
        return "NA"
    if expected == "convergent":
        return "OK" if rho >= EXTVAL_CONVERGENT_MIN else "FAIL_CONVERGENT"
    if expected == "discriminant":
        return "OK" if abs(rho) <= EXTVAL_DISCRIMINANT_MAX else "FAIL_DISCRIMINANT"
    return "INFO"


# =============================================================================
# KORRELATIONS-TABELLE
# =============================================================================

def compute_correlations(topic_external: pd.DataFrame,
                          indicator_df: pd.DataFrame) -> pd.DataFrame:
    """Spearman-ρ jeder Indikator-Spalte gegen RTW und CTW."""
    joined = topic_external[["RTW", "CTW"]].join(indicator_df, how="inner")
    records = []
    for ext_col in ["RTW", "CTW"]:
        for ind_col in indicator_df.columns:
            rho, p, n = _spearman_corr(joined[ext_col], joined[ind_col])
            expected = EXPECTED_PROFILE.get(ext_col, {}).get(ind_col, "neutral")
            records.append({
                "external": ext_col,
                "indicator": ind_col,
                "spearman_rho": rho,
                "p_value": p,
                "n_topics": n,
                "expected": expected,
                "verdict": _classify_correlation(rho, expected),
            })
    return pd.DataFrame(records)


# =============================================================================
# MTMM-DIAGNOSTIK
# =============================================================================

def compute_mtmm_diagnostics(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregiert die Korrelations-Tabelle zu vier MTMM-Kennzahlen pro
    externem Maß:

      - mean_convergent_rho        Mittelwert der konvergent-erwarteten ρ
      - mean_abs_discriminant_rho  Mittelwert |ρ| der diskriminant-erwarteten
      - n_violations               Anzahl FAIL_*-Befunde
      - validity_pattern           "supported" / "partial" / "rejected"
    """
    records = []
    for ext in ["RTW", "CTW"]:
        sub = corr_df[corr_df["external"] == ext]
        conv = sub[sub["expected"] == "convergent"]
        disc = sub[sub["expected"] == "discriminant"]

        mean_conv = float(conv["spearman_rho"].mean()) if len(conv) else float("nan")
        mean_disc = float(disc["spearman_rho"].abs().mean()) if len(disc) else float("nan")
        violations = int((sub["verdict"].astype(str).str.startswith("FAIL")).sum())

        # Heuristisches Gesamturteil:
        #   - alle konvergenten Erwartungen erfüllt UND keine Diskriminanz-FAILs → supported
        #   - mind. eine konvergente Erwartung erfüllt                            → partial
        #   - sonst                                                               → rejected
        any_conv_ok = (conv["verdict"] == "OK").any()
        all_conv_ok = (conv["verdict"] == "OK").all()
        any_disc_fail = (disc["verdict"] == "FAIL_DISCRIMINANT").any()

        if all_conv_ok and not any_disc_fail:
            verdict = "supported"
        elif any_conv_ok:
            verdict = "partial"
        else:
            verdict = "rejected"

        records.append({
            "external": ext,
            "mean_convergent_rho": mean_conv,
            "mean_abs_discriminant_rho": mean_disc,
            "n_violations": violations,
            "validity_pattern": verdict,
        })
    return pd.DataFrame(records)


# =============================================================================
# REPORT
# =============================================================================

def write_report(output_dir: Path,
                  topic_external: pd.DataFrame,
                  corr_df: pd.DataFrame,
                  mtmm_df: pd.DataFrame) -> None:
    """Konsolidierter Markdown-Bericht."""
    md = ["# Externe Validierung der Indikator-Konstrukte (Schritt 3b)\n"]
    md.append(
        "Die topic-level RTW/CTW-Mediane werden Spearman-korreliert "
        "gegen die intern berechneten 16 Indikatoren. Die Beurteilung "
        f"erfolgt MTMM-basiert (ρ ≥ {EXTVAL_CONVERGENT_MIN:.2f} → konvergent; "
        f"|ρ| ≤ {EXTVAL_DISCRIMINANT_MAX:.2f} → diskriminant).\n"
    )

    md.append("## 1. Topic-level Aggregation\n")
    md.append(f"Aggregation: **{EXTERNAL_VALIDATION_AGGREGATION}** über die "
              "paper-level RTW/CTW-Werte.\n")
    md.append(f"Topics gesamt: **{len(topic_external)}**.\n")
    md.append(topic_external.head(20).round(3).to_markdown() + "\n")

    md.append("## 2. Spearman-ρ — Indikatoren × {RTW, CTW}\n")
    md.append(corr_df.round(3).to_markdown(index=False) + "\n")

    md.append("## 3. MTMM-Diagnostik\n")
    md.append(mtmm_df.round(3).to_markdown(index=False) + "\n")

    md.append("## 4. Interpretation\n")
    for _, row in mtmm_df.iterrows():
        md.append(
            f"- **{row['external']}**: Validierungsmuster "
            f"`{row['validity_pattern']}` (mean ρ_convergent = "
            f"{row['mean_convergent_rho']:.2f}; mean |ρ|_discriminant = "
            f"{row['mean_abs_discriminant_rho']:.2f}; "
            f"{row['n_violations']} Verletzungen)."
        )
    md.append("")

    report_path = output_dir / "external_validation_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"  [step03b] Bericht gespeichert: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def run(data_path: Path = None,
        output_dir: Path = None,
        agg: str = None) -> dict:
    """Externe Validierung gegen RTW/CTW.

    Args:
        data_path: Pfad zur Eingangs-CSV (default: config.DATA_PATH).
        output_dir: Output-Verzeichnis (default: config.OUTPUT_DIR).
        agg: "median" oder "mean" (default: config.EXTERNAL_VALIDATION_AGGREGATION).

    Returns:
        dict mit Schlüsseln "topic_external", "corr", "mtmm".
    """
    data_path = Path(data_path or DATA_PATH)
    output_dir = Path(output_dir or OUTPUT_DIR)
    agg = agg or EXTERNAL_VALIDATION_AGGREGATION
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SCHRITT 3b: Externe Validierung (RTW/CTW)")
    print("=" * 60)

    # 1. Paper-Ebene laden + Topic-Zuordnung joinen
    paper_external = _load_paper_level_rtw_ctw(data_path)
    assignments = _load_topic_assignments(output_dir)

    join_key = None
    for k in ["UID", "DOI", "Title"]:
        if k in paper_external.columns and k in assignments.columns:
            join_key = k
            break
    if join_key is None:
        raise ValueError(
            "Kein gemeinsamer Identifier (UID/DOI/Title) zwischen "
            "Eingangs-CSV und topic_assignments.csv gefunden."
        )
    print(f"  Join-Key: {join_key}")

    paper_df = assignments[[join_key, "topic"]].merge(
        paper_external, on=join_key, how="left"
    )
    paper_df.to_csv(output_dir / "external_validation_paper.csv", index=False)
    print(f"  Paper-Ebene: {len(paper_df)} Records "
          f"({paper_df['RTW'].notna().sum()} mit RTW, "
          f"{paper_df['CTW'].notna().sum()} mit CTW)")

    # 2. Topic-Aggregation
    topic_external = _aggregate_by_topic(paper_df, agg=agg)
    topic_external.to_csv(output_dir / "external_validation_topic.csv")
    print(f"  Topic-Ebene: {len(topic_external)} Topics aggregiert ({agg})")

    # 3. Korrelationen
    indicator_df = _load_indicator_matrix(output_dir)
    corr_df = compute_correlations(topic_external, indicator_df)
    corr_df.to_csv(output_dir / "external_validation_corr.csv", index=False)
    print(f"  Spearman-ρ-Tabelle: {len(corr_df)} Paare berechnet")

    # 4. MTMM-Diagnostik
    mtmm_df = compute_mtmm_diagnostics(corr_df)
    mtmm_df.to_csv(output_dir / "external_validation_mtmm.csv", index=False)
    for _, row in mtmm_df.iterrows():
        print(f"  {row['external']}: "
              f"mean ρ_conv={row['mean_convergent_rho']:.2f}, "
              f"mean |ρ|_disc={row['mean_abs_discriminant_rho']:.2f}, "
              f"verdict={row['validity_pattern']}")

    # 5. Report
    write_report(output_dir, topic_external, corr_df, mtmm_df)

    # 6. Zusammenfassende JSON für maschinellen Konsum
    summary = {
        "n_topics": int(len(topic_external)),
        "n_papers": int(len(paper_df)),
        "aggregation": agg,
        "mtmm": mtmm_df.to_dict(orient="records"),
    }
    (output_dir / "external_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("Fertig (Schritt 3b).")
    print("=" * 60)

    return {
        "topic_external": topic_external,
        "corr": corr_df,
        "mtmm": mtmm_df,
    }


if __name__ == "__main__":
    run()
