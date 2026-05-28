
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



EXPECTED_PROFILE = {
    "RTW": {
        "disciplinary_entropy":     "convergent",
        "geographic_concentration": "discriminant",
        "author_concentration":     "discriminant",
        "temporal_novelty":         "discriminant",
        "field_breadth":            "neutral",
        "keyword_volatility":       "neutral",
    },
    "CTW": {
        "field_breadth":            "convergent",
        "geographic_concentration": "discriminant",
        "author_concentration":     "discriminant",
        "temporal_novelty":         "discriminant",
        "disciplinary_entropy":     "neutral",
        "review_absence":           "neutral",
    },
}



def _load_paper_level_rtw_ctw(data_path: Path) -> pd.DataFrame:
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
    path = output_dir / "topic_assignments.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"topic_assignments.csv nicht gefunden in {output_dir}. "
            "Schritt 1 (step01_topic_modeling.py) muss vorher gelaufen sein."
        )
    return pd.read_csv(path)


def _load_indicator_matrix(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "indicators_16.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"indicators_16.csv nicht gefunden in {output_dir}. "
            "Schritt 2 (step02_indicators.py) muss vorher gelaufen sein."
        )
    return pd.read_csv(path, index_col="topic")


def _aggregate_by_topic(paper_df: pd.DataFrame,
                         agg: str = "median") -> pd.DataFrame:
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
    df = pd.concat([s1, s2], axis=1).dropna()
    if len(df) < 5:
        return float("nan"), float("nan"), len(df)
    rho, p = spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return float(rho), float(p), int(len(df))


def _classify_correlation(rho: float, expected: str) -> str:
    if np.isnan(rho):
        return "NA"
    if expected == "convergent":
        return "OK" if rho >= EXTVAL_CONVERGENT_MIN else "FAIL_CONVERGENT"
    if expected == "discriminant":
        return "OK" if abs(rho) <= EXTVAL_DISCRIMINANT_MAX else "FAIL_DISCRIMINANT"
    return "INFO"



def compute_correlations(topic_external: pd.DataFrame,
                          indicator_df: pd.DataFrame) -> pd.DataFrame:
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



def compute_mtmm_diagnostics(corr_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for ext in ["RTW", "CTW"]:
        sub = corr_df[corr_df["external"] == ext]
        conv = sub[sub["expected"] == "convergent"]
        disc = sub[sub["expected"] == "discriminant"]

        mean_conv = float(conv["spearman_rho"].mean()) if len(conv) else float("nan")
        mean_disc = float(disc["spearman_rho"].abs().mean()) if len(disc) else float("nan")
        violations = int((sub["verdict"].astype(str).str.startswith("FAIL")).sum())

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



def write_report(output_dir: Path,
                  topic_external: pd.DataFrame,
                  corr_df: pd.DataFrame,
                  mtmm_df: pd.DataFrame) -> None:
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



def run(data_path: Path = None,
        output_dir: Path = None,
        agg: str = None) -> dict:
    data_path = Path(data_path or DATA_PATH)
    output_dir = Path(output_dir or OUTPUT_DIR)
    agg = agg or EXTERNAL_VALIDATION_AGGREGATION
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SCHRITT 3b: Externe Validierung (RTW/CTW)")
    print("=" * 60)

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

    topic_external = _aggregate_by_topic(paper_df, agg=agg)
    topic_external.to_csv(output_dir / "external_validation_topic.csv")
    print(f"  Topic-Ebene: {len(topic_external)} Topics aggregiert ({agg})")

    indicator_df = _load_indicator_matrix(output_dir)
    corr_df = compute_correlations(topic_external, indicator_df)
    corr_df.to_csv(output_dir / "external_validation_corr.csv", index=False)
    print(f"  Spearman-ρ-Tabelle: {len(corr_df)} Paare berechnet")

    mtmm_df = compute_mtmm_diagnostics(corr_df)
    mtmm_df.to_csv(output_dir / "external_validation_mtmm.csv", index=False)
    for _, row in mtmm_df.iterrows():
        print(f"  {row['external']}: "
              f"mean ρ_conv={row['mean_convergent_rho']:.2f}, "
              f"mean |ρ|_disc={row['mean_abs_discriminant_rho']:.2f}, "
              f"verdict={row['validity_pattern']}")

    write_report(output_dir, topic_external, corr_df, mtmm_df)

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
