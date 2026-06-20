
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

PHASE_DIRS = {
    "1": BASE_DIR / "output_phase1",
    "2": BASE_DIR / "output_phase2",
}

PARTIAL_YEAR_THRESHOLD = 0.6



def detect_partial_years(
    year_counts: pd.Series, threshold: float = PARTIAL_YEAR_THRESHOLD
) -> list[int]:
    yc = year_counts.sort_index()
    if len(yc) < 4:
        return []

    partial = []
    for i in range(len(yc) - 1, max(len(yc) - 4, 2) - 1, -1):
        year = yc.index[i]
        if i < 3:
            continue
        prior_window = yc.iloc[max(0, i - 3): i]
        if len(prior_window) < 2:
            continue
        ref = prior_window.median()
        if ref <= 0:
            continue
        ratio = yc.iloc[i] / ref
        if ratio < threshold:
            partial.append((int(year), float(ratio), int(yc.iloc[i]), int(ref)))

    contiguous_partial = []
    for year, ratio, vol, ref in partial:
        contiguous_partial.append((year, ratio, vol, ref))
    return contiguous_partial



def compute_topic_proportions(
    df_assign: pd.DataFrame,
    end_year: int | None = None,
) -> pd.DataFrame:
    df = df_assign[["Year", "topic"]].dropna().copy()
    df["Year"] = df["Year"].astype(int)
    if end_year is not None:
        df = df[df["Year"] <= end_year].copy()

    yearly_counts = (df.groupby(["Year", "topic"]).size()
                       .unstack(fill_value=0))
    year_totals = yearly_counts.sum(axis=1)
    proportions = yearly_counts.div(year_totals, axis=0)

    long = proportions.reset_index().melt(
        id_vars="Year", var_name="topic", value_name="proportion"
    )
    long = long[long["topic"] != -1].copy()
    return long.sort_values(["topic", "Year"]).reset_index(drop=True)


def linear_growth_rate(years: np.ndarray, props: np.ndarray) -> float:
    if len(years) < 2:
        return 0.0
    avg = float(np.mean(props))
    if avg <= 0:
        return 0.0
    x = years.astype(float)
    y = props.astype(float)
    n = len(x)
    sx = x.sum()
    sy = y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    denom = (n * sxx - sx * sx)
    if denom == 0:
        return 0.0
    slope = (n * sxy - sx * sy) / denom
    return float(slope / avg)


def half_split_growth_rate(years: np.ndarray, props: np.ndarray) -> float:
    if len(years) < 4:
        return 0.0
    order = np.argsort(years)
    p = props[order]
    half = len(p) // 2
    first = p[:half]
    second = p[-half:]
    overall_med = float(np.median(p))
    if overall_med <= 0:
        return 0.0
    return float((np.median(second) - np.median(first)) / overall_med)


def compute_tem(
    proportions_long: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for t, g in proportions_long.groupby("topic"):
        years = g["Year"].to_numpy()
        props = g["proportion"].to_numpy()
        rows.append(dict(
            topic=int(t),
            n_years=int(len(years)),
            avg_proportion=float(np.mean(props)),
            max_proportion=float(np.max(props)),
            growth_rate_ols=linear_growth_rate(years, props),
            growth_rate_halfsplit=half_split_growth_rate(years, props),
        ))
    return pd.DataFrame(rows).sort_values("topic").reset_index(drop=True)



def write_outputs(
    df_assign: pd.DataFrame,
    out_dir: Path,
    end_year_full: int,
    end_year_trim: int | None,
    partial_years: list,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    prop_full = compute_topic_proportions(df_assign, end_year=end_year_full)
    tem_full = compute_tem(prop_full)
    tem_full.to_csv(out_dir / "tem_metrics_full.csv", index=False)
    prop_full.to_csv(out_dir / "topic_proportions_full.csv", index=False)
    print(f"  voll  ({end_year_full}): {len(tem_full)} Topics, "
          f"{out_dir.name}/tem_metrics_full.csv")

    if end_year_trim is not None and end_year_trim < end_year_full:
        prop_trim = compute_topic_proportions(df_assign, end_year=end_year_trim)
        tem_trim = compute_tem(prop_trim)
        tem_trim.to_csv(out_dir / f"tem_metrics_trim_{end_year_trim}.csv",
                        index=False)
        prop_trim.to_csv(out_dir / f"topic_proportions_trim_{end_year_trim}.csv",
                         index=False)
        print(f"  trim  ({end_year_trim}): {len(tem_trim)} Topics, "
              f"{out_dir.name}/tem_metrics_trim_{end_year_trim}.csv")

        cmp = tem_full.merge(
            tem_trim, on="topic", suffixes=("_full", "_trim"),
        )
        cmp["growth_rate_delta"] = (
            cmp["growth_rate_ols_trim"] - cmp["growth_rate_ols_full"]
        )
        cmp.to_csv(out_dir / "growth_rate_comparison.csv", index=False)

        delta_abs = cmp["growth_rate_delta"].abs()
        print(f"  growth_rate-Verschiebung |Δ|: "
              f"mean={delta_abs.mean():.4f}, max={delta_abs.max():.4f}")
        flipped = ((cmp["growth_rate_ols_full"] > 0) !=
                   (cmp["growth_rate_ols_trim"] > 0)).sum()
        print(f"  Topics mit Vorzeichenwechsel des growth_rate: {flipped} "
              f"/ {len(cmp)}")

    diag = []
    diag.append("TEM-Robustheits-Bericht")
    diag.append("=" * 60)
    diag.append(f"Records (mit Year): {len(df_assign):,}")
    diag.append(f"Year-Spanne:        {df_assign['Year'].min()} – "
                f"{df_assign['Year'].max()}")
    diag.append(f"Voll-Endjahr:       {end_year_full}")
    diag.append(f"Trim-Endjahr:       "
                f"{end_year_trim if end_year_trim else '(nicht aktiv)'}")
    diag.append("")
    diag.append("Partial-Year-Detektion (Vol < "
                f"{PARTIAL_YEAR_THRESHOLD:.0%} des 3-Jahres-Median-Trends):")
    if not partial_years:
        diag.append("  Keine partiellen Endjahre erkannt.")
    else:
        for year, ratio, vol, ref in partial_years:
            diag.append(f"  {year}: Volumen {vol:,} vs. Trend-Median {ref:,}  "
                        f"(Verhältnis {ratio:.2f})")
    diag.append("")
    diag.append("Jahresvolumina:")
    yc = df_assign["Year"].value_counts().sort_index()
    for year, n in yc.items():
        diag.append(f"  {int(year)}: {int(n):,}")

    diag_path = out_dir / "tem_diagnostics.txt"
    diag_path.write_text("\n".join(diag), encoding="utf-8")
    print(f"  Diagnostik:  {out_dir.name}/tem_diagnostics.txt")



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["1", "2"], required=True,
                        help="Phase auswählen")
    parser.add_argument("--end-year", type=int, default=None,
                        help="Manuelles Trim-Endjahr (überschreibt --auto-trim)")
    parser.add_argument("--auto-trim", action="store_true",
                        help="Erkannte Partial-Years automatisch ausschließen")
    parser.add_argument("--diagnose-only", action="store_true",
                        help="Nur Partial-Year-Diagnose, keine TEM-Berechnung")
    args = parser.parse_args()

    phase_dir = PHASE_DIRS[args.phase]
    if not phase_dir.exists():
        raise FileNotFoundError(f"Phasen-Output fehlt: {phase_dir}")

    assign_path = phase_dir / "topic_assignments.csv"
    if not assign_path.exists():
        raise FileNotFoundError(f"topic_assignments.csv fehlt in {phase_dir}")

    out_dir = phase_dir / "tem_robust"

    print(f"TEM-Robustheits-Analyse: Phase {args.phase}")
    print("=" * 60)

    df_assign = pd.read_csv(assign_path)
    df_assign = df_assign.dropna(subset=["Year"]).copy()
    df_assign["Year"] = df_assign["Year"].astype(int)
    print(f"Records: {len(df_assign):,}   "
          f"Year-Range: {df_assign['Year'].min()} – {df_assign['Year'].max()}")

    yc = df_assign["Year"].value_counts().sort_index()
    partial = detect_partial_years(yc)
    if partial:
        print("\nPartial-Year-Detektion:")
        for year, ratio, vol, ref in partial:
            print(f"  {year}: Volumen {vol:,} vs. Trend-Median {ref:,}  "
                  f"(Verhältnis {ratio:.2f})  → potenziell unvollständig")
    else:
        print("\nKeine partiellen Endjahre erkannt.")

    if args.diagnose_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_outputs(
            df_assign=df_assign,
            out_dir=out_dir,
            end_year_full=int(df_assign["Year"].max()),
            end_year_trim=None,
            partial_years=partial,
        )
        print("\nFertig (diagnose-only).")
        return

    end_year_full = int(df_assign["Year"].max())
    if args.end_year is not None:
        end_year_trim = args.end_year
    elif args.auto_trim and partial:
        end_year_trim = min(year for year, *_ in partial) - 1
        print(f"\nAuto-Trim aktiv → end_year_trim = {end_year_trim}")
    else:
        end_year_trim = None

    print()
    print("Berechne TEM-Varianten:")
    write_outputs(
        df_assign=df_assign,
        out_dir=out_dir,
        end_year_full=end_year_full,
        end_year_trim=end_year_trim,
        partial_years=partial,
    )

    print("\nFertig.")


if __name__ == "__main__":
    main()
