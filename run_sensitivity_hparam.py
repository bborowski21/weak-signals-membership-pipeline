
import argparse
import pickle
import sys
from pathlib import Path

from config import (
    OUTPUT_DIR,
    SENSITIVITY_MIN_CLUSTER_GRID,
    SENSITIVITY_MIN_TOPIC_SIZE_GRID,
    SENSITIVITY_N_NEIGHBORS_GRID,
)
from step05_sensitivity import bertopic_hyperparameter_sensitivity


def _load_artifacts(output_dir: Path):
    with open(output_dir / "step1_artifacts.pkl", "rb") as f:
        art1 = pickle.load(f)
    with open(output_dir / "step2_artifacts.pkl", "rb") as f:
        art2 = pickle.load(f)
    return {
        "df":            art1["df"],
        "labels":        art1["labels"],
        "emb_sbert":     art1["embeddings_sbert"],
        "indicator_df":  art2["indicator_df"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BERTopic-Hyperparameter-Sensitivität (Master-Batch)")
    parser.add_argument(
        "--only", choices=["min_cluster_size", "min_topic_size", "n_neighbors"],
        help="Nur einen der drei Hyperparameter ausführen.")
    parser.add_argument(
        "--skip", choices=["min_cluster_size", "min_topic_size", "n_neighbors"],
        action="append", default=[],
        help="Einen Hyperparameter überspringen (wiederholbar).")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                         help="Alternatives Ausgabeverzeichnis.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not (output_dir / "step1_artifacts.pkl").exists():
        print(f"[run_sensitivity_hparam] Fehlende Artefakte in {output_dir}.",
              file=sys.stderr)
        print("  Bitte zuerst step01 und step02 ausführen.", file=sys.stderr)
        return 1

    print("=" * 70)
    print("BERTopic-Hyperparameter-Sensitivität — Master-Batch")
    print("=" * 70)
    artifacts = _load_artifacts(output_dir)

    run_mcs = (args.only in (None, "min_cluster_size")
                 and "min_cluster_size" not in args.skip)
    run_mts = (args.only in (None, "min_topic_size")
                 and "min_topic_size" not in args.skip)
    run_nn = (args.only in (None, "n_neighbors")
                and "n_neighbors" not in args.skip)

    mcs_grid = SENSITIVITY_MIN_CLUSTER_GRID if run_mcs else []
    mts_grid = SENSITIVITY_MIN_TOPIC_SIZE_GRID if run_mts else []
    nn_grid  = SENSITIVITY_N_NEIGHBORS_GRID  if run_nn  else []

    total = len(mcs_grid) + len(mts_grid) + len(nn_grid)
    print(f"Gridzellen: {total} "
          f"(mcs={len(mcs_grid)}, mts={len(mts_grid)}, nn={len(nn_grid)})")
    print(f"Geschätzte Laufzeit: ~{total * 10} min\n")

    df_out = bertopic_hyperparameter_sensitivity(
        df=artifacts["df"],
        baseline_labels=artifacts["labels"],
        baseline_indicators=artifacts["indicator_df"],
        embeddings_sbert=artifacts["emb_sbert"],
        min_cluster_grid=mcs_grid,
        min_topic_size_grid=mts_grid,
        n_neighbors_grid=nn_grid,
    )

    out_path = output_dir / "sensitivity_parameter_hparam.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n[run_sensitivity_hparam] Ergebnisse gespeichert: {out_path}")

    print("\n--- Zusammenfassung Spearman-ρ (mean über 17 Indikatoren) ---")
    if "spearman_rho_mean" in df_out.columns:
        summary = df_out[["param", "value", "n_topics",
                          "n_matched_pairs", "spearman_rho_mean"]]
        print(summary.to_string(index=False))
        print("\nRobustheits-Konvention: ρ > 0.9 robust; ρ < 0.7 → Reflexion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
