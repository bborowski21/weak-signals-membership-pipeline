
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.metrics import adjusted_rand_score, v_measure_score
from scipy.stats import spearmanr

try:
    from config import (
        OUTPUT_DIR,
        SENSITIVITY_ALPHA_GRID,
        SENSITIVITY_HYBRID_ALPHA_GRID,
        SENSITIVITY_MIN_CLUSTER_GRID,
        SENSITIVITY_MIN_SAMPLES_GRID,
        SENSITIVITY_MIN_TOPIC_SIZE_GRID,
        SENSITIVITY_N_NEIGHBORS_GRID,
        SENSITIVITY_SIGMOID_K_GRID,
        SENSITIVITY_LAMBDA_GRID,
        SENSITIVITY_SEEDS,
        SENSITIVITY_PHASE_SPLITS,
        SENSITIVITY_INDEXING_CUTOFF_MONTHS,
        MEMBERSHIP_SIGMOID_K,
        MEMBERSHIP_LAMBDA_WP,
        HDBSCAN_MIN_CLUSTER_SIZE,
        HDBSCAN_MIN_SAMPLES,
        REVIEW_ABSENCE_ALPHA,
        INDICATOR_DIMENSIONS,
    )
except ImportError:
    OUTPUT_DIR = Path(__file__).parent / "output"
    SENSITIVITY_ALPHA_GRID = [2, 5, 10]
    SENSITIVITY_HYBRID_ALPHA_GRID = [0.4, 0.6, 0.8]
    SENSITIVITY_MIN_CLUSTER_GRID = [15, 20, 25, 30, 40]
    SENSITIVITY_MIN_SAMPLES_GRID = [5, 8, 12]
    SENSITIVITY_MIN_TOPIC_SIZE_GRID = [5, 10, 15, 20]
    SENSITIVITY_N_NEIGHBORS_GRID = [10, 15, 30]
    SENSITIVITY_SIGMOID_K_GRID = [0.5, 1.0, 2.0]
    SENSITIVITY_LAMBDA_GRID = [0.3, 0.5, 0.7]
    SENSITIVITY_SEEDS = [0, 1, 7, 42, 2026]
    SENSITIVITY_PHASE_SPLITS = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
    SENSITIVITY_INDEXING_CUTOFF_MONTHS = 12
    MEMBERSHIP_SIGMOID_K = 1.0
    MEMBERSHIP_LAMBDA_WP = 0.5
    HDBSCAN_MIN_CLUSTER_SIZE = 25
    HDBSCAN_MIN_SAMPLES = 8
    REVIEW_ABSENCE_ALPHA = 5



MEMBERSHIP_COLS = ["m_ws", "m_trend", "m_ec", "m_latent"]


def _spearman_per_membership(base_memb: pd.DataFrame,
                              new_memb: pd.DataFrame) -> dict:
    common = base_memb.index.intersection(new_memb.index)
    if len(common) < 3:
        nan_dict = {f"rho_{c}": np.nan for c in MEMBERSHIP_COLS}
        nan_dict.update({"rho_mean": np.nan, "rho_min": np.nan,
                         "n_common": len(common)})
        return nan_dict

    rhos = {}
    for col in MEMBERSHIP_COLS:
        rho, _ = spearmanr(base_memb.loc[common, col],
                           new_memb.loc[common, col],
                           nan_policy="omit")
        rhos[f"rho_{col}"] = float(rho) if not np.isnan(rho) else np.nan

    vals = [v for v in rhos.values() if not np.isnan(v)]
    rhos["rho_mean"] = float(np.mean(vals)) if vals else np.nan
    rhos["rho_min"] = float(np.min(vals)) if vals else np.nan
    rhos["n_common"] = int(len(common))
    return rhos


def _margin_shift(base_memb: pd.DataFrame,
                    new_memb: pd.DataFrame) -> dict:
    common = base_memb.index.intersection(new_memb.index)
    if len(common) < 3 or "margin" not in base_memb.columns \
            or "margin" not in new_memb.columns:
        return {"delta_median_margin": np.nan,
                "delta_share_unklar": np.nan,
                "rho_margin": np.nan}

    bm = base_memb.loc[common, "margin"]
    nm = new_memb.loc[common, "margin"]
    rho, _ = spearmanr(bm, nm, nan_policy="omit")

    return {
        "delta_median_margin": float(nm.median() - bm.median()),
        "delta_share_unklar": float(((nm < 0.10).mean()) - ((bm < 0.10).mean())),
        "rho_margin": float(rho) if not np.isnan(rho) else np.nan,
    }


def _jaccard_sets(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _compute_memberships_from_dims(
        dim_scores: pd.DataFrame,
        indicator_df: pd.DataFrame,
        k: float = MEMBERSHIP_SIGMOID_K,
        lambda_wp: float = MEMBERSHIP_LAMBDA_WP) -> pd.DataFrame:
    from step02_memberships import compute_memberships
    return compute_memberships(
        indicator_df=indicator_df,
        dim_scores=dim_scores,
        lambda_wp=lambda_wp,
        k=k,
    )


def _aggregate_dimensions(indicator_df: pd.DataFrame,
                           dimensions: dict = None) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler
    dims = dimensions or INDICATOR_DIMENSIONS
    scaler = StandardScaler()
    z = pd.DataFrame(scaler.fit_transform(indicator_df),
                     index=indicator_df.index, columns=indicator_df.columns)
    out = pd.DataFrame(index=indicator_df.index)
    for dim_name, inds in dims.items():
        valid = [i for i in inds if i in z.columns and z[i].std() > 0.01]
        out[dim_name] = z[valid].mean(axis=1) if valid else 0.0
    return out



def growth_rate_rightedge(
        output_dir: Path,
        baseline_memberships: pd.DataFrame,
        indicator_df: pd.DataFrame,
        df: pd.DataFrame,
        labels: np.ndarray) -> pd.DataFrame:
    """WP1-Rechtsrand-Bias-Diagnostik (Kap. 3): CAGR-Referenz (step01) gegen
    OLS-Steigung und Half-Split-Median (step01c), jeweils voll vs. mit
    entferntem (unvollstaendig indexiertem) Endjahr, plus Durchgriff auf die
    Membership-Klassifikation unter de-biased WP1.

    Schreibt growth_rate_comparison.csv und wp1_membership_stability.csv und
    gibt eine einzeilige Kennzahl-Zusammenfassung fuer den Bericht zurueck.
    Entscheidungsregel (Methodenteil): rho(OLS, Half-Split) > 0.9 => CAGR
    beibehalten; sonst Half-Split-Fallback.
    """
    from step01c_tem_robustness import (
        compute_topic_proportions, linear_growth_rate, half_split_growth_rate,
        detect_partial_years,
    )

    def _cagr(years: np.ndarray, props: np.ndarray) -> float:
        # = step01 compute_tem_metrics: erste/letzte Nicht-Null-Proportion
        s = pd.Series(props, index=years).sort_index()
        nz = s[s > 0]
        if len(nz) < 2:
            return 0.0
        ps, pe = nz.iloc[0], nz.iloc[-1]
        ny = nz.index[-1] - nz.index[0]
        return (pe / ps) ** (1 / ny) - 1 if ny > 0 and ps > 0 else 0.0

    dfa = df.copy()
    dfa["topic"] = labels
    dfa = dfa[dfa["topic"] >= 0]
    end_year = int(dfa["Year"].dropna().astype(int).max())

    # Fallback nur, wenn das Endjahr tatsaechlich unvollstaendig indexiert ist
    # (Teiljahr-Detektor aus step01c). Phasen mit vollstaendigem Endjahr (z. B.
    # P1/2015) sind Kontrolle -- die Regel rho>0.9 wird dort nicht angewandt.
    year_counts = dfa["Year"].dropna().astype(int).value_counts().sort_index()
    partial_end = any(int(p[0]) == end_year for p in detect_partial_years(year_counts))

    prop_full = compute_topic_proportions(dfa, end_year=end_year)
    prop_trim = compute_topic_proportions(dfa, end_year=end_year - 1)
    trim_by_topic = {t: g for t, g in prop_trim.groupby("topic")}

    rows = []
    for t, g in prop_full.groupby("topic"):
        yf, pf = g["Year"].to_numpy(), g["proportion"].to_numpy()
        gt = trim_by_topic.get(t)
        if gt is not None:
            yt, pt = gt["Year"].to_numpy(), gt["proportion"].to_numpy()
        else:
            yt, pt = yf[:-1], pf[:-1]
        rows.append(dict(
            topic=int(t),
            growth_rate_cagr_full=_cagr(yf, pf),
            growth_rate_cagr_trim=_cagr(yt, pt),
            growth_rate_ols_full=linear_growth_rate(yf, pf),
            growth_rate_ols_trim=linear_growth_rate(yt, pt),
            growth_rate_halfsplit=half_split_growth_rate(yf, pf),
        ))
    R = pd.DataFrame(rows).set_index("topic").sort_index()
    R["growth_rate_delta"] = R["growth_rate_cagr_trim"] - R["growth_rate_cagr_full"]
    R.round(6).to_csv(output_dir / "growth_rate_comparison.csv")

    rho_ols_hs = float(spearmanr(R["growth_rate_ols_full"],
                                 R["growth_rate_halfsplit"])[0])
    rho_cagr = float(spearmanr(R["growth_rate_cagr_full"],
                               R["growth_rate_cagr_trim"])[0])

    # Durchgriff: Memberships mit de-biased (getrimmtem) Wachstumsindikator
    ind_trim = indicator_df.copy()
    common = R.index.intersection(ind_trim.index)
    ind_trim.loc[common, "growth_rate"] = R.loc[common, "growth_rate_cagr_trim"]
    memb_trim = _compute_memberships_from_dims(
        _aggregate_dimensions(ind_trim), ind_trim)

    a_base = baseline_memberships[MEMBERSHIP_COLS].idxmax(axis=1)
    a_trim = memb_trim[MEMBERSHIP_COLS].idxmax(axis=1)
    idx = a_base.index.intersection(a_trim.index)
    changed = a_base.loc[idx] != a_trim.loc[idx]
    pd.DataFrame({"argmax_baseline": a_base.loc[idx],
                  "argmax_detrend": a_trim.loc[idx],
                  "changed": changed}).to_csv(
        output_dir / "wp1_membership_stability.csv")

    if not partial_end:
        decision = "CAGR beibehalten (kein unvollstaendiges Endjahr -- Kontrolle)"
    elif rho_ols_hs > 0.9:
        decision = "CAGR beibehalten"
    else:
        decision = "Half-Split-Fallback"

    d = R["growth_rate_delta"]
    return pd.DataFrame([{
        "end_year": end_year,
        "partial_end_year": "ja" if partial_end else "nein",
        "n_topics": int(len(R)),
        "rho_ols_vs_halfsplit": round(rho_ols_hs, 3),
        "rho_cagr_full_vs_trim": round(rho_cagr, 3),
        "median_shift_trim_minus_full": round(float(d.median()), 4),
        "mean_abs_delta": round(float(d.abs().mean()), 4),
        "topics_biased_down": int((d > 0).sum()),
        "argmax_stable_pct": round(float((~changed).mean()) * 100, 1),
        "ws_baseline": int((a_base == "m_ws").sum()),
        "ws_detrend": int((a_trim == "m_ws").sum()),
        "decision": decision,
    }])


def membership_sensitivity_k_lambda(
        baseline_memberships: pd.DataFrame,
        indicator_df: pd.DataFrame,
        dim_scores: pd.DataFrame,
        k_grid=SENSITIVITY_SIGMOID_K_GRID,
        lambda_grid=SENSITIVITY_LAMBDA_GRID) -> pd.DataFrame:
    records = []
    for k in k_grid:
        for lam in lambda_grid:
            new_memb = _compute_memberships_from_dims(
                dim_scores=dim_scores, indicator_df=indicator_df,
                k=k, lambda_wp=lam,
            )
            rhos = _spearman_per_membership(baseline_memberships, new_memb)
            mshift = _margin_shift(baseline_memberships, new_memb)
            records.append({
                "rule": "membership_k_lambda",
                "k": k, "lambda_wp": lam,
                **rhos,
                **mshift,
            })
    return pd.DataFrame(records)



def parameter_sensitivity_alpha(
        baseline_memberships: pd.DataFrame,
        indicator_df: pd.DataFrame,
        df: pd.DataFrame,
        labels: np.ndarray,
        alpha_grid=SENSITIVITY_ALPHA_GRID) -> pd.DataFrame:
    from step02_indicators import compute_review_absence

    records = []
    topic_ids = sorted(indicator_df.index.tolist())
    ref_ep3 = indicator_df["review_absence"].reindex(topic_ids)

    for alpha in alpha_grid:
        ep3_new = compute_review_absence(df, topic_ids, labels, alpha=alpha)
        ep3_series = pd.Series(ep3_new).reindex(topic_ids)

        ind_df_new = indicator_df.copy()
        ind_df_new["review_absence"] = ep3_series.values

        dims_new = _aggregate_dimensions(ind_df_new)
        memb_new = _compute_memberships_from_dims(
            dim_scores=dims_new, indicator_df=ind_df_new,
        )

        rhos = _spearman_per_membership(baseline_memberships, memb_new)
        mshift = _margin_shift(baseline_memberships, memb_new)
        rho_ep3, pval_ep3 = spearmanr(ref_ep3.values, ep3_series.values,
                                       nan_policy="omit")
        records.append({
            "param": "review_absence_alpha",
            "value": alpha,
            **rhos,
            **mshift,
            "spearman_rho_ep3": float(rho_ep3) if not np.isnan(rho_ep3) else np.nan,
            "spearman_pvalue_ep3": float(pval_ep3) if not np.isnan(pval_ep3) else np.nan,
        })
    return pd.DataFrame(records)



def _match_topics_by_overlap(baseline_labels: np.ndarray,
                              new_labels: np.ndarray) -> dict:
    base_tids = sorted(set(baseline_labels[baseline_labels >= 0]))
    new_tids = sorted(set(new_labels[new_labels >= 0]))
    mapping = {}
    base_sets = {t: set(np.where(baseline_labels == t)[0]) for t in base_tids}
    for nt in new_tids:
        new_set = set(np.where(new_labels == nt)[0])
        best, best_j = None, 0.0
        for bt, bset in base_sets.items():
            union = new_set | bset
            if not union:
                continue
            j = len(new_set & bset) / len(union)
            if j > best_j:
                best, best_j = bt, j
        if best is not None and best_j >= 0.05:
            mapping[nt] = best
    return mapping


def _rebuild_indicators(df: pd.DataFrame,
                         new_labels: np.ndarray,
                         embeddings_sbert: np.ndarray,
                         embeddings_reduced: np.ndarray) -> pd.DataFrame:
    from step01_topic_modeling import compute_tem_metrics
    from step02_indicators import compute_all_indicators

    tem_metrics, proportions = compute_tem_metrics(df, new_labels)
    return compute_all_indicators(
        df=df, labels=new_labels,
        embeddings_sbert=embeddings_sbert,
        embeddings_reduced=embeddings_reduced,
        proportions=proportions,
        tem_metrics=tem_metrics,
    )


def _spearman_membership_aligned(baseline_memberships: pd.DataFrame,
                                   new_memberships: pd.DataFrame,
                                   mapping: dict) -> dict:
    pairs_base, pairs_new = [], []
    for nt, bt in mapping.items():
        if bt not in baseline_memberships.index or nt not in new_memberships.index:
            continue
        pairs_base.append(baseline_memberships.loc[bt, MEMBERSHIP_COLS])
        pairs_new.append(new_memberships.loc[nt, MEMBERSHIP_COLS])
    if len(pairs_base) < 3:
        nan_dict = {f"rho_{c}": np.nan for c in MEMBERSHIP_COLS}
        nan_dict.update({"rho_mean": np.nan, "rho_min": np.nan,
                         "n_matched_pairs": len(pairs_base)})
        return nan_dict

    base_df = pd.DataFrame(pairs_base).reset_index(drop=True)
    new_df = pd.DataFrame(pairs_new).reset_index(drop=True)

    rhos = {}
    for col in MEMBERSHIP_COLS:
        rho, _ = spearmanr(base_df[col], new_df[col], nan_policy="omit")
        rhos[f"rho_{col}"] = float(rho) if not np.isnan(rho) else np.nan

    vals = [v for v in rhos.values() if not np.isnan(v)]
    rhos["rho_mean"] = float(np.mean(vals)) if vals else np.nan
    rhos["rho_min"] = float(np.min(vals)) if vals else np.nan
    rhos["n_matched_pairs"] = int(len(pairs_base))
    return rhos


def bertopic_hyperparameter_sensitivity(
        df: pd.DataFrame,
        baseline_labels: np.ndarray,
        baseline_indicators: pd.DataFrame,
        baseline_memberships: pd.DataFrame,
        embeddings_sbert: np.ndarray,
        min_cluster_grid=SENSITIVITY_MIN_CLUSTER_GRID,
        min_samples_grid=SENSITIVITY_MIN_SAMPLES_GRID,
        min_topic_size_grid=SENSITIVITY_MIN_TOPIC_SIZE_GRID,
        n_neighbors_grid=SENSITIVITY_N_NEIGHBORS_GRID) -> pd.DataFrame:
    import umap
    import hdbscan
    from config import (UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
                         UMAP_METRIC, HDBSCAN_CLUSTER_METHOD)

    records = []

    def _rerun_units(min_cluster: int, n_neigh: int,
                      min_samples: int = HDBSCAN_MIN_SAMPLES) -> tuple:
        reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS,
                             n_neighbors=n_neigh,
                             min_dist=UMAP_MIN_DIST,
                             metric=UMAP_METRIC, random_state=42)
        reduced = reducer.fit_transform(embeddings_sbert)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster,
                                      min_samples=min_samples,
                                      cluster_selection_method=HDBSCAN_CLUSTER_METHOD)
        return clusterer.fit_predict(reduced), reduced

    def _apply_min_topic_size_filter(labels: np.ndarray,
                                       min_size: int) -> np.ndarray:
        filtered = labels.copy()
        tids, counts = np.unique(filtered[filtered >= 0], return_counts=True)
        to_drop = set(tids[counts < min_size].tolist())
        filtered[np.isin(filtered, list(to_drop))] = -1
        return filtered

    def _eval_cell(new_labels, reduced, param_name, value):
        mapping = _match_topics_by_overlap(baseline_labels, new_labels)
        new_ind = _rebuild_indicators(df, new_labels,
                                        embeddings_sbert, reduced)
        new_dims = _aggregate_dimensions(new_ind)
        new_memb = _compute_memberships_from_dims(
            dim_scores=new_dims, indicator_df=new_ind,
        )
        rhos = _spearman_membership_aligned(
            baseline_memberships, new_memb, mapping)
        return {
            "param": param_name, "value": value,
            "n_topics": len(set(new_labels[new_labels >= 0])),
            "n_noise": int((new_labels == -1).sum()),
            **rhos,
        }

    for mcs in min_cluster_grid:
        print(f"  [hparam] min_cluster_size={mcs} …", flush=True)
        new_labels, reduced = _rerun_units(min_cluster=mcs,
                                             n_neigh=UMAP_N_NEIGHBORS)
        records.append(_eval_cell(new_labels, reduced,
                                   "hdbscan_min_cluster_size", mcs))

    for mts in min_topic_size_grid:
        print(f"  [hparam] min_topic_size={mts} …", flush=True)
        new_labels = _apply_min_topic_size_filter(baseline_labels, mts)
        reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS,
                             n_neighbors=UMAP_N_NEIGHBORS,
                             min_dist=UMAP_MIN_DIST,
                             metric=UMAP_METRIC, random_state=42)
        reduced = reducer.fit_transform(embeddings_sbert)
        records.append(_eval_cell(new_labels, reduced,
                                   "bertopic_min_topic_size", mts))

    for nn in n_neighbors_grid:
        print(f"  [hparam] n_neighbors={nn} …", flush=True)
        new_labels, reduced = _rerun_units(
            min_cluster=HDBSCAN_MIN_CLUSTER_SIZE, n_neigh=nn)
        records.append(_eval_cell(new_labels, reduced,
                                   "umap_n_neighbors", nn))

    for ms in min_samples_grid:
        print(f"  [hparam] min_samples={ms} …", flush=True)
        new_labels, reduced = _rerun_units(
            min_cluster=HDBSCAN_MIN_CLUSTER_SIZE,
            n_neigh=UMAP_N_NEIGHBORS,
            min_samples=ms)
        records.append(_eval_cell(new_labels, reduced,
                                   "hdbscan_min_samples", ms))

    return pd.DataFrame(records)



def indicator_ablation(baseline_memberships: pd.DataFrame,
                        indicator_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for ind in indicator_df.columns:
        ind_df_reduced = indicator_df.drop(columns=[ind])
        dims_reduced = _aggregate_dimensions(ind_df_reduced)
        new_memb = _compute_memberships_from_dims(
            dim_scores=dims_reduced, indicator_df=ind_df_reduced,
        ) if ind not in {"temporal_novelty", "growth_rate",
                           "citation_momentum", "field_breadth"} else None

        if new_memb is None:
            try:
                new_memb = _compute_memberships_from_dims(
                    dim_scores=dims_reduced, indicator_df=indicator_df,
                )
            except Exception:
                records.append({
                    "indicator_removed": ind,
                    **{f"rho_{c}": np.nan for c in MEMBERSHIP_COLS},
                    "rho_mean": np.nan, "rho_min": np.nan,
                    "n_common": 0,
                    "note": "EC-Subindikator; m_ec nicht direkt eliminierbar",
                })
                continue

        rhos = _spearman_per_membership(baseline_memberships, new_memb)
        records.append({"indicator_removed": ind, **rhos})
    return pd.DataFrame(records).sort_values("rho_mean", ascending=True)



def field_alternative_sensitivity(df: pd.DataFrame,
                                   labels: np.ndarray) -> pd.DataFrame:
    from step02_indicators import (compute_disciplinary_entropy,
                                    compute_field_breadth)

    topic_ids = sorted(set(labels[labels >= 0]))
    records = []

    if "Research Areas" in df.columns and "WoS Categories" in df.columns:
        df_wc = df.copy()
        df_ra = df.copy().rename(columns={"Research Areas": "WoS Categories",
                                          "WoS Categories": "_orig_wc"})
        v_wc = compute_disciplinary_entropy(df_wc, topic_ids, labels)
        v_ra = compute_disciplinary_entropy(df_ra, topic_ids, labels)
        rho, pval = spearmanr([v_wc[t] for t in topic_ids],
                               [v_ra[t] for t in topic_ids])
        records.append({"indicator": "disciplinary_entropy",
                        "field_A": "WoS Categories",
                        "field_B": "Research Areas",
                        "spearman_rho": rho, "p_value": pval})

    if "Author Keywords" in df.columns and "Keywords Plus" in df.columns:
        df_kw_plus = df.copy()
        df_kw_auth = df.copy().rename(
            columns={"Author Keywords": "Keywords Plus",
                     "Keywords Plus": "_orig_kwp"})
        v_plus = compute_field_breadth(df_kw_plus, topic_ids, labels)
        v_auth = compute_field_breadth(df_kw_auth, topic_ids, labels)
        rho, pval = spearmanr([v_plus[t] for t in topic_ids],
                               [v_auth[t] for t in topic_ids])
        records.append({"indicator": "field_breadth",
                        "field_A": "Keywords Plus",
                        "field_B": "Author Keywords",
                        "spearman_rho": rho, "p_value": pval})

    return pd.DataFrame(records)



def random_seed_stability(embeddings_sbert: np.ndarray,
                            baseline_labels: np.ndarray,
                            seeds=SENSITIVITY_SEEDS) -> pd.DataFrame:
    import umap
    import hdbscan
    from config import (UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
                        UMAP_METRIC, HDBSCAN_CLUSTER_METHOD)

    records = []
    for seed in seeds:
        reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS,
                            n_neighbors=UMAP_N_NEIGHBORS,
                            min_dist=UMAP_MIN_DIST,
                            metric=UMAP_METRIC,
                            random_state=seed)
        reduced = reducer.fit_transform(embeddings_sbert)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                                     min_samples=HDBSCAN_MIN_SAMPLES,
                                     cluster_selection_method=HDBSCAN_CLUSTER_METHOD)
        new_labels = clusterer.fit_predict(reduced)
        ari = adjusted_rand_score(baseline_labels, new_labels)
        vmea = v_measure_score(baseline_labels, new_labels)
        records.append({"seed": seed, "ari": ari, "v_measure": vmea,
                        "n_topics": len(set(new_labels[new_labels >= 0])),
                        "n_noise": int((new_labels == -1).sum())})
    return pd.DataFrame(records)



def phase_alternative(df: pd.DataFrame,
                       split_years=SENSITIVITY_PHASE_SPLITS) -> pd.DataFrame:
    if "Year" not in df.columns:
        return pd.DataFrame([{"error": "No Year column"}])

    records = []
    for sy in split_years:
        n_p1 = int((df["Year"] <= sy).sum())
        n_p2 = int((df["Year"] > sy).sum())
        records.append({
            "alt_split_year": sy,
            "n_phase1_alt": n_p1,
            "n_phase2_alt": n_p2,
            "ratio_p2_p1": (n_p2 / n_p1) if n_p1 > 0 else np.nan,
            "note": ("Referenzwahl 2015/2016" if sy == 2015 else
                     "Splitjahr ±1" if sy in (2014, 2016, 2017) else
                     "Doubling-Time-Referenz (Scheidsteger 2021)"),
        })
    return pd.DataFrame(records)



def indexing_latency_variant(
        df: pd.DataFrame,
        labels: np.ndarray,
        cutoff_months: int = SENSITIVITY_INDEXING_CUTOFF_MONTHS) -> pd.DataFrame:
    if "Year" not in df.columns:
        return pd.DataFrame([{"error": "No Year column"}])

    max_year = int(df["Year"].max())
    months_per_year = 12
    cutoff_years = cutoff_months / months_per_year
    cutoff_boundary = max_year - cutoff_years

    mask_excluded = df["Year"] > cutoff_boundary
    n_excluded = int(mask_excluded.sum())
    n_total = len(df)

    return pd.DataFrame([{
        "cutoff_months": cutoff_months,
        "max_year": max_year,
        "cutoff_boundary": cutoff_boundary,
        "n_excluded": n_excluded,
        "n_total": n_total,
        "share_excluded": n_excluded / n_total if n_total else np.nan,
        "affected_indicators": "DI4, WP2",
        "note": ("Master-Batch: Re-Run dieser zwei Indikatoren unter "
                 "Maske (Year <= cutoff_boundary); Spearman-Vergleich."),
    }])



def hybrid_alpha_sensitivity_cross_phase(
        phase1_dir: Path,
        phase2_dir: Path,
        alpha_grid=SENSITIVITY_HYBRID_ALPHA_GRID,
        baseline_alpha: float = 0.6,
        topk: int = 15,
        use_sbert: bool = True) -> pd.DataFrame:
    from step01b_cross_phase_matching import (
        load_topic_keywords, compute_pairwise_scores, best_matches,
    )

    p1_kw = load_topic_keywords(phase1_dir, topk)
    p2_kw = load_topic_keywords(phase2_dir, topk)

    base_scores = compute_pairwise_scores(
        p1_kw, p2_kw, alpha=baseline_alpha,
        use_sbert=use_sbert, p1_dir=phase1_dir, p2_dir=phase2_dir,
    )
    base_matches = best_matches(base_scores)
    base_mutual = set(
        tuple(r) for r in base_matches[base_matches["mutual_best"]][
            ["phase1_topic", "phase2_topic"]].itertuples(index=False, name=None)
    )
    base_scores_sorted = base_scores.sort_values(
        ["phase1_topic", "phase2_topic"]).reset_index(drop=True)
    ref_hybrid = base_scores_sorted["hybrid"].to_numpy()

    records = []
    for a in alpha_grid:
        if abs(a - baseline_alpha) < 1e-9:
            scores_a = base_scores
            matches_a = base_matches
        else:
            scores_a = compute_pairwise_scores(
                p1_kw, p2_kw, alpha=a,
                use_sbert=use_sbert, p1_dir=phase1_dir, p2_dir=phase2_dir,
            )
            matches_a = best_matches(scores_a)

        mutual_a = set(
            tuple(r) for r in matches_a[matches_a["mutual_best"]][
                ["phase1_topic", "phase2_topic"]].itertuples(index=False, name=None)
        )
        scores_a_sorted = scores_a.sort_values(
            ["phase1_topic", "phase2_topic"]).reset_index(drop=True)
        rho, pval = spearmanr(ref_hybrid, scores_a_sorted["hybrid"].to_numpy(),
                                nan_policy="omit")

        best_p1_a = matches_a[matches_a["rank_p1_to_p2"] == 1][
            ["phase1_topic", "phase2_topic"]].set_index("phase1_topic")
        base_best_p1_map = base_matches[base_matches["rank_p1_to_p2"] == 1][
            ["phase1_topic", "phase2_topic"]].set_index("phase1_topic")
        common_p1 = base_best_p1_map.index.intersection(best_p1_a.index)
        n_best_changed = int(
            (base_best_p1_map.loc[common_p1, "phase2_topic"] !=
             best_p1_a.loc[common_p1, "phase2_topic"]).sum()
        )

        mutual_hyb = matches_a[matches_a["mutual_best"]]["hybrid"]
        n_unsicher = int((mutual_hyb < 0.25).sum())

        records.append({
            "alpha_hybrid": a,
            "n_mutual_best": len(mutual_a),
            "n_best_p1": int((matches_a["rank_p1_to_p2"] == 1).sum()),
            "n_best_p1_changed_vs_ref": n_best_changed,
            "jaccard_mutual_vs_ref": _jaccard_sets(base_mutual, mutual_a),
            "spearman_rho_hybrid_vs_ref": float(rho) if rho is not None else np.nan,
            "spearman_pvalue_hybrid_vs_ref": float(pval) if pval is not None else np.nan,
            "mean_hybrid_mutual": float(mutual_hyb.mean()) if len(mutual_hyb) else np.nan,
            "median_hybrid_mutual": float(mutual_hyb.median()) if len(mutual_hyb) else np.nan,
            "n_unsicher_mutual_lt_0_25": n_unsicher,
        })

    return pd.DataFrame(records)


def step05c_cross_phase_sensitivity(
        phase1_dir: Path,
        phase2_dir: Path,
        output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== SCHRITT 5 (Cross-Phase): Hybrid-α-Sensitivität ===")
    df_alpha = hybrid_alpha_sensitivity_cross_phase(
        phase1_dir=phase1_dir, phase2_dir=phase2_dir)
    out_path = output_dir / "sensitivity_hybrid_alpha.csv"
    df_alpha.to_csv(out_path, index=False)
    print(f"  [step05x] Geschrieben: {out_path}")
    print(df_alpha.to_string(index=False))
    return df_alpha



def write_report(output_dir: Path, results: dict):
    md = ["# Sensitivitätsanalyse F3 — Konsolidierter Bericht (Pipeline V2)\n",
          f"Ausgabeordner: `{output_dir}`\n",
          "Stabilitätsmaß: Spearman-ρ pro Membership-Spalte (m_ws, m_trend, "
          "m_ec, m_latent) plus Margin-Verschiebung. ρ > 0.9 robust; ρ < 0.7 "
          "Reflexion erforderlich.\n"]

    md.append("## 1a. Membership-Hyperparameter — 2D-Grid (k × λ)\n")
    md.append("Variation Sigmoid-k und WP-Gewicht λ. ρ pro Membership-Spalte "
              "gegen Referenzlauf (k=1.0, λ=0.5).\n")
    md.append(results["membership_kl"].to_markdown(index=False) + "\n")

    md.append("## 1b. Bayes-Prior α (EP3) — Vollständiger Re-Run\n")
    md.append("α ∈ {2, 5, 10}: schwache vs. starke Regularisierung der "
              "Review-Absenz; Spearman-ρ pro Membership-Spalte sowie für "
              "EP3-Werte selbst.\n")
    md.append(results["parameter_alpha"].to_markdown(index=False) + "\n")

    md.append("## 1c. Einheitenbildung — BERTopic-Hyperparameter\n")
    md.append("Grid über min_cluster_size, min_topic_size, n_neighbors, "
              "min_samples. Membership-Spearman über gematchte Topic-Paare.\n")
    md.append(results["parameter_hparam"].to_markdown(index=False) + "\n")

    md.append("## 2. Indikator-Ablation (Leave-One-Out)\n")
    md.append("16 Durchläufe, je ein Indikator entfernt. Niedrige ρ_mean-"
              "Werte identifizieren Indikatoren mit hohem Einfluss auf die "
              "Membership-Struktur.\n")
    md.append(results["ablation"].to_markdown(index=False) + "\n")

    md.append("## 3. Feld-Alternativen\n")
    md.append("Spearman-Korrelationen für EO2 (WC vs. Research Areas) und "
              "WP3 (Keywords Plus vs. Author Keywords).\n")
    md.append(results["fields"].to_markdown(index=False) + "\n")

    md.append("## 4. Seed-Stabilität (UMAP-Stochastik)\n")
    md.append("UMAP + HDBSCAN über fünf Seeds; ARI und V-Measure auf "
              "Clustering-Ebene (vor Membership-Berechnung).\n")
    md.append(results["seeds"].to_markdown(index=False) + "\n")

    md.append("## 5. Phasen-Alternativen (Splitjahr ±1 + Doubling-Time)\n")
    md.append("Referenzgrenze 2015/2016, Variation ±1 Jahr sowie "
              "Doubling-Time-Referenz (Scheidsteger 2021).\n")
    md.append(results["phase"].to_markdown(index=False) + "\n")

    md.append("## 6. Indexierungs-/Rezeptionslatenz\n")
    md.append("Variante mit Exklusion der letzten 12 Monate für DI4, WP2.\n")
    md.append(results["latency"].to_markdown(index=False) + "\n")

    if "growth_rightedge" in results:
        md.append("## 7. WP1-Rechtsrand-Bias (Wachstumsraten-Schätzung)\n")
        md.append("CAGR (Referenz, step01) gegen OLS-Steigung und "
                  "Half-Split-Median (step01c), full vs. Endjahr-Trim. "
                  "Entscheidungsregel: ρ(OLS, Half-Split) > 0.9 ⇒ CAGR "
                  "beibehalten. 'argmax_stable_pct' misst den Durchgriff auf "
                  "die Membership-Klassifikation unter de-biased WP1. "
                  "Artefakte: growth_rate_comparison.csv, "
                  "wp1_membership_stability.csv.\n")
        md.append(results["growth_rightedge"].to_markdown(index=False) + "\n")

    report_path = output_dir / "sensitivity_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"  [step05] Bericht gespeichert: {report_path}")



def run(output_dir: Path = None):
    output_dir = Path(output_dir or OUTPUT_DIR)

    with open(output_dir / "step1_artifacts.pkl", "rb") as f:
        art1 = pickle.load(f)

    df = art1["df"]
    labels = art1["labels"]
    emb_sbert = art1["embeddings_sbert"]

    indicator_df = pd.read_csv(output_dir / "indicators_16.csv", index_col=0)
    dim_scores = pd.read_csv(output_dir / "dimension_scores.csv", index_col=0)
    baseline_memberships = pd.read_csv(
        output_dir / "signal_memberships.csv", index_col=0)

    print("\n=== SCHRITT 5: Sensitivitätsanalyse (Pipeline V2) ===")
    results = {}

    print("  (1a/7) Membership-Hyperparameter — 2D-Grid (k × λ) …")
    results["membership_kl"] = membership_sensitivity_k_lambda(
        baseline_memberships, indicator_df, dim_scores)
    results["membership_kl"].to_csv(
        output_dir / "sensitivity_membership_kl.csv", index=False)

    print("  (1b/7) Bayes-Prior α (EP3) …")
    results["parameter_alpha"] = parameter_sensitivity_alpha(
        baseline_memberships, indicator_df, df, labels)
    results["parameter_alpha"].to_csv(
        output_dir / "sensitivity_parameter_alpha.csv", index=False)

    print("  (1c/7) BERTopic-Hyperparameter (vollständige Re-Indikator-Läufe) …")
    results["parameter_hparam"] = bertopic_hyperparameter_sensitivity(
        df=df, baseline_labels=labels,
        baseline_indicators=indicator_df,
        baseline_memberships=baseline_memberships,
        embeddings_sbert=emb_sbert)
    results["parameter_hparam"].to_csv(
        output_dir / "sensitivity_parameter_hparam.csv", index=False)

    print("  (2/7) Indikator-Ablation …")
    results["ablation"] = indicator_ablation(baseline_memberships, indicator_df)
    results["ablation"].to_csv(output_dir / "sensitivity_ablation.csv", index=False)

    print("  (3/7) Feld-Alternativen …")
    results["fields"] = field_alternative_sensitivity(df, labels)
    results["fields"].to_csv(output_dir / "sensitivity_fields.csv", index=False)

    print("  (4/7) Seed-Stabilität …")
    results["seeds"] = random_seed_stability(emb_sbert, labels)
    results["seeds"].to_csv(output_dir / "sensitivity_seeds.csv", index=False)

    print("  (5/7) Phasen-Alternativen …")
    results["phase"] = phase_alternative(df)
    results["phase"].to_csv(output_dir / "sensitivity_phase.csv", index=False)

    print("  (6/7) Indexierungs-/Rezeptionslatenz …")
    results["latency"] = indexing_latency_variant(df, labels)
    results["latency"].to_csv(output_dir / "sensitivity_latency.csv", index=False)

    print("  (7/7) WP1-Rechtsrand-Bias (CAGR vs. OLS/Half-Split) …")
    results["growth_rightedge"] = growth_rate_rightedge(
        output_dir, baseline_memberships, indicator_df, df, labels)
    results["growth_rightedge"].to_csv(
        output_dir / "sensitivity_growth_rightedge.csv", index=False)

    write_report(output_dir, results)
    print("=== SCHRITT 5 abgeschlossen (Pipeline V2) ===")
    return results


if __name__ == "__main__":
    run()
