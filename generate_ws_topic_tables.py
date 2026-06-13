"""
generate_ws_topic_tables.py — Reproduzierbare Erzeugung der WS-Topic-Anhangtabellen.

Erzeugt die drei LaTeX-Tabellen fuer Anhang F3.F2 der Masterarbeit:
  1. Alle WS-dominanten Topics Phase 1 (Argmax = m_ws)
  2. Alle WS-dominanten Topics Phase 2
  3. Phasenuebergreifend WS-dominante Mutual-Best-Paare (WS in beiden Phasen)

Quelle ist ausschliesslich der Pipeline-Output:
  - output_phase{1,2}/signal_memberships.csv   (m_ws, m_trend, m_ec, m_latent, margin)
  - output_phase{1,2}/indicators_16.csv         (Konsistenz-Re-Check der Memberships)
  - output_phase{1,2}/topic_keywords.csv        (Top-Keywords)
  - ../<repo>/output_cross_phase/topic_matches_mutual.csv (Cross-Phase)
  - <thesis>/step2b_reference_overlap/reference_overlap_p{1,2}.csv (rho_t)

Eine eingebaute Konsistenzpruefung rechnet die Memberships aus indicators_16.csv
exakt nach (StandardScaler -> Dimensionsmittel -> robust-z -> Sigmoid, vgl.
step02_indicators.py / step02b_memberships.py) und vergleicht sie mit
signal_memberships.csv. Weicht ein Wert ab, bricht das Skript ab.

Aufruf:  python generate_ws_topic_tables.py [--out ws_tables.tex]
Ausgabe: LaTeX-Tabellenrumpf (drei \\begin{longtable}/\\begin{table}-Bloecke),
         zusaetzlich eine Verifikationszusammenfassung auf stdout.

Hinweis: ALLE LaTeX-Strings sind Raw-Strings (r"..."), damit Steuerzeichen wie
\\b (Backspace) nicht versehentlich entstehen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent


def _first_existing(*candidates: Path) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


def reference_overlap_path(ph: int) -> Path | None:
    """Pipeline-natives reference_overlap_p{ph}.csv; faellt auf Geschwister-Repos
    und den Thesis-Projektordner zurueck."""
    return _first_existing(
        BASE / f"output_phase{ph}" / f"reference_overlap_p{ph}.csv",
        BASE.parent / "sbert_pipeline_membership" / f"output_phase{ph}"
        / f"reference_overlap_p{ph}.csv",
        BASE.parent.parent / "Masterthesis_Projektordner_PDF"
        / "step2b_reference_overlap" / f"reference_overlap_p{ph}.csv",
    )


def cross_phase_dir() -> Path | None:
    return _first_existing(
        BASE / "output_cross_phase",
        BASE.parent / "sbert_pipeline_membership" / "output_cross_phase",
    )

INDICATOR_DIMENSIONS = {
    "Epistemische Offenheit": ["keyword_volatility", "disciplinary_entropy",
                               "semantic_incoherence"],
    "Wahrnehmbarkeit": ["relative_proportion_inv", "noise_ratio",
                        "journal_specificity"],
    "Entwicklungsphase": ["temporal_novelty", "terminological_instability",
                          "review_absence"],
    "Diffusion": ["author_concentration", "institutional_concentration",
                  "geographic_concentration", "citation_concentration"],
    "Wirkungspotenzial": ["growth_rate", "citation_momentum", "field_breadth"],
}
CORE_DIMS = ["Epistemische Offenheit", "Wahrnehmbarkeit",
             "Entwicklungsphase", "Diffusion"]
WP_DIM = "Wirkungspotenzial"
EC_SUBINDICATORS = ["temporal_novelty", "growth_rate",
                    "citation_momentum", "field_breadth"]


# ---------------------------------------------------------------------------
# Pipeline-identische Membership-Mechanik (Konsistenz-Re-Check)
# ---------------------------------------------------------------------------
def robust_z(s: pd.Series, eps: float = 1e-9) -> pd.Series:
    iqr = s.quantile(0.75) - s.quantile(0.25)
    return (s - s.median()) / max(iqr / 2.0, eps)


def sigmoid(x, k: float = 1.0):
    return 1.0 / (1.0 + np.exp(-k * x))


def recompute_memberships(ind: pd.DataFrame) -> pd.DataFrame:
    z = pd.DataFrame(StandardScaler().fit_transform(ind),
                     index=ind.index, columns=ind.columns)
    dim = pd.DataFrame(index=ind.index)
    for name, inds in INDICATOR_DIMENSIONS.items():
        valid = [c for c in inds if z[c].std() > 0.01]
        dim[name] = z[valid].mean(axis=1)
    zd = dim.apply(robust_z, axis=0)
    z_core = zd[CORE_DIMS].mean(axis=1)
    z_ec = ind[EC_SUBINDICATORS].apply(robust_z, axis=0).mean(axis=1)
    z_all = zd[CORE_DIMS + [WP_DIM]].mean(axis=1)
    out = pd.DataFrame({
        "m_ws": sigmoid(z_core),
        "m_trend": sigmoid(-z_core + 0.5 * zd[WP_DIM]),
        "m_ec": sigmoid(z_ec),
        "m_latent": sigmoid(-z_all),
    }, index=ind.index)
    sv = np.sort(out.values, axis=1)
    out["margin"] = sv[:, -1] - sv[:, -2]
    return out


# ---------------------------------------------------------------------------
# LaTeX-Helfer
# ---------------------------------------------------------------------------
def esc(s: str) -> str:
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_") \
            .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def comma(x: float) -> str:
    return f"{x:.2f}".replace(".", "{,}")


def rho_str(rt: float) -> str:
    if np.isnan(rt):
        return "--"
    return r"$\approx$0" if rt < 0.01 else f"{rt:.2f}".replace(".", "{,}")


def top_keywords(kwdf: pd.DataFrame, tcol: str, kcol: str, t: int,
                 n: int = 3, maxlen: int = 44) -> str:
    out: list[str] = []
    for v in (str(x).strip() for x in kwdf.loc[kwdf[tcol] == t, kcol].values):
        if v and not any(v in o or o in v for o in out):
            out.append(v)
        if len(out) >= n:
            break
    return esc(", ".join(out))[:maxlen]


def load_phase(ph: int):
    ind = pd.read_csv(BASE / f"output_phase{ph}" / "indicators_16.csv",
                      index_col="topic")
    mem = pd.read_csv(BASE / f"output_phase{ph}" / "signal_memberships.csv",
                      index_col="topic")
    kw = pd.read_csv(BASE / f"output_phase{ph}" / "topic_keywords.csv")
    tcol = next(c for c in kw.columns if "topic" in c.lower())
    kcol = next(c for c in kw.columns if "keyword" in c.lower()
                or "word" in c.lower())
    ro_path = reference_overlap_path(ph)
    if ro_path is None:
        raise FileNotFoundError(
            f"reference_overlap_p{ph}.csv nicht gefunden (output_phase{ph}/ "
            f"oder Geschwister-Repo).")
    ro = pd.read_csv(ro_path).set_index("topic")["ratio_vs_global"]
    return ind, mem, kw, tcol, kcol, ro


def consistency_check(ph: int, ind: pd.DataFrame, mem: pd.DataFrame) -> float:
    rc = recompute_memberships(ind)
    cols = ["m_ws", "m_trend", "m_ec", "m_latent", "margin"]
    err = (rc[cols] - mem[cols]).abs().max().max()
    return float(err)


def phase_longtable(ph: int, ind, mem, kw, tcol, kcol, ro) -> str:
    arg = mem[["m_ws", "m_trend", "m_ec", "m_latent"]].idxmax(axis=1)
    ws = mem[arg == "m_ws"].sort_values("m_ws", ascending=False)
    years = "2000--2015" if ph == 1 else "2016--2025"
    rows = []
    for t in ws.index:
        rt = ro.get(t, np.nan)
        r = ws.loc[t]
        rows.append(
            f"{t} & {comma(r['m_ws'])} & {comma(r['m_trend'])} & "
            f"{comma(r['m_ec'])} & {comma(r['m_latent'])} & "
            f"{comma(r['margin'])} & {rho_str(rt)} & "
            f"{top_keywords(kw, tcol, kcol, t)} \\\\")
    header = (
        r"{\footnotesize" "\n"
        r"\begin{longtable}{@{}rrrrrrr p{4.6cm}@{}}" "\n"
        r"\caption[WS-dominante Topics in Phase " + str(ph) + r"]{Vollst\"andige "
        r"Liste der " + str(len(ws)) + r" WS-dominanten Topics in Phase~" + str(ph)
        + r" (" + years + r") mit vollst\"andigem Membership-Vektor "
        r"(\textsf{m\textsubscript{ws}}, \textsf{m\textsubscript{tr}}=Trend, "
        r"\textsf{m\textsubscript{ec}}, \textsf{m\textsubscript{lat}}=Latent), "
        r"Margin (Mrg.) und Referenzkoh\"arenz $\rho_t$; sortiert nach absteigendem "
        r"\textsf{m\textsubscript{ws}}. (Eigene Darstellung)}\\" "\n"
        r"\label{tab:app_ws_p" + str(ph) + r"}\\" "\n"
        r"\toprule" "\n"
        r"\textbf{Topic} & \textsf{m\textsubscript{ws}} & \textsf{m\textsubscript{tr}} "
        r"& \textsf{m\textsubscript{ec}} & \textsf{m\textsubscript{lat}} & "
        r"\textbf{Mrg.} & \textbf{$\rho_t$} & \textbf{Top-Keywords} \\" "\n"
        r"\midrule" "\n"
        r"\endfirsthead" "\n"
        r"\toprule" "\n"
        r"\textbf{Topic} & \textsf{m\textsubscript{ws}} & \textsf{m\textsubscript{tr}} "
        r"& \textsf{m\textsubscript{ec}} & \textsf{m\textsubscript{lat}} & "
        r"\textbf{Mrg.} & \textbf{$\rho_t$} & \textbf{Top-Keywords} \\" "\n"
        r"\midrule" "\n"
        r"\endhead" "\n"
        r"\midrule \multicolumn{8}{r@{}}{\footnotesize\textit{Fortsetzung "
        r"n\"achste Seite}}\\" "\n"
        r"\endfoot" "\n"
        r"\bottomrule" "\n"
        r"\endlastfoot" "\n")
    return header + "\n".join(rows) + "\n" + r"\end{longtable}}" + "\n"


def crossphase_table(mems: dict, kws: dict) -> str:
    cross = cross_phase_dir()
    if cross is None:
        raise FileNotFoundError("output_cross_phase/ nicht gefunden.")
    mut = pd.read_csv(cross / "topic_matches_mutual.csv")
    p1c = next(c for c in mut.columns if "phase1" in c.lower() and "topic" in c.lower())
    p2c = next(c for c in mut.columns if "phase2" in c.lower() and "topic" in c.lower())
    hc = "hybrid" if "hybrid" in mut.columns else next(c for c in mut.columns if "hybrid" in c.lower())
    k1c = next(c for c in mut.columns if "phase1_key" in c.lower())
    a1 = mems[1][["m_ws", "m_trend", "m_ec", "m_latent"]].idxmax(axis=1)
    a2 = mems[2][["m_ws", "m_trend", "m_ec", "m_latent"]].idxmax(axis=1)
    ws1 = set(mems[1][a1 == "m_ws"].index)
    ws2 = set(mems[2][a2 == "m_ws"].index)
    both = mut[mut[p1c].isin(ws1) & mut[p2c].isin(ws2)].sort_values(hc, ascending=False)
    rows = []
    for _, r in both.iterrows():
        t1, t2 = int(r[p1c]), int(r[p2c])
        kwtext = esc(", ".join(s.strip() for s in str(r[k1c]).split(",")[:3]))[:40]
        rows.append(f"{t1} & {t2} & {comma(r[hc])} & {comma(mems[1].loc[t1,'m_ws'])} "
                    f"& {comma(mems[2].loc[t2,'m_ws'])} & {kwtext} \\\\")
    return (
        r"\begin{table}[H]\centering\small" "\n"
        r"\begin{tabularx}{\textwidth}{@{}rrrrr X@{}}" "\n"
        r"\toprule" "\n"
        r"\textbf{P1-Topic} & \textbf{P2-Topic} & \textbf{Hybrid-Score} & "
        r"\textsf{m\textsubscript{ws}} (P1) & \textsf{m\textsubscript{ws}} (P2) & "
        r"\textbf{Keywords (P1-Seite)} \\" "\n"
        r"\midrule" "\n" + "\n".join(rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabularx}" "\n"
        r"\caption[Phasen\"ubergreifend WS-dominante Topic-Paare]{Die "
        + str(len(both)) + r" Mutual-Best-Paare, die in \emph{beiden} Phasen "
        r"WS-dominant sind (Argmax~$=\textsf{m\textsubscript{ws}}$), sortiert nach "
        r"absteigendem Hybrid-Score. Keywords zeigen die P1-Seite des Paares. "
        r"(Eigene Darstellung)}" "\n"
        r"\label{tab:app_ws_crossphase}" "\n"
        r"\end{table}" "\n"), len(both)


def plot_ws_scatter(data: dict, out_dir: Path) -> tuple[Path, int]:
    """Erzeugt das WS-Membership-Streudiagramm.

    Delegiert an die kanonische Pipeline-Funktion
    step04c_cross_phase_viz.plot_ws_membership_scatter, damit die Abbildung
    nur EINE Implementierung besitzt (Single Source of Truth). Die hier
    geladenen Memberships werden um die Signal-Typ-Spalte ergaenzt und an die
    Pipeline-Funktion uebergeben.
    """
    from step04c_cross_phase_viz import (
        plot_ws_membership_scatter, MEMBERSHIP_COLUMNS, MEMBERSHIP_LABELS)

    def with_signal_type(mem: pd.DataFrame) -> pd.DataFrame:
        df = mem.copy()
        df["signal_type"] = df[MEMBERSHIP_COLUMNS].idxmax(axis=1).map(MEMBERSHIP_LABELS)
        return df

    df1 = with_signal_type(data[1][1])
    df2 = with_signal_type(data[2][1])
    ro1, ro2 = data[1][5], data[2][5]
    png = out_dir / "ws_scatter.png"
    info = plot_ws_membership_scatter(df1, df2, ro1, ro2, png)
    return png, info["n_ws_p1"] + info["n_ws_p2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "ws_topic_tables.tex"))
    ap.add_argument("--figdir", default=str(BASE / "figures_ws"))
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()

    print("=" * 70)
    print("WS-TOPIC-TABELLEN — Generierung & Pipeline-Konsistenzpruefung")
    print("=" * 70)

    data, mems, kws = {}, {}, {}
    for ph in (1, 2):
        ind, mem, kw, tcol, kcol, ro = load_phase(ph)
        err = consistency_check(ph, ind, mem)
        status = "OK" if err < args.tol else "FEHLER"
        print(f"P{ph}: Re-Check signal_memberships vs. indicators_16 "
              f"-> max-Abweichung {err:.2e} [{status}]")
        if err >= args.tol:
            raise SystemExit(f"Konsistenzpruefung P{ph} fehlgeschlagen ({err:.2e}).")
        data[ph] = (ind, mem, kw, tcol, kcol, ro)
        mems[ph] = mem
        kws[ph] = (kw, tcol, kcol)

    blocks = []
    for ph in (1, 2):
        ind, mem, kw, tcol, kcol, ro = data[ph]
        arg = mem[["m_ws", "m_trend", "m_ec", "m_latent"]].idxmax(axis=1)
        n_ws = int((arg == "m_ws").sum())
        n_inc = int(sum(ro.get(t, 1.0) < 1.0
                        for t in mem[arg == "m_ws"].index)) if len(ro) else 0
        print(f"   -> {n_ws} WS-dominante Topics, davon {n_inc} mit rho_t < 1")
        blocks.append(phase_longtable(ph, ind, mem, kw, tcol, kcol, ro))

    cross, n_both = crossphase_table(mems, kws)
    print(f"Cross-Phase: {n_both} WS<->WS Mutual-Best-Paare")
    blocks.append(cross)

    fig_dir = Path(args.figdir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    png, n_scatter = plot_ws_scatter(data, fig_dir)
    print(f"Streudiagramm: {png}  ({n_scatter} WS-Topics geplottet)")

    out = Path(args.out)
    out.write_text("\n".join(blocks), encoding="utf-8")
    # Steuerzeichen-Selbsttest
    txt = out.read_text(encoding="utf-8")
    ctrl = sum(1 for c in txt if ord(c) < 32 and c not in "\n\t")
    print(f"\nGeschrieben: {out}  ({len(txt)} Zeichen, {ctrl} Steuerzeichen)")
    assert ctrl == 0, "Steuerzeichen im Output!"
    print("Fertig.")


if __name__ == "__main__":
    main()
