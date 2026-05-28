
import os
from pathlib import Path

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_KATI_PATH = _REPO_ROOT / "data" / "kati"

KATI_BASE_CANDIDATES = [
    Path(os.environ["KATI_DATA_DIR"]) if "KATI_DATA_DIR" in os.environ else None,
    _DEFAULT_KATI_PATH,
]
KATI_BASE = next(
    (p for p in KATI_BASE_CANDIDATES if p is not None and p.exists()),
    None,
)

OUT_DIR = Path(__file__).parent.parent

PHASES = [
    {
        "subdir": "Phase 1 2000-2015",
        "tag":    "2000-2015",
        "dst":    "wos_qc_phase1_2000_2015.csv",
        "label":  "Phase 1 (2000-2015)",
    },
    {
        "subdir": "Phase 2 2016-2025",
        "tag":    "2016-2025",
        "dst":    "wos_qc_phase2_2016_2025.csv",
        "label":  "Phase 2 (2016-2025)",
    },
]

PIPELINE_OUTPUT_COLUMNS = [
    "Title",
    "Year",
    "Abstract",
    "Document Type",
    "Source Title",
    "Author Keywords",
    "Keywords Plus",
    "WoS Categories",
    "Times Cited, WoS Core",
    "RTW",
    "CTW",
    "UID",
    "DOI",
    "Affiliations",
    "Addresses",
    "Country List",
    "Country Count",
    "Citation Topic Macro",
    "Citation Topic Meso",
    "Citation Topic Micro",
    "Author Full Names",
    "ORCIDs",
    "Cited References",
]



def _read_kati_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin({"nan", "None", ""}), col] = ""

    return df


def clean_doctype(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    return value.split("|")[0].strip()


def normalize_wos_categories(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    parts = [p.strip() for p in value.split("|") if p.strip()]
    return "; ".join(parts)


def normalize_keyword_list(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    parts = [p.strip() for p in value.split("|") if p.strip()]
    return "; ".join(parts)


def normalize_country_list(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    parts = [p.strip().upper() for p in value.split("|") if p.strip()]
    return "; ".join(parts)


def normalize_org_list(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    parts = [p.strip() for p in value.split("|") if p.strip()]
    return "; ".join(parts)


def pivot_citation_topics(ct_df: pd.DataFrame) -> pd.DataFrame:
    if ct_df.empty:
        return pd.DataFrame(columns=[
            "UID", "Citation Topic Macro", "Citation Topic Meso",
            "Citation Topic Micro",
        ])

    df = ct_df.copy()
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df["cTopicLabel"] = df["cTopicLabel"].astype(str).str.strip()

    df = df.drop_duplicates(subset=["UID", "type"], keep="first")

    wide = df.pivot(index="UID", columns="type", values="cTopicLabel")
    wide = wide.rename(columns={
        "macro": "Citation Topic Macro",
        "meso":  "Citation Topic Meso",
        "micro": "Citation Topic Micro",
    })
    for col in ["Citation Topic Macro", "Citation Topic Meso",
                "Citation Topic Micro"]:
        if col not in wide.columns:
            wide[col] = ""
    wide = wide[["Citation Topic Macro", "Citation Topic Meso",
                 "Citation Topic Micro"]].fillna("")
    wide = wide.reset_index()
    return wide



def prepare_phase(phase_dir: Path, tag: str, dst_path: Path, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"Quelle: {phase_dir}")

    baseline_path = phase_dir / f"QC_{tag}_01.csv"
    df = _read_kati_csv(baseline_path)
    n_in = len(df)

    df = df[df["abstract"].notna() & (df["abstract"] != "")].copy()
    n_after_abstract = len(df)
    print(f"  Baseline: {n_in} → {n_after_abstract} (Abstracts nicht leer)")

    df["docTypeList"] = df["docTypeList"].apply(clean_doctype)

    df = df.drop_duplicates(subset="UID", keep="first").copy()
    print(f"  Baseline: dedupliziert auf {len(df)} eindeutige UIDs")

    journals_path = phase_dir / f"QC_{tag} journals.csv"
    if journals_path.exists():
        jdf = _read_kati_csv(journals_path)
        jdf = jdf[["UID", "journal"]].drop_duplicates(subset="UID")
        df = df.merge(jdf, on="UID", how="left")
        n_jour = df["journal"].fillna("").astype(bool).sum()
        print(f"  Journals: {n_jour}/{len(df)} Records gematcht")
    else:
        print(f"  Journals: Datei fehlt — Source Title bleibt leer")
        df["journal"] = ""

    auto_path = phase_dir / f"QC_{tag}_auto KW.csv"
    if auto_path.exists():
        adf = _read_kati_csv(auto_path)
        adf = adf.rename(columns={"uid": "UID"})
        adf = adf[["UID", "autotagList"]].drop_duplicates(subset="UID")
        df = df.merge(adf, on="UID", how="left")
        df["autotagList"] = df["autotagList"].fillna("").apply(normalize_keyword_list)
        n_auto = df["autotagList"].astype(bool).sum()
        print(f"  Keywords Plus (auto): {n_auto}/{len(df)} Records gematcht")
    else:
        print(f"  Keywords Plus: Datei fehlt — Spalte bleibt leer")
        df["autotagList"] = ""

    manual_path = phase_dir / f"QC_{tag}_manual KW.csv"
    if manual_path.exists():
        mdf = _read_kati_csv(manual_path)
        mdf = mdf.rename(columns={"uid": "UID"})
        mdf = mdf[["UID", "manualtagList"]].drop_duplicates(subset="UID")
        df = df.merge(mdf, on="UID", how="left")
        df["manualtagList"] = df["manualtagList"].fillna("").apply(
            normalize_keyword_list
        )
        n_man = df["manualtagList"].astype(bool).sum()
        print(f"  Author Keywords (manual): {n_man}/{len(df)} Records gematcht")
    else:
        print(f"  Author Keywords: Datei fehlt — Spalte bleibt leer")
        df["manualtagList"] = ""

    wc_path = phase_dir / f"QC_{tag} WoS Categories.csv"
    if wc_path.exists():
        wdf = _read_kati_csv(wc_path)
        wdf = wdf.rename(columns={"uid": "UID"})
        keep = [c for c in ["UID", "topicNameL", "RTW", "CTW"] if c in wdf.columns]
        wdf = wdf[keep].drop_duplicates(subset="UID")
        df = df.merge(wdf, on="UID", how="left")
        if "topicNameL" in df.columns:
            df["topicNameL"] = df["topicNameL"].fillna("").apply(
                normalize_wos_categories
            )
            n_wc = df["topicNameL"].astype(bool).sum()
            print(f"  WoS Categories: {n_wc}/{len(df)} Records gematcht")
        if "RTW" in df.columns:
            n_rtw = pd.to_numeric(df["RTW"], errors="coerce").notna().sum()
            print(f"  RTW: {n_rtw}/{len(df)} Records mit numerischem Wert")
        if "CTW" in df.columns:
            n_ctw = pd.to_numeric(df["CTW"], errors="coerce").notna().sum()
            print(f"  CTW: {n_ctw}/{len(df)} Records mit numerischem Wert")
    else:
        print(f"  WoS Categories: Datei fehlt — EO2/RTW/CTW bleiben leer")
        df["topicNameL"] = ""
        df["RTW"] = ""
        df["CTW"] = ""

    addr_path = phase_dir / f"QC_{tag} Adress Org.csv"
    if addr_path.exists():
        odf = _read_kati_csv(addr_path)
        odf = odf.rename(columns={"uid": "UID"})
        keep = [c for c in ["UID", "orgList", "adrList"] if c in odf.columns]
        odf = odf[keep].drop_duplicates(subset="UID")
        df = df.merge(odf, on="UID", how="left")
        if "orgList" in df.columns:
            df["orgList"] = df["orgList"].fillna("").apply(normalize_org_list)
            n_org = df["orgList"].astype(bool).sum()
            print(f"  Affiliations (orgList): {n_org}/{len(df)} Records "
                  f"gematcht")
        if "adrList" in df.columns:
            df["adrList"] = df["adrList"].fillna("").apply(normalize_org_list)
            n_adr = df["adrList"].astype(bool).sum()
            print(f"  Addresses (adrList):    {n_adr}/{len(df)} Records "
                  f"gematcht")
    else:
        print(f"  Adress Org: Datei fehlt — DI2/DI3-Fallback bleiben leer")
        df["orgList"] = ""
        df["adrList"] = ""

    cntry_path = phase_dir / f"QC_{tag} Countries.csv"
    if cntry_path.exists():
        cdf = _read_kati_csv(cntry_path)
        cdf = cdf.rename(columns={"uid": "UID"})
        keep = [c for c in ["UID", "countryList", "countryCount"]
                if c in cdf.columns]
        cdf = cdf[keep].drop_duplicates(subset="UID")
        df = df.merge(cdf, on="UID", how="left")
        if "countryList" in df.columns:
            df["countryList"] = df["countryList"].fillna("").apply(
                normalize_country_list
            )
            n_cl = df["countryList"].astype(bool).sum()
            print(f"  Country List:           {n_cl}/{len(df)} Records "
                  f"gematcht")
        if "countryCount" in df.columns:
            n_cc = pd.to_numeric(df["countryCount"], errors="coerce") \
                     .notna().sum()
            print(f"  Country Count:          {n_cc}/{len(df)} Records "
                  f"numerisch")
    else:
        print(f"  Countries: Datei fehlt — DI3 fällt auf Addresses zurück")
        df["countryList"] = ""
        df["countryCount"] = ""

    ct_path = phase_dir / f"QC_{tag} Citation Topics.csv"
    if ct_path.exists():
        ctdf_raw = _read_kati_csv(ct_path)
        needed = {"UID", "cTopicLabel", "type"}
        if needed.issubset(ctdf_raw.columns):
            ctdf = pivot_citation_topics(ctdf_raw[list(needed)])
            df = df.merge(ctdf, on="UID", how="left")
            for col in ["Citation Topic Macro", "Citation Topic Meso",
                        "Citation Topic Micro"]:
                df[col] = df[col].fillna("")
                n_ct = df[col].astype(bool).sum()
                print(f"  {col}: {n_ct}/{len(df)} Records gematcht")
        else:
            print(f"  Citation Topics: Spalten fehlen ({needed - set(ctdf_raw.columns)})")
            df["Citation Topic Macro"] = ""
            df["Citation Topic Meso"] = ""
            df["Citation Topic Micro"] = ""
    else:
        print(f"  Citation Topics: Datei fehlt — Spalten bleiben leer")
        df["Citation Topic Macro"] = ""
        df["Citation Topic Meso"] = ""
        df["Citation Topic Micro"] = ""

    authors_path = phase_dir / f"QC_{tag} Authors ORCID.csv"
    if authors_path.exists():
        adf = _read_kati_csv(authors_path)
        adf = adf.rename(columns={"uid": "UID"})
        for col in ("author", "orcid"):
            if col not in adf.columns:
                adf[col] = ""
        adf["author"] = adf["author"].fillna("").astype(str).str.strip()
        adf["orcid"]  = adf["orcid"].fillna("").astype(str).str.strip()

        def _agg(group: pd.DataFrame) -> pd.Series:
            names  = group["author"].tolist()
            orcids = group["orcid"].tolist()
            afn = "; ".join(names)
            orc_entries = [
                f"{n}/{o}" if (o and len(o) >= 10) else ""
                for n, o in zip(names, orcids)
            ]
            return pd.Series({
                "Author Full Names": afn,
                "ORCIDs":            "; ".join(orc_entries),
            })

        try:
            auth_wide = adf.groupby("UID", sort=False, group_keys=False) \
                           .apply(_agg, include_groups=False) \
                           .reset_index()
        except TypeError:
            auth_wide = adf.groupby("UID", sort=False, group_keys=False) \
                           .apply(_agg) \
                           .reset_index()

        df = df.merge(auth_wide, on="UID", how="left")
        df["Author Full Names"] = df["Author Full Names"].fillna("")
        df["ORCIDs"]            = df["ORCIDs"].fillna("")

        n_afn      = df["Author Full Names"].astype(bool).sum()
        n_orc_any  = df["ORCIDs"].apply(lambda x: "/" in str(x)).sum()
        n_authors  = df["Author Full Names"].apply(
            lambda x: len([p for p in str(x).split(";") if p.strip()])
        ).sum()
        n_with_id  = df["ORCIDs"].apply(
            lambda x: sum(1 for p in str(x).split(";") if "/" in p)
        ).sum()
        cov_aut    = 100 * n_with_id / n_authors if n_authors else 0.0
        print(f"  Author Full Names:       {n_afn}/{len(df)} Records gematcht")
        print(f"  ORCIDs (Pub mit >=1):    {n_orc_any}/{len(df)} Records "
              f"| Autor-ORCID-Coverage: {cov_aut:.1f}%")
    else:
        print(f"  Authors ORCID: Datei fehlt — DI1 bleibt blockiert")
        df["Author Full Names"] = ""
        df["ORCIDs"]            = ""

    out = pd.DataFrame({
        "Title":                  df["title"],
        "Year":                   pd.to_numeric(df["year"], errors="coerce")
                                    .astype("Int64"),
        "Abstract":               df["abstract"],
        "Document Type":          df["docTypeList"],
        "Source Title":           df.get("journal", ""),
        "Author Keywords":        df.get("manualtagList", ""),
        "Keywords Plus":          df.get("autotagList", ""),
        "WoS Categories":         df.get("topicNameL", ""),
        "Times Cited, WoS Core":  pd.to_numeric(df["timescited"], errors="coerce")
                                    .fillna(0).astype(int),
        "RTW":                    pd.to_numeric(df.get("RTW", pd.NA),
                                                errors="coerce"),
        "CTW":                    pd.to_numeric(df.get("CTW", pd.NA),
                                                errors="coerce"),
        "UID":                    df["UID"],
        "DOI":                    df["doi"],
        "Affiliations":           df.get("orgList", ""),
        "Addresses":              df.get("adrList", ""),
        "Country List":           df.get("countryList", ""),
        "Country Count":          pd.to_numeric(df.get("countryCount", pd.NA),
                                                errors="coerce").astype("Int64"),
        "Citation Topic Macro":   df.get("Citation Topic Macro", ""),
        "Citation Topic Meso":    df.get("Citation Topic Meso", ""),
        "Citation Topic Micro":   df.get("Citation Topic Micro", ""),
        "Author Full Names":      df.get("Author Full Names", ""),
        "ORCIDs":                 df.get("ORCIDs", ""),
        "Cited References":       "",
    })

    n_before_year = len(out)
    out = out.dropna(subset=["Year"]).copy()
    print(f"  Records: {n_before_year} → {len(out)} (Jahresangabe vorhanden)")

    out = out[PIPELINE_OUTPUT_COLUMNS]
    out.to_csv(dst_path, index=False)
    print(f"\nGeschrieben: {dst_path}  ({len(out)} Records)")
    print(f"  Year-Range: {out['Year'].min()} – {out['Year'].max()}")
    print(f"  DocType-Top-5:")
    for dt, n in out["Document Type"].value_counts().head(5).items():
        print(f"    {dt}: {n}")

    print("  Feld-Vollständigkeit:")
    for col in ["Source Title", "Author Keywords", "Keywords Plus",
                "WoS Categories", "Affiliations", "Addresses", "Country List",
                "Author Full Names", "ORCIDs",
                "Citation Topic Macro", "Citation Topic Meso",
                "Citation Topic Micro"]:
        n_filled = out[col].astype(str).str.len().gt(0).sum()
        pct = 100 * n_filled / len(out) if len(out) else 0
        print(f"    {col}: {n_filled}/{len(out)} ({pct:.1f}%)")
    for col in ["RTW", "CTW", "Country Count"]:
        n_filled = out[col].notna().sum()
        pct = 100 * n_filled / len(out) if len(out) else 0
        print(f"    {col}: {n_filled}/{len(out)} ({pct:.1f}%)")



def main(force: bool = False) -> None:
    print("KATI → Pipeline-Format Konvertierung")
    print("=" * 60)
    if KATI_BASE is None or not KATI_BASE.exists():
        raise FileNotFoundError(
            "KATI-Basis-Verzeichnis nicht gefunden. Geprüfte Pfade:\n  "
            + "\n  ".join(str(p) for p in KATI_BASE_CANDIDATES)
        )
    print(f"Basisverzeichnis: {KATI_BASE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for phase in PHASES:
        dst = OUT_DIR / phase["dst"]
        if dst.exists() and not force:
            print(f"\n=== {phase['label']} ===  (übersprungen — Datei "
                  f"vorhanden: {dst.name})")
            continue
        prepare_phase(
            phase_dir=KATI_BASE / phase["subdir"],
            tag=phase["tag"],
            dst_path=dst,
            label=phase["label"],
        )

    print("\n" + "=" * 60)
    print("Fertig. Nächste Schritte:")
    print("  1. config.py: DATA_PATH auf eine der beiden Dateien setzen")
    print("     (Phase 1: wos_qc_phase1_2000_2015.csv,")
    print("      Phase 2: wos_qc_phase2_2016_2025.csv).")
    print("  2. python step01_topic_modeling.py")
    print("  3. python step02_indicators.py   # 16/16 Indikatoren operativ")
    print("     (DI1 nun aktiv durch Authors+ORCIDs aus KATI Mai-2026)")
    print("  4. python step03_efa_pca.py")
    print("  5. python step03b_external_validation.py   # RTW/CTW")
    print("  6. python step04_visualizations.py")
    print("  7. python step05_sensitivity.py")


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
