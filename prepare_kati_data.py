"""
Datenaufbereitung: KATI-CSVs (Dr. John, Fraunhofer FKIE) → Pipeline-Format
==========================================================================

Konvertiert die KATI-Lieferung in das von step01_topic_modeling.py erwartete
Eingabeformat (Title, Abstract, Year, Document Type, Source Title, Author
Keywords, Keywords Plus, WoS Categories, RTW, CTW …).

Pro Phase werden acht KATI-CSVs auf der gemeinsamen UID gemergt:
  - QC_<phase>_01.csv               → Baseline (Title, Year, DOI, Times Cited,
                                       DocType, Abstract)
  - QC_<phase> journals.csv         → Source Title          (← "journal")
  - QC_<phase>_auto KW.csv          → Keywords Plus          (← "autotagList")
  - QC_<phase>_manual KW.csv        → Author Keywords        (← "manualtagList")
  - QC_<phase> WoS Categories.csv   → WoS Categories + RTW + CTW
                                       (← "topicNameL", "RTW", "CTW")
  - QC_<phase> Adress Org.csv       → Affiliations + Addresses
                                       (← "orgList" / "adrList")           — DI2/DI3
  - QC_<phase> Countries.csv        → Country List           (← "countryList")
                                       (algorithmisch bereinigte ISO-3-Codes) — DI3
  - QC_<phase> Citation Topics.csv  → Citation Topic Macro/Meso/Micro
                                       (← "cTopicLabel" je nach "type")     — post-hoc

Die Spaltennamen werden exakt auf die kanonischen WoS-Spalten in
``config.WOS_COLUMNS`` abgebildet, damit step02_indicators.py via CANON_COLUMNS
ohne Sonderbehandlung lesen kann.

Mit der Mai-2026-Tranche von Dr. John liegen Author Full Names und ORCIDs
nun vor; DI1 ist damit operativ. Cited References werden nicht als
Indikator-Eingang verwendet (Validierungs-Spur, separat zu evaluieren) und
bleiben als leere Spalte erhalten, ohne die Pipeline abzubrechen.

Citation Topics werden nicht als Indikator verwendet, sondern als deskriptive
Charakterisierungsschicht für die Ergebnisinterpretation (step04
Visualisierungen / Cross-Phase-Tabellen) im Output mitgeführt.

Aufruf:
    python prepare_kati_data.py            # idempotent
    python prepare_kati_data.py --force    # neu konvertieren

Erzeugt:
    ../wos_qc_phase1_2000_2015.csv
    ../wos_qc_phase2_2016_2025.csv

Autor: Ben Borowski
"""

import os
from pathlib import Path

import pandas as pd

# =============================================================================
# PFADE — KATI-Datenverzeichnis konfigurierbar via Umgebungsvariable
# =============================================================================
#
# Setze die Umgebungsvariable ``KATI_DATA_DIR`` auf das Verzeichnis, das
# die beiden Phasen-Unterordner ("Phase 1 2000-2015", "Phase 2 2016-2025")
# enthaelt. Standard-Fallback: ``./data/kati`` relativ zum Repository-Root.
#
# Beispiel:
#     export KATI_DATA_DIR="/pfad/zu/Data Kati"
#     python prepare_kati_data.py
#
# Der eigentliche WoS/KATI-Korpus ist aus Lizenzgruenden nicht im
# Repository enthalten; siehe README.md ("Daten").

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

# Ziel-Verzeichnis: gleiche Ebene wie das bestehende
# scopus_f3_v2_original.csv, damit DATA_PATH in config.py minimal-invasiv
# umzustellen ist.
OUT_DIR = Path(__file__).parent.parent  # → F3_Prototyp/

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

# Spalten, die step01_topic_modeling.py / step02_indicators.py erwarten.
# Felder, die in der aktuellen KATI-Lieferung noch nicht enthalten sind
# (Author Full Names, ORCIDs, Cited References), werden mit Leerstring
# befüllt — die zugehörigen Indikatoren (DI1, Schritt 2b) liefern dann NaN.
PIPELINE_OUTPUT_COLUMNS = [
    "Title",
    "Year",
    "Abstract",
    "Document Type",
    "Source Title",          # ← KATI: journal
    "Author Keywords",       # ← KATI: manualtagList
    "Keywords Plus",         # ← KATI: autotagList
    "WoS Categories",        # ← KATI: topicNameL
    "Times Cited, WoS Core", # ← KATI: timescited
    "RTW",                   # ← KATI: RTW (Reference Topic Width)
    "CTW",                   # ← KATI: CTW (Citation Topic Width)
    "UID",
    "DOI",
    # Adress Org (DI2/DI3) — Dr. John Lieferung Mai 2026
    "Affiliations",          # ← KATI: orgList    (Semikolon-normalisiert)
    "Addresses",             # ← KATI: adrList    (Semikolon-normalisiert)
    # Countries (DI3, bevorzugte Quelle) — Dr. John Lieferung Mai 2026
    "Country List",          # ← KATI: countryList (ISO-3, Semikolon-norm.)
    "Country Count",         # ← KATI: countryCount
    # Citation Topics (post-hoc, kein Indikator) — Dr. John Lieferung Mai 2026
    "Citation Topic Macro",  # ← KATI: cTopicLabel where type=macro
    "Citation Topic Meso",   # ← KATI: cTopicLabel where type=meso
    "Citation Topic Micro",  # ← KATI: cTopicLabel where type=micro
    # Authors + ORCIDs (DI1) — Dr. John Lieferung Mai 2026
    "Author Full Names",     # ← KATI Authors ORCID: author (wide, sem.-getr.)
    "ORCIDs",                # ← KATI Authors ORCID: WoS-Format Name/ORCID
    # Cited References (Validierungs-Spur, kein Indikator-Eingang)
    "Cited References",
]


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def _read_kati_csv(path: Path) -> pd.DataFrame:
    """Liest eine KATI-CSV ein und säubert KATI-typische Artefakte:
    trailing Whitespaces in Headern, leere Trailing-Spalten ("Unnamed:..."),
    quotierte Werte mit Leading/Trailing-Spaces, "nan"-Strings.
    """
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin({"nan", "None", ""}), col] = ""

    return df


def clean_doctype(value: str) -> str:
    """KATI liefert teils 'article|article' oder 'article|early access article'.
    Wir nehmen den ersten Teil als kanonischen Document Type."""
    if not isinstance(value, str) or not value.strip():
        return ""
    return value.split("|")[0].strip()


def normalize_wos_categories(value: str) -> str:
    """KATI liefert WoS Categories als Pipe-getrennte Liste (`topicNameL`).
    Wandelt in Semikolon-getrennte Form um — kompatibel mit step02_indicators.
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    parts = [p.strip() for p in value.split("|") if p.strip()]
    return "; ".join(parts)


def normalize_keyword_list(value: str) -> str:
    """KATI liefert Keyword-Listen ebenfalls Pipe-getrennt; auf Semikolon
    normalisieren (Pipeline-Konvention für Author Keywords / Keywords Plus).
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    parts = [p.strip() for p in value.split("|") if p.strip()]
    return "; ".join(parts)


def normalize_country_list(value: str) -> str:
    """KATI Countries.csv liefert `countryList` als pipe-getrennte Liste
    kleingeschriebener ISO-3-Codes (z. B. "che|usa"). Wir normalisieren auf
    Semikolon-getrennte Großschreibung ("CHE; USA"), damit step02_indicators
    deterministisch parsen kann (Counter, HHI).
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    parts = [p.strip().upper() for p in value.split("|") if p.strip()]
    return "; ".join(parts)


def normalize_org_list(value: str) -> str:
    """KATI Adress Org.csv `orgList` / `adrList` sind pipe-getrennte Listen.
    Auf Semikolon normalisieren — kompatibel mit step02_indicators
    (DI2 splittet `Affiliations` auf ";", DI3 splittet `Addresses` auf ";").
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    parts = [p.strip() for p in value.split("|") if p.strip()]
    return "; ".join(parts)


def pivot_citation_topics(ct_df: pd.DataFrame) -> pd.DataFrame:
    """Citation Topics liegen in long format vor: pro UID je eine Zeile für
    `type ∈ {macro, meso, micro}`. Pivot auf wide-format mit drei Spalten
    `Citation Topic Macro/Meso/Micro` (Wert = `cTopicLabel` auf jeweiliger
    Hierarchieebene). Bei mehrfacher Zuweisung auf gleicher Ebene wird der
    erste Eintrag genommen (Stabilität für post-hoc Charakterisierung).
    """
    if ct_df.empty:
        return pd.DataFrame(columns=[
            "UID", "Citation Topic Macro", "Citation Topic Meso",
            "Citation Topic Micro",
        ])

    df = ct_df.copy()
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df["cTopicLabel"] = df["cTopicLabel"].astype(str).str.strip()

    # Erste Zuweisung pro (UID, type) behalten
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


# =============================================================================
# PHASEN-MERGE
# =============================================================================

def prepare_phase(phase_dir: Path, tag: str, dst_path: Path, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"Quelle: {phase_dir}")

    # --- Baseline ---
    baseline_path = phase_dir / f"QC_{tag}_01.csv"
    df = _read_kati_csv(baseline_path)
    n_in = len(df)

    # Fehlende Abstracts droppen — ohne Text kein SBERT-Embedding
    df = df[df["abstract"].notna() & (df["abstract"] != "")].copy()
    n_after_abstract = len(df)
    print(f"  Baseline: {n_in} → {n_after_abstract} (Abstracts nicht leer)")

    # Document Type konsolidieren
    df["docTypeList"] = df["docTypeList"].apply(clean_doctype)

    # Nach UID deduplizieren
    df = df.drop_duplicates(subset="UID", keep="first").copy()
    print(f"  Baseline: dedupliziert auf {len(df)} eindeutige UIDs")

    # --- Journals (Source Title) ---
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

    # --- Auto-Keywords (Keywords Plus) ---
    auto_path = phase_dir / f"QC_{tag}_auto KW.csv"
    if auto_path.exists():
        adf = _read_kati_csv(auto_path)
        # Spaltenname `uid` (lowercase) — auf UID umbenennen
        adf = adf.rename(columns={"uid": "UID"})
        adf = adf[["UID", "autotagList"]].drop_duplicates(subset="UID")
        df = df.merge(adf, on="UID", how="left")
        df["autotagList"] = df["autotagList"].fillna("").apply(normalize_keyword_list)
        n_auto = df["autotagList"].astype(bool).sum()
        print(f"  Keywords Plus (auto): {n_auto}/{len(df)} Records gematcht")
    else:
        print(f"  Keywords Plus: Datei fehlt — Spalte bleibt leer")
        df["autotagList"] = ""

    # --- Manual-Keywords (Author Keywords) ---
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

    # --- WoS Categories + RTW + CTW ---
    wc_path = phase_dir / f"QC_{tag} WoS Categories.csv"
    if wc_path.exists():
        wdf = _read_kati_csv(wc_path)
        wdf = wdf.rename(columns={"uid": "UID"})
        # Spalten: UID, year, TC, RTW, CTW, topicNameL
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

    # --- Adress Org (Affiliations + Addresses) ---
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

    # --- Countries (saubere ISO-Codes für DI3) ---
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

    # --- Citation Topics (post-hoc Charakterisierung) ---
    ct_path = phase_dir / f"QC_{tag} Citation Topics.csv"
    if ct_path.exists():
        ctdf_raw = _read_kati_csv(ct_path)
        # Spalten: title, UID, year, cTopicLabel, topCLabel, type
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

    # --- Authors + ORCIDs (DI1) — Dr. John Lieferung Mai 2026 ---
    # Long format: eine Zeile pro (Publikation, Autor); Reihenfolge in der
    # CSV entspricht der Autor-Position auf dem Paper. Pivot auf wide
    # erhaelt diese Reihenfolge, damit DI1 die ORCIDs positional matchen
    # kann.
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
            # WoS-Format "Name/ORCID" emittieren; leerer Slot bei fehlender
            # ORCID — DI1 erkennt das "/" als ORCID-Trigger.
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
        except TypeError:  # pandas < 2.2
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

    # --- Spalten-Mapping → Pipeline-Format ---
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
        # Adress Org (DI2/DI3)
        "Affiliations":           df.get("orgList", ""),
        "Addresses":              df.get("adrList", ""),
        # Countries (DI3, bevorzugte Quelle)
        "Country List":           df.get("countryList", ""),
        "Country Count":          pd.to_numeric(df.get("countryCount", pd.NA),
                                                errors="coerce").astype("Int64"),
        # Citation Topics (post-hoc, kein Indikator)
        "Citation Topic Macro":   df.get("Citation Topic Macro", ""),
        "Citation Topic Meso":    df.get("Citation Topic Meso", ""),
        "Citation Topic Micro":   df.get("Citation Topic Micro", ""),
        # Authors + ORCIDs (DI1) — KATI Mai-2026-Tranche
        "Author Full Names":      df.get("Author Full Names", ""),
        "ORCIDs":                 df.get("ORCIDs", ""),
        # Cited References: Validierungs-Spur, kein 16-Indikator-Eingang
        "Cited References":       "",
    })

    # Year-NaN-Records droppen
    n_before_year = len(out)
    out = out.dropna(subset=["Year"]).copy()
    print(f"  Records: {n_before_year} → {len(out)} (Jahresangabe vorhanden)")

    # Speichern
    out = out[PIPELINE_OUTPUT_COLUMNS]
    out.to_csv(dst_path, index=False)
    print(f"\nGeschrieben: {dst_path}  ({len(out)} Records)")
    print(f"  Year-Range: {out['Year'].min()} – {out['Year'].max()}")
    print(f"  DocType-Top-5:")
    for dt, n in out["Document Type"].value_counts().head(5).items():
        print(f"    {dt}: {n}")

    # Diagnose: Vollständigkeit der wichtigsten Felder
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


# =============================================================================
# EINSTIEGSPUNKT
# =============================================================================

def main(force: bool = False) -> None:
    """Konvertiert die KATI-Tranchen in Pipeline-Format.

    Standard: idempotent — Phasen, deren Ziel-CSV bereits existiert, werden
    übersprungen. Mit ``force=True`` wird neu konvertiert.
    """
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
