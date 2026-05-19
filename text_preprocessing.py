"""
Text-Preprocessing-Utilities für die SBERT-Pipeline
====================================================

Reine Hilfsfunktionen für die Bereinigung von Title- und Abstract-Text vor
dem Einsatz in SBERT-Embedding und c-TF-IDF-Topic-Repräsentation.

Motivation (empirisch fundiert):
  Die Phase-2-Topic-Modeling-Ergebnisse haben "usepackage" als Top-Cumulative-
  Keyword aufgedeckt. Ursache: Einige WoS-Abstracts enthalten LaTeX-Reste
  (\\usepackage{...}, \\cite{...}, $...$ etc.), die als gewöhnliche Wörter
  in die TF-IDF-Repräsentation einfließen und Topic-Keywords verzerren.

Designprinzip:
  Kein Information-Loss: nur eindeutig nicht-semantische Marker werden
  entfernt. Inhaltliche Begriffe (auch fachspezifische Akronyme, Formel-
  Variablen in Klartext) bleiben erhalten.

Funktionen sind reine Funktionen ohne Seiteneffekte und einzeln testbar.

Autor: Ben Borowski (vorbereitet im Cowork-Modus)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

# =============================================================================
# REGEX-PATTERNS — kompiliert (modullokal cached)
# =============================================================================

# LaTeX-Befehle der Form \cmd{arg} oder \cmd[opt]{arg} oder \cmd
# Beispiele: \usepackage{amsmath}, \cite{Smith2020}, \emph{wichtig}, \alpha
_LATEX_CMD_WITH_ARG = re.compile(
    r"\\[a-zA-Z]+\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}"
)
_LATEX_CMD_BARE = re.compile(r"\\[a-zA-Z]+\b")

# Inline-Math: $...$ oder \(...\) — ersetzen durch Platzhalter, weil komplette
# Entfernung Sätze beschneiden würde. Platzhalter " MATH " bleibt als generischer
# Marker erkennbar, fließt aber nicht als Vokabel ins TF-IDF (Stopwort-Filter
# in step01 entfernt zu kurze Token, plus min_df-Schwelle).
_LATEX_INLINE_MATH = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)")
_LATEX_DISPLAY_MATH = re.compile(r"\$\$[^$]*\$\$|\\\[[^\]]*\\\]")

# HTML/XML-Tags (gelegentlich in WoS-Abstracts: <sub>, <sup>, <i>)
_HTML_TAG = re.compile(r"<[^<>]{1,40}>")

# Copyright-Floskeln am Abstract-Ende (typisches WoS-Artefakt)
_COPYRIGHT_TAIL = re.compile(
    r"\(c\)\s*\d{4}.*?$|\u00a9\s*\d{4}.*?$|"
    r"copyright\s*\(c\)\s*\d{4}.*?$|"
    r"all rights reserved\.?\s*$",
    flags=re.IGNORECASE,
)

# Mehrfach-Whitespace
_MULTI_WS = re.compile(r"\s+")

# Reine Punktreste nach Cleaning
_DANGLING_PUNCT = re.compile(r"\s+([,.;:!?])")


# =============================================================================
# ÖFFENTLICHE API
# =============================================================================

def strip_latex(text: str) -> str:
    """Entfernt LaTeX-Befehle und Math-Markup.

    Reihenfolge ist semantisch wichtig:
      1. Display-Math zuerst ($$...$$, \\[...\\]) — sonst frisst Inline-Math
         die äußeren $$ falsch.
      2. Inline-Math ($...$, \\(...\\)) → Platzhalter " ".
      3. LaTeX-Befehle mit Argument (\\cmd{arg}) — frisst Befehl + Argument.
      4. Bare Befehle (\\alpha, \\textbf) — übrig gebliebene.
    """
    if not isinstance(text, str) or not text:
        return ""
    text = _LATEX_DISPLAY_MATH.sub(" ", text)
    text = _LATEX_INLINE_MATH.sub(" ", text)
    text = _LATEX_CMD_WITH_ARG.sub(" ", text)
    text = _LATEX_CMD_BARE.sub(" ", text)
    return text


def strip_html(text: str) -> str:
    """Entfernt HTML-/XML-Tags (sub, sup, i, b, etc.)."""
    if not isinstance(text, str) or not text:
        return ""
    return _HTML_TAG.sub(" ", text)


def strip_copyright_tail(text: str) -> str:
    """Entfernt typische Copyright-Floskeln am Abstract-Ende."""
    if not isinstance(text, str) or not text:
        return ""
    return _COPYRIGHT_TAIL.sub("", text).rstrip()


def normalize_unicode(text: str) -> str:
    """NFKC-Normalisierung: vereinheitlicht Ligaturen, Sonderzeichen."""
    if not isinstance(text, str) or not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    """Mehrfach-Whitespace zu Single-Space, dangling Punctuation, trim."""
    if not isinstance(text, str) or not text:
        return ""
    text = _MULTI_WS.sub(" ", text)
    text = _DANGLING_PUNCT.sub(r"\1", text)
    return text.strip()


def clean_text(
    text: Optional[str],
    *,
    drop_latex: bool = True,
    drop_html: bool = True,
    drop_copyright: bool = True,
    nfkc: bool = True,
) -> str:
    """Vollständige Text-Bereinigung in stabiler Reihenfolge.

    Reihenfolge (alle Schritte konfigurierbar):
      1. Unicode-Normalisierung (NFKC) — vereinheitlicht Code-Points vor
         Regex-Matching.
      2. LaTeX-Cleaning — entfernt \\cmd{arg}, \\cmd, $...$.
      3. HTML-Cleaning — entfernt <tag>.
      4. Copyright-Tail — entfernt "(c) 2020 Elsevier..." am Ende.
      5. Whitespace-Normalisierung — final.

    Beispiele:
      >>> clean_text(r"We use \\usepackage{amsmath} and \\(\\alpha = 1\\).")
      'We use and .'
      >>> clean_text("CO<sub>2</sub> emissions")
      'CO emissions'
      >>> clean_text("Result is high. (c) 2020 Elsevier. All rights reserved.")
      'Result is high.'
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return ""

    if nfkc:
        text = normalize_unicode(text)
    if drop_latex:
        text = strip_latex(text)
    if drop_html:
        text = strip_html(text)
    if drop_copyright:
        text = strip_copyright_tail(text)
    text = normalize_whitespace(text)
    return text


def clean_series(
    series: Iterable[Optional[str]],
    **kwargs,
) -> list[str]:
    """Vektorisierte Variante für pandas.Series oder Listen."""
    return [clean_text(x, **kwargs) for x in series]


# =============================================================================
# DIAGNOSTIK
# =============================================================================

def diagnose_artifacts(texts: Iterable[Optional[str]]) -> dict:
    """Zählt vermutete Artefakte vor dem Cleaning. Für QC-Reports.

    Returns:
        dict mit Zählern: latex_cmd, inline_math, html_tag, copyright_tail
    """
    counts = dict(latex_cmd=0, inline_math=0, html_tag=0, copyright_tail=0,
                  n_total=0, n_with_any=0)
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        counts["n_total"] += 1
        has_any = False
        if _LATEX_CMD_BARE.search(t) or _LATEX_CMD_WITH_ARG.search(t):
            counts["latex_cmd"] += 1
            has_any = True
        if _LATEX_INLINE_MATH.search(t) or _LATEX_DISPLAY_MATH.search(t):
            counts["inline_math"] += 1
            has_any = True
        if _HTML_TAG.search(t):
            counts["html_tag"] += 1
            has_any = True
        if _COPYRIGHT_TAIL.search(t):
            counts["copyright_tail"] += 1
            has_any = True
        if has_any:
            counts["n_with_any"] += 1
    return counts


if __name__ == "__main__":
    # Mini-Selbsttest
    samples = [
        r"We use \usepackage{amsmath} and \(\alpha = 1\).",
        "CO<sub>2</sub> emissions are high.",
        "Result is high. (c) 2020 Elsevier. All rights reserved.",
        r"$$E = mc^2$$ governs energy.",
        r"See \cite{Smith2020} for details about \emph{quantum} effects.",
        "",
        None,
    ]
    print("Selbsttest text_preprocessing.clean_text:")
    print("-" * 70)
    for s in samples:
        repr_in = repr(s) if s else repr(s)
        print(f"  IN : {repr_in[:60]}")
        print(f"  OUT: {clean_text(s)!r}")
        print()
    print("diagnose_artifacts:", diagnose_artifacts(samples))
