from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional


_LATEX_CMD_WITH_ARG = re.compile(
    r"\\[a-zA-Z]+\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}"
)
_LATEX_CMD_BARE = re.compile(r"\\[a-zA-Z]+\b")

_LATEX_INLINE_MATH = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)")
_LATEX_DISPLAY_MATH = re.compile(r"\$\$[^$]*\$\$|\\\[[^\]]*\\\]")

_HTML_TAG = re.compile(r"<[^<>]{1,40}>")

_COPYRIGHT_TAIL = re.compile(
    r"\(c\)\s*\d{4}.*?$|\u00a9\s*\d{4}.*?$|"
    r"copyright\s*\(c\)\s*\d{4}.*?$|"
    r"all rights reserved\.?\s*$",
    flags=re.IGNORECASE,
)

_MULTI_WS = re.compile(r"\s+")

_DANGLING_PUNCT = re.compile(r"\s+([,.;:!?])")



def strip_latex(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = _LATEX_DISPLAY_MATH.sub(" ", text)
    text = _LATEX_INLINE_MATH.sub(" ", text)
    text = _LATEX_CMD_WITH_ARG.sub(" ", text)
    text = _LATEX_CMD_BARE.sub(" ", text)
    return text


def strip_html(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    return _HTML_TAG.sub(" ", text)


def strip_copyright_tail(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    return _COPYRIGHT_TAIL.sub("", text).rstrip()


def normalize_unicode(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
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
    return [clean_text(x, **kwargs) for x in series]



def diagnose_artifacts(texts: Iterable[Optional[str]]) -> dict:
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
