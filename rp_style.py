"""rp_style.py

Abbildungsstil fuer die Einreichung bei Research Policy (Elsevier).
Alle Groessen sind auf die Endbreite im Satz ausgelegt, damit nichts skaliert
werden muss (Elsevier Artwork sizing: 90 / 140 / 190 mm; Schrift im Druck
mindestens 7 pt, Indizes 6 pt; Linien 0.10 bis 1.5 pt; Farbe RGB, Druck in
Graustufen; Titel gehoeren in die Caption, nicht ins Bild).

Verwendung:
    import rp_style as rp
    fig, axes = rp.figure("double", height_mm=70, ncols=2)
    ...
    rp.panel(axes[0], "A"); rp.panel(axes[1], "B")
    rp.save(fig, "Fig2_margin_distribution", width="double")

Palette (validiert mit dem dataviz-Validator, alle Paare, Weissflaeche):
    Weak signal      #bb4717  (Vermillion, OKLCH L 0.55)
    Emerging concept #07519d  (Blau, L 0.44)
    Trend            #d6a20a  (Gold, L 0.74; Kontrast auf Weiss 2.3:1, deshalb
                              immer mit dunklem Rand oder Linientyp)
    Latent/mixed     #808080  (Grau, Restklasse)
    Schlechtestes CVD-Paar (protan/deutan) Delta E 11.8, Normalsicht >= 16.8;
    Graustufenwerte 100 / 68 / 160 / 128 (0 bis 255), also auch im Druck trennbar.
Zweitkodierung ist Pflicht: Marker o / s / ^ / D, Linientypen -, --, -., :,
Schraffuren fuer Flaechen. Phasenvergleiche laufen monochrom (Schwarz gegen Grau).
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ------------------------------------------------------------ Zielbreiten (mm)
WIDTH_MM = {"single": 90.0, "mid": 140.0, "double": 190.0}
MAX_HEIGHT_MM = 230.0          # Sicherheitsgrenze fuer eine Satzseite mit Caption
MM = 1.0 / 25.4

# ------------------------------------------------------------------ Schrift
FS = {"base": 8.0, "label": 8.0, "tick": 7.0, "legend": 7.0, "panel": 9.0,
      "annot": 7.0, "small": 7.0, "cell": 6.5}   # Elsevier: gedruckt >= 7 pt; 6.5 nur fuer Zellwerte in Matrizen
_FONT_STACK = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def active_font() -> str:
    """Name der tatsaechlich verwendeten Schrift (Arial auf dem Mac, sonst Liberation Sans)."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for f in _FONT_STACK:
        if f in available:
            return f
    return "DejaVu Sans"


RC = {
    "font.family": "sans-serif",
    "font.sans-serif": _FONT_STACK,
    "font.size": FS["base"],
    "axes.titlesize": FS["base"],
    "axes.labelsize": FS["label"],
    "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "legend.fontsize": FS["legend"], "legend.frameon": False,
    "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.labelspacing": 0.35,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "lines.linewidth": 1.0, "lines.markersize": 4.0,
    "patch.linewidth": 0.6, "hatch.linewidth": 0.35, "hatch.color": "white",
    "grid.linewidth": 0.4, "grid.color": "#d9d9d9", "grid.alpha": 1.0,
    "axes.edgecolor": "#333333", "axes.labelcolor": "#111111",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "figure.dpi": 100,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",   # Schrift einbetten
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold", "mathtext.cal": "sans", "mathtext.sf": "sans", "mathtext.tt": "monospace",
    "mathtext.default": "it",   # Variablen kursiv (n, m, rho, p, C); Delta bleibt aufrecht. Wortindizes mit \mathrm{} setzen.
}
matplotlib.rcParams.update(RC)

# Klassenfarben sind fuer Konfigurationen reserviert. Groessen, die keine Konfiguration
# sind (Haeufigkeiten, Wahrscheinlichkeiten, eine einzelne Serie), tragen NEUTRAL_FILL,
# geordnete Stufen die Graustufen in STEP_GREY.
NEUTRAL_FILL = "#3a74b0"
NEUTRAL_EDGE = "#1f4e79"
STEP_GREY = ["#d0d0d0", "#8f8f8f", "#4a4a4a"]

TEXT = "#111111"
TEXT_MUTED = "#555555"
GRID = "#d9d9d9"
REF_LINE = "#7a7a7a"

# --------------------------------------------------------------- Klassen
CLASS_ORDER = ["Weak Signal", "Emerging Concept", "Trend", "Latent/Mixed"]  # Pipeline-Schluessel
CLASS_EN = {"Weak Signal": "Weak signal", "Emerging Concept": "Emerging concept",
            "Trend": "Trend", "Latent/Mixed": "Latent/mixed"}
CLASS_COLOR = {"Weak Signal": "#bb4717", "Emerging Concept": "#07519d",
               "Trend": "#d6a20a", "Latent/Mixed": "#808080"}
CLASS_EDGE = {"Weak Signal": "#6e2a0d", "Emerging Concept": "#032f5e",
              "Trend": "#6b5000", "Latent/Mixed": "#4d4d4d"}   # dunkler Rand (Relief fuer Gold)
CLASS_MARKER = {"Weak Signal": "o", "Emerging Concept": "s", "Trend": "^", "Latent/Mixed": "D"}
CLASS_LS = {"Weak Signal": "-", "Emerging Concept": "--", "Trend": "-.", "Latent/Mixed": ":"}
CLASS_HATCH = {"Weak Signal": "", "Emerging Concept": "//", "Trend": "..", "Latent/Mixed": "x"}
MEMBERSHIP_COL = {"Weak Signal": "m_ws", "Emerging Concept": "m_ec", "Trend": "m_trend", "Latent/Mixed": "m_latent"}

# --------------------------------------------------------------- Phasen (monochrom)
PHASE_COLOR = {1: "#111111", 2: "#8a8a8a"}
PHASE_MARKER = {1: "o", 2: "s"}
PHASE_LS = {1: "-", 2: "--"}
PHASE_LABEL = {1: "Phase 1 (2000 to 2015)", 2: "Phase 2 (2016 to 2025)"}

# --------------------------------------------------------------- Dimensionen (EN provisorisch)
DIM_DE = ["Epistemische Offenheit", "Wahrnehmbarkeit", "Entwicklungsphase", "Diffusion", "Wirkungspotenzial"]
# WICHTIG (04.09., beim Konsistenzaudit geprueft): Alle fuenf Dimensionen sind in der
# WEAK-SIGNAL-RICHTUNG gebaut, ein hoher Score heisst also nicht immer "viel davon":
#   Wahrnehmbarkeit  = relative_proportion_inv + noise_ratio + journal_specificity
#                      -> hoher Wert = GERINGE Wahrnehmbarkeit
#   Entwicklungsphase = temporal_novelty + terminological_instability + review_absence
#                      -> hoher Wert = FRUEHE Phase
#   Diffusion        = vier Konzentrationsmasse; step02_indicators.py sagt es selbst:
#                      "Diffusion (invertiert: hoher Wert = geringe Diffusion)"
# Empirisch bestaetigt: Spearman(m_ws, Dimension) ist fuer alle fuenf positiv
# (P1 +0.60/+0.47/+0.54/+0.05/+0.40, P2 +0.63/+0.42/+0.55/-0.00/+0.36).
# Deshalb tragen die Achsen die Richtung im Namen; die neutralen Rahmenbegriffe stehen
# in DIM_EN_NEUTRAL und gehoeren in den Fliesstext. Endgueltige Wortwahl am 09.10.
DIM_EN = {  # deutsch -> (Achsenname mit Richtung, Kurzcode)
    "Epistemische Offenheit": ("Epistemic openness", "EO"),
    "Wahrnehmbarkeit":        ("Low perceptibility", "PE"),
    "Entwicklungsphase":      ("Early developmental stage", "DS"),   # „stage" statt „phase": „Phase" ist im Paper fuer die zwei Korpusphasen belegt
    "Diffusion":              ("Low diffusion", "DI"),
    "Wirkungspotenzial":      ("Impact potential", "IP"),
}
DIM_EN_NEUTRAL = {  # Rahmenbegriffe ohne Richtung, fuer Fliesstext und Tabellen
    "Epistemische Offenheit": "Epistemic openness",
    "Wahrnehmbarkeit":        "Perceptibility",
    "Entwicklungsphase":      "Developmental stage",
    "Diffusion":              "Diffusion",
    "Wirkungspotenzial":      "Impact potential",
}
DIM_DIRECTION_NOTE = ("All dimensions are scored in the weak-signal direction: a higher value means greater "
                      "epistemic openness, lower perceptibility, an earlier developmental stage, lower diffusion "
                      "and higher impact potential.")
DIM_EN_NAMES = [DIM_EN[d][0] for d in DIM_DE]
DIM_EN_CODES = [DIM_EN[d][1] for d in DIM_DE]

# --------------------------------------------------------------- Farbkarten
# sequentiell: ein Farbton (Blau), hell nach dunkel; Graustufen monoton
CMAP_SEQ = LinearSegmentedColormap.from_list("rp_blues", ["#f5f8fc", "#c6d7ea", "#7fa5cf", "#3a74b0", "#07519d"])
# divergierend: Blau, Weiss, Vermillion (Vorzeichen zusaetzlich als Zahl in die Zelle schreiben)
CMAP_DIV = LinearSegmentedColormap.from_list("rp_div", ["#07519d", "#7fa5cf", "#f4f4f4", "#e39a7a", "#bb4717"])


# ------------------------------------------------------------------ Helfer
def figure(width: str = "double", height_mm: float = 70.0, nrows: int = 1, ncols: int = 1,
           polar: bool = False, **kw):
    """Erzeugt eine Abbildung mit exakter Endbreite. Rueckgabe wie plt.subplots."""
    w = WIDTH_MM[width]
    assert height_mm <= MAX_HEIGHT_MM, f"Hoehe {height_mm} mm ueberschreitet {MAX_HEIGHT_MM} mm"
    subplot_kw = kw.pop("subplot_kw", {})
    if polar:
        subplot_kw["polar"] = True
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * MM, height_mm * MM),
                             subplot_kw=subplot_kw, **kw)
    return fig, axes


def panel(ax, letter: str, x: float = -0.02, y: float = 1.02, polar: bool = False) -> None:
    """Panel-Kennung (A), (B) ... links oberhalb der Achse, fett, 9 pt."""
    if polar:
        x, y = -0.12, 1.05
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=FS["panel"],
            fontweight="bold", ha="right" if not polar else "left", va="bottom", color=TEXT)


def labels_on_top(ax, pad: float = 1.8) -> None:
    """Stellt Achsenbeschriftungen frei und legt sie ueber Gitter, Rahmen und Daten.

    Noetig bei Polarplots: dort liegen die Beschriftungen auf dem aeusseren Kreis und
    werden sonst von Rahmen und Gitterlinien durchschnitten. Die Freistellflaeche ist
    weiss und damit auf der Zeichenflaeche unsichtbar; nur die Linie darunter verschwindet.
    """
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        if not lab.get_text():
            continue
        lab.set_zorder(20)
        lab.set_bbox(dict(facecolor="white", edgecolor="none", pad=pad))


def radial_ticks(ax, values, labels=None, angle_deg: float = 54.0, fontsize=None, pad: float = 1.2) -> None:
    """Radiuswerte als EIGENE Textebene ueber den Daten.

    set_yticklabels reicht nicht: die Ticklabels gehoeren zum Achsenartist, und der wird
    komplett vor oder nach den Linien gezeichnet. Ein zorder am Text aendert daran nichts.
    Deshalb werden die Werte hier abgeschaltet und als Text mit weisser Freistellflaeche
    und zorder 25 neu gesetzt.
    """
    import numpy as _np
    labels = labels if labels is not None else [f"{v:g}" for v in values]
    ax.set_yticks(list(values))
    ax.set_yticklabels([""] * len(values))
    th = _np.deg2rad(angle_deg)
    for v, lab in zip(values, labels):
        if not lab:
            continue
        ax.text(th, v, lab, ha="center", va="center", fontsize=fontsize or FS["small"],
                color=TEXT_MUTED, zorder=25, clip_on=False,
                bbox=dict(facecolor="white", edgecolor="none", pad=pad))


def ink(color) -> str:
    """Schwarz oder Weiss, je nach Helligkeit der Flaeche (statt nach dem Betrag des Werts)."""
    import matplotlib.colors as _mc
    r, g, b = _mc.to_rgb(color)[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.45 else TEXT


def grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)


def class_handles(classes=None, kind: str = "line", counts: dict | None = None):
    """Legendeneintraege mit Farbe UND Marker/Linientyp (Zweitkodierung)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    classes = classes or CLASS_ORDER
    hs = []
    for c in classes:
        lab = CLASS_EN[c] + (f" ($n$ = {counts[c]})" if counts and c in counts else "")
        if kind == "line":
            hs.append(Line2D([0], [0], color=CLASS_COLOR[c], ls=CLASS_LS[c], marker=CLASS_MARKER[c],
                             markerfacecolor=CLASS_COLOR[c], markeredgecolor=CLASS_EDGE[c],
                             markeredgewidth=0.5, lw=1.2, markersize=4, label=lab))
        elif kind == "marker":
            hs.append(Line2D([0], [0], color="none", marker=CLASS_MARKER[c], markerfacecolor=CLASS_COLOR[c],
                             markeredgecolor=CLASS_EDGE[c], markeredgewidth=0.5, markersize=4.5, label=lab))
        else:
            hs.append(Patch(facecolor=CLASS_COLOR[c], edgecolor=CLASS_EDGE[c], hatch=CLASS_HATCH[c],
                            linewidth=0.5, label=lab))
    return hs


_MANIFEST: list = []


def save(fig, name: str, width: str, out_dir: Path | str = "figures_rp", tiff_dpi: int = 600,
         png_dpi: int = 300, greyscale_check: bool = True, note: str = "") -> dict:
    """Speichert PDF (Vektor, exakte Breite), TIFF (RGB, LZW), PNG (Sichtprobe) und eine Graustufenprobe.

    Kein bbox_inches='tight': die physische Breite bleibt die Zielbreite. Layout vorher
    mit constrained_layout oder subplots_adjust setzen.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w_in, h_in = fig.get_size_inches()
    meta = {"Title": name, "Author": "", "Creator": "matplotlib", "Subject": ""}   # keine Autorenhinweise (Doppelblind)
    fig.savefig(out / f"{name}.pdf", format="pdf", metadata=meta)
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=tiff_dpi)                 # Raster in Zielaufloesung
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[3])                             # RGB ohne Alphakanal (Elsevier: RGB)
    bg.save(out / f"{name}.tif", format="TIFF", compression="tiff_lzw", dpi=(tiff_dpi, tiff_dpi))
    fig.savefig(out / f"{name}.png", format="png", dpi=png_dpi)
    if greyscale_check:
        Image.open(out / f"{name}.png").convert("L").save(out / f"{name}_greyscale.png")
    plt.close(fig)
    rec = {"file": name, "width_class": width, "width_mm": round(w_in * 25.4, 1),
           "height_mm": round(h_in * 25.4, 1), "tiff_dpi": tiff_dpi, "font": active_font(), "note": note}
    _MANIFEST.append(rec)
    print(f"  {name}: {rec['width_mm']} x {rec['height_mm']} mm, {width}")
    return rec


def write_manifest(out_dir: Path | str = "figures_rp") -> None:
    import csv
    out = Path(out_dir)
    if not _MANIFEST:
        return
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_MANIFEST[0].keys()))
        w.writeheader()
        w.writerows(_MANIFEST)
    (out / "palette.json").write_text(json.dumps({
        "classes": {CLASS_EN[c]: CLASS_COLOR[c] for c in CLASS_ORDER},
        "markers": {CLASS_EN[c]: CLASS_MARKER[c] for c in CLASS_ORDER},
        "linestyles": {CLASS_EN[c]: CLASS_LS[c] for c in CLASS_ORDER},
        "phases": {str(k): v for k, v in PHASE_COLOR.items()},
        "sequential": ["#f5f8fc", "#c6d7ea", "#7fa5cf", "#3a74b0", "#07519d"],
        "diverging": ["#07519d", "#7fa5cf", "#f4f4f4", "#e39a7a", "#bb4717"],
        "font": active_font(), "font_sizes_pt": FS, "widths_mm": WIDTH_MM}, indent=2))
