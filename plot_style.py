"""
plot_style.py — Einheitliches, publikationsreifes Matplotlib-Styling fuer die F3-Figures.

Verwendung in einem Plot-Skript:
    from plot_style import apply_pub_style, save_fig
    apply_pub_style()
    ...
    save_fig(fig, output_dir / "radar_profiles.png")   # schreibt .png (300 dpi) + .pdf (Vektor)

Die Signal- und Dimensionsfarben werden aus config.py uebernommen, damit alle
Abbildungen dieselbe Farbzuordnung verwenden (Konsistenz ueber alle Figures).
"""
from __future__ import annotations

from pathlib import Path
import matplotlib as mpl

try:
    from config import SIGNAL_COLORS, DIM_COLORS  # einheitliche Palette
except Exception:  # pragma: no cover
    SIGNAL_COLORS, DIM_COLORS = {}, {}


PUB_RCPARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    # Vektor-PDF mit einbettbarem, durchsuchbarem Text:
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
}


def apply_pub_style() -> None:
    """Setzt die einheitlichen rcParams (Schriftgroessen, DPI, Vektor-Fonts)."""
    mpl.rcParams.update(PUB_RCPARAMS)


def save_fig(fig, path, formats=("png", "pdf"), dpi=300) -> None:
    """Speichert eine Figure konsistent als 300-dpi-PNG und als Vektor-PDF."""
    p = Path(path)
    stem = p.with_suffix("")
    for fmt in formats:
        fig.savefig(f"{stem}.{fmt}", dpi=dpi, bbox_inches="tight", facecolor="white")
