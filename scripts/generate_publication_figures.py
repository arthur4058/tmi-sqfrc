"""Generate publication-ready figures for the SORF-TMI manuscript.

The script uses a restrained, color-blind-friendly visual system and writes
both SVG (for the Markdown manuscript) and PDF (for final typesetting).
Experimental plots are derived from audited JSON artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]

INK = "#1F2937"
MUTED = "#667085"
GRID = "#D9E0E7"
BLUE = "#0072B2"
BLUE_DARK = "#005A8D"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7A5195"
GREY = "#5F6B7A"
LIGHT_BLUE = "#EEF6FA"
LIGHT_ORANGE = "#FCF3EC"
LIGHT_GREEN = "#ECF7F3"
LIGHT_PURPLE = "#F4EFF7"
LIGHT_GREY = "#F5F7F9"
WHITE = "#FFFFFF"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.7,
            "lines.markersize": 5.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
        }
    )


def clean_axes(axis, grid: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(MUTED)
    axis.spines["bottom"].set_color(MUTED)
    axis.tick_params(colors=INK, width=0.7, length=3)
    if grid:
        axis.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
    axis.set_axisbelow(True)


def save_figure(figure, output: Path, preview_dir: Path | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "SORF-TMI reproducible figure generator"}
    figure.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata=metadata,
    )
    svg = output.read_text(encoding="utf-8")
    output.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
    )
    figure.savefig(
        output.with_suffix(".pdf"),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata=metadata,
    )
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            preview_dir / f"{output.stem}.png",
            dpi=180,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(figure)


def rounded_box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    face: str,
    edge: str,
    linewidth: float = 0.9,
    radius: float = 0.014,
    linestyle: str = "-",
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        transform=axis.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=1,
    )
    axis.add_patch(patch)
    return patch


def text_block(
    axis,
    x: float,
    y: float,
    title: str,
    lines: list[str],
    title_size: float = 8.5,
    body_size: float = 7.3,
    align: str = "center",
    line_gap: float = 0.06,
) -> None:
    axis.text(
        x,
        y,
        title,
        transform=axis.transAxes,
        ha=align,
        va="top",
        fontsize=title_size,
        fontweight="semibold",
        color=INK,
        zorder=3,
    )
    for index, line in enumerate(lines):
        axis.text(
            x,
            y - 0.085 - index * line_gap,
            line,
            transform=axis.transAxes,
            ha=align,
            va="top",
            fontsize=body_size,
            color=MUTED,
            zorder=3,
        )


def arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = GREY,
    linewidth: float = 1.1,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0",
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        transform=axis.transAxes,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=4,
    )
    axis.add_patch(patch)


def step_badge(axis, x: float, y: float, label: str, color: str) -> None:
    axis.add_patch(
        Circle(
            (x, y),
            0.017,
            transform=axis.transAxes,
            facecolor=color,
            edgecolor=WHITE,
            linewidth=0.7,
            zorder=5,
        )
    )
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color=WHITE,
        zorder=6,
    )


def figure_data_protocol(output_dir: Path, preview_dir: Path | None) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.05))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    boxes = [
        (0.015, 0.22, 0.16, 0.68, LIGHT_BLUE, BLUE, "GeoLife 1.3", ["Timestamped GPS", "User ID + mode label"]),
        (0.205, 0.22, 0.17, 0.68, LIGHT_GREEN, GREEN, "User split", ["Train: 45 users", "Val: 6 users", "Test: 13 users"]),
        (0.405, 0.22, 0.17, 0.68, LIGHT_ORANGE, ORANGE, "Time windows", ["Length: 300 s", "Stride: 150 s", "Single-mode only"]),
        (0.605, 0.22, 0.20, 0.68, LIGHT_PURPLE, PURPLE, "Sampling views", ["5 / 10 / 20 / 30 / 60 s", "First point per bin", "No interpolation"]),
        (0.835, 0.22, 0.15, 0.68, LIGHT_GREY, GREY, "Matched test", ["Same-rate", "train / val / test", "Acc. + Macro-F1"]),
    ]
    for index, (x, y, width, height, face, edge, title, lines) in enumerate(boxes, start=1):
        rounded_box(axis, x, y, width, height, face, edge)
        step_badge(axis, x + 0.025, y + height - 0.035, str(index), edge)
        text_block(axis, x + width / 2, y + height - 0.075, title, lines, title_size=8.0 if index in (1, 5) else 8.3, body_size=6.6 if index in (1, 5) else 7.0)
        if index < len(boxes):
            arrow(axis, (x + width + 0.008, 0.56), (boxes[index][0] - 0.008, 0.56))

    # Source trajectory.
    xs = np.array([0.035, 0.055, 0.078, 0.100, 0.128, 0.153])
    ys = np.array([0.38, 0.48, 0.43, 0.61, 0.50, 0.66])
    axis.plot(xs, ys, transform=axis.transAxes, color=BLUE, linewidth=1.4, zorder=3)
    axis.scatter(xs, ys, transform=axis.transAxes, s=14, color=BLUE, edgecolor=WHITE, linewidth=0.5, zorder=4)

    # Split bands.
    band_y = [0.42, 0.50, 0.58]
    band_w = [0.095, 0.055, 0.040]
    for y, width, color in zip(band_y, band_w, [GREEN, BLUE, ORANGE]):
        axis.add_patch(Rectangle((0.245, y), width, 0.026, transform=axis.transAxes, facecolor=color, edgecolor="none", zorder=3))

    # Sliding windows.
    axis.plot([0.455, 0.565], [0.52, 0.52], transform=axis.transAxes, color=ORANGE, linewidth=1.2, zorder=3)
    axis.add_patch(Rectangle((0.455, 0.48), 0.072, 0.08, transform=axis.transAxes, facecolor="none", edgecolor=ORANGE, linewidth=1.0, zorder=3))
    axis.add_patch(Rectangle((0.493, 0.48), 0.072, 0.08, transform=axis.transAxes, facecolor=ORANGE, alpha=0.12, edgecolor=ORANGE, linewidth=1.0, zorder=3))

    # Five sampling rows.
    for row, count in enumerate([11, 8, 6, 4, 3]):
        y = 0.59 - row * 0.052
        axis.plot([0.665, 0.805], [y, y], transform=axis.transAxes, color="#C9B6D4", linewidth=0.6, zorder=2)
        locations = np.linspace(0.674, 0.797, count)
        axis.scatter(locations, np.full_like(locations, y), transform=axis.transAxes, s=6 + row * 2, color=PURPLE, zorder=3)

    rounded_box(axis, 0.015, 0.055, 0.97, 0.095, WHITE, GRID, linewidth=0.8, radius=0.008)
    axis.text(
        0.5,
        0.102,
        "Controlled invariants: users, physical windows, labels and pair IDs   |   Controlled variable: GPS sampling interval",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=7.4,
        color=INK,
    )
    save_figure(figure, output_dir / "fig1_data_protocol.svg", preview_dir)


def figure_overall_framework(output_dir: Path, preview_dir: Path | None) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.9))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    # Shared low-rate input.
    rounded_box(axis, 0.02, 0.20, 0.15, 0.63, LIGHT_GREY, GREY)
    axis.text(0.095, 0.785, "Low-rate GPS", transform=axis.transAxes, ha="center", va="top", fontsize=9.0, fontweight="semibold", color=INK)
    axis.text(0.095, 0.710, "real observations only", transform=axis.transAxes, ha="center", va="top", fontsize=7.0, color=MUTED)
    axis.text(0.095, 0.655, "coordinates + motion", transform=axis.transAxes, ha="center", va="top", fontsize=7.0, color=MUTED)
    xs = np.linspace(0.045, 0.145, 6)
    ys = np.array([0.35, 0.45, 0.40, 0.56, 0.48, 0.62])
    axis.plot(xs, ys, transform=axis.transAxes, color=GREY, linewidth=1.2)
    axis.scatter(xs, ys, transform=axis.transAxes, s=16, color=GREY, edgecolor=WHITE, linewidth=0.5, zorder=4)
    rounded_box(axis, 0.047, 0.245, 0.096, 0.055, WHITE, GREY, linewidth=0.7, radius=0.006)
    axis.text(0.095, 0.272, "validity masks", transform=axis.transAxes, ha="center", va="center", fontsize=6.8, color=INK)

    # Base expert.
    rounded_box(axis, 0.24, 0.57, 0.43, 0.30, LIGHT_BLUE, BLUE)
    axis.text(0.265, 0.825, "Base expert B0", transform=axis.transAxes, ha="left", va="top", fontsize=9.2, fontweight="semibold", color=BLUE_DARK)
    rounded_box(axis, 0.29, 0.665, 0.13, 0.095, WHITE, BLUE, linewidth=0.8)
    rounded_box(axis, 0.49, 0.665, 0.13, 0.095, WHITE, BLUE, linewidth=0.8)
    axis.text(0.355, 0.712, "Trajectory encoder", transform=axis.transAxes, ha="center", va="center", fontsize=7.2, color=INK)
    axis.text(0.555, 0.712, "Feature encoder", transform=axis.transAxes, ha="center", va="center", fontsize=7.2, color=INK)
    axis.text(0.455, 0.712, "+", transform=axis.transAxes, ha="center", va="center", fontsize=9.0, color=BLUE_DARK)
    axis.text(0.455, 0.612, "path + motion evidence  →  p_B", transform=axis.transAxes, ha="center", va="center", fontsize=7.2, color=INK)

    # Relational expert.
    rounded_box(axis, 0.24, 0.12, 0.43, 0.34, LIGHT_ORANGE, ORANGE)
    axis.text(0.265, 0.420, "Sparse-relation expert R", transform=axis.transAxes, ha="left", va="top", fontsize=9.0, fontweight="semibold", color=ORANGE)
    component_x = [0.285, 0.405, 0.525]
    component_titles = ["Point MLP", "Relations", "Rel. encoder"]
    for x, title in zip(component_x, component_titles):
        rounded_box(axis, x, 0.265, 0.105, 0.085, WHITE, ORANGE, linewidth=0.8)
        axis.text(x + 0.0525, 0.307, title, transform=axis.transAxes, ha="center", va="center", fontsize=6.8, color=INK, linespacing=1.05)
    arrow(axis, (0.390, 0.307), (0.405, 0.307), color=ORANGE, linewidth=0.8)
    arrow(axis, (0.510, 0.307), (0.525, 0.307), color=ORANGE, linewidth=0.8)
    axis.text(0.455, 0.220, "long-range relational evidence  →  p_R", transform=axis.transAxes, ha="center", va="center", fontsize=7.2, color=INK)
    axis.text(0.455, 0.158, "training regularization: point drop + consistency", transform=axis.transAxes, ha="center", va="center", fontsize=6.7, color=MUTED)

    # Validation-controlled fusion.
    rounded_box(axis, 0.76, 0.24, 0.21, 0.50, LIGHT_PURPLE, PURPLE)
    axis.text(0.865, 0.695, "Validation fusion", transform=axis.transAxes, ha="center", va="top", fontsize=8.8, fontweight="semibold", color=PURPLE)
    rounded_box(axis, 0.79, 0.535, 0.15, 0.065, WHITE, PURPLE, linewidth=0.8)
    axis.text(0.865, 0.567, "temperature scale", transform=axis.transAxes, ha="center", va="center", fontsize=7.0, color=INK)
    rounded_box(axis, 0.79, 0.405, 0.15, 0.075, WHITE, PURPLE, linewidth=0.8)
    axis.text(0.865, 0.442, "p = (1−w) pB + w pR", transform=axis.transAxes, ha="center", va="center", fontsize=7.4, color=INK)
    rounded_box(axis, 0.79, 0.295, 0.15, 0.06, PURPLE, PURPLE, linewidth=0.8)
    axis.text(0.865, 0.325, "five-mode prediction", transform=axis.transAxes, ha="center", va="center", fontsize=7.2, color=WHITE, fontweight="semibold")

    arrow(axis, (0.17, 0.57), (0.24, 0.72), color=BLUE)
    arrow(axis, (0.17, 0.40), (0.24, 0.29), color=ORANGE)
    arrow(axis, (0.67, 0.71), (0.76, 0.565), color=BLUE)
    arrow(axis, (0.67, 0.28), (0.76, 0.455), color=ORANGE)
    axis.text(0.716, 0.655, "p_B", transform=axis.transAxes, ha="center", fontsize=6.8, color=BLUE_DARK)
    axis.text(0.716, 0.365, "p_R", transform=axis.transAxes, ha="center", fontsize=6.8, color=ORANGE)
    arrow(axis, (0.865, 0.535), (0.865, 0.480), color=PURPLE, linewidth=0.8)
    arrow(axis, (0.865, 0.405), (0.865, 0.355), color=PURPLE, linewidth=0.8)

    axis.text(0.5, 0.045, "Checkpoint selection, calibration and fusion use validation data only; test labels are never accessed.", transform=axis.transAxes, ha="center", va="center", fontsize=7.0, color=MUTED)
    save_figure(figure, output_dir / "fig2_sorf_tmi_overall.svg", preview_dir)


def figure_relation_encoder(output_dir: Path, preview_dir: Path | None) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.25))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    blocks = [
        (0.02, 0.23, 0.18, 0.66, LIGHT_BLUE, BLUE, "Observed anchors", ["M real GPS points", "12-D point descriptors"]),
        (0.27, 0.23, 0.25, 0.66, LIGHT_ORANGE, ORANGE, "Relational tokenization", ["M point tokens", "M(M−1)/2 pair tokens"]),
        (0.59, 0.23, 0.17, 0.66, LIGHT_GREEN, GREEN, "Relation Transformer", ["2 layers / 4 heads", "Masked self-attention"]),
        (0.83, 0.23, 0.15, 0.66, LIGHT_PURPLE, PURPLE, "Trajectory representation", ["mean / std / max", "statistics + quality", "expert probability p_R"]),
    ]
    for index, (x, y, width, height, face, edge, title, lines) in enumerate(blocks):
        rounded_box(axis, x, y, width, height, face, edge)
        text_block(axis, x + width / 2, y + height - 0.06, title, lines, title_size=8.2, body_size=7.0)
        if index < len(blocks) - 1:
            arrow(axis, (x + width + 0.012, 0.56), (blocks[index + 1][0] - 0.012, 0.56))

    # Six real anchors.
    xs = np.linspace(0.045, 0.175, 6)
    ys = np.array([0.40, 0.48, 0.43, 0.61, 0.53, 0.69])
    axis.plot(xs, ys, transform=axis.transAxes, color=BLUE, linewidth=1.2, zorder=3)
    axis.scatter(xs, ys, transform=axis.transAxes, s=22, color=BLUE, edgecolor=WHITE, linewidth=0.6, zorder=4)
    for index, (x, y) in enumerate(zip(xs, ys), start=1):
        axis.text(x, y - 0.045, f"p{index}", transform=axis.transAxes, ha="center", fontsize=6.2, color=MUTED)

    # Point and pair-token motif.
    token_x = np.array([0.305, 0.350, 0.395, 0.440, 0.485])
    token_y = np.array([0.57, 0.66, 0.55, 0.68, 0.59])
    for i in range(len(token_x)):
        for j in range(i + 1, len(token_x)):
            axis.plot([token_x[i], token_x[j]], [token_y[i], token_y[j]], transform=axis.transAxes, color=ORANGE, alpha=0.28, linewidth=0.6, zorder=2)
    axis.scatter(token_x, token_y, transform=axis.transAxes, s=28, color=ORANGE, edgecolor=WHITE, linewidth=0.6, zorder=3)
    rounded_box(axis, 0.305, 0.34, 0.18, 0.105, WHITE, ORANGE, linewidth=0.75, radius=0.007)
    axis.text(0.395, 0.405, "r_ij = MLP[x_i, x_j,", transform=axis.transAxes, ha="center", va="center", fontsize=6.9, color=INK)
    axis.text(0.395, 0.370, "x_j−x_i, Δt_ij]", transform=axis.transAxes, ha="center", va="center", fontsize=6.9, color=INK)

    # Transformer stack.
    for y, label in [(0.61, "Layer 1"), (0.48, "Layer 2"), (0.35, "Masked pool")]:
        rounded_box(axis, 0.62, y, 0.11, 0.075, WHITE, GREEN, linewidth=0.75, radius=0.006)
        axis.text(0.675, y + 0.037, label, transform=axis.transAxes, ha="center", va="center", fontsize=7.0, color=INK)
    arrow(axis, (0.675, 0.61), (0.675, 0.555), color=GREEN, linewidth=0.8)
    arrow(axis, (0.675, 0.48), (0.675, 0.425), color=GREEN, linewidth=0.8)

    # Pooled output bars.
    labels = ["μ", "σ", "max", "q"]
    for index, label in enumerate(labels):
        x = 0.85 + index * 0.028
        height = [0.13, 0.09, 0.16, 0.11][index]
        axis.add_patch(Rectangle((x, 0.43), 0.018, height, transform=axis.transAxes, facecolor=PURPLE, alpha=0.82, edgecolor="none", zorder=3))
        axis.text(x + 0.009, 0.405, label, transform=axis.transAxes, ha="center", va="top", fontsize=6.4, color=MUTED)

    rounded_box(axis, 0.02, 0.055, 0.96, 0.095, WHITE, GRID, linewidth=0.8, radius=0.008)
    axis.text(
        0.5,
        0.103,
        "No coordinate interpolation   |   Long-range relations from real observations   |   Training regularized by point dropping and consistency",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=7.1,
        color=INK,
    )
    save_figure(figure, output_dir / "fig3_relation_encoder.svg", preview_dir)


def load_main_results() -> list[dict]:
    path = ROOT / "reports/experiments/geolife_v74_five_rate_multiseed.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = [row for row in payload["results"] if row["seed"] == 10086]
    selected.sort(key=lambda row: row["rate_seconds"])
    if [row["rate_seconds"] for row in selected] != [5, 10, 20, 30, 60]:
        raise AssertionError("seed10086 five-rate results are incomplete")
    return selected


def figure_main_results(output_dir: Path, preview_dir: Path | None) -> None:
    rows = load_main_results()
    rates = np.array([row["rate_seconds"] for row in rows])
    x = np.arange(len(rates))
    b0_acc = np.array([100 * row["test"]["b0"]["accuracy"] for row in rows])
    sorf_acc = np.array([100 * row["test"]["v74"]["accuracy"] for row in rows])
    b0_f1 = np.array([100 * row["test"]["b0"]["macro_f1"] for row in rows])
    sorf_f1 = np.array([100 * row["test"]["v74"]["macro_f1"] for row in rows])

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    panels = [
        (axes[0], b0_acc, sorf_acc, "(a) Accuracy", (68, 83)),
        (axes[1], b0_f1, sorf_f1, "(b) Macro-F1", (60, 80)),
    ]
    for axis, baseline, proposed, title, limits in panels:
        axis.axvspan(2.55, 4.18, color=LIGHT_ORANGE, alpha=0.55, zorder=0)
        axis.text(3.37, limits[1] - 0.7, "low-rate focus", ha="center", va="top", fontsize=6.8, color=MUTED)
        axis.plot(x, baseline, marker="o", color=GREY, label="B0", zorder=3)
        axis.plot(x, proposed, marker="s", color=BLUE, label="SORF-TMI", zorder=4)
        for xi, yi, delta in zip(x, proposed, proposed - baseline):
            axis.annotate(
                f"+{delta:.2f}",
                (xi, yi),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.7,
                color=BLUE_DARK,
            )
        axis.set_title(title, loc="left", fontweight="semibold", color=INK)
        axis.set_xticks(x, [str(rate) for rate in rates])
        axis.set_xlabel("Sampling interval (s)")
        axis.set_ylabel("Score (%)")
        axis.set_ylim(*limits)
        axis.set_xlim(-0.2, len(x) - 0.8)
        clean_axes(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.83, wspace=0.27)
    save_figure(figure, output_dir / "fig4_five_rate_results.svg", preview_dir)


def load_confusion_results() -> dict:
    path = ROOT / "reports/experiments/sorf_tmi_confusion_30s_60s_seed10086.json"
    return json.loads(path.read_text(encoding="utf-8"))


def annotate_heatmap(axis, values: np.ndarray, limit: float | None = None) -> None:
    threshold = 58.0 if limit is None else 0.52 * limit
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = f"{value:.1f}" if limit is None else f"{value:+.1f}"
            color = WHITE if abs(value) >= threshold else INK
            axis.text(column, row, label, ha="center", va="center", fontsize=7.1, color=color)


def figure_confusion_matrices(output_dir: Path, preview_dir: Path | None) -> None:
    payload = load_confusion_results()
    classes = payload["class_names"]
    cmap = LinearSegmentedColormap.from_list(
        "paper_blues", ["#F7FAFC", "#D8EAF3", "#8FC2DA", "#2F7FB6", "#0B3C5D"]
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    image = None
    panel = 0
    for row_index, result in enumerate(payload["results"]):
        for column_index, model in enumerate(("b0", "sorf_tmi")):
            axis = axes[row_index, column_index]
            values = 100 * np.asarray(result["matrices"][model]["row_normalized"])
            image = axis.imshow(values, vmin=0, vmax=100, cmap=cmap, aspect="equal")
            annotate_heatmap(axis, values)
            model_label = "B0" if model == "b0" else "SORF-TMI"
            axis.set_title(
                f"({chr(97 + panel)}) {result['rate_seconds']} s — {model_label}",
                loc="left",
                fontweight="semibold",
                color=INK,
                pad=6,
            )
            panel += 1
            axis.set_xticks(np.arange(len(classes)), classes, rotation=25, ha="right")
            axis.set_yticks(np.arange(len(classes)), classes)
            axis.tick_params(length=0)
            if row_index == 1:
                axis.set_xlabel("Predicted class")
            if column_index == 0:
                axis.set_ylabel("True class")
            else:
                axis.set_yticklabels([])
            for spine in axis.spines.values():
                spine.set_color(WHITE)
                spine.set_linewidth(1.2)
            axis.set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
            axis.grid(which="minor", color=WHITE, linewidth=1.0)
            axis.tick_params(which="minor", bottom=False, left=False)
    colorbar = figure.colorbar(image, ax=axes, fraction=0.028, pad=0.025)
    colorbar.set_label("Within-class proportion (%)")
    colorbar.outline.set_linewidth(0.6)
    figure.subplots_adjust(left=0.10, right=0.91, bottom=0.10, top=0.95, wspace=0.10, hspace=0.34)
    save_figure(figure, output_dir / "fig5_confusion_matrices.svg", preview_dir)


def figure_confusion_delta(output_dir: Path, preview_dir: Path | None) -> None:
    payload = load_confusion_results()
    classes = payload["class_names"]
    matrices = [100 * np.asarray(result["row_normalized_delta"]) for result in payload["results"]]
    limit = math.ceil(max(float(np.abs(matrix).max()) for matrix in matrices))
    cmap = LinearSegmentedColormap.from_list(
        "paper_diverging", ["#B2182B", "#EF8A8C", "#F7F7F7", "#8FC2DA", "#0B5A8C"]
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.95))
    image = None
    for index, (axis, result, values) in enumerate(zip(axes, payload["results"], matrices)):
        image = axis.imshow(values, vmin=-limit, vmax=limit, cmap=cmap, aspect="equal")
        annotate_heatmap(axis, values, limit=float(limit))
        axis.set_title(
            f"({chr(97 + index)}) {result['rate_seconds']} s: SORF-TMI − B0",
            loc="left",
            fontweight="semibold",
            color=INK,
            pad=6,
        )
        axis.set_xticks(np.arange(len(classes)), classes, rotation=25, ha="right")
        axis.set_yticks(np.arange(len(classes)), classes if index == 0 else [])
        axis.set_xlabel("Predicted class")
        if index == 0:
            axis.set_ylabel("True class")
        axis.tick_params(length=0)
        for diagonal in range(len(classes)):
            axis.add_patch(Rectangle((diagonal - 0.49, diagonal - 0.49), 0.98, 0.98, facecolor="none", edgecolor=INK, linewidth=0.65))
        axis.set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
        axis.grid(which="minor", color=WHITE, linewidth=1.0)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_color(WHITE)
            spine.set_linewidth(1.2)
    colorbar = figure.colorbar(image, ax=axes, fraction=0.035, pad=0.03)
    colorbar.set_label("Change (percentage points)")
    colorbar.outline.set_linewidth(0.6)
    figure.subplots_adjust(left=0.10, right=0.90, bottom=0.22, top=0.90, wspace=0.18)
    save_figure(figure, output_dir / "fig6_confusion_delta.svg", preview_dir)


def load_ablation() -> dict:
    path = ROOT / "reports/experiments/geolife_v74_ablation_30s_60s.json"
    return json.loads(path.read_text(encoding="utf-8"))


def figure_ablation(output_dir: Path, preview_dir: Path | None) -> None:
    payload = load_ablation()
    variants = payload["variants"]
    ids = [variant["id"] for variant in variants]
    x = np.arange(len(ids))
    colors = [GREY, "#AAB2BD", "#8FC2DA", "#4D9BC4", BLUE]
    hatches = ["", "//", "", "..", ""]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), sharey=True)
    for index, (axis, rate) in enumerate(zip(axes, (30, 60))):
        values = np.array([100 * variant["rates"][str(rate)]["macro_f1"] for variant in variants])
        bars = axis.bar(x, values, width=0.68, color=colors, edgecolor=INK, linewidth=0.55, zorder=3)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        for xi, value in zip(x, values):
            axis.text(xi, value + 0.25, f"{value:.2f}", ha="center", va="bottom", fontsize=6.8, color=INK)
        axis.set_title(f"({chr(97 + index)}) {rate} s", loc="left", fontweight="semibold", color=INK)
        axis.set_xticks(x, ["B0", "A1", "A2", "A3", "A4\n(full)"])
        axis.set_xlabel("Ablation variant")
        axis.set_ylim(60, 73)
        axis.set_yticks([60, 64, 68, 72])
        axis.set_xlim(-0.55, len(ids) - 0.45)
        clean_axes(axis)
    axes[0].set_ylabel("Macro-F1 (%)")
    figure.text(
        0.5,
        0.03,
        "A1: relational expert   A2: A1 + B0 fusion   A3: A2 + observation drop   A4: A3 + consistency",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=MUTED,
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.27, top=0.91, wspace=0.18)
    save_figure(figure, output_dir / "fig7_ablation.svg", preview_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="docs/figures")
    parser.add_argument("--preview-dir", default=None)
    args = parser.parse_args()

    configure_style()
    output_dir = ROOT / args.output_dir
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    generators = [
        figure_data_protocol,
        figure_overall_framework,
        figure_relation_encoder,
        figure_main_results,
        figure_confusion_matrices,
        figure_confusion_delta,
        figure_ablation,
    ]
    for generator in generators:
        generator(output_dir, preview_dir)
    print(f"Generated {len(generators)} SVG/PDF figure pairs in {output_dir}")


if __name__ == "__main__":
    main()
