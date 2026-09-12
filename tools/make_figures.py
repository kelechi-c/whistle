"""Regenerate the four publication figures for the Whistle article.

Run from the repository root (matplotlib is not a runtime dependency, and
--no-project keeps uv from syncing the qwen-tts/torch environment):

    uv run --no-project --with matplotlib python tools/make_figures.py [--only STEM]

Outputs PNG (220 dpi) and SVG into local/article/article/. The SVG keeps glyph
outlines so it renders identically where SF Pro is not installed.

Data is read from the saved evidence JSON files, never transcribed from images:

    quality/speed  evidence/{official_current_p50_5runs,v7_current_p50_5runs,
                   victoria_faster_0.6b_3runs}.json  -> p50 RTF
                   results/victoria_fresh_{official,whistle}_wer.json
                   sandbox/shrink/results/wer_alicia_faster.json -> WER
    overall RTF    same three p50 files, plus the 1.7B files and the RTX PRO
                   6000 rows of docs/latency_report.md (Modal; no raw JSON)
    module latency evidence/victoria_modular_*.json and
                   local/benchmarks/modal/modular_*.json

Font: SF Pro, the Sciel body font. Resolution order is tools/fonts, the SF Pro
Display download, then Sciel's bundled sfpromedium.otf. Missing weights fall
back to the nearest available one. No silent substitution happens: if no SF Pro
file is found the script stops with the directories it searched.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "local" / "article" / "article"
EVIDENCE = ROOT / "evidence"
RESULTS = ROOT / "results"
MODAL = ROOT / "local" / "benchmarks" / "modal"

FONT_DIRS = (
    ROOT / "tools" / "fonts",
    Path("/home/tensor/Shared/Downloads/sf-pro-display"),
    Path("/home/tensor/Shared/code/miscellaneous/sciel"),
)
# Weight -> filename fragment, in preference order. Italics are never used.
FONT_WEIGHTS = {
    "regular": ("regular",),
    "medium": ("medium", "semibold", "bold"),
    "bold": ("bold", "semibold", "medium"),
}

BACKGROUND = "#f3f1ed"
INK = "#1c1c1c"
INK_2 = "#55524e"
MUTED = "#8b8781"
GRID = "#dedad3"
RAIL = "#e9e5de"
AXIS = "#b9b4ac"

# Grayscale series ramp. Values are ordered so the highlighted runtime stays the
# highest-contrast mark on the muted-white canvas; the same ordering also reads
# correctly after the Sciel dark-theme CSS inverts the figure.
FILL = {"whistle": "#2f2f2f", "faster": "#8d8d8d", "official": "#dcd8d1"}
EDGE = {"whistle": "#1c1c1c", "faster": "#6e6e6e", "official": "#a5a099"}
NAME = {
    "whistle": "Whistle",
    "faster": "faster-qwen3-tts",
    "official": "official (qwen-tts)",
}
RUNTIMES = ("official", "faster", "whistle")


def load_fonts() -> dict[str, Path]:
    """Resolve SF Pro files per weight, falling back to whichever weight exists."""
    found: list[Path] = [
        path
        for directory in FONT_DIRS
        if directory.is_dir()
        for path in sorted(directory.iterdir())
        if path.suffix.lower() in {".otf", ".ttf"} and "italic" not in path.name.lower()
    ]
    if not found:
        raise SystemExit(f"no SF Pro file found in: {', '.join(map(str, FONT_DIRS))}")

    picks: dict[str, Path] = {}
    for weight, fragments in FONT_WEIGHTS.items():
        for fragment in fragments:
            match = next((p for p in found if fragment in p.name.lower()), None)
            if match is not None:
                picks[weight] = match
                break
    spare = next(iter(picks.values()))
    return {weight: picks.get(weight, spare) for weight in FONT_WEIGHTS}


FONT_FILE = load_fonts()
for _path in set(FONT_FILE.values()):
    fontManager.addfont(str(_path))


def font(weight: str, size: float) -> FontProperties:
    """FontProperties for one weight/size pair; weights never share a family name."""
    return FontProperties(fname=str(FONT_FILE[weight]), size=size)


def style_axes(ax: plt.Axes, *, ygrid: bool = False, left_spine: bool = True) -> None:
    """Strip chart junk: no top/right frame, hairlines, zero-length ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left_spine)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["left"].set_color(AXIS)
    ax.spines["left"].set_linewidth(0.9)
    ax.tick_params(length=0, pad=7)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontproperties(font("regular", 10.5))
        label.set_color(INK_2)


def markers(ax: plt.Axes, x: float, y: float, key: str, *, size: float = 9.0) -> None:
    """Single filled dot in the runtime's grayscale."""
    ax.plot(
        [x],
        [y],
        marker="o",
        markersize=size,
        markerfacecolor=FILL[key],
        markeredgecolor=EDGE[key],
        markeredgewidth=1.0,
        linestyle="none",
        zorder=3,
    )


def value_text(ax: plt.Axes, x: float, y: float, text: str, *, ha: str = "left", pad: float = 13.0) -> None:
    """Numeric label beside a marker, offset by a fixed point gap so the distance
    stays the same at every scale instead of collapsing against the dot."""
    direction = -1 if ha == "right" else 1
    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(direction * pad, 0),
        ha=ha,
        va="center",
        fontproperties=font("regular", 10.5),
        color=INK_2,
        zorder=4,
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=220, facecolor=BACKGROUND)
    fig.savefig(OUT / f"{stem}.svg", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"wrote {OUT / stem}.png and .svg")


def p50_rtf(path: Path) -> float:
    """Median of per-run RTF ratios in a bench record."""
    runs = json.loads(path.read_text())["iterations"]
    return statistics.median(run["rtf"] for run in runs)


def wer_percent(path: Path) -> float:
    """Word error rate from a Qwen3-ASR evaluation record, in percent."""
    return 100.0 * json.loads(path.read_text())["wer"]


def module_ms(path: Path) -> tuple[float, float]:
    """(predictor, talker) CUDA-event milliseconds per call."""
    record = json.loads(path.read_text())
    return record["predictor_ms_per_call"], record["talker_ms_per_call"]


def overall_rtf() -> dict[str, dict[str, dict[str, float]]]:
    """p50 generation RTF per GPU, model size, and runtime."""
    return {
        "RTX 3050 Laptop · 6 GB": {
            "0.6B": {
                "official": p50_rtf(EVIDENCE / "official_current_p50_5runs.json"),
                "faster": p50_rtf(EVIDENCE / "victoria_faster_0.6b_3runs.json"),
                "whistle": p50_rtf(EVIDENCE / "v7_current_p50_5runs.json"),
            },
            "1.7B": {
                "official": p50_rtf(EVIDENCE / "official_1.7b_3runs.json"),
                "faster": p50_rtf(EVIDENCE / "victoria_faster_1.7b_3runs.json"),
                "whistle": p50_rtf(EVIDENCE / "v7_1.7b_3runs.json"),
            },
        },
        # Modal rows: docs/latency_report.md, 3-5 runs, no raw JSON retained.
        "RTX PRO 6000 Blackwell · 96 GB": {
            "0.6B": {"official": 0.713, "faster": 0.199, "whistle": 0.260},
            "1.7B": {"official": 0.668, "faster": 0.235, "whistle": 0.265},
        },
    }


def module_latency() -> dict[str, tuple[str, dict[str, dict[str, tuple[float, float]]]]]:
    """Per GPU: display name and per model size the (predictor, talker) ms/call pair."""
    return {
        "victoria": (
            "RTX 3050 Laptop · 6 GB",
            {
                size: {
                    "whistle": module_ms(EVIDENCE / f"victoria_modular_whistle_{size}.json"),
                    "faster": module_ms(EVIDENCE / f"victoria_modular_faster_{size}.json"),
                }
                for size in ("0.6B", "1.7B")
            },
        ),
        "modal": (
            "RTX PRO 6000 Blackwell · 96 GB",
            {
                size: {
                    "whistle": module_ms(MODAL / f"modular_whistle_{size}.json"),
                    "faster": module_ms(MODAL / f"modular_faster_{size}.json"),
                }
                for size in ("0.6B", "1.7B")
            },
        ),
    }


def figure_title(fig: plt.Figure, x: float, title: str, subtitle: str, *, top: float) -> None:
    """Left-aligned title plus a muted subtitle line, both at the plot's left edge."""
    fig.text(x, top, title, ha="left", va="top", fontproperties=font("medium", 16.5), color=INK)
    fig.text(
        x,
        top - 0.062,
        subtitle,
        ha="left",
        va="top",
        fontproperties=font("regular", 11),
        color=INK_2,
    )


def footprint(fig: plt.Figure, x: float, y: float, text: str) -> None:
    """Muted provenance note pinned to the figure's bottom-left corner."""
    fig.text(x, y, text, ha="left", va="bottom", fontproperties=font("regular", 9.5), color=MUTED)


def fig_quality_speed() -> None:
    """Scatter: transcription quality against speed; two runtimes share a WER row."""
    points = [
        ("official", p50_rtf(EVIDENCE / "official_current_p50_5runs.json"), wer_percent(RESULTS / "victoria_fresh_official_wer.json")),
        ("whistle", p50_rtf(EVIDENCE / "v7_current_p50_5runs.json"), wer_percent(RESULTS / "victoria_fresh_whistle_wer.json")),
        ("faster", p50_rtf(EVIDENCE / "victoria_faster_0.6b_3runs.json"), wer_percent(ROOT / "sandbox" / "shrink" / "results" / "wer_alicia_faster.json")),
    ]
    wer, whistle = points[0][2], points[1][1]

    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    fig.subplots_adjust(left=0.125, right=0.965, top=0.845, bottom=0.16)
    figure_title(
        fig,
        ax.get_position().x0,
        "Transcription quality versus generation speed",
        "Whistle matches the official runtime's transcript at a lower real-time factor",
        top=0.975,
    )

    ax.plot(
        [points[1][1], points[0][1]],
        [wer, wer],
        color="#c9c4bc",
        linewidth=1.0,
        dashes=(5, 3),
        zorder=1,
    )
    ax.text(
        (points[0][1] + points[1][1]) / 2,
        wer - 0.13,
        "identical transcript (3.02% WER)",
        ha="center",
        va="top",
        fontproperties=font("regular", 10),
        color=MUTED,
    )
    for key, x, y in points:
        markers(ax, x, y, key, size=11)
    ax.annotate(
        NAME["official"], (points[0][1], wer), textcoords="offset points", xytext=(0, -18),
        ha="center", va="top", fontproperties=font("medium", 11.5), color=INK,
    )
    ax.annotate(
        NAME["whistle"], (points[1][1], wer), textcoords="offset points", xytext=(0, 16),
        ha="center", va="bottom", fontproperties=font("medium", 11.5), color=INK,
    )
    ax.annotate(
        NAME["faster"], (points[2][1], points[2][2]), textcoords="offset points", xytext=(0, 16),
        ha="center", va="bottom", fontproperties=font("medium", 11.5), color=INK,
    )

    ax.set_xlim(0.50, 0.93)
    ax.set_ylim(2.70, 5.15)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticks([3.0, 3.5, 4.0, 4.5, 5.0])
    ax.set_xlabel("Generation RTF ↓", fontproperties=font("regular", 12), color=INK, labelpad=9)
    ax.set_ylabel("WER (%) ↓", fontproperties=font("regular", 12), color=INK, labelpad=9)
    style_axes(ax, ygrid=True)
    footprint(
        fig,
        ax.get_position().x0,
        0.035,
        "Alicia · 0.6B · RTF is the p50 of repeated runs · WER from Qwen3-ASR-0.6B transcription of the generated audio",
    )
    save(fig, "whistle_wer_rtf_scatter")


def fig_gpu_rtf() -> None:
    """Two GPUs by two model sizes, one shared RTF scale.

    Each GPU is a band of two panels with its own heading, so the hardware name
    stays horizontal; the model size is a column heading shared by both bands.
    """
    data = overall_rtf()
    gpus = list(data)
    sizes = ("0.6B", "1.7B")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex=True)
    layout = {"left": 0.165, "right": 0.975, "top": 0.79, "bottom": 0.185, "hspace": 0.5, "wspace": 0.18}
    fig.subplots_adjust(**layout)
    xmax, xticks = 1.30, [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]

    for row, gpu in enumerate(gpus):
        for col, size in enumerate(sizes):
            ax = axes[row, col]
            values = data[gpu][size]
            for index, key in enumerate(RUNTIMES):
                y = len(RUNTIMES) - 1 - index
                markers(ax, values[key], y, key)
                value_text(ax, values[key], y, f"{values[key]:.2f}")
            ax.set_yticks(range(len(RUNTIMES)))
            ax.set_yticklabels([NAME[key] for key in reversed(RUNTIMES)])
            ax.set_ylim(-0.75, len(RUNTIMES) - 0.25)
            ax.set_xlim(0, xmax)
            ax.set_xticks(xticks, [f"{tick:.2f}" for tick in xticks])
            style_axes(ax, left_spine=False)
            if row == 0:
                ax.tick_params(labelbottom=False)
            if col == 1:
                ax.tick_params(labelleft=False)
            fig.text(
                ax.get_position().x0 + ax.get_position().width / 2,
                axes[0, 0].get_position().y1 + 0.078,
                size,
                ha="center",
                va="center",
                fontproperties=font("medium", 12.5),
                color=INK,
            )
        fig.text(
            layout["left"],
            axes[row, 0].get_position().y1 + 0.022,
            gpu,
            ha="left",
            va="bottom",
            fontproperties=font("medium", 12),
            color=INK,
        )
    axes[1, 0].set_xlabel("Generation RTF ↓", fontproperties=font("regular", 12), color=INK, labelpad=9)
    figure_title(
        fig,
        layout["left"],
        "End-to-end generation speed across GPUs and model sizes",
        "Alicia · bf16 · greedy decoding · one RTF scale shared by all four panels",
        top=0.975,
    )
    footprint(
        fig,
        layout["left"],
        0.038,
        "p50 of 3–5 measured runs · ↓ lower is better · official RTF is the historical Alicia-parity measurement",
    )
    save(fig, "whistle_gpu_rtf_comparison")


def module_markers(ax: plt.Axes, predictor: float, talker: float, y: float, key: str) -> None:
    """Draw the talker ring first and the predictor dot on top.

    The talker marker is larger so that where the two values coincide (measured
    to within 0.01 ms on the PRO 6000) the predictor dot still shows inside the
    ring instead of being hidden behind it.
    """
    ax.plot(
        [talker],
        [y],
        marker="o",
        markersize=12,
        markerfacecolor=BACKGROUND,
        markeredgecolor=EDGE[key],
        markeredgewidth=1.6,
        linestyle="none",
        zorder=2,
    )
    ax.plot(
        [predictor],
        [y],
        marker="o",
        markersize=8.5,
        markerfacecolor=FILL[key],
        markeredgecolor=EDGE[key],
        markeredgewidth=1.0,
        linestyle="none",
        zorder=3,
    )


def fig_module(slug: str, gpu: str, models: dict[str, dict[str, tuple[float, float]]]) -> None:
    """Dumbbell panels: predictor-to-talker spread per runtime, per model size."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.7))
    layout = {"left": 0.135, "right": 0.975, "top": 0.745, "bottom": 0.27, "wspace": 0.14}
    fig.subplots_adjust(**layout)
    xmax = max(max(pair) for model in models.values() for pair in model.values()) * 1.28

    for col, size in enumerate(("0.6B", "1.7B")):
        ax = axes[col]
        ax.set_xlim(0, xmax)
        for index, key in enumerate(("whistle", "faster")):
            y = 1 - index
            predictor, talker = models[size][key]
            ax.plot([0, xmax], [y, y], color=RAIL, linewidth=0.9, zorder=0)
            ax.plot([predictor, talker], [y, y], color="#8f8b84", linewidth=1.2, zorder=1)
            module_markers(ax, predictor, talker, y, key)
            # The two labels flank the dumbbell so a near-equal pair never stacks.
            value_text(ax, predictor, y, f"{predictor:.2f}", ha="left" if predictor >= talker else "right")
            value_text(ax, talker, y, f"{talker:.2f}", ha="left" if talker > predictor else "right")
        ax.set_yticks([1, 0])
        ax.set_yticklabels([NAME["whistle"], NAME["faster"]])
        ax.set_ylim(-0.45, 1.45)
        ax.set_title(size, fontproperties=font("medium", 12.5), color=INK, pad=10)
        style_axes(ax, left_spine=False)
        if col == 1:
            ax.tick_params(labelleft=False)

    axes[0].set_xlabel("Latency per call (ms)", fontproperties=font("regular", 12), color=INK, labelpad=9)
    axes[1].set_xlabel("Latency per call (ms)", fontproperties=font("regular", 12), color=INK, labelpad=9)
    # A real legend, so the key is drawn with the same glyphs as the data.
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markersize=8.5, markerfacecolor=INK,
                   markeredgecolor=INK, label="predictor"),
            Line2D([], [], marker="o", linestyle="none", markersize=12, markerfacecolor=BACKGROUND,
                   markeredgecolor=INK_2, markeredgewidth=1.6, label="talker"),
        ],
        loc="upper right",
        bbox_to_anchor=(layout["right"], 0.985),
        frameon=False,
        ncols=2,
        handletextpad=0.5,
        columnspacing=1.8,
        prop=font("regular", 10.5),
        labelcolor=INK_2,
    )
    figure_title(
        fig,
        layout["left"],
        f"Module latency on {gpu}",
        "CUDA-event time per call · 128-frame diagnostic after a 32-frame warmup",
        top=0.975,
    )
    footprint(
        fig,
        layout["left"],
        0.05,
        "Official module timing unavailable — absent, not zero · single diagnostic run per runtime",
    )
    save(fig, f"whistle_module_latency_{slug}")


FIGURES = {
    "whistle_wer_rtf_scatter": fig_quality_speed,
    "whistle_gpu_rtf_comparison": fig_gpu_rtf,
    "whistle_module_latency_victoria": lambda: fig_module("victoria", *module_latency()["victoria"]),
    "whistle_module_latency_modal": lambda: fig_module("modal", *module_latency()["modal"]),
}


def main() -> None:
    """Render every figure, or only the --only stem."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(FIGURES), help="render a single figure")
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FontProperties(fname=str(FONT_FILE["regular"])).get_name()],
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "svg.fonttype": "path",
        }
    )
    for stem in ([args.only] if args.only else FIGURES):
        FIGURES[stem]()


if __name__ == "__main__":
    main()
