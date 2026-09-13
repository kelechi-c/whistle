"""Regenerate the four publication figures for the Whistle article.

Run from the repository root (matplotlib is not a runtime dependency, and
--no-project keeps uv from syncing the qwen-tts/torch environment):

    uv run --no-project --with matplotlib --with click \
        python tools/make_figures.py [--only STEM] [--variant both] [--font-dir DIR]

Each figure is emitted in two geometries: the wide desktop panel layout, and a
single-column mobile variant (suffix -mobile) drawn about as wide as a phone's
content column, so the same type sizes read at roughly 1:1 instead of being
scaled down to illegibility. Outputs PNG (220 dpi) and SVG into
article/. The SVG keeps glyph outlines so it renders identically
where SF Pro is not installed.

Data is read from the saved evidence JSON files, never transcribed from images:

    quality/speed  evidence/{official_current_p50_5runs,v7_current_p50_5runs,
                   victoria_faster_0.6b_3runs}.json  -> p50 RTF
                   evidence/victoria_fresh_{official,whistle}_wer.json,
                   evidence/wer_alicia_faster.json -> WER
    overall RTF    same three p50 files, plus the 1.7B files and the RTX PRO
                   6000 rows of latency_report.md (Modal; no raw JSON)
    module latency evidence/victoria_modular_*.json and
                   evidence/modular_*.json

Font: SF Pro, the Sciel body font. It is not redistributable here, so it must be
installed locally; point --font-dir or WHISTLE_FONT_DIR at it if the default
search misses. Missing weights fall back to the nearest available one, and no
silent substitution happens: if no SF Pro file is found the script stops with
the directories it searched.
"""

import json
import os
import statistics
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
# figures render into the local article bundle; the records they read are tracked
ARTICLE = ROOT / "article"
OUT = ARTICLE
EVIDENCE = ROOT / "evidence"

# SF Pro is not redistributable here, so the figures need it installed locally.
# Resolution order: --font-dir, WHISTLE_FONT_DIR, a vendored tools/fonts, then the
# author's local copies. Nothing is silently substituted: if none of these yields
# an SF Pro file the script stops and lists what it searched.
FONT_DIR_ENV = "WHISTLE_FONT_DIR"
FALLBACK_FONT_DIRS = (
    ROOT / "tools" / "fonts",
    Path.home() / ".local" / "share" / "fonts",
    Path("/usr/share/fonts"),
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

DESKTOP, MOBILE = "desktop", "mobile"
VARIANTS = (DESKTOP, MOBILE)


def typography(variant: str) -> dict[str, float]:
    """Point sizes per variant.

    The mobile figures are drawn at roughly the width of a phone's content
    column (~354 CSS px), so these sizes land close to 1:1 on screen; the
    desktop figures are shown at a fraction of their width and need larger type.
    """
    wide = variant == DESKTOP
    return {
        "title": 16.5 if wide else 13.0,
        "subtitle": 11.0 if wide else 9.5,
        "axis": 12.0 if wide else 10.0,
        "tick": 10.5 if wide else 8.5,
        "value": 10.5 if wide else 8.5,
        "label": 11.5 if wide else 10.0,
        "note": 10.0 if wide else 8.5,
        "foot": 9.5 if wide else 8.0,
        "title_gap": 0.062 if wide else 0.050,
        "linespacing": 1.2 if wide else 1.35,
    }


def font_dirs(extra: Path | None = None) -> list[Path]:
    """Directories to search for SF Pro, most specific first."""
    dirs: list[Path] = []
    if extra is not None:
        dirs.append(Path(extra))
    from_env = os.environ.get(FONT_DIR_ENV)
    if from_env:
        dirs.extend(Path(part) for part in from_env.split(os.pathsep) if part)
    dirs.extend(FALLBACK_FONT_DIRS)
    return dirs


def load_fonts(dirs: list[Path] | None = None) -> dict[str, Path]:
    """Resolve SF Pro files per weight, falling back to whichever weight exists.

    Searches recursively so a normal font install (for example
    ``~/.local/share/fonts/SF-Pro/``) works without extra flags.
    """
    searched = dirs if dirs is not None else font_dirs()
    found: list[Path] = [
        path
        for directory in searched
        if directory.is_dir()
        for path in sorted(directory.rglob("*"))
        if path.suffix.lower() in {".otf", ".ttf"} and "sf" in path.name.lower()
        and "italic" not in path.name.lower()
    ]
    if not found:
        raise SystemExit(
            "no SF Pro font found. Install it, pass --font-dir, or set "
            f"{FONT_DIR_ENV}. Searched: {', '.join(map(str, searched))}"
        )

    picks: dict[str, Path] = {}
    for weight, fragments in FONT_WEIGHTS.items():
        for fragment in fragments:
            match = next((p for p in found if fragment in p.name.lower()), None)
            if match is not None:
                picks[weight] = match
                break
    spare = next(iter(picks.values()))
    return {weight: picks.get(weight, spare) for weight in FONT_WEIGHTS}


FONT_FILE: dict[str, Path] = {}


def configure_fonts(extra: Path | None = None) -> None:
    """Resolve and register the SF Pro files used by every figure."""
    global FONT_FILE
    FONT_FILE = load_fonts(font_dirs(extra))
    for path in set(FONT_FILE.values()):
        fontManager.addfont(str(path))


def font(weight: str, size: float) -> FontProperties:
    """FontProperties for one weight/size pair; weights never share a family name.

    Fonts are resolved in main() so --font-dir and WHISTLE_FONT_DIR take effect.
    """
    if not FONT_FILE:
        raise SystemExit("fonts are not configured yet; call configure_fonts() first")
    return FontProperties(fname=str(FONT_FILE[weight]), size=size)


def style_axes(ax: plt.Axes, *, ygrid: bool = False, left_spine: bool = True, tick: float = 10.5) -> None:
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
        label.set_fontproperties(font("regular", tick))
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


def value_text(
    ax: plt.Axes, x: float, y: float, text: str, *, ha: str = "left", pad: float = 13.0, size: float = 10.5
) -> None:
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
        fontproperties=font("regular", size),
        color=INK_2,
        zorder=4,
    )


def save(fig: plt.Figure, stem: str) -> None:
    """Write one figure as PNG + SVG under the exact stem the caller chose."""
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
        # Modal rows: latency_report.md, 3-5 runs, no raw JSON retained.
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
                    "whistle": module_ms(EVIDENCE / f"modular_whistle_{size}.json"),
                    "faster": module_ms(EVIDENCE / f"modular_faster_{size}.json"),
                }
                for size in ("0.6B", "1.7B")
            },
        ),
    }


def figure_title(fig: plt.Figure, x: float, title: str, subtitle: str, *, top: float, t: dict[str, float]) -> None:
    """Left-aligned title plus a muted subtitle line.

    Mobile titles and subtitles carry explicit newlines: the strings are too long
    for a single phone-width line at a legible size.
    """
    fig.text(x, top, title, ha="left", va="top", fontproperties=font("medium", t["title"]), color=INK)
    fig.text(
        x,
        top - t["title_gap"],
        subtitle,
        ha="left",
        va="top",
        fontproperties=font("regular", t["subtitle"]),
        color=INK_2,
        linespacing=t["linespacing"],
    )


def footprint(fig: plt.Figure, x: float, y: float, text: str, *, size: float = 9.5, linespacing: float = 1.2) -> None:
    """Muted provenance note pinned to the figure's bottom-left corner."""
    fig.text(
        x, y, text, ha="left", va="bottom", fontproperties=font("regular", size), color=MUTED,
        linespacing=linespacing,
    )


def panel_heading(ax: plt.Axes, text: str) -> None:
    """Heading strip above a standalone mobile panel, left-aligned with the plot."""
    ax.annotate(
        text,
        (0, 1.0),
        xycoords="axes fraction",
        textcoords="offset points",
        xytext=(0, 30),
        ha="left",
        va="bottom",
        fontproperties=font("medium", 9.5),
        color=INK,
    )


def fig_quality_speed(variant: str = DESKTOP) -> None:
    """Scatter: transcription quality against speed; two runtimes share a WER row.

    Mobile is the same single panel drawn about a phone-column wide, with the
    official label anchored to the left of its dot so it cannot run off the edge.
    """
    points = [
        ("official", p50_rtf(EVIDENCE / "official_current_p50_5runs.json"), wer_percent(EVIDENCE / "victoria_fresh_official_wer.json")),
        ("whistle", p50_rtf(EVIDENCE / "v7_current_p50_5runs.json"), wer_percent(EVIDENCE / "victoria_fresh_whistle_wer.json")),
        ("faster", p50_rtf(EVIDENCE / "victoria_faster_0.6b_3runs.json"), wer_percent(EVIDENCE / "wer_alicia_faster.json")),
    ]
    wer = points[0][2]
    t = typography(variant)

    if variant == DESKTOP:
        fig, ax = plt.subplots(figsize=(10.6, 6.2))
        fig.subplots_adjust(left=0.125, right=0.965, top=0.845, bottom=0.16)
        title_x, foot_x, foot_y = ax.get_position().x0, ax.get_position().x0, 0.035
        subtitle = "Whistle matches the official runtime's transcript at a lower real-time factor"
        foot = "Alicia · 0.6B · RTF is the p50 of repeated runs · WER from Qwen3-ASR-0.6B transcription of the generated audio"
        offsets = {"official": (0, -18, "center"), "whistle": (0, 16, "center"), "faster": (0, 16, "center")}
        note_pos = ((points[0][1] + points[1][1]) / 2, "center")
    else:
        fig, ax = plt.subplots(figsize=(4.9, 4.9))
        fig.subplots_adjust(left=0.195, right=0.975, top=0.80, bottom=0.215)
        title_x, foot_x, foot_y = 0.04, 0.04, 0.025
        subtitle = "Whistle matches the official runtime's transcript\nat a lower real-time factor"
        foot = "Alicia · 0.6B · RTF is the p50 of repeated runs\nWER from Qwen3-ASR-0.6B transcription of the audio"
        # official sits above-left of its dot and the parity note drops to the left
        # edge, otherwise the two labels land on the same line and collide.
        offsets = {"official": (-7, 14, "right"), "whistle": (0, 15, "center"), "faster": (0, 15, "center")}
        note_pos = (0.51, "left")

    figure_title(
        fig,
        title_x,
        "Transcription quality versus generation speed",
        subtitle,
        top=0.975,
        t=t,
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
        note_pos[0],
        wer - (0.13 if variant == DESKTOP else 0.10),
        f"identical transcript ({wer:.2f}% WER)",
        ha=note_pos[1],
        va="top",
        fontproperties=font("regular", t["note"]),
        color=MUTED,
    )
    for key, x, y in points:
        markers(ax, x, y, key, size=11)
    for key, x, y in points:
        dx, dy, ha = offsets[key]
        ax.annotate(
            NAME[key], (x, y), textcoords="offset points", xytext=(dx, dy),
            ha=ha, va="top" if dy < 0 else "bottom",
            fontproperties=font("medium", t["label"]), color=INK,
        )

    ax.set_xlim(0.50, 0.93)
    ax.set_ylim(2.70, 5.15)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticks([3.0, 3.5, 4.0, 4.5, 5.0])
    ax.set_xlabel("Generation RTF ↓", fontproperties=font("regular", t["axis"]), color=INK, labelpad=9)
    ax.set_ylabel("WER (%) ↓", fontproperties=font("regular", t["axis"]), color=INK, labelpad=9)
    style_axes(ax, ygrid=True, tick=t["tick"])
    footprint(fig, foot_x, foot_y, foot, size=t["foot"], linespacing=t["linespacing"])
    save(fig, "whistle_wer_rtf_scatter" if variant == DESKTOP else "whistle_wer_rtf_scatter-mobile")


GPU_SLUG = {"RTX 3050 Laptop · 6 GB": "rtx3050", "RTX PRO 6000 Blackwell · 96 GB": "rtxpro6000"}


def size_slug(size: str) -> str:
    """0.6B -> 0p6b, so a model size can live in a filename."""
    return size.replace(".", "p").lower()


def draw_rtf_rows(ax: plt.Axes, values: dict[str, float], *, t: dict[str, float], mobile: bool, xmax: float) -> None:
    """The three runtime rows of one RTF panel, shared by both variants."""
    for position, key in enumerate(RUNTIMES):
        y = len(RUNTIMES) - 1 - position
        markers(ax, values[key], y, key, size=9.0 if not mobile else 8.0)
        value_text(ax, values[key], y, f"{values[key]:.2f}", pad=13.0 if not mobile else 10.0, size=t["value"])
    ax.set_yticks(range(len(RUNTIMES)))
    ax.set_yticklabels([NAME[key] for key in reversed(RUNTIMES)])
    ax.set_ylim(-0.75, len(RUNTIMES) - 0.25)
    ax.set_xlim(0, xmax)


def fig_gpu_rtf(variant: str = DESKTOP) -> None:
    """Two GPUs by two model sizes, one shared RTF scale.

    Desktop keeps the hardware name horizontal as a band heading over two panels
    and shares the model size as a column heading. Mobile emits one image per
    GPU/size panel instead of one tall composite, so each panel is a single
    readable figure on a phone.
    """
    data = overall_rtf()
    gpus = list(data)
    sizes = ("0.6B", "1.7B")
    panels = [(gpu, size) for gpu in gpus for size in sizes]
    t = typography(variant)
    xmax = 1.30
    title = "End-to-end generation speed across GPUs and model sizes"
    subtitle = "Alicia · bf16 · greedy decoding · one RTF scale shared by all four panels"
    foot = "p50 of 3–5 measured runs · ↓ lower is better · official RTF is the historical Alicia-parity measurement"

    if variant == MOBILE:
        for gpu, size in panels:
            fig, ax = plt.subplots(figsize=(4.9, 2.45))
            fig.subplots_adjust(left=0.30, right=0.975, top=0.74, bottom=0.30)
            draw_rtf_rows(ax, data[gpu][size], t=t, mobile=True, xmax=xmax)
            ax.set_xticks([0.0, 0.5, 1.0], ["0.00", "0.50", "1.00"])
            style_axes(ax, left_spine=False, tick=t["tick"])
            ax.set_xlabel("Generation RTF ↓", fontproperties=font("regular", t["axis"]), color=INK, labelpad=9)
            panel_heading(ax, f"{gpu} · {size}")
            save(fig, f"whistle_gpu_rtf_comparison-mobile-{GPU_SLUG[gpu]}-{size_slug(size)}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex=True)
    layout = {"left": 0.165, "right": 0.975, "top": 0.79, "bottom": 0.185, "hspace": 0.5, "wspace": 0.18}
    fig.subplots_adjust(**layout)
    xticks = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]

    for index, (gpu, size) in enumerate(panels):
        ax = axes[index // 2, index % 2]
        draw_rtf_rows(ax, data[gpu][size], t=t, mobile=False, xmax=xmax)
        ax.set_xticks(xticks, [f"{tick:.2f}" for tick in xticks])
        style_axes(ax, left_spine=False, tick=t["tick"])
        if index < 2:
            ax.tick_params(labelbottom=False)
        if index % 2 == 1:
            ax.tick_params(labelleft=False)

    for row, gpu in enumerate(gpus):
        fig.text(
            layout["left"],
            axes[row, 0].get_position().y1 + 0.022,
            gpu,
            ha="left",
            va="bottom",
            fontproperties=font("medium", 12),
            color=INK,
        )
    for col, size in enumerate(sizes):
        fig.text(
            axes[0, col].get_position().x0 + axes[0, col].get_position().width / 2,
            axes[0, 0].get_position().y1 + 0.078,
            size,
            ha="center",
            va="center",
            fontproperties=font("medium", 12.5),
            color=INK,
        )

    axes[1, 0].set_xlabel("Generation RTF ↓", fontproperties=font("regular", t["axis"]), color=INK, labelpad=9)
    figure_title(fig, layout["left"], title, subtitle, top=0.975, t=t)
    footprint(fig, layout["left"], 0.038, foot, size=t["foot"], linespacing=t["linespacing"])
    save(fig, "whistle_gpu_rtf_comparison")


def module_markers(
    ax: plt.Axes, predictor: float, talker: float, y: float, key: str, *, ring: float = 12.0, dot: float = 8.5
) -> None:
    """Draw the talker ring first and the predictor dot on top.

    The talker marker is larger so that where the two values coincide (measured
    to within 0.01 ms on the PRO 6000) the predictor dot still shows inside the
    ring instead of being hidden behind it.
    """
    ax.plot(
        [talker],
        [y],
        marker="o",
        markersize=ring,
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
        markersize=dot,
        markerfacecolor=FILL[key],
        markeredgecolor=EDGE[key],
        markeredgewidth=1.0,
        linestyle="none",
        zorder=3,
    )


def draw_module_panel(
    ax: plt.Axes, models: dict[str, tuple[float, float]], *, t: dict[str, float], mobile: bool, xmax: float
) -> None:
    """One model size: a predictor-to-talker dumbbell per runtime."""
    ax.set_xlim(0, xmax)
    for position, key in enumerate(("whistle", "faster")):
        y = 1 - position
        predictor, talker = models[key]
        ax.plot([0, xmax], [y, y], color=RAIL, linewidth=0.9, zorder=0)
        ax.plot([predictor, talker], [y, y], color="#8f8b84", linewidth=1.2, zorder=1)
        module_markers(
            ax, predictor, talker, y, key,
            ring=12.0 if not mobile else 10.0,
            dot=8.5 if not mobile else 7.1,
        )
        # The two labels flank the dumbbell so a near-equal pair never stacks.
        value_text(ax, predictor, y, f"{predictor:.2f}", ha="left" if predictor >= talker else "right",
                   pad=13.0 if not mobile else 10.0, size=t["value"])
        value_text(ax, talker, y, f"{talker:.2f}", ha="left" if talker > predictor else "right",
                   pad=13.0 if not mobile else 10.0, size=t["value"])
    ax.set_yticks([1, 0])
    ax.set_yticklabels([NAME["whistle"], NAME["faster"]])
    ax.set_ylim(-0.45, 1.45)
    ax.set_xlabel("Latency per call (ms)", fontproperties=font("regular", t["axis"]), color=INK, labelpad=9)


def module_legend(target, *, size: float, marker: float, loc: str = "upper right", anchor: tuple[float, float] | None = None) -> None:
    """A real legend, so the key is drawn with the same glyphs as the data.

    Pass an Axes to keep the key inside a standalone panel, or a Figure with an
    anchor to place it over a multi-panel figure.
    """
    target.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markersize=marker,
                   markerfacecolor=INK, markeredgecolor=INK, label="predictor"),
            Line2D([], [], marker="o", linestyle="none", markersize=marker * 1.41,
                   markerfacecolor=BACKGROUND, markeredgecolor=INK_2, markeredgewidth=1.6, label="talker"),
        ],
        loc=loc,
        **({"bbox_to_anchor": anchor} if anchor is not None else {}),
        frameon=False,
        ncols=2,
        handletextpad=0.5,
        columnspacing=1.8,
        prop=font("regular", size),
        labelcolor=INK_2,
        borderaxespad=0.0,
    )


def fig_module(slug: str, gpu: str, models: dict[str, dict[str, tuple[float, float]]], variant: str = DESKTOP) -> None:
    """Dumbbell panels: predictor-to-talker spread per runtime, per model size.

    Mobile emits one image per model size, each self-contained with its own
    legend, instead of one two-panel composite.
    """
    t = typography(variant)
    subtitle = "CUDA-event time per call · 128-frame diagnostic after a 32-frame warmup"
    foot = "Official module timing unavailable — absent, not zero · single diagnostic run per runtime"
    xmax = max(max(pair) for model in models.values() for pair in model.values()) * 1.28

    if variant == MOBILE:
        for size in ("0.6B", "1.7B"):
            fig, ax = plt.subplots(figsize=(4.9, 2.75))
            fig.subplots_adjust(left=0.30, right=0.975, top=0.76, bottom=0.40)
            draw_module_panel(ax, models[size], t=t, mobile=True, xmax=xmax)
            style_axes(ax, left_spine=False, tick=t["tick"])
            panel_heading(ax, f"{gpu} · {size}")
            # Each panel stands alone, so it carries its own key. It sits below the
            # axis: inside the plot the key's own markers read as data points.
            module_legend(ax, size=t["value"], marker=7.5, loc="upper center", anchor=(0.5, -0.55))
            save(fig, f"whistle_module_latency_{slug}-mobile-{size_slug(size)}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.7))
    layout = {"left": 0.135, "right": 0.975, "top": 0.745, "bottom": 0.27, "wspace": 0.14}
    fig.subplots_adjust(**layout)
    for index, size in enumerate(("0.6B", "1.7B")):
        ax = axes[index]
        draw_module_panel(ax, models[size], t=t, mobile=False, xmax=xmax)
        style_axes(ax, left_spine=False, tick=t["tick"])
        ax.set_title(size, fontproperties=font("medium", 12.5), color=INK, pad=10)
        if index == 1:
            ax.tick_params(labelleft=False)
    axes[1].set_xlabel("Latency per call (ms)", fontproperties=font("regular", t["axis"]), color=INK, labelpad=9)

    module_legend(fig, loc="upper right", anchor=(layout["right"], 0.985), size=t["value"], marker=8.5)
    figure_title(fig, layout["left"], f"Module latency on {gpu}", subtitle, top=0.975, t=t)
    footprint(fig, layout["left"], 0.05, foot, size=t["foot"], linespacing=t["linespacing"])
    save(fig, f"whistle_module_latency_{slug}")


FIGURES = {
    "whistle_wer_rtf_scatter": fig_quality_speed,
    "whistle_gpu_rtf_comparison": fig_gpu_rtf,
    "whistle_module_latency_victoria": lambda variant: fig_module("victoria", *module_latency()["victoria"], variant),
    "whistle_module_latency_modal": lambda variant: fig_module("modal", *module_latency()["modal"], variant),
}


@click.command()
@click.option("--only", type=click.Choice(sorted(FIGURES)), default=None, help="Render a single figure.")
@click.option(
    "--variant",
    type=click.Choice((*VARIANTS, "both")),
    default="both",
    show_default=True,
    help="Desktop panels, the phone-width mobile variant, or both.",
)
@click.option(
    "--font-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Extra directory to search for SF Pro (also WHISTLE_FONT_DIR).",
)
def main(only: str | None, variant: str, font_dir: Path | None) -> None:
    """Render every Whistle article figure into article/ as PNG and SVG."""
    configure_fonts(font_dir)
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
    variants = VARIANTS if variant == "both" else (variant,)
    for name in variants:
        for stem in ([only] if only else FIGURES):
            FIGURES[stem](name)


if __name__ == "__main__":
    main()
