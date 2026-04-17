"""
Parse Stage 1 & Stage 2 training logs and generate publication-quality loss curves.

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --output figures/training_curves.pdf
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Log Parsing ────────────────────────────────────────────────────────────────

TQDM_RE = re.compile(
    r"(Stage1|Specialization) Epoch (\d+).*\|\s*(\d+)/(\d+).*loss=([\d.]+)"
)

CKPT_RE = re.compile(r"^Epoch (\d+) saved to (.+\.pt)$")


def parse_log(path: Path) -> list[dict]:
    """Parse a tqdm-formatted training log into step records.

    Returns list of {stage, epoch, step, total_steps, loss}.
    Deduplicates the double-line-per-step pattern by keeping the last
    loss value for each (epoch, step) pair.
    """
    raw: dict[tuple[int, int], dict] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TQDM_RE.search(line)
            if m:
                stage, epoch, step, total, loss = m.groups()
                key = (int(epoch), int(step))
                raw[key] = {
                    "stage": stage,
                    "epoch": int(epoch),
                    "step": int(step),
                    "total_steps": int(total),
                    "loss": float(loss),
                }
    # Sort by (epoch, step)
    return [raw[k] for k in sorted(raw.keys())]


def smooth(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Exponential moving average for smoothed loss curves."""
    alpha = 2.0 / (window + 1)
    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


# ── Plotting ───────────────────────────────────────────────────────────────────


def plot_combined(
    s1_records: list[dict],
    s2_records: list[dict],
    output: Path,
    smooth_window: int = 100,
):
    """Generate a 1×2 figure with Stage 1 and Stage 2 loss curves."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    for ax, records, title, color_raw, color_smooth in [
        (axes[0], s1_records, "Stage 1: Alignment", "#c5c5c5", "#2563eb"),
        (axes[1], s2_records, "Stage 2: Specialization", "#c5c5c5", "#dc2626"),
    ]:
        if not records:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        epochs = sorted(set(r["epoch"] for r in records))
        total_per_epoch = records[0]["total_steps"]

        # Build global step axis
        global_steps = []
        losses = []
        for r in records:
            gs = (r["epoch"] - 1) * total_per_epoch + r["step"]
            global_steps.append(gs)
            losses.append(r["loss"])

        gs = np.array(global_steps)
        ls = np.array(losses)

        # Raw scatter (faint)
        ax.plot(gs, ls, linewidth=0.3, alpha=0.25, color=color_raw, rasterized=True)

        # Smoothed curve
        ls_smooth = smooth(ls, window=smooth_window)
        ax.plot(gs, ls_smooth, linewidth=1.8, color=color_smooth, label="EMA-smoothed loss")

        # Epoch boundaries
        for ep in epochs[1:]:
            boundary = (ep - 1) * total_per_epoch
            ax.axvline(boundary, color="#6b7280", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.text(boundary, ax.get_ylim()[1] * 0.95, f" Epoch {ep}", fontsize=7, color="#6b7280")

        ax.set_xlabel("Training Step", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

        # Annotate start/end loss
        ax.annotate(
            f"{ls[0]:.2f}",
            xy=(gs[0], ls_smooth[0]),
            fontsize=7,
            color=color_smooth,
            ha="left",
        )
        ax.annotate(
            f"{ls_smooth[-1]:.3f}",
            xy=(gs[-1], ls_smooth[-1]),
            fontsize=7,
            color=color_smooth,
            ha="right",
        )

    fig.suptitle("AgriPerceiver Training Loss", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output}")


def plot_per_epoch_summary(
    s1_records: list[dict],
    s2_records: list[dict],
    output: Path,
):
    """Generate per-epoch mean loss bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)

    all_bars = []
    colors = []
    labels = []

    for records, stage_name, color in [
        (s1_records, "S1", "#2563eb"),
        (s2_records, "S2", "#dc2626"),
    ]:
        epochs = sorted(set(r["epoch"] for r in records))
        for ep in epochs:
            ep_losses = [r["loss"] for r in records if r["epoch"] == ep]
            mean_loss = np.mean(ep_losses)
            all_bars.append(mean_loss)
            colors.append(color)
            labels.append(f"{stage_name} E{ep}")

    x = np.arange(len(all_bars))
    bars = ax.bar(x, all_bars, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, all_bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Epoch Loss", fontsize=10)
    ax.set_title("Per-Epoch Mean Loss", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.15)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Plot AgriPerceiver training loss curves")
    parser.add_argument("--s1-log", type=str, default="docs/logs/stage1_alignment.log")
    parser.add_argument("--s2-log", type=str, default="docs/logs/stage2_specialization.log")
    parser.add_argument("--output", type=str, default="figures/training_curves.pdf")
    parser.add_argument("--smooth", type=int, default=100)
    args = parser.parse_args()

    s1 = parse_log(Path(args.s1_log))
    s2 = parse_log(Path(args.s2_log))

    print(f"Stage 1: {len(s1)} steps across {len(set(r['epoch'] for r in s1))} epochs")
    print(f"Stage 2: {len(s2)} steps across {len(set(r['epoch'] for r in s2))} epochs")

    out = Path(args.output)
    plot_combined(s1, s2, out, smooth_window=args.smooth)

    # Also generate per-epoch summary
    summary_out = out.parent / "epoch_summary.pdf"
    plot_per_epoch_summary(s1, s2, summary_out)


if __name__ == "__main__":
    main()
