"""Parse midtraining and SFT logs and save loss curves matching d12_loss_curve.png style."""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BG = "#0b0f1a"
BLUE = "#5b9bd5"
ORANGE = "#f5a623"

TRAIN_RE = re.compile(r"step (\d+)/\d+ \([\d.]+%\) \| loss: ([\d.]+)")
VAL_RE = re.compile(r"step (\d+) \| val bpb: ([\d.]+)")


def parse_log(path: Path):
    train_steps, train_loss = [], []
    val_steps, val_bpb = [], []
    for line in path.read_text(errors="replace").splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = TRAIN_RE.search(line)
        if m:
            train_steps.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            continue
        m = VAL_RE.search(line)
        if m:
            val_steps.append(int(m.group(1)))
            val_bpb.append(float(m.group(2)))
    return (
        np.array(train_steps), np.array(train_loss),
        np.array(val_steps),   np.array(val_bpb),
    )


def smooth(values, window=50):
    if len(values) < window:
        window = max(1, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_phase(title: str, train_steps, train_loss, val_steps, val_bpb, out: Path):
    fig, ax1 = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor(BG)
    ax1.set_facecolor(BG)

    # Raw train loss
    ax1.plot(train_steps, train_loss, color=BLUE, alpha=0.25, linewidth=0.8,
             label="Train loss (raw)")

    # Smoothed train loss
    s = smooth(train_loss, 50)
    offset = len(train_loss) - len(s)
    ax1.plot(train_steps[offset:], s, color=BLUE, linewidth=1.8,
             label="Train loss (smooth-50)")

    ax1.set_xlabel("Step", color="white", fontsize=11)
    ax1.set_ylabel("Cross-entropy loss", color=BLUE, fontsize=11)
    ax1.tick_params(axis="x", colors="white")
    ax1.tick_params(axis="y", colors=BLUE)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333")

    # Val bpb on right axis
    ax2 = ax1.twinx()
    ax2.set_facecolor(BG)
    ax2.plot(val_steps, val_bpb, color=ORANGE, linewidth=1.8,
             marker="o", markersize=4, label="Val bpb")
    ax2.set_ylabel("Validation bpb", color=ORANGE, fontsize=11)
    ax2.tick_params(axis="y", colors=ORANGE)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333")

    # Final bpb annotation
    if len(val_steps) and len(val_bpb):
        ax2.annotate(
            f"final bpb {val_bpb[-1]:.4f}",
            xy=(val_steps[-1], val_bpb[-1]),
            xytext=(-90, 12),
            textcoords="offset points",
            color=ORANGE,
            fontsize=10,
        )

    # Legend (merge both axes)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", facecolor="#1a1f2e", edgecolor="#444",
               labelcolor="white", fontsize=9)

    ax1.set_title(title, color="white", fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    log_dir = Path(__file__).parent / "logs"
    logs = {
        "d12 midtraining — cross-entropy loss & validation bpb": (
            log_dir / "d12_midtrain.log",
            log_dir / "d12_midtrain_loss_curve.png",
        ),
        "d12 SFT — cross-entropy loss & validation bpb": (
            log_dir / "d12_sft.log",
            log_dir / "d12_sft_loss_curve.png",
        ),
    }

    for title, (log_path, out_path) in logs.items():
        if not log_path.exists():
            print(f"Missing: {log_path}", file=sys.stderr)
            sys.exit(1)
        ts, tl, vs, vb = parse_log(log_path)
        plot_phase(title, ts, tl, vs, vb, out_path)


if __name__ == "__main__":
    main()
