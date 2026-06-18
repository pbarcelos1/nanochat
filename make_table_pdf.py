import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# ── Data ────────────────────────────────────────────────────────────────────
rows = [
    # text_type, bytes, gpt2_tok, gpt2_ratio, ours_tok, ours_ratio, rel_diff, better
    ("news",      1813,    398,  4.56,  592,  3.06, -48.7, "GPT-2"),
    ("korean",     877,    735,  1.19,  844,  1.04, -14.8, "GPT-2"),
    ("code",      1259,    576,  2.19,  672,  1.87, -16.7, "GPT-2"),
    ("math",      1830,    933,  1.96, 1107,  1.65, -18.6, "GPT-2"),
    ("science",   1110,    258,  4.30,  372,  2.98, -44.2, "GPT-2"),
    ("pt-news",    795,    282,  2.82,  156,  5.10, +44.7, "Ours"),
    ("fwe-train", 3209616, 1149595, 2.79, 701754, 4.57, +39.0, "Ours"),
    ("fwe-val",   2661219,  955342, 2.79, 571106, 4.66, +40.2, "Ours"),
]

rows4 = [
    ("news",      1813,    387,  4.68,  592,  3.06, -53.0, "GPT-4"),
    ("korean",     877,    364,  2.41,  844,  1.04,-131.9, "GPT-4"),
    ("code",      1259,    309,  4.07,  672,  1.87,-117.5, "GPT-4"),
    ("math",      1830,    831,  2.20, 1107,  1.65, -33.2, "GPT-4"),
    ("science",   1110,    249,  4.46,  372,  2.98, -49.4, "GPT-4"),
    ("pt-news",    795,    215,  3.70,  156,  5.10, +27.4, "Ours"),
    ("fwe-train", 3209616,  883058, 3.63, 701754, 4.57, +20.5, "Ours"),
    ("fwe-val",   2661219,  729099, 3.65, 571106, 4.66, +21.7, "Ours"),
]

col_headers = ["Text Type", "Bytes", "Tokens", "Ratio", "Tokens", "Ratio", "Rel. Diff %", "Winner"]

GREEN = "#2d8a4e"
RED   = "#c0392b"
LIGHT_GREEN = "#d4edda"
LIGHT_RED   = "#fde8e8"
HEADER_BG   = "#2c3e50"
HEADER_FG   = "white"
ALT_ROW     = "#f5f5f5"

def fmt(v):
    if isinstance(v, float):
        return f"{v:+.1f}%" if abs(v) < 200 else f"{v:.1f}%"
    if isinstance(v, int) and v > 9999:
        return f"{v:,}"
    return str(v)

def draw_table(ax, rows, compare_label):
    ax.axis("off")
    n = len(rows)
    ncols = len(col_headers)

    # Column widths (relative)
    col_w = [0.13, 0.11, 0.10, 0.08, 0.10, 0.08, 0.13, 0.10]
    # Normalize
    total = sum(col_w)
    col_w = [w / total for w in col_w]

    row_h = 0.072
    header_h = 0.10
    top = 0.97

    # ── Subheader groups ──────────────────────────────────────────────────
    group_y = top
    groups = [
        ("", 0, 2),
        (compare_label, 2, 4),
        ("Ours", 4, 6),
        ("", 6, 8),
    ]
    for label, c_start, c_end in groups:
        if not label:
            continue
        x = sum(col_w[:c_start])
        w = sum(col_w[c_start:c_end])
        rect = mpatches.FancyBboxPatch(
            (x, group_y - 0.045), w, 0.045,
            boxstyle="square,pad=0",
            linewidth=0, facecolor="#34495e", transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, group_y - 0.022, label,
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="white", transform=ax.transAxes)

    # ── Column headers ────────────────────────────────────────────────────
    header_y = group_y - 0.045
    for ci, (hdr, w) in enumerate(zip(col_headers, col_w)):
        x = sum(col_w[:ci])
        rect = mpatches.FancyBboxPatch(
            (x, header_y - header_h), w, header_h,
            boxstyle="square,pad=0",
            linewidth=0.5, edgecolor="white", facecolor=HEADER_BG,
            transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, header_y - header_h / 2, hdr,
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color=HEADER_FG, transform=ax.transAxes)

    # ── Data rows ─────────────────────────────────────────────────────────
    data_top = header_y - header_h
    for ri, row in enumerate(rows):
        y = data_top - ri * row_h
        winner = row[-1]
        bg = LIGHT_GREEN if winner == "Ours" else LIGHT_RED
        if ri % 2 == 1 and winner not in ("Ours",):
            bg = "#f8d7da"  # slightly darker red for alt rows
        if ri % 2 == 1 and winner == "Ours":
            bg = "#c3e6cb"

        for ci, (val, w) in enumerate(zip(row, col_w)):
            x = sum(col_w[:ci])
            cell_bg = bg
            # rel diff cell: color by sign
            if ci == 6:
                cell_bg = LIGHT_GREEN if val > 0 else LIGHT_RED
                txt_color = GREEN if val > 0 else RED
            elif ci == 7:
                txt_color = GREEN if val == "Ours" else RED
            else:
                txt_color = "black"

            rect = mpatches.FancyBboxPatch(
                (x, y - row_h), w, row_h,
                boxstyle="square,pad=0",
                linewidth=0.3, edgecolor="#cccccc", facecolor=cell_bg,
                transform=ax.transAxes, clip_on=False
            )
            ax.add_patch(rect)

            display = fmt(val) if ci == 6 else fmt(val)
            fw = "bold" if ci in (6, 7) else "normal"
            ax.text(x + w / 2, y - row_h / 2, display,
                    ha="center", va="center", fontsize=8,
                    color=txt_color, fontweight=fw, transform=ax.transAxes)

    # bottom border
    bottom_y = data_top - n * row_h
    ax.plot([0, 1], [bottom_y, bottom_y], color="#aaaaaa", linewidth=0.5, transform=ax.transAxes)

# ── PDF ──────────────────────────────────────────────────────────────────────
with PdfPages("table.pdf") as pdf:
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.97, "Tokenizer Compression Comparison",
             ha="center", va="top", fontsize=14, fontweight="bold", color="#1a1a2e")
    fig.text(0.5, 0.935, "Bytes-per-token ratio: higher is better (fewer tokens per byte)",
             ha="center", va="top", fontsize=9, color="#555555", style="italic")

    draw_table(axes[0], rows,  "GPT-2")
    draw_table(axes[1], rows4, "GPT-4")

    axes[0].set_title("vs. GPT-2  (vocab 50 257)", fontsize=10, fontweight="bold",
                       color="#2c3e50", pad=12, loc="left")
    axes[1].set_title("vs. GPT-4  (vocab 100 277)", fontsize=10, fontweight="bold",
                       color="#2c3e50", pad=12, loc="left")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

print("table.pdf written.")
