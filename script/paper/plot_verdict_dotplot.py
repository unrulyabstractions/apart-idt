"""Verdict-at-a-glance figure for the main paper.

    uv run python script/paper/plot_verdict_dotplot.py

One row per run: the observed statistic S as a dot, the permutation null's 95th
percentile as a tick, and the named favoured candidate on rejecting rows. Every
value is read from out/r2/compare/*/comparison_summary.json, never hand-typed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.common.file_io import load_json

# Paper palette (paperdefs.tex): candpink, baseink, and neutral inks.
PINK = "#C2447E"
GREY = "#6B7280"
INK = "#111827"

RUNS = [
    ("calibration_blind", "12-mar-gen9-1.5b, blind"),
    ("calibration_typed", "12-mar-gen9-1.5b, typed"),
    ("calibration_informed", "12-mar-gen9-1.5b, informed"),
    ("challenge_organism_a", "organism a"),
    ("challenge_organism_b", "organism b"),
    ("challenge_organism_c", "organism c"),
]


def main() -> None:
    root = Path("out/r2/compare")
    rows = []
    for run, label in RUNS:
        summary = load_json(root / run / "comparison_summary.json")
        rc = summary["reference_contrast"]
        level = sorted(k for k in rc if k.startswith("L"))[-1]
        pm = rc[level]["paired_max_test"]
        display = {k.lower().replace(" ", "_"): v for k, v in
                   load_json(Path("out/r2/score") / run / "prompt_sets.json")
                   .get("principals", {}).items()}
        principal = pm["principal"]
        name = display.get(principal, principal) if principal else None
        top = pm["top_pairs"][0]
        favoured = top["mean_excess"] > 0
        rows.append({"label": label, "s": pm["statistic"], "null95": pm["null_p95"],
                     "p": pm["p_family_wise"], "loyal": pm["loyal"],
                     "name": name, "favoured": favoured})

    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    ys = range(len(rows) - 1, -1, -1)
    for y, r in zip(ys, rows):
        ax.plot([min(r["s"], r["null95"]), max(r["s"], r["null95"])], [y, y],
                color="#D1D5DB", lw=1, zorder=1)
        ax.plot(r["null95"], y, marker="|", color=INK, ms=11, mew=1.6, zorder=2)
        if r["loyal"]:
            ax.plot(r["s"], y, "o", color=PINK, ms=7, zorder=3)
            tag = r["name"] if r["favoured"] else f"{r['name']} (disfavoured)"
            ax.annotate(tag, (r["s"], y), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
        else:
            ax.plot(r["s"], y, "o", mfc="white", mec=GREY, ms=7, mew=1.4, zorder=3)
            ax.annotate("no rejection", (max(r["s"], r["null95"]), y),
                        xytext=(7, 0), textcoords="offset points", va="center",
                        fontsize=8, color=GREY)

    ax.set_yticks(list(ys))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8.5)
    ax.set_xlabel("largest standardized excess $S$", fontsize=9)
    ax.set_xlim(3.0, 8.6)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E5E7EB", lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax.plot([], [], "o", color=PINK, ms=7, label="observed $S$, rejects")
    ax.plot([], [], "o", mfc="white", mec=GREY, ms=7, mew=1.4,
            label="observed $S$, does not reject")
    ax.plot([], [], marker="|", color=INK, ms=11, mew=1.6, ls="none",
            label="null 95th percentile")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3,
              fontsize=7.5, frameon=False, borderaxespad=0)

    out = Path("paper/figures/verdict_dotplot.pdf")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
