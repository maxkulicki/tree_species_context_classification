#!/usr/bin/env python3
"""
Visualize district-level leave-one-out cross-validation results.

Reads CSVs from data/district_cv_results/ and generates 3 PDF figures:
  1. Heatmap — balanced accuracy by district × combination
  2. Heatmap — per-species F1 across districts (best combination only)
  3. Bar chart — district CV vs random split comparison

Usage:
    python plot_district_cv.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants (reuse patterns from plot_importance.py)
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = DATA_DIR / "district_cv_results"
RANDOM_RESULTS_DIR = DATA_DIR / "experiment_results"

SHORT_LABELS = {
    "geometry": "geo",
    "geometry + bdl": "geo+bdl",
    "geometry + alphaearth": "geo+ae",
    "geometry + rpp": "geo+rpp",
    "geometry + bdl + alphaearth": "geo+bdl+ae",
    "geometry + bdl + rpp": "geo+bdl+rpp",
    "geometry + alphaearth + rpp": "geo+ae+rpp",
    "geometry + bdl + alphaearth + rpp": "all",
}

COMBO_ORDER = [
    "geometry",
    "geometry + bdl",
    "geometry + alphaearth",
    "geometry + rpp",
    "geometry + bdl + alphaearth",
    "geometry + bdl + rpp",
    "geometry + alphaearth + rpp",
    "geometry + bdl + alphaearth + rpp",
]


# ---------------------------------------------------------------------------
# Plot 1: District performance heatmap
# ---------------------------------------------------------------------------
def plot_district_performance_heatmap(summary):
    """Heatmap of balanced accuracy: rows = districts, columns = combinations."""
    ordered_combos = [c for c in COMBO_ORDER if c in summary["combination"].values]

    pivot = summary.pivot_table(
        index="district", columns="combination", values="balanced_accuracy",
    )
    pivot = pivot[ordered_combos]
    pivot.columns = [SHORT_LABELS[c] for c in ordered_combos]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGn",
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Balanced accuracy"},
    )
    ax.set_title("Balanced accuracy by district and feature combination",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("District")
    ax.set_xlabel("")

    fig.tight_layout()
    out = RESULTS_DIR / "district_performance_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Plot 2: Per-species F1 across districts (best combination = all features)
# ---------------------------------------------------------------------------
def plot_species_f1_heatmap(per_species):
    """Heatmap of F1-score per species × district for the all-features combination."""
    best_combo = "geometry + bdl + alphaearth + rpp"
    df = per_species[per_species["combination"] == best_combo].copy()

    if len(df) == 0:
        print("  Skipping species F1 heatmap: no data for all-features combination")
        return

    # Build species labels sorted by species ID
    df["species"] = df["species"].astype(int)
    species_ids = sorted(df["species"].unique())
    species_labels = {}
    for sp_id in species_ids:
        row = df[df["species"] == sp_id].iloc[0]
        common = row.get("common_name", "")
        latin = row.get("latin_name", "")
        if pd.notna(latin) and latin:
            species_labels[sp_id] = f"{common} ({latin})"
        else:
            species_labels[sp_id] = str(sp_id)

    # F1 pivot
    f1_pivot = df.pivot_table(index="species", columns="district", values="f1")
    f1_pivot = f1_pivot.loc[species_ids]

    # Support pivot for annotations
    support_pivot = df.pivot_table(index="species", columns="district", values="support")
    support_pivot = support_pivot.loc[species_ids]

    # Build annotation: "F1\n(n=support)"
    annot = f1_pivot.copy().astype(object)
    for col in f1_pivot.columns:
        for sp in f1_pivot.index:
            f1_val = f1_pivot.loc[sp, col]
            sup_val = support_pivot.loc[sp, col]
            if pd.isna(f1_val):
                annot.loc[sp, col] = ""
            else:
                annot.loc[sp, col] = f"{f1_val:.2f}\n(n={int(sup_val)})"

    # Relabel index
    f1_pivot.index = [species_labels[sp] for sp in f1_pivot.index]
    annot.index = f1_pivot.index

    fig, ax = plt.subplots(figsize=(12, max(6, len(f1_pivot) * 0.5)))
    sns.heatmap(
        f1_pivot,
        annot=annot,
        fmt="",
        cmap="YlGn",
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "F1-score"},
    )
    ax.set_title(f"Per-species F1-score across districts ({SHORT_LABELS[best_combo]})",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("")
    ax.set_xlabel("")

    fig.tight_layout()
    out = RESULTS_DIR / "district_species_f1_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Plot 3: District CV vs random split comparison
# ---------------------------------------------------------------------------
def plot_cv_vs_random(aggregated):
    """Grouped bar chart comparing random-split and district CV balanced accuracy."""
    random_path = RANDOM_RESULTS_DIR / "comparison.csv"
    if not random_path.exists():
        print(f"  Skipping CV vs random plot: {random_path} not found")
        return

    random_df = pd.read_csv(random_path)

    ordered_combos = [c for c in COMBO_ORDER
                      if c in aggregated["combination"].values
                      and c in random_df["combination"].values]

    random_vals = random_df.set_index("combination").loc[ordered_combos, "balanced_accuracy"]
    cv_means = aggregated.set_index("combination").loc[ordered_combos, "balanced_accuracy_mean"]
    cv_stds = aggregated.set_index("combination").loc[ordered_combos, "balanced_accuracy_std"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ordered_combos))
    width = 0.35

    ax.bar(x - width / 2, random_vals.values, width, label="Random split (80/20)",
           color="#4C72B0", alpha=0.85)
    ax.bar(x + width / 2, cv_means.values, width, label="District CV (mean)",
           color="#DD8452", alpha=0.85,
           yerr=cv_stds.values, capsize=4, error_kw={"linewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[c] for c in ordered_combos], rotation=30, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Random split vs. district-level cross-validation",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    out = RESULTS_DIR / "district_cv_vs_random.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading district CV results...")
    summary = pd.read_csv(RESULTS_DIR / "district_cv_summary.csv")
    per_species = pd.read_csv(RESULTS_DIR / "district_cv_per_species.csv")
    aggregated = pd.read_csv(RESULTS_DIR / "district_cv_aggregated.csv")
    print(f"  Summary: {len(summary)} rows")
    print(f"  Per-species: {len(per_species)} rows")
    print(f"  Aggregated: {len(aggregated)} rows\n")

    print("Generating plots...")
    plot_district_performance_heatmap(summary)
    plot_species_f1_heatmap(per_species)
    plot_cv_vs_random(aggregated)

    print("\nDone.")


if __name__ == "__main__":
    main()
