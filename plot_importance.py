#!/usr/bin/env python3
"""
Visualize feature importance analysis from experiment results.

Reads CSVs produced by run_experiments.py and generates 4 PDF figures:
  1. Stacked bar — source importance by combination
  2. Grouped bar — performance metrics comparison
  3. Top-15 features grid — per-combination horizontal bars
  4. Heatmap — feature importance across combinations

Usage:
    python plot_importance.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "data" / "experiment_results"

SOURCE_COLORS = {
    "geometry": "#4C72B0",
    "bdl": "#DD8452",
    "alphaearth": "#55A868",
    "rpp": "#C44E52",
}

SOURCE_ORDER = ["geometry", "bdl", "alphaearth", "rpp"]

# Short labels for x-axis readability
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

# Combination display order: baseline first, then by number of sources
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
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load comparison CSV and all per-combination importance CSVs."""
    comparison = pd.read_csv(RESULTS_DIR / "comparison.csv")

    imp_dfs = {}
    for _, row in comparison.iterrows():
        combo = row["combination"]
        safe_name = combo.replace(" + ", "_").replace(" ", "_")
        path = RESULTS_DIR / f"importance_{safe_name}.csv"
        if path.exists():
            imp_dfs[combo] = pd.read_csv(path)

    return comparison, imp_dfs


# ---------------------------------------------------------------------------
# Plot 1: Stacked bar — source importance by combination
# ---------------------------------------------------------------------------
def plot_source_importance(comparison):
    """Stacked bar chart showing source importance proportions."""
    # Order combinations
    ordered = [c for c in COMBO_ORDER if c in comparison["combination"].values]
    df = comparison.set_index("combination").loc[ordered]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(ordered))
    bottom = np.zeros(len(ordered))

    for src in SOURCE_ORDER:
        col = f"importance_{src}"
        vals = df[col].fillna(0).values
        ax.bar(x, vals, bottom=bottom, label=src, color=SOURCE_COLORS[src], width=0.7)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[c] for c in ordered], rotation=30, ha="right")
    ax.set_ylabel("Cumulative MDI importance")
    ax.set_title("Feature importance by data source")
    ax.legend(title="Source", loc="upper right")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    out = RESULTS_DIR / "importance_by_source.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Plot 2: Grouped bar — performance comparison
# ---------------------------------------------------------------------------
def plot_performance(comparison):
    """Grouped bar chart comparing balanced accuracy and F1 macro."""
    # Order by balanced accuracy descending
    df = comparison.set_index("combination")
    ordered = [c for c in COMBO_ORDER if c in df.index]
    df = df.loc[ordered]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(ordered))
    width = 0.3

    ax.bar(x - width / 2, df["balanced_accuracy"], width, label="Balanced accuracy",
           color="#4C72B0", alpha=0.85)
    ax.bar(x + width / 2, df["f1_macro"], width, label="F1 macro",
           color="#55A868", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[c] for c in ordered], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Classification performance by feature combination")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=df["balanced_accuracy"].max(), color="grey", linestyle="--",
               linewidth=0.7, alpha=0.5)

    fig.tight_layout()
    out = RESULTS_DIR / "performance_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Plot 3: Top-15 features grid
# ---------------------------------------------------------------------------
def plot_top_features_grid(imp_dfs):
    """2x4 grid of horizontal bar charts, top 15 features per combination."""
    ordered = [c for c in COMBO_ORDER if c in imp_dfs]

    fig, axes = plt.subplots(2, 4, figsize=(22, 14))
    axes_flat = axes.flatten()

    for idx, combo in enumerate(ordered):
        ax = axes_flat[idx]
        df = imp_dfs[combo].head(15).iloc[::-1]  # reverse for bottom-to-top

        colors = [SOURCE_COLORS.get(s, "grey") for s in df["source"]]
        ax.barh(range(len(df)), df["importance"], color=colors, height=0.7)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["feature"], fontsize=8)
        ax.set_title(SHORT_LABELS.get(combo, combo), fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance", fontsize=9)

    # Hide unused subplots
    for idx in range(len(ordered), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=SOURCE_COLORS[s], label=s) for s in SOURCE_ORDER]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, fontsize=11,
               title="Source", title_fontsize=12, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Top 15 features per experiment combination", fontsize=14,
                 fontweight="bold", y=1.04)
    fig.tight_layout()
    out = RESULTS_DIR / "top_features_grid.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Plot 4: Heatmap — feature importance across combinations
# ---------------------------------------------------------------------------
def plot_importance_heatmap(imp_dfs):
    """Heatmap of top features across all combinations."""
    ordered = [c for c in COMBO_ORDER if c in imp_dfs]

    # Collect top 20 features from each combination
    top_features = []
    feature_source = {}
    for combo in ordered:
        df = imp_dfs[combo].head(20)
        for _, row in df.iterrows():
            feat = row["feature"]
            if feat not in feature_source:
                feature_source[feat] = row["source"]
            top_features.append(feat)

    # Deduplicate preserving order
    seen = set()
    unique_features = []
    for f in top_features:
        if f not in seen:
            seen.add(f)
            unique_features.append(f)

    # Build matrix
    matrix = pd.DataFrame(0.0, index=unique_features,
                          columns=[SHORT_LABELS[c] for c in ordered])
    for combo in ordered:
        df = imp_dfs[combo].set_index("feature")
        short = SHORT_LABELS[combo]
        for feat in unique_features:
            if feat in df.index:
                matrix.loc[feat, short] = df.loc[feat, "importance"]

    # Sort by max importance across combinations
    matrix["_max"] = matrix.max(axis=1)
    matrix = matrix.sort_values("_max", ascending=True).drop(columns=["_max"])

    fig, ax = plt.subplots(figsize=(12, max(8, len(matrix) * 0.3)))

    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.5,
                ax=ax, cbar_kws={"label": "MDI importance"})

    # Color y-tick labels by source
    for label in ax.get_yticklabels():
        feat = label.get_text()
        src = feature_source.get(feat, "unknown")
        label.set_color(SOURCE_COLORS.get(src, "black"))
        label.set_fontweight("bold")

    ax.set_title("Feature importance across experiment combinations", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_ylabel("")

    # Source color legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=SOURCE_COLORS[s], label=s) for s in SOURCE_ORDER]
    ax.legend(handles=legend_handles, loc="lower right", title="Source",
              fontsize=9, title_fontsize=10)

    fig.tight_layout()
    out = RESULTS_DIR / "importance_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    comparison, imp_dfs = load_data()
    print(f"  {len(comparison)} combinations, {len(imp_dfs)} importance files\n")

    print("Generating plots...")
    plot_source_importance(comparison)
    plot_performance(comparison)
    plot_top_features_grid(imp_dfs)
    plot_importance_heatmap(imp_dfs)

    print("\nDone.")


if __name__ == "__main__":
    main()
