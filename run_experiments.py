#!/usr/bin/env python3
"""
Systematic evaluation of feature combinations for tree species classification.

Runs all 8 combinations of contextual data sources (BDL, AlphaEarth, RPP)
on top of the geometry-only baseline, using a Random Forest classifier.

Usage:
    python run_experiments.py
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
TREE_FEATURES_PATH = DATA_DIR / "tree_features" / "all_tree_features.csv"
BDL_PATH = DATA_DIR / "plots_bdl_fused.csv"
ALPHAEARTH_PATH = DATA_DIR / "plots_alphaearth_2018.csv"
RPP_PATH = DATA_DIR / "plots_rpp_probabilities.csv"
RESULTS_DIR = DATA_DIR / "experiment_results"

SEED = 42
MIN_SAMPLES = 10

# Columns that are identifiers / labels, not geometric features
NON_FEATURE_COLS = {"plot_id", "tree_id", "treeSP", "completelyInside"}

# BDL categorical columns to one-hot encode
BDL_CATEGORICALS = [
    "site_type",
    "silvicult",
    "forest_fun",
    "stand_stru",
    "moisture_cd",
    "degradation_cd",
    "species_cd",
]

# BDL numeric columns (plot-level constants)
BDL_NUMERICS = ["rotat_age", "sub_area", "part_cd", "spec_age", "damage_degree"]


# ---------------------------------------------------------------------------
# BDL flattening
# ---------------------------------------------------------------------------
def flatten_bdl(bdl_raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten long-form BDL data to one row per plot."""
    # De-duplicate plot-level constants: take the first row per plot
    plot_const = bdl_raw.groupby("num").first().reset_index()

    # --- Categorical one-hot encoding ---
    cat_frames = []
    for col in BDL_CATEGORICALS:
        dummies = pd.get_dummies(plot_const[col], prefix=f"bdl_{col}", dtype=float)
        cat_frames.append(dummies)
    cats = pd.concat(cat_frames, axis=1)

    # --- Numeric features ---
    nums = plot_const[["num"] + BDL_NUMERICS].copy()
    for col in BDL_NUMERICS:
        nums[col] = pd.to_numeric(nums[col], errors="coerce")

    # --- Aggregated features from storey/species rows ---
    # Number of storeys per plot
    n_storeys = (
        bdl_raw.groupby("num")["storey_cd"]
        .nunique()
        .reset_index()
        .rename(columns={"storey_cd": "n_storeys"})
    )

    # Number of unique species per plot (from species rows in the main storey)
    n_species = (
        bdl_raw.groupby("num")["sp_species_cd"]
        .nunique()
        .reset_index()
        .rename(columns={"sp_species_cd": "n_species"})
    )

    # Main storey density: storey_density_index where storey_rank == 1
    main_storey = bdl_raw[bdl_raw["storey_rank"] == 1]
    main_density = (
        main_storey.groupby("num")["storey_density_index"]
        .first()
        .reset_index()
        .rename(columns={"storey_density_index": "main_storey_density"})
    )

    # Conifer ratio: proportion of species rows in the main storey that are conifers
    # sp_wood_kind == 'I' means conifer (Iglaste), 'L' means deciduous (Liściaste)
    main_with_wood = main_storey[main_storey["sp_wood_kind"].isin(["I", "L"])].copy()
    if len(main_with_wood) > 0:
        main_with_wood["is_conifer"] = (main_with_wood["sp_wood_kind"] == "I").astype(float)
        conifer_ratio = (
            main_with_wood.groupby("num")["is_conifer"]
            .mean()
            .reset_index()
            .rename(columns={"is_conifer": "conifer_ratio"})
        )
    else:
        conifer_ratio = pd.DataFrame(columns=["num", "conifer_ratio"])

    # --- Merge everything ---
    result = nums.copy()
    result = pd.concat([result, cats], axis=1)
    for agg_df in [n_storeys, n_species, main_density, conifer_ratio]:
        result = result.merge(agg_df, on="num", how="left")

    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load and prepare all data sources, returning a merged DataFrame and feature groups."""
    print("Loading data...")

    # --- Tree features (geometry) ---
    trees = pd.read_csv(TREE_FEATURES_PATH, sep=";")
    print(f"  Tree features: {len(trees)} rows")

    # Filter
    trees = trees[
        (trees["tree_id"] > 0)
        & (trees["completelyInside"] == 1)
        & (trees["treeSP"] > 0)
    ].copy()
    print(f"  After filtering: {len(trees)} trees")

    # Drop rare species
    species_counts = trees["treeSP"].value_counts()
    keep_species = species_counts[species_counts >= MIN_SAMPLES].index
    dropped = species_counts[species_counts < MIN_SAMPLES]
    trees = trees[trees["treeSP"].isin(keep_species)].copy()
    print(f"  After dropping species with < {MIN_SAMPLES} samples: {len(trees)} trees")
    if len(dropped) > 0:
        print(f"  Dropped species: {dict(dropped)}")

    geometry_cols = [c for c in trees.columns if c not in NON_FEATURE_COLS]
    print(f"  Geometry features: {len(geometry_cols)} columns")

    # --- BDL ---
    bdl_raw = pd.read_csv(BDL_PATH, sep=";")
    print(f"  BDL raw: {len(bdl_raw)} rows")
    bdl = flatten_bdl(bdl_raw)
    bdl_cols = [c for c in bdl.columns if c != "num"]
    print(f"  BDL flattened: {len(bdl)} plots, {len(bdl_cols)} features")

    # --- AlphaEarth ---
    ae = pd.read_csv(ALPHAEARTH_PATH, sep=";")
    ae_feature_cols = [c for c in ae.columns if c.startswith("A") and c[1:].isdigit()]
    ae = ae[["num"] + ae_feature_cols]
    print(f"  AlphaEarth: {len(ae)} plots, {len(ae_feature_cols)} features")

    # --- RPP ---
    rpp = pd.read_csv(RPP_PATH, sep=";")
    rpp_species_cols = [
        c for c in rpp.columns
        if c not in ("source", "file", "year", "num", "num_txt", "X", "Y")
    ]
    rpp = rpp[["num"] + rpp_species_cols]
    print(f"  RPP: {len(rpp)} plots, {len(rpp_species_cols)} features")

    # --- Merge all onto tree features ---
    merged = trees.copy()
    merged = merged.merge(bdl, left_on="plot_id", right_on="num", how="left").drop(
        columns=["num"], errors="ignore"
    )
    merged = merged.merge(ae, left_on="plot_id", right_on="num", how="left").drop(
        columns=["num"], errors="ignore"
    )
    merged = merged.merge(rpp, left_on="plot_id", right_on="num", how="left").drop(
        columns=["num"], errors="ignore"
    )
    print(f"\n  Merged dataset: {len(merged)} trees, {len(merged.columns)} columns")

    # Check coverage
    for name, cols in [("BDL", bdl_cols), ("AlphaEarth", ae_feature_cols), ("RPP", rpp_species_cols)]:
        non_null = merged[cols[0]].notna().sum()
        print(f"  {name} coverage: {non_null}/{len(merged)} trees ({100*non_null/len(merged):.1f}%)")

    feature_groups = {
        "geometry": geometry_cols,
        "bdl": bdl_cols,
        "alphaearth": ae_feature_cols,
        "rpp": rpp_species_cols,
    }

    return merged, feature_groups


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    merged: pd.DataFrame,
    feature_cols: list[str],
    combo_name: str,
) -> dict:
    """Train and evaluate a single experiment combination."""
    # Drop rows with NaN in selected features
    subset = merged.dropna(subset=feature_cols)
    X = subset[feature_cols].values
    y = subset["treeSP"].values

    # Re-filter rare species after subsetting (some may drop below threshold)
    species, counts = np.unique(y, return_counts=True)
    keep = species[counts >= MIN_SAMPLES]
    mask = np.isin(y, keep)
    X, y = X[mask], y[mask]

    if len(np.unique(y)) < 2:
        print(f"  {combo_name}: not enough classes after filtering, skipping")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    report = classification_report(y_test, y_pred, labels=labels, digits=3, zero_division=0)

    return {
        "combination": combo_name,
        "n_trees": len(y),
        "n_features": len(feature_cols),
        "n_classes": len(np.unique(y)),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "cohen_kappa": kappa,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report,
    }


def main():
    print("=" * 70)
    print("Feature Combination Experiment for Species Classification")
    print("=" * 70)

    merged, feature_groups = load_data()

    # Print species distribution
    print(f"\nSpecies distribution (full geometry set):")
    for sp, count in merged["treeSP"].value_counts().sort_index().items():
        print(f"  {sp:>4d}: {count:>5d} trees")
    print(f"  Total: {len(merged)} trees, {merged['treeSP'].nunique()} species")

    # Define the 8 combinations (geometry is always included)
    context_sources = ["bdl", "alphaearth", "rpp"]
    combinations = []
    for r in range(len(context_sources) + 1):
        for combo in itertools.combinations(context_sources, r):
            sources = ["geometry"] + list(combo)
            name = " + ".join(sources)
            cols = []
            for s in sources:
                cols.extend(feature_groups[s])
            combinations.append((name, cols))

    print(f"\nRunning {len(combinations)} experiments...\n")

    results = []
    for name, cols in combinations:
        print(f"--- {name} ({len(cols)} features) ---")
        result = run_experiment(merged, cols, name)
        if result is not None:
            results.append(result)
            print(
                f"  n={result['n_trees']}, classes={result['n_classes']}, "
                f"acc={result['accuracy']:.4f}, bal_acc={result['balanced_accuracy']:.4f}, "
                f"kappa={result['cohen_kappa']:.4f}, F1m={result['f1_macro']:.4f}, "
                f"F1w={result['f1_weighted']:.4f}"
            )
        print()

    # --- Summary table ---
    summary = pd.DataFrame(results).drop(columns=["report"])
    summary = summary.sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)

    print("=" * 70)
    print("COMPARISON TABLE (sorted by balanced accuracy)")
    print("=" * 70)
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    print(f"\nSaved comparison to {RESULTS_DIR / 'comparison.csv'}")

    for result in results:
        safe_name = result["combination"].replace(" + ", "_").replace(" ", "_")
        report_path = RESULTS_DIR / f"report_{safe_name}.txt"
        with open(report_path, "w") as f:
            f.write(f"Combination: {result['combination']}\n")
            f.write(f"Trees: {result['n_trees']}, Features: {result['n_features']}, Classes: {result['n_classes']}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"Balanced accuracy: {result['balanced_accuracy']:.4f}\n")
            f.write(f"Cohen's kappa: {result['cohen_kappa']:.4f}\n")
            f.write(f"F1 macro: {result['f1_macro']:.4f}\n")
            f.write(f"F1 weighted: {result['f1_weighted']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(result["report"])
        print(f"  Saved {report_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
