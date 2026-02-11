#!/usr/bin/env python3
"""
District-level leave-one-out cross-validation for tree species classification.

Trains on 5 districts, tests on the held-out district, repeats for all 6 districts.
Species 21 (Silver fir) and 72 (Grey alder) are excluded because they only
occur in Gorlice — no training data exists when Gorlice is held out.

Usage:
    python run_district_cv.py
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

from run_experiments import (
    SEED,
    MIN_SAMPLES,
    SPECIES_NAMES_PATH,
    load_data,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
PLOT_LOCATIONS_PATH = DATA_DIR / "TreeScanPL_plot_locations.csv"
RESULTS_DIR = DATA_DIR / "district_cv_results"

EXCLUDE_SPECIES = [21, 72]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("District-level Leave-One-Out Cross-Validation")
    print("=" * 70)

    # --- Load data ---
    merged, feature_groups = load_data()

    # --- Load district mapping ---
    plot_locs = pd.read_csv(PLOT_LOCATIONS_PATH, sep=";")
    district_map = plot_locs.set_index("num")["file"].to_dict()
    merged["district"] = merged["plot_id"].map(district_map)

    unmapped = merged["district"].isna().sum()
    if unmapped > 0:
        print(f"  Warning: {unmapped} trees could not be mapped to a district (dropped)")
        merged = merged.dropna(subset=["district"]).copy()

    districts = sorted(merged["district"].unique())
    print(f"\nDistricts ({len(districts)}): {', '.join(districts)}")

    # --- Exclude species 21 and 72 ---
    before = len(merged)
    merged = merged[~merged["treeSP"].isin(EXCLUDE_SPECIES)].copy()
    print(f"Excluded species {EXCLUDE_SPECIES}: {before} -> {len(merged)} trees")

    # --- Re-apply MIN_SAMPLES filtering ---
    species_counts = merged["treeSP"].value_counts()
    keep_species = species_counts[species_counts >= MIN_SAMPLES].index
    merged = merged[merged["treeSP"].isin(keep_species)].copy()
    print(f"After MIN_SAMPLES={MIN_SAMPLES} filtering: {len(merged)} trees, "
          f"{merged['treeSP'].nunique()} species")

    # --- Print district × species distribution ---
    print("\nTrees per district:")
    for d in districts:
        n = (merged["district"] == d).sum()
        n_sp = merged.loc[merged["district"] == d, "treeSP"].nunique()
        print(f"  {d:>12s}: {n:>5d} trees, {n_sp:>2d} species")

    # --- Load species names ---
    species_names = {}
    if SPECIES_NAMES_PATH.exists():
        sn = pd.read_csv(SPECIES_NAMES_PATH)
        for _, row in sn.iterrows():
            species_names[int(row["CODE"])] = {
                "latin_name": row["LATIN_NAME"],
                "common_name": row["COMMON_NAME"],
            }

    # --- Define 8 combinations ---
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

    print(f"\nRunning {len(combinations)} combinations × {len(districts)} folds "
          f"= {len(combinations) * len(districts)} experiments...\n")

    # --- CV loop ---
    summary_rows = []
    per_species_rows = []

    for combo_name, feature_cols in combinations:
        print(f"--- {combo_name} ({len(feature_cols)} features) ---")

        for district in districts:
            # Split by district
            test_mask = merged["district"] == district
            train_full = merged[~test_mask].copy()
            test_full = merged[test_mask].copy()

            # Drop NaN in selected features
            train_full = train_full.dropna(subset=feature_cols)
            test_full = test_full.dropna(subset=feature_cols)

            if len(test_full) == 0:
                print(f"  {district}: no test samples after NaN drop, skipping")
                continue

            # Identify species present in both sets with >= MIN_SAMPLES in train
            train_species_counts = train_full["treeSP"].value_counts()
            train_valid_species = set(
                train_species_counts[train_species_counts >= MIN_SAMPLES].index
            )
            test_species = set(test_full["treeSP"].unique())
            shared_species = sorted(train_valid_species & test_species)

            if len(shared_species) < 2:
                print(f"  {district}: only {len(shared_species)} shared species, skipping")
                continue

            # Filter both sets to shared species
            train_sub = train_full[train_full["treeSP"].isin(shared_species)]
            test_sub = test_full[test_full["treeSP"].isin(shared_species)]

            X_train = train_sub[feature_cols].values
            y_train = train_sub["treeSP"].values
            X_test = test_sub[feature_cols].values
            y_test = test_sub["treeSP"].values

            # Train RF with same hyperparams as run_experiments.py
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

            # Overall metrics
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            summary_rows.append({
                "combination": combo_name,
                "district": district,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_species": len(shared_species),
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "cohen_kappa": kappa,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
            })

            # Per-species metrics
            labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
            report_dict = classification_report(
                y_test, y_pred, labels=labels, digits=3,
                zero_division=0, output_dict=True,
            )
            for key, metrics in report_dict.items():
                if not isinstance(metrics, dict) or "precision" not in metrics:
                    continue
                if key in ("accuracy", "macro avg", "weighted avg"):
                    continue
                sp_id = int(key)
                names = species_names.get(sp_id, {})
                per_species_rows.append({
                    "combination": combo_name,
                    "district": district,
                    "species": sp_id,
                    "latin_name": names.get("latin_name", ""),
                    "common_name": names.get("common_name", ""),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1-score"],
                    "support": int(metrics["support"]),
                })

            print(f"  {district:>12s}: n_train={len(y_train):>5d}, n_test={len(y_test):>4d}, "
                  f"species={len(shared_species):>2d}, bal_acc={bal_acc:.4f}, "
                  f"kappa={kappa:.4f}")

        print()

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "district_cv_summary.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'district_cv_summary.csv'} ({len(summary_df)} rows)")

    # Per-species CSV
    per_species_df = pd.DataFrame(per_species_rows)
    per_species_df.to_csv(RESULTS_DIR / "district_cv_per_species.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'district_cv_per_species.csv'} ({len(per_species_df)} rows)")

    # Aggregated CSV: mean ± std per combination
    metrics_cols = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_macro", "f1_weighted"]
    agg_rows = []
    for combo_name, _ in combinations:
        combo_data = summary_df[summary_df["combination"] == combo_name]
        if len(combo_data) == 0:
            continue
        row = {"combination": combo_name}
        for m in metrics_cols:
            row[f"{m}_mean"] = combo_data[m].mean()
            row[f"{m}_std"] = combo_data[m].std()
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(RESULTS_DIR / "district_cv_aggregated.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'district_cv_aggregated.csv'} ({len(agg_df)} rows)")

    # Print aggregated summary
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS (mean ± std across districts)")
    print("=" * 70)
    for _, row in agg_df.iterrows():
        print(f"\n  {row['combination']}:")
        for m in metrics_cols:
            print(f"    {m:>20s}: {row[f'{m}_mean']:.4f} ± {row[f'{m}_std']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
