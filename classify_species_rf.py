#!/usr/bin/env python3
"""
Random Forest baseline for tree species classification from geometric features.

Filters to labeled trees (treeSP > 0) that are completely inside the plot,
trains an 80/20 random forest classifier, and reports standard metrics.

Usage:
    python classify_species_rf.py
    python classify_species_rf.py --input data/tree_features/all_tree_features.csv
    python classify_species_rf.py --min-samples 20
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_INPUT = DATA_DIR / "tree_features" / "all_tree_features.csv"

# Columns that are not features
NON_FEATURE_COLS = {"plot_id", "tree_id", "treeSP", "completelyInside"}


def main():
    parser = argparse.ArgumentParser(
        description="Random Forest species classification from tree geometric features."
    )
    parser.add_argument(
        "--input", "-i", default=str(DEFAULT_INPUT),
        help="Input CSV with tree features (default: data/tree_features/all_tree_features.csv)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=10,
        help="Minimum number of samples per species to include (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Random Forest Species Classification")
    print("=" * 60)

    # Load and filter
    df = pd.read_csv(args.input, sep=";")
    print(f"\nLoaded {len(df)} trees from {args.input}")

    df = df[(df["tree_id"] > 0) & (df["completelyInside"] == 1) & (df["treeSP"] > 0)]
    print(f"After filtering (treeID>0, completelyInside=1, treeSP>0): {len(df)}")

    # Drop rare species
    species_counts = df["treeSP"].value_counts()
    keep_species = species_counts[species_counts >= args.min_samples].index
    dropped = species_counts[species_counts < args.min_samples]
    df = df[df["treeSP"].isin(keep_species)]
    print(f"After dropping species with < {args.min_samples} samples: {len(df)}")
    if len(dropped) > 0:
        print(f"  Dropped species (code: count): {dict(dropped)}")

    print(f"\nSpecies distribution:")
    for sp, count in df["treeSP"].value_counts().sort_index().items():
        print(f"  {sp:>4d}: {count:>5d} trees")
    print(f"  Total: {len(df)} trees, {df['treeSP'].nunique()} species")

    # Prepare features and labels
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values
    y = df["treeSP"].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y,
    )
    print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

    # Train
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced accuracy: {bal_acc:.4f}")
    print(f"  Cohen's kappa:     {kappa:.4f}")
    print(f"  F1 (macro):        {f1_macro:.4f}")
    print(f"  F1 (weighted):     {f1_weighted:.4f}")

    # Per-class report
    labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    print(f"\nPer-class classification report:")
    print(classification_report(y_test, y_pred, labels=labels, digits=3, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("Confusion matrix:")
    print(cm_df.to_string())

    # Feature importance (top 15)
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]
    print(f"\nTop 15 features by importance:")
    for rank, i in enumerate(idx[:15], 1):
        print(f"  {rank:2d}. {feature_cols[i]:30s} {importances[i]:.4f}")


if __name__ == "__main__":
    main()
