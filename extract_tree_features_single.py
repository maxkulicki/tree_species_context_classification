#!/usr/bin/env python3
"""
Extract per-tree geometric features from TLS plot LAZ files.

Reads LAZ file(s), groups points by treeID, runs extract_tree_features()
for each tree (treeID > 0), and saves the results with speciesID and
completelyInside flag to a semicolon-delimited CSV.

Usage:
    # Single file:
    python extract_tree_features_single.py data/TreeScanPL_downsapled_2cm_corrected/Rem_Gorlice_2015_0101703.laz

    # Entire directory (all .laz files, combined output):
    python extract_tree_features_single.py data/TreeScanPL_downsapled_2cm_corrected/

    # With custom output:
    python extract_tree_features_single.py data/TreeScanPL_downsapled_2cm_corrected/ -o data/all_tree_features.csv
"""

import argparse
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

from extract_features import extract_tree_features


def process_single_file(laz_path: Path) -> pd.DataFrame:
    """Extract per-tree features from a single LAZ file. Returns a DataFrame."""
    las = laspy.read(str(laz_path))

    points = np.vstack([las.x, las.y, las.z]).T
    tree_ids = np.array(las["treeID"])
    tree_sp = np.array(las["treeSP"])
    completely_inside = np.array(las["completelyInside"])

    unique_trees = sorted(set(tree_ids[tree_ids > 0]))

    rows = []
    for tid in unique_trees:
        mask = tree_ids == tid
        tree_points = points[mask]

        sp = int(np.bincount(tree_sp[mask].astype(int)).argmax())
        inside = int(np.bincount(completely_inside[mask].astype(int)).argmax())

        features = extract_tree_features(tree_points, tree_id=tid)
        if features is None:
            continue

        features["treeSP"] = sp
        features["completelyInside"] = inside
        rows.append(features)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-tree features from TLS plot LAZ file(s)."
    )
    parser.add_argument(
        "input",
        help="Input LAZ/LAS file or directory containing .laz files",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path (default: {input_stem}_tree_features.csv "
             "for single file, or data/all_tree_features.csv for directory)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    # Determine file list
    if input_path.is_dir():
        laz_files = sorted(input_path.glob("*.laz"))
        if not laz_files:
            print(f"No .laz files found in {input_path}")
            return
        output_path = Path(args.output) if args.output else (
            Path("data") / "tree_features" / "all_tree_features.csv"
        )
    else:
        laz_files = [input_path]
        output_path = Path(args.output) if args.output else (
            Path("data") / "tree_features" / (input_path.stem + "_tree_features.csv")
        )

    print(f"Processing {len(laz_files)} file(s)...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Column order for the output
    front = ["plot_id", "tree_id", "treeSP", "completelyInside", "n_points"]

    # Load existing results if resuming a previous run
    if output_path.exists():
        existing = pd.read_csv(output_path, sep=";")
        done_plots = set(existing["plot_id"].astype(str).unique())
        all_frames = [existing]
        print(f"  Resuming: {len(done_plots)} plots already processed")
    else:
        done_plots = set()
        all_frames = []

    for i, laz_path in enumerate(laz_files, 1):
        plot_id = laz_path.stem.split("_")[-1]

        if plot_id in done_plots:
            print(f"  [{i}/{len(laz_files)}] {laz_path.name} ... already done, skipping")
            continue

        print(f"  [{i}/{len(laz_files)}] {laz_path.name} ...", end=" ", flush=True)
        df = process_single_file(laz_path)

        if len(df) == 0:
            print("no trees found, skipping")
            continue

        df.insert(0, "plot_id", plot_id)
        n_labeled = (df["treeSP"] > 0).sum()
        print(f"{len(df)} trees ({n_labeled} labeled)")
        all_frames.append(df)

        # Save intermediate result after each file
        result = pd.concat(all_frames, ignore_index=True)
        rest = [c for c in result.columns if c not in front]
        result[front + rest].to_csv(output_path, index=False, sep=";")

    if not all_frames:
        print("No trees extracted from any file.")
        return

    result = pd.concat(all_frames, ignore_index=True)
    rest = [c for c in result.columns if c not in front]
    result = result[front + rest]

    n_labeled = (result["treeSP"] > 0).sum()
    print(f"\nOutput saved to {output_path}")
    print(f"  Plots: {result['plot_id'].nunique()}")
    print(f"  Trees: {len(result)} ({n_labeled} labeled, {len(result) - n_labeled} unlabeled)")
    print(f"  Columns: {len(result.columns)}")
    print(f"  Species codes: {sorted(result['treeSP'].unique())}")


if __name__ == "__main__":
    main()
