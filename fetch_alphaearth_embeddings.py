#!/usr/bin/env python3
"""
Fetch AlphaEarth Satellite Embeddings for TLS plot locations.

Queries the Google Earth Engine Satellite Embedding V1 Annual dataset
(GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL) for the 2018 annual image,
extracts the 64-dimensional embedding vector at each TLS plot location,
and saves the results to a semicolon-delimited CSV.

Prerequisites:
    - earthengine-api installed
    - Authenticated via ee.Authenticate() (run once)
    - Google account registered for Earth Engine

Usage:
    python fetch_alphaearth_embeddings.py
    python fetch_alphaearth_embeddings.py --year 2019
    python fetch_alphaearth_embeddings.py --output results/embeddings.csv
"""

import argparse
from pathlib import Path

import ee
import pandas as pd
from pyproj import Transformer


# =============================================================================
# PATHS & CONSTANTS
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"
PLOTS_CSV = DATA_DIR / "TreeScanPL_plot_locations.csv"

COLLECTION_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]

CRS_PUWG92 = "EPSG:2180"
CRS_WGS84 = "EPSG:4326"


# =============================================================================
# HELPERS
# =============================================================================

def reproject_plots(df: pd.DataFrame) -> pd.DataFrame:
    """Add lon/lat columns by reprojecting X/Y from PUWG 1992 to WGS84."""
    transformer = Transformer.from_crs(CRS_PUWG92, CRS_WGS84, always_xy=True)
    lons, lats = transformer.transform(df["X"].values, df["Y"].values)
    df = df.copy()
    df["lon"] = lons
    df["lat"] = lats
    return df


def make_ee_features(df: pd.DataFrame) -> ee.FeatureCollection:
    """Create an EE FeatureCollection of points from the plots dataframe."""
    features = []
    for _, row in df.iterrows():
        geom = ee.Geometry.Point([row["lon"], row["lat"]])
        props = {
            "num": int(row["num"]),
            "file": row["file"],
            "X": float(row["X"]),
            "Y": float(row["Y"]),
        }
        features.append(ee.Feature(geom, props))
    return ee.FeatureCollection(features)


def extract_embeddings(
    plots_fc: ee.FeatureCollection,
    year: int,
) -> pd.DataFrame:
    """
    Sample the AlphaEarth embedding image for the given year at all plot points.

    Uses sampleRegions() which executes server-side and returns all 64 bands
    for each point in one call.
    """
    collection = ee.ImageCollection(COLLECTION_ID)
    image = (
        collection
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select(EMBEDDING_BANDS)
        .mosaic()
    )

    sampled = image.sampleRegions(
        collection=plots_fc,
        scale=10,
        geometries=False,
    )

    results = sampled.getInfo()
    rows = []
    for feat in results["features"]:
        props = feat["properties"]
        rows.append(props)

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch AlphaEarth embeddings for TLS plot locations."
    )
    parser.add_argument(
        "--year", type=int, default=2018,
        help="Embedding year to fetch (default: 2018)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path (default: data/plots_alphaearth_{year}.csv)",
    )
    parser.add_argument(
        "--project", "-p", default=None,
        help="Google Cloud project ID for Earth Engine initialization",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else DATA_DIR / f"plots_alphaearth_{args.year}.csv"

    print("=" * 60)
    print(f"AlphaEarth Embedding Extraction (year={args.year})")
    print("=" * 60)

    # Authenticate & initialize
    print("\nInitializing Earth Engine...")
    ee.Authenticate()
    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()
    print("  Authenticated and initialized.")

    # Load and reproject plots
    print("\nLoading plot locations...")
    plots_df = pd.read_csv(PLOTS_CSV, sep=";")
    plots_df = reproject_plots(plots_df)
    print(f"  {len(plots_df)} plots loaded and reprojected to WGS84")
    print(f"  Lon range: {plots_df['lon'].min():.4f} to {plots_df['lon'].max():.4f}")
    print(f"  Lat range: {plots_df['lat'].min():.4f} to {plots_df['lat'].max():.4f}")

    # Build EE feature collection
    print("\nBuilding Earth Engine FeatureCollection...")
    plots_fc = make_ee_features(plots_df)

    # Extract embeddings
    print(f"\nQuerying {COLLECTION_ID} for {args.year}...")
    print("  This may take a minute (server-side computation)...")
    embeddings_df = extract_embeddings(plots_fc, args.year)

    print(f"  Retrieved embeddings for {len(embeddings_df)} plots")

    # Merge with original plot info to ensure alignment
    result = plots_df[["source", "file", "year", "num", "num_txt", "X", "Y", "lon", "lat"]].copy()
    emb_subset = embeddings_df[["num"] + [b for b in EMBEDDING_BANDS if b in embeddings_df.columns]]
    result = result.merge(emb_subset, on="num", how="left")

    # Report any missing
    n_missing = result[EMBEDDING_BANDS[0]].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} plots have no embedding (masked/no-data pixels)")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, sep=";")

    print(f"\nOutput saved to {output_path}")
    print(f"  Rows: {len(result)}")
    print(f"  Columns: {len(result.columns)} (9 plot attrs + 64 embedding dims)")


if __name__ == "__main__":
    main()
