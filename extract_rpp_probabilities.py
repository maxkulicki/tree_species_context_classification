#!/usr/bin/env python3
"""
Extract species probability distribution values at TLS plot locations.

Reads each GeoTIFF raster in the rpp_probability_distribution directory,
samples the pixel value at each plot location (reprojected to the raster CRS),
and saves the results to a semicolon-delimited CSV.

Values:
    0.0 - 1.0 : probability of species presence
    -1.0      : outside species modeled range
    NaN       : outside raster extent / nodata

Usage:
    python extract_rpp_probabilities.py
    python extract_rpp_probabilities.py --output results/rpp_values.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer


# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"
RPP_DIR = DATA_DIR / "rpp_probability_distribution"
PLOTS_CSV = DATA_DIR / "TreeScanPL_plot_locations.csv"

CRS_PUWG92 = "EPSG:2180"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract species probability values at TLS plot locations."
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DATA_DIR / "plots_rpp_probabilities.csv"),
        help="Output CSV path (default: data/plots_rpp_probabilities.csv)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Species Probability Distribution Extraction")
    print("=" * 60)

    # Load plots
    print("\nLoading plot locations...")
    plots = pd.read_csv(PLOTS_CSV, sep=";")
    print(f"  {len(plots)} plots loaded")

    coords_puwg = np.column_stack([plots["X"].values, plots["Y"].values])

    # Discover rasters
    tif_files = sorted(RPP_DIR.glob("*.tif"))
    print(f"\nFound {len(tif_files)} rasters:")
    for f in tif_files:
        print(f"  {f.name}")

    # Sample each raster at plot locations
    species_values = {}
    transformer_cache = {}

    for tif_path in tif_files:
        species_name = tif_path.stem.replace("_rpp", "").replace("-", "_")

        with rasterio.open(tif_path) as src:
            raster_crs = str(src.crs)

            # Reproject plot coordinates to raster CRS (cache transformer per CRS)
            if raster_crs not in transformer_cache:
                transformer_cache[raster_crs] = Transformer.from_crs(
                    CRS_PUWG92, raster_crs, always_xy=True
                )
            transformer = transformer_cache[raster_crs]
            xs, ys = transformer.transform(coords_puwg[:, 0], coords_puwg[:, 1])

            # Sample pixel values at each point
            coord_pairs = list(zip(xs, ys))
            values = np.array([v[0] for v in src.sample(coord_pairs)])

            # Replace nodata with NaN
            if src.nodata is not None:
                values = np.where(values == src.nodata, np.nan, values)

        species_values[species_name] = values
        n_valid = np.sum(~np.isnan(values) & (values >= 0))
        n_absent = np.sum(values == -1.0)
        n_nodata = np.sum(np.isnan(values))
        print(f"  {species_name:30s}  valid={n_valid:3d}  absent={n_absent:3d}  nodata={n_nodata:3d}")

    # Build result
    result = plots[["source", "file", "year", "num", "num_txt", "X", "Y"]].copy()
    for species_name, values in species_values.items():
        result[species_name] = values

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, sep=";")

    species_cols = list(species_values.keys())
    print(f"\nOutput saved to {output_path}")
    print(f"  Rows: {len(result)}")
    print(f"  Columns: {len(result.columns)} (7 plot attrs + {len(species_cols)} species)")


if __name__ == "__main__":
    main()
