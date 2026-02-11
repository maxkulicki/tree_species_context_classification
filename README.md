# Context-Enhanced Tree Species Classification from TLS Point Clouds

Tree species classification using geometric features extracted from Terrestrial Laser Scanning (TLS) point clouds, enhanced with contextual data sources from forest inventories, satellite embeddings, and species distribution models.

## Overview

Individual trees segmented from TLS plots (272 plots, ~6,800 trees, 18 species) are classified using a Random Forest model. We systematically evaluate how three contextual data sources improve upon a geometry-only baseline:

| Source | Description | Features |
|--------|-------------|----------|
| **Geometry** | Per-tree point cloud descriptors (height stats, crown shape, eigenvalues, etc.) | 68 |
| **BDL** | Polish Forest Data Bank attributes (site type, stand structure, species composition) | ~62 |
| **AlphaEarth** | Google Earth Engine satellite embedding vectors (2018) | 64 |
| **RPP** | Range Prediction Probabilities for 17 tree species | 17 |

## Results

All 8 combinations of context sources (geometry always included):

| Combination | Trees | Accuracy | Balanced Acc. | Cohen's Kappa |
|---|---|---|---|---|
| Geometry + BDL + AlphaEarth | 4,147 | 90.7% | **81.1%** | 0.840 |
| Geometry + BDL + AlphaEarth + RPP | 4,147 | 90.8% | 80.7% | 0.843 |
| Geometry + BDL | 4,147 | 88.4% | 72.3% | 0.792 |
| Geometry + BDL + RPP | 4,147 | 89.2% | 71.4% | 0.809 |
| Geometry + AlphaEarth | 6,845 | 89.0% | 65.6% | 0.841 |
| Geometry + AlphaEarth + RPP | 6,845 | 89.2% | 64.9% | 0.844 |
| Geometry + RPP | 6,845 | 87.2% | 57.0% | 0.811 |
| Geometry (baseline) | 6,845 | 74.3% | 25.7% | 0.602 |

## Geographic Generalization (District-Level CV)

To test whether models generalize across geographic regions, we run leave-one-district-out cross-validation: train on 5 districts, test on the held-out district, repeat for all 6 districts (Gorlice, Herby, Katrynka, Milicz, Piensk, Suprasl). Species 21 (Silver fir) and 72 (Grey alder) are excluded as they only occur in Gorlice.

| Combination | Balanced Acc. (mean ± std) | Cohen's Kappa (mean ± std) |
|---|---|---|
| Geometry + BDL | 0.239 ± 0.081 | 0.396 ± 0.200 |
| Geometry + BDL + AlphaEarth | 0.216 ± 0.029 | 0.412 ± 0.209 |
| Geometry + AlphaEarth | 0.213 ± 0.027 | 0.413 ± 0.215 |
| Geometry + BDL + RPP | 0.208 ± 0.055 | 0.348 ± 0.218 |
| Geometry (baseline) | 0.206 ± 0.022 | 0.333 ± 0.172 |
| Geometry + BDL + AlphaEarth + RPP | 0.204 ± 0.051 | 0.393 ± 0.211 |
| Geometry + AlphaEarth + RPP | 0.196 ± 0.049 | 0.400 ± 0.219 |
| Geometry + RPP | 0.176 ± 0.043 | 0.321 ± 0.231 |

Performance drops significantly compared to random splits (e.g. best balanced accuracy: 0.239 vs 0.677), showing that random splits inflate metrics by including same-district trees in both train and test sets.

## Scripts

| Script | Purpose |
|--------|---------|
| `extract_features.py` | Extract per-tree geometric features from a LAZ file |
| `extract_tree_features_single.py` | Batch extraction across all plot LAZ files |
| `fuse_bdl_plots.py` | Spatial join of TLS plots with BDL forest inventory polygons |
| `fetch_alphaearth_embeddings.py` | Query Google Earth Engine for satellite embeddings at plot locations |
| `extract_rpp_probabilities.py` | Sample species probability rasters at plot locations |
| `classify_species_rf.py` | Geometry-only RF baseline classifier |
| `run_experiments.py` | Systematic evaluation of all 8 feature combinations |
| `run_district_cv.py` | District-level leave-one-out cross-validation |
| `plot_importance.py` | Feature importance visualizations |
| `plot_district_cv.py` | District CV visualizations (heatmaps, comparison chart) |

## Data

Data files are not included in the repository. Expected structure under `data/`:

```
data/
  TreeScanPL_downsapled_2cm_corrected/   # TLS point clouds (LAZ)
  TreeScanPL_plot_locations.csv          # Plot coordinates
  tree_features/all_tree_features.csv    # Extracted geometric features
  plots_bdl_fused.csv                    # BDL forest attributes (long-form)
  plots_alphaearth_2018.csv              # AlphaEarth satellite embeddings
  plots_rpp_probabilities.csv            # RPP species probabilities
  experiment_results/                    # Output from run_experiments.py
  district_cv_results/                   # Output from run_district_cv.py
```

## Usage

```bash
# Geometry-only baseline
python classify_species_rf.py

# Full feature combination experiment
python run_experiments.py

# District-level leave-one-out cross-validation
python run_district_cv.py

# Generate plots
python plot_importance.py
python plot_district_cv.py
```

Results are saved to `data/experiment_results/` and `data/district_cv_results/`.
