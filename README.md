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
```

## Usage

```bash
# Geometry-only baseline
python classify_species_rf.py

# Full feature combination experiment
python run_experiments.py
```

Results are saved to `data/experiment_results/`.
