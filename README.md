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

## Feature Descriptions

### Geometry (68 features) — per-tree point cloud descriptors

Extracted from individual tree point clouds segmented from TLS plots using `extract_features.py`.

**Height statistics (8):** `n_points`, `h_max`, `h_mean`, `h_std`, `h_cv`, `h_skewness`, `h_kurtosis`, `h_iqr`, `h_mad` — basic distributional statistics of point heights within the tree crown.

**Height percentiles (12):** `h_p5` through `h_p95` — absolute height at the 5th, 10th, 20th, 25th, 30th, 40th, 50th, 60th, 70th, 75th, 90th, and 95th percentiles.

**Normalized height percentiles (12):** `hn_p5` through `hn_p95` — same percentiles normalized to [0, 1] by the tree's height range, making them scale-invariant.

**Vertical density profile (10):** `vd_0_10` through `vd_90_100` — proportion of points in each of 10 equal vertical slices of the tree crown. Captures the vertical distribution of foliage.

**Horizontal spread (8):** `xy_range_x`, `xy_range_y`, `xy_range_ratio` — crown bounding box dimensions and elongation. `r_mean`, `r_std`, `r_cv`, `r_max`, `r_p95` — radial distance statistics from the crown centroid.

**Eigenvalue features (9):** `eig_linearity`, `eig_planarity`, `eig_sphericity`, `eig_omnivariance`, `eig_anisotropy`, `eig_eigenentropy`, `eig_sum`, `eig_ratio_12`, `eig_ratio_13` — derived from the 3D covariance matrix eigenvalues of the point cloud. Describe the overall 3D shape (linear vs. planar vs. spherical).

**Convex hull metrics (5):** `hull_area_2d`, `hull_perimeter_2d`, `hull_compactness_2d` — 2D crown projection shape. `hull_volume_3d`, `hull_density` — 3D crown volume and point density.

**Vertical profile shape (3):** `vp_centroid_height` — relative height of the point mass center. `vp_concentration` — IQR relative to total height (how concentrated the crown is). `vp_top_heaviness` — fraction of points above median height.

### BDL (61 features) — Polish Forest Data Bank (Bank Danych o Lasach)

Plot-level forest inventory attributes from the BDL, spatially joined to TLS plot locations. Source data is flattened from a long-form table describing forest subcompartments (wydzielenia).

**Numeric stand attributes (4):**
- `rotat_age` — rotation age (wiek rebnosci): planned harvest age for the stand
- `sub_area` — subcompartment area in hectares
- `part_cd` — share of the dominant species in the stand (udział gatunku panującego)
- `spec_age` — age of the dominant species

**Site type (one-hot, 20):** `bdl_site_type_*` — forest habitat type (typ siedliskowy lasu), e.g. BMB (fresh mixed coniferous forest), LW (mesic broadleaf forest), OL (alder swamp forest). Encodes soil fertility and moisture conditions.

**Silvicultural regime (one-hot, 4):** `bdl_silvicult_*` — management type (gospodarstwo): GPZ (shelterwood), GZ (clearcut), O (conservation), S (seed-tree).

**Forest function (one-hot, 3):** `bdl_forest_fun_*` — designated function: GOSP (economic/production), OCHR (protective), REZ (nature reserve).

**Stand structure (one-hot, 5):** `bdl_stand_stru_*` — vertical structure of the stand: 2 PIETR (two-storied), DRZEW (timber stand), KDO (shelterwood regeneration), KO (natural regeneration), SP (even-aged).

**Soil moisture (one-hot, 9):** `bdl_moisture_cd_*` — soil moisture variant, ranging from very dry (BO) through moist (WW) to waterlogged/marshy conditions.

**Degradation (one-hot, 3):** `bdl_degradation_cd_*` — site degradation level: N1/N2 (chemical degradation levels), Z1 (mechanical degradation).

**Dominant species (one-hot, 11):** `bdl_species_cd_*` — code of the dominant tree species in the stand (e.g. SO = Scots pine, BK = European beech, DB = oak, SW = Norway spruce, JD = Silver fir).

**Aggregated features (4):**
- `n_storeys` — number of distinct canopy storeys in the stand
- `n_species` — number of tree species recorded in the stand
- `main_storey_density` — canopy closure index of the main storey
- `conifer_ratio` — proportion of conifer species in the main storey

### AlphaEarth (64 features) — satellite embedding vectors

`A00` through `A63` — 64-dimensional embedding vectors from the Google Earth Engine Satellite Embedding V1 Annual dataset (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`), sampled at each plot location for the year 2018. These are learned representations from satellite imagery that encode land cover, vegetation structure, and phenological patterns into a compact feature vector. Individual dimensions do not have direct physical interpretations.

### RPP (17 features) — Range Prediction Probabilities

Species-level presence probabilities from the European Atlas of Forest Tree Species (Caudullo et al., 2017), published by the EU Joint Research Centre (JRC). Each feature is the modelled probability of finding at least one individual of that species within the 1 km x 1 km grid cell containing the plot. Values range from 0.0 (absent) to 1.0 (high probability), with -1.0 indicating the location is outside the species' modelled range. The 17 species with RPP maps used here are:

| Feature | Species |
|---------|---------|
| `Abies_alba` | Silver fir |
| `Acer_pseudoplatanus` | Sycamore maple |
| `Alnus_glutinosa` | Black alder |
| `Alnus_incana` | Grey alder |
| `Betula_sp` | Birch (genus) |
| `Carpinus_betulus` | Common hornbeam |
| `Fagus_sylvatica` | European beech |
| `Fraxinus_excelsior` | European ash |
| `Larix_decidua` | European larch |
| `Picea_abies` | Norway spruce |
| `Pinus_sylvestris` | Scots pine |
| `Populus_tremula` | Aspen |
| `Prunus_avium` | Wild cherry |
| `Quercus_robur` | English oak |
| `Salix_caprea` | Goat willow |
| `Sorbus_aucuparia` | Rowan |
| `Tilia_sp` | Lime (genus) |

Maps were produced using the Relative Distance Similarity (RDS) method, linking National Forest Inventory occurrence records across Europe with bioclimatic and topographic predictors at 1 km resolution.

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
