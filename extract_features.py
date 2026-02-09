#!/usr/bin/env python3
"""
Point Cloud Feature Extraction for Tree Species Classification

Extracts two sets of features from a LAZ file with annotated individual trees:
1. Per-tree features (67 features per tree) - saved to {input}_tree_features.csv
2. Plot-level features (72 features per plot) - saved to {input}_plot_features.csv

Requirements:
    pip install laspy numpy scipy pandas

Usage:
    python extract_features.py input.laz
    python extract_features.py input.laz --output-dir ./results
    python extract_features.py input.laz --tree-id-field treeID

Input:
    LAZ/LAS file with a point attribute for tree IDs (default: 'treeID')
    treeID = 0 indicates non-tree points (ground, understory, unassigned canopy)
    treeID > 0 indicates individual tree membership

Output:
    - {basename}_tree_features.csv: One row per tree with 67 geometric features
    - {basename}_plot_features.csv: One row with 72 plot-level features

Author: Generated for TLS species classification research
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.stats import skew, kurtosis
import pandas as pd

try:
    import laspy
except ImportError:
    print("Error: laspy is required. Install with: pip install laspy")
    sys.exit(1)


# =============================================================================
# CONSTANTS
# =============================================================================

PLOT_RADIUS = 15.0  # meters
PLOT_AREA = np.pi * PLOT_RADIUS ** 2  # ~706.86 m²

# Height thresholds for non-tree classification (meters)
GROUND_THRESHOLD = 0.5
UNDERSTORY_THRESHOLD = 2.0
MIDSTORY_THRESHOLD = 10.0


# =============================================================================
# PER-TREE FEATURE EXTRACTION (67 features)
# =============================================================================

def extract_tree_features(points: np.ndarray, tree_id: int) -> dict:
    """
    Extract 67 geometric features from a single tree point cloud.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud of shape (n, 3) with columns [x, y, z]
    tree_id : int
        Tree identifier for output
    
    Returns
    -------
    dict
        Dictionary with 67 features plus tree_id
    """
    if len(points) < 4:
        # Not enough points for meaningful features
        return None
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    n = len(z)
    
    features = {'tree_id': tree_id, 'n_points': n}
    
    # -------------------------------------------------------------------------
    # 1. Height Statistics (8 features)
    # -------------------------------------------------------------------------
    h_max = np.max(z)
    h_min = np.min(z)
    h_mean = np.mean(z)
    h_std = np.std(z, ddof=1) if n > 1 else 0
    
    features['h_max'] = h_max
    features['h_mean'] = h_mean
    features['h_std'] = h_std
    features['h_cv'] = h_std / h_mean if h_mean > 0 else 0
    features['h_skewness'] = skew(z) if n > 2 else 0
    features['h_kurtosis'] = kurtosis(z) if n > 3 else 0
    features['h_iqr'] = np.percentile(z, 75) - np.percentile(z, 25)
    features['h_mad'] = np.median(np.abs(z - np.median(z)))
    
    # -------------------------------------------------------------------------
    # 2. Height Percentiles (12 features)
    # -------------------------------------------------------------------------
    percentiles = [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 90, 95]
    for p in percentiles:
        features[f'h_p{p}'] = np.percentile(z, p)
    
    # -------------------------------------------------------------------------
    # 3. Normalized Height Percentiles (12 features)
    # -------------------------------------------------------------------------
    h_range = h_max - h_min
    for p in percentiles:
        if h_range > 0:
            features[f'hn_p{p}'] = (np.percentile(z, p) - h_min) / h_range
        else:
            features[f'hn_p{p}'] = 0
    
    # -------------------------------------------------------------------------
    # 4. Vertical Layer Density (10 features)
    # -------------------------------------------------------------------------
    if h_range > 0:
        z_norm = (z - h_min) / h_range
    else:
        z_norm = np.zeros_like(z)
    
    for i in range(10):
        lower, upper = i * 0.1, (i + 1) * 0.1
        if i == 9:  # Include upper bound for last bin
            count = np.sum((z_norm >= lower) & (z_norm <= upper))
        else:
            count = np.sum((z_norm >= lower) & (z_norm < upper))
        features[f'vd_{i*10}_{(i+1)*10}'] = count / n
    
    # -------------------------------------------------------------------------
    # 5. Horizontal Spread Statistics (8 features)
    # -------------------------------------------------------------------------
    x_c, y_c = np.mean(x), np.mean(y)
    r = np.sqrt((x - x_c)**2 + (y - y_c)**2)
    
    features['xy_range_x'] = np.max(x) - np.min(x)
    features['xy_range_y'] = np.max(y) - np.min(y)
    range_max = max(features['xy_range_x'], features['xy_range_y'])
    range_min = min(features['xy_range_x'], features['xy_range_y'])
    features['xy_range_ratio'] = range_min / range_max if range_max > 0 else 1
    
    r_mean = np.mean(r)
    r_std = np.std(r, ddof=1) if n > 1 else 0
    features['r_mean'] = r_mean
    features['r_std'] = r_std
    features['r_cv'] = r_std / r_mean if r_mean > 0 else 0
    features['r_max'] = np.max(r)
    features['r_p95'] = np.percentile(r, 95)
    
    # -------------------------------------------------------------------------
    # 6. Eigenvalue Features (9 features)
    # -------------------------------------------------------------------------
    try:
        cov_matrix = np.cov(points.T)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # λ1 >= λ2 >= λ3
        l1, l2, l3 = eigenvalues
        
        features['eig_linearity'] = (l1 - l2) / l1 if l1 > 0 else 0
        features['eig_planarity'] = (l2 - l3) / l1 if l1 > 0 else 0
        features['eig_sphericity'] = l3 / l1 if l1 > 0 else 0
        features['eig_omnivariance'] = (max(l1, 1e-10) * max(l2, 1e-10) * max(l3, 1e-10)) ** (1/3)
        features['eig_anisotropy'] = (l1 - l3) / l1 if l1 > 0 else 0
        
        l_sum = l1 + l2 + l3
        if l_sum > 0:
            l_norm = eigenvalues / l_sum
            l_norm = np.clip(l_norm, 1e-10, None)
            features['eig_eigenentropy'] = -np.sum(l_norm * np.log(l_norm))
        else:
            features['eig_eigenentropy'] = 0
        
        features['eig_sum'] = l_sum
        features['eig_ratio_12'] = l2 / l1 if l1 > 0 else 0
        features['eig_ratio_13'] = l3 / l1 if l1 > 0 else 0
    except:
        for key in ['eig_linearity', 'eig_planarity', 'eig_sphericity', 
                    'eig_omnivariance', 'eig_anisotropy', 'eig_eigenentropy',
                    'eig_sum', 'eig_ratio_12', 'eig_ratio_13']:
            features[key] = 0
    
    # -------------------------------------------------------------------------
    # 7. Convex Hull Metrics (5 features)
    # -------------------------------------------------------------------------
    # 2D hull (XY projection)
    try:
        xy_points = points[:, :2]
        if len(np.unique(xy_points, axis=0)) >= 3:
            hull_2d = ConvexHull(xy_points)
            features['hull_area_2d'] = hull_2d.volume  # In 2D, volume = area
            features['hull_perimeter_2d'] = hull_2d.area  # In 2D, area = perimeter
            if features['hull_perimeter_2d'] > 0:
                features['hull_compactness_2d'] = (4 * np.pi * features['hull_area_2d']) / (features['hull_perimeter_2d']**2)
            else:
                features['hull_compactness_2d'] = 0
        else:
            features['hull_area_2d'] = 0
            features['hull_perimeter_2d'] = 0
            features['hull_compactness_2d'] = 0
    except:
        features['hull_area_2d'] = 0
        features['hull_perimeter_2d'] = 0
        features['hull_compactness_2d'] = 0
    
    # 3D hull
    try:
        if len(np.unique(points, axis=0)) >= 4:
            hull_3d = ConvexHull(points)
            features['hull_volume_3d'] = hull_3d.volume
            features['hull_density'] = n / hull_3d.volume if hull_3d.volume > 0 else 0
        else:
            features['hull_volume_3d'] = 0
            features['hull_density'] = 0
    except:
        features['hull_volume_3d'] = 0
        features['hull_density'] = 0
    
    # -------------------------------------------------------------------------
    # 8. Vertical Profile Shape (3 features)
    # -------------------------------------------------------------------------
    features['vp_centroid_height'] = (h_mean - h_min) / h_range if h_range > 0 else 0
    features['vp_concentration'] = features['h_iqr'] / h_range if h_range > 0 else 0
    h_median = np.median(z)
    features['vp_top_heaviness'] = np.sum(z > h_median) / n
    
    return features


# =============================================================================
# PLOT-LEVEL FEATURE EXTRACTION (72 features)
# =============================================================================

def extract_plot_features(
    all_points: np.ndarray,
    tree_ids: np.ndarray,
    tree_features_list: list,
    plot_id: str = "plot"
) -> dict:
    """
    Extract 72 plot-level features from the entire plot point cloud.
    
    Parameters
    ----------
    all_points : np.ndarray
        Full plot point cloud of shape (n, 3)
    tree_ids : np.ndarray
        Array of tree IDs for each point (0 = non-tree)
    tree_features_list : list
        List of per-tree feature dictionaries (from extract_tree_features)
    plot_id : str
        Identifier for the plot
    
    Returns
    -------
    dict
        Dictionary with 72 plot-level features
    """
    features = {'plot_id': plot_id}
    
    x, y, z = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    n_total = len(z)
    
    # Separate tree and non-tree points
    tree_mask = tree_ids > 0
    nontree_mask = tree_ids == 0
    
    tree_points = all_points[tree_mask]
    nontree_points = all_points[nontree_mask]
    
    unique_trees = [tid for tid in np.unique(tree_ids) if tid > 0]
    n_trees = len(unique_trees)
    
    # -------------------------------------------------------------------------
    # 1. Stand Composition Features (8 features)
    # -------------------------------------------------------------------------
    features['stand_n_trees'] = n_trees
    features['stand_tree_density'] = n_trees / PLOT_AREA
    features['stand_total_points'] = n_total
    features['stand_tree_points_ratio'] = np.sum(tree_mask) / n_total if n_total > 0 else 0
    features['stand_nontree_points_ratio'] = np.sum(nontree_mask) / n_total if n_total > 0 else 0
    
    if n_trees > 0 and tree_features_list:
        tree_sizes = [tf['n_points'] for tf in tree_features_list if tf is not None]
        features['stand_mean_tree_size'] = np.mean(tree_sizes) if tree_sizes else 0
        features['stand_std_tree_size'] = np.std(tree_sizes, ddof=1) if len(tree_sizes) > 1 else 0
        features['stand_cv_tree_size'] = features['stand_std_tree_size'] / features['stand_mean_tree_size'] if features['stand_mean_tree_size'] > 0 else 0
    else:
        features['stand_mean_tree_size'] = 0
        features['stand_std_tree_size'] = 0
        features['stand_cv_tree_size'] = 0
    
    # -------------------------------------------------------------------------
    # 2. Plot-Wide Height Statistics (10 features)
    # -------------------------------------------------------------------------
    h_max = np.max(z)
    h_min = np.min(z)
    h_mean = np.mean(z)
    h_std = np.std(z, ddof=1) if n_total > 1 else 0
    
    features['plot_h_max'] = h_max
    features['plot_h_mean'] = h_mean
    features['plot_h_std'] = h_std
    features['plot_h_cv'] = h_std / h_mean if h_mean > 0 else 0
    features['plot_h_skewness'] = skew(z) if n_total > 2 else 0
    features['plot_h_kurtosis'] = kurtosis(z) if n_total > 3 else 0
    features['plot_h_p25'] = np.percentile(z, 25)
    features['plot_h_p50'] = np.percentile(z, 50)
    features['plot_h_p75'] = np.percentile(z, 75)
    features['plot_h_p95'] = np.percentile(z, 95)
    
    # -------------------------------------------------------------------------
    # 3. Aggregated Tree Statistics (12 features)
    # -------------------------------------------------------------------------
    if tree_features_list and any(tf is not None for tf in tree_features_list):
        valid_trees = [tf for tf in tree_features_list if tf is not None]
        
        tree_h_maxs = [tf['h_max'] for tf in valid_trees]
        tree_h_means = [tf['h_mean'] for tf in valid_trees]
        tree_spreads = [tf['r_max'] for tf in valid_trees]
        tree_volumes = [tf['hull_volume_3d'] for tf in valid_trees]
        
        features['agg_tree_h_max_mean'] = np.mean(tree_h_maxs)
        features['agg_tree_h_max_std'] = np.std(tree_h_maxs, ddof=1) if len(tree_h_maxs) > 1 else 0
        features['agg_tree_h_max_min'] = np.min(tree_h_maxs)
        features['agg_tree_h_max_max'] = np.max(tree_h_maxs)
        features['agg_tree_h_max_range'] = np.max(tree_h_maxs) - np.min(tree_h_maxs)
        features['agg_tree_h_mean_mean'] = np.mean(tree_h_means)
        features['agg_tree_h_mean_std'] = np.std(tree_h_means, ddof=1) if len(tree_h_means) > 1 else 0
        features['agg_tree_spread_mean'] = np.mean(tree_spreads)
        features['agg_tree_spread_std'] = np.std(tree_spreads, ddof=1) if len(tree_spreads) > 1 else 0
        features['agg_tree_volume_mean'] = np.mean(tree_volumes)
        features['agg_tree_volume_std'] = np.std(tree_volumes, ddof=1) if len(tree_volumes) > 1 else 0
        features['agg_tree_volume_total'] = np.sum(tree_volumes)
    else:
        for key in ['agg_tree_h_max_mean', 'agg_tree_h_max_std', 'agg_tree_h_max_min',
                    'agg_tree_h_max_max', 'agg_tree_h_max_range', 'agg_tree_h_mean_mean',
                    'agg_tree_h_mean_std', 'agg_tree_spread_mean', 'agg_tree_spread_std',
                    'agg_tree_volume_mean', 'agg_tree_volume_std', 'agg_tree_volume_total']:
            features[key] = 0
    
    # -------------------------------------------------------------------------
    # 4. Canopy Layer Distribution (10 features)
    # -------------------------------------------------------------------------
    h_range = h_max - h_min
    if h_range > 0:
        z_norm = (z - h_min) / h_range
    else:
        z_norm = np.zeros_like(z)
    
    for i in range(10):
        lower, upper = i * 0.1, (i + 1) * 0.1
        if i == 9:
            count = np.sum((z_norm >= lower) & (z_norm <= upper))
        else:
            count = np.sum((z_norm >= lower) & (z_norm < upper))
        features[f'canopy_vd_{i*10}_{(i+1)*10}'] = count / n_total if n_total > 0 else 0
    
    # -------------------------------------------------------------------------
    # 5. Non-Tree Point Cloud Features (14 features)
    # -------------------------------------------------------------------------
    n_nontree = len(nontree_points)
    features['nontree_n_points'] = n_nontree
    
    if n_nontree > 0:
        nt_z = nontree_points[:, 2]
        features['nontree_h_max'] = np.max(nt_z)
        features['nontree_h_mean'] = np.mean(nt_z)
        features['nontree_h_std'] = np.std(nt_z, ddof=1) if n_nontree > 1 else 0
        features['nontree_h_p25'] = np.percentile(nt_z, 25)
        features['nontree_h_p50'] = np.percentile(nt_z, 50)
        features['nontree_h_p75'] = np.percentile(nt_z, 75)
        features['nontree_h_p95'] = np.percentile(nt_z, 95)
        
        # Height-based classification
        n_ground = np.sum(nt_z < GROUND_THRESHOLD)
        n_understory = np.sum((nt_z >= GROUND_THRESHOLD) & (nt_z < UNDERSTORY_THRESHOLD))
        n_midstory = np.sum((nt_z >= UNDERSTORY_THRESHOLD) & (nt_z < MIDSTORY_THRESHOLD))
        n_canopy = np.sum(nt_z >= MIDSTORY_THRESHOLD)
        
        features['nontree_ground_ratio'] = n_ground / n_nontree
        features['nontree_understory_ratio'] = n_understory / n_nontree
        features['nontree_midstory_ratio'] = n_midstory / n_nontree
        features['nontree_canopy_ratio'] = n_canopy / n_nontree
        
        # Horizontal spread (from plot center assumed at 0,0 or computed)
        nt_x, nt_y = nontree_points[:, 0], nontree_points[:, 1]
        plot_center_x, plot_center_y = np.mean(x), np.mean(y)
        nt_r = np.sqrt((nt_x - plot_center_x)**2 + (nt_y - plot_center_y)**2)
        features['nontree_horizontal_spread'] = np.max(nt_r)
        features['nontree_density_ground'] = n_ground / PLOT_AREA
    else:
        for key in ['nontree_h_max', 'nontree_h_mean', 'nontree_h_std',
                    'nontree_h_p25', 'nontree_h_p50', 'nontree_h_p75', 'nontree_h_p95',
                    'nontree_ground_ratio', 'nontree_understory_ratio',
                    'nontree_midstory_ratio', 'nontree_canopy_ratio',
                    'nontree_horizontal_spread', 'nontree_density_ground']:
            features[key] = 0
    
    # -------------------------------------------------------------------------
    # 6. Plot-Level 3D Structure (10 features)
    # -------------------------------------------------------------------------
    # Eigenvalue features
    try:
        cov_matrix = np.cov(all_points.T)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        l1, l2, l3 = eigenvalues
        
        features['plot_eig_linearity'] = (l1 - l2) / l1 if l1 > 0 else 0
        features['plot_eig_planarity'] = (l2 - l3) / l1 if l1 > 0 else 0
        features['plot_eig_sphericity'] = l3 / l1 if l1 > 0 else 0
        features['plot_eig_omnivariance'] = (max(l1, 1e-10) * max(l2, 1e-10) * max(l3, 1e-10)) ** (1/3)
        features['plot_eig_anisotropy'] = (l1 - l3) / l1 if l1 > 0 else 0
        
        l_sum = l1 + l2 + l3
        if l_sum > 0:
            l_norm = eigenvalues / l_sum
            l_norm = np.clip(l_norm, 1e-10, None)
            features['plot_eig_eigenentropy'] = -np.sum(l_norm * np.log(l_norm))
        else:
            features['plot_eig_eigenentropy'] = 0
    except:
        for key in ['plot_eig_linearity', 'plot_eig_planarity', 'plot_eig_sphericity',
                    'plot_eig_omnivariance', 'plot_eig_anisotropy', 'plot_eig_eigenentropy']:
            features[key] = 0
    
    # Convex hull features
    try:
        if len(np.unique(all_points, axis=0)) >= 4:
            hull_3d = ConvexHull(all_points)
            features['plot_hull_volume_3d'] = hull_3d.volume
            features['plot_hull_density'] = n_total / hull_3d.volume if hull_3d.volume > 0 else 0
        else:
            features['plot_hull_volume_3d'] = 0
            features['plot_hull_density'] = 0
    except:
        features['plot_hull_volume_3d'] = 0
        features['plot_hull_density'] = 0
    
    try:
        xy_points = all_points[:, :2]
        if len(np.unique(xy_points, axis=0)) >= 3:
            hull_2d = ConvexHull(xy_points)
            features['plot_hull_area_2d'] = hull_2d.volume
            perimeter = hull_2d.area
            features['plot_hull_compactness_2d'] = (4 * np.pi * features['plot_hull_area_2d']) / (perimeter**2) if perimeter > 0 else 0
        else:
            features['plot_hull_area_2d'] = 0
            features['plot_hull_compactness_2d'] = 0
    except:
        features['plot_hull_area_2d'] = 0
        features['plot_hull_compactness_2d'] = 0
    
    # -------------------------------------------------------------------------
    # 7. Spatial Distribution Features (8 features)
    # -------------------------------------------------------------------------
    if n_trees > 0 and tree_features_list:
        # Compute tree centroids
        tree_centroids = []
        tree_hull_areas = []
        
        for tid in unique_trees:
            tree_mask_i = tree_ids == tid
            tree_pts = all_points[tree_mask_i]
            if len(tree_pts) > 0:
                centroid = np.mean(tree_pts, axis=0)
                tree_centroids.append(centroid[:2])  # XY only
                
                # 2D hull area for coverage
                try:
                    if len(np.unique(tree_pts[:, :2], axis=0)) >= 3:
                        hull = ConvexHull(tree_pts[:, :2])
                        tree_hull_areas.append(hull.volume)
                    else:
                        tree_hull_areas.append(0)
                except:
                    tree_hull_areas.append(0)
        
        if tree_centroids:
            tree_centroids = np.array(tree_centroids)
            plot_center = np.array([np.mean(x), np.mean(y)])
            
            # Distance from plot center
            centroid_distances = np.sqrt(np.sum((tree_centroids - plot_center)**2, axis=1))
            features['spatial_tree_centroid_r_mean'] = np.mean(centroid_distances)
            features['spatial_tree_centroid_r_std'] = np.std(centroid_distances, ddof=1) if len(centroid_distances) > 1 else 0
            features['spatial_tree_centroid_r_max'] = np.max(centroid_distances)
            
            # Nearest neighbor distances
            if len(tree_centroids) > 1:
                dist_matrix = cdist(tree_centroids, tree_centroids)
                np.fill_diagonal(dist_matrix, np.inf)
                nn_distances = np.min(dist_matrix, axis=1)
                
                features['spatial_tree_spacing_mean'] = np.mean(nn_distances)
                features['spatial_tree_spacing_std'] = np.std(nn_distances, ddof=1)
                features['spatial_tree_spacing_min'] = np.min(nn_distances)
                
                # Clark-Evans index: R = mean_NN / (0.5 * sqrt(area/n))
                expected_nn = 0.5 * np.sqrt(PLOT_AREA / n_trees)
                features['spatial_clark_evans'] = features['spatial_tree_spacing_mean'] / expected_nn if expected_nn > 0 else 0
            else:
                features['spatial_tree_spacing_mean'] = 0
                features['spatial_tree_spacing_std'] = 0
                features['spatial_tree_spacing_min'] = 0
                features['spatial_clark_evans'] = 0
            
            # Coverage ratio
            features['spatial_coverage_ratio'] = sum(tree_hull_areas) / PLOT_AREA
        else:
            for key in ['spatial_tree_centroid_r_mean', 'spatial_tree_centroid_r_std',
                        'spatial_tree_centroid_r_max', 'spatial_tree_spacing_mean',
                        'spatial_tree_spacing_std', 'spatial_tree_spacing_min',
                        'spatial_clark_evans', 'spatial_coverage_ratio']:
                features[key] = 0
    else:
        for key in ['spatial_tree_centroid_r_mean', 'spatial_tree_centroid_r_std',
                    'spatial_tree_centroid_r_max', 'spatial_tree_spacing_mean',
                    'spatial_tree_spacing_std', 'spatial_tree_spacing_min',
                    'spatial_clark_evans', 'spatial_coverage_ratio']:
            features[key] = 0
    
    return features


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_laz_file(
    input_path: str,
    output_dir: str = None,
    tree_id_field: str = 'treeID'
) -> tuple:
    """
    Process a LAZ file and extract both per-tree and plot-level features.
    
    Parameters
    ----------
    input_path : str
        Path to input LAZ/LAS file
    output_dir : str, optional
        Output directory (default: same as input)
    tree_id_field : str
        Name of the point attribute containing tree IDs
    
    Returns
    -------
    tuple
        (tree_features_df, plot_features_df)
    """
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    basename = input_path.stem
    
    print(f"Loading {input_path}...")
    las = laspy.read(str(input_path))
    
    # Extract coordinates
    points = np.vstack([las.x, las.y, las.z]).T
    print(f"  Total points: {len(points):,}")
    
    # Extract tree IDs
    if tree_id_field in las.point_format.dimension_names:
        tree_ids = np.array(las[tree_id_field])
    elif tree_id_field.lower() in [d.lower() for d in las.point_format.dimension_names]:
        # Case-insensitive fallback
        for dim in las.point_format.dimension_names:
            if dim.lower() == tree_id_field.lower():
                tree_ids = np.array(las[dim])
                break
    else:
        available = list(las.point_format.dimension_names)
        raise ValueError(f"Tree ID field '{tree_id_field}' not found. Available fields: {available}")
    
    unique_trees = [tid for tid in np.unique(tree_ids) if tid > 0]
    n_trees = len(unique_trees)
    print(f"  Found {n_trees} trees (treeID > 0)")
    print(f"  Non-tree points (treeID=0): {np.sum(tree_ids == 0):,}")
    
    # -------------------------------------------------------------------------
    # Extract per-tree features
    # -------------------------------------------------------------------------
    print("\nExtracting per-tree features...")
    tree_features_list = []
    
    for i, tid in enumerate(unique_trees):
        mask = tree_ids == tid
        tree_points = points[mask]
        
        features = extract_tree_features(tree_points, tree_id=tid)
        if features is not None:
            tree_features_list.append(features)
        
        if (i + 1) % 50 == 0 or (i + 1) == n_trees:
            print(f"  Processed {i+1}/{n_trees} trees")
    
    tree_df = pd.DataFrame(tree_features_list)
    
    # -------------------------------------------------------------------------
    # Extract plot-level features
    # -------------------------------------------------------------------------
    print("\nExtracting plot-level features...")
    plot_features = extract_plot_features(
        all_points=points,
        tree_ids=tree_ids,
        tree_features_list=tree_features_list,
        plot_id=basename
    )
    plot_df = pd.DataFrame([plot_features])
    
    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    tree_output = output_dir / f"{basename}_tree_features.csv"
    plot_output = output_dir / f"{basename}_plot_features.csv"
    
    tree_df.to_csv(tree_output, index=False)
    plot_df.to_csv(plot_output, index=False)
    
    print(f"\nOutputs saved:")
    print(f"  Tree features ({len(tree_df)} trees, {len(tree_df.columns)} columns): {tree_output}")
    print(f"  Plot features (1 row, {len(plot_df.columns)} columns): {plot_output}")
    
    return tree_df, plot_df


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract geometric features from LAZ point cloud for tree species classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_features.py forest_plot.laz
  python extract_features.py forest_plot.laz --output-dir ./results
  python extract_features.py forest_plot.laz --tree-id-field TreeID
        """
    )
    
    parser.add_argument('input', help='Input LAZ/LAS file path')
    parser.add_argument('--output-dir', '-o', help='Output directory (default: same as input)')
    parser.add_argument('--tree-id-field', '-t', default='treeID',
                        help='Name of tree ID attribute (default: treeID)')
    
    args = parser.parse_args()
    
    try:
        tree_df, plot_df = process_laz_file(
            input_path=args.input,
            output_dir=args.output_dir,
            tree_id_field=args.tree_id_field
        )
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\nPer-tree features: {len(tree_df.columns) - 2} features + tree_id + n_points")
        print(f"Plot features: {len(plot_df.columns) - 1} features + plot_id")
        
        if len(tree_df) > 0:
            print(f"\nTree height range: {tree_df['h_max'].min():.2f} - {tree_df['h_max'].max():.2f} m")
            print(f"Mean tree height: {tree_df['h_max'].mean():.2f} m")
            print(f"Mean points per tree: {tree_df['n_points'].mean():.0f}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
