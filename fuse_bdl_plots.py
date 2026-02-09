#!/usr/bin/env python3
"""
Fuse TLS plot locations with BDL forest subdivision data.

For each TLS plot point, performs a spatial join against G_SUBAREA polygons
from all BDL districts, then enriches with all available attributes from
the BDL relational text tables (subarea details, storey layers, species
composition) and resolves coded values via dictionary lookup tables.

Output is a long-form table: one row per plot x storey x species combination.

Usage:
    python fuse_bdl_plots.py
    python fuse_bdl_plots.py --output results/fused_dataset.csv
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"
BDL_DIR = DATA_DIR / "BDL"
PLOTS_CSV = DATA_DIR / "TreeScanPL_plot_locations.csv"

# PUWG 1992 EPSG code
CRS_PUWG92 = "EPSG:2180"


# =============================================================================
# LOADING HELPERS
# =============================================================================

def load_all_subareas(bdl_dir: Path) -> gpd.GeoDataFrame:
    """Load G_SUBAREA.shp from every BDL subdirectory and concatenate."""
    frames = []
    for subdir in sorted(bdl_dir.iterdir()):
        shp_path = subdir / "G_SUBAREA.shp"
        if not shp_path.exists():
            continue
        gdf = gpd.read_file(shp_path)
        gdf["bdl_source"] = subdir.name
        frames.append(gdf)
        print(f"  Loaded {len(gdf):>5} subareas from {subdir.name}")

    all_subareas = pd.concat(frames, ignore_index=True)
    all_subareas = all_subareas.set_crs(CRS_PUWG92, allow_override=True)

    # Strip whitespace from string columns carried over from DBF
    str_cols = all_subareas.select_dtypes(include="object").columns
    for col in str_cols:
        all_subareas[col] = all_subareas[col].str.strip()

    print(f"  Total subareas: {len(all_subareas)}")
    return all_subareas


def load_plots(csv_path: Path) -> gpd.GeoDataFrame:
    """Load TLS plot locations CSV and return a GeoDataFrame in PUWG 1992."""
    df = pd.read_csv(csv_path, sep=";")
    geometry = [Point(xy) for xy in zip(df["X"], df["Y"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_PUWG92)
    print(f"  Loaded {len(gdf)} plot locations")
    return gdf


def load_tab_file(path: Path) -> pd.DataFrame:
    """Load a tab-delimited BDL text file, stripping whitespace from values."""
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def load_bdl_text_tables(bdl_dir: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load all relational text files from each BDL subdirectory.

    Returns a dict keyed by subdirectory name, each containing a dict of
    table_name -> DataFrame.
    """
    tables_by_source = {}
    for subdir in sorted(bdl_dir.iterdir()):
        if not (subdir / "G_SUBAREA.shp").exists():
            continue

        tables = {}
        for txt_file in subdir.glob("f_*.txt"):
            table_name = txt_file.stem  # e.g. "f_subarea"
            tables[table_name] = load_tab_file(txt_file)

        tables_by_source[subdir.name] = tables

    return tables_by_source


# =============================================================================
# DICTIONARY RESOLUTION
# =============================================================================

def build_dict_lookup(dic_df: pd.DataFrame, code_col: str, name_col: str) -> dict:
    """
    Build a code -> name mapping from a dictionary table.
    Uses only current entries (date_to is empty/NaN).
    """
    if "date_to" in dic_df.columns:
        current = dic_df[dic_df["date_to"].isna() | (dic_df["date_to"] == "")]
    else:
        current = dic_df
    return dict(zip(current[code_col], current[name_col]))


# =============================================================================
# MAIN FUSION LOGIC
# =============================================================================

def fuse_data(
    plots: gpd.GeoDataFrame,
    subareas: gpd.GeoDataFrame,
    tables_by_source: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """
    Spatial join plots to subareas, then enrich with all BDL text tables.
    Returns a long-form DataFrame (one row per plot x storey x species).
    """

    # --- 1. Spatial join: point-in-polygon -----------------------------------
    print("\nPerforming spatial join...")
    joined = gpd.sjoin(plots, subareas, how="left", predicate="within")

    n_matched = joined["a_i_num"].notna().sum()
    n_missed = joined["a_i_num"].isna().sum()
    print(f"  Matched: {n_matched} / {len(plots)} plots")
    if n_missed > 0:
        missed = joined[joined["a_i_num"].isna()]
        print(f"  Unmatched plots ({n_missed}):")
        for _, row in missed.iterrows():
            print(f"    {row['file']} num={row['num']} X={row['X']:.1f} Y={row['Y']:.1f}")

    # Keep relevant columns from the spatial join
    plot_cols = ["source", "file", "year", "num", "num_txt", "X", "Y"]
    subarea_shape_cols = [
        "a_i_num", "adr_for", "area_type", "site_type", "silvicult",
        "forest_fun", "stand_stru", "rotat_age", "sub_area", "prot_categ",
        "species_cd", "part_cd", "spec_age", "a_year", "bdl_source",
    ]
    keep_cols = plot_cols + [c for c in subarea_shape_cols if c in joined.columns]
    result = pd.DataFrame(joined[keep_cols])

    # --- 2. Enrich with f_subarea.txt (1:1) ----------------------------------
    print("Enriching with f_subarea.txt...")
    subarea_extras = []
    for src_name, tables in tables_by_source.items():
        if "f_subarea" in tables:
            df = tables["f_subarea"].copy()
            df["bdl_source"] = src_name
            subarea_extras.append(df)

    if subarea_extras:
        sub_df = pd.concat(subarea_extras, ignore_index=True)
        # Columns already present from the shapefile — drop to avoid duplication
        shapefile_overlap = {"a_year", "area_type_cd", "site_type_cd",
                             "stand_struct_cd", "forest_func_cd",
                             "silviculture_cd", "rotation_age", "sub_area"}
        extra_cols = [c for c in sub_df.columns
                      if c not in shapefile_overlap and c not in ("bdl_source",)]
        merge_cols = ["arodes_int_num", "bdl_source"] + [
            c for c in extra_cols if c != "arodes_int_num"
        ]
        sub_merge = sub_df[merge_cols].drop_duplicates()

        result = result.merge(
            sub_merge,
            left_on=["a_i_num", "bdl_source"],
            right_on=["arodes_int_num", "bdl_source"],
            how="left",
        )
        result.drop(columns=["arodes_int_num"], inplace=True, errors="ignore")

    # --- 3. Enrich with f_arod_storey.txt (1:N per subarea) ------------------
    print("Enriching with f_arod_storey.txt (storey layers)...")
    storey_frames = []
    for src_name, tables in tables_by_source.items():
        if "f_arod_storey" in tables:
            df = tables["f_arod_storey"].copy()
            df["bdl_source"] = src_name
            storey_frames.append(df)

    if storey_frames:
        storey_df = pd.concat(storey_frames, ignore_index=True)
        storey_df.rename(columns={
            "density_cd": "storey_density_cd",
            "mixture_cd": "storey_mixture_cd",
            "standdensity_index": "storey_density_index",
            "tree_stock_cd": "storey_tree_stock_cd",
            "st_rank_order_act": "storey_rank",
            "storey_cd": "storey_cd",
        }, inplace=True)
        storey_merge_cols = [
            "arodes_int_num", "bdl_source", "storey_cd", "storey_rank",
            "storey_density_cd", "storey_mixture_cd", "storey_density_index",
            "storey_tree_stock_cd",
        ]
        storey_merge = storey_df[[c for c in storey_merge_cols
                                   if c in storey_df.columns]].drop_duplicates()

        result = result.merge(
            storey_merge,
            left_on=["a_i_num", "bdl_source"],
            right_on=["arodes_int_num", "bdl_source"],
            how="left",
        )
        result.drop(columns=["arodes_int_num"], inplace=True, errors="ignore")

    # --- 4. Enrich with f_storey_species.txt (1:N per storey) ----------------
    print("Enriching with f_storey_species.txt (species per storey)...")
    species_frames = []
    for src_name, tables in tables_by_source.items():
        if "f_storey_species" in tables:
            df = tables["f_storey_species"].copy()
            df["bdl_source"] = src_name
            species_frames.append(df)

    if species_frames:
        sp_df = pd.concat(species_frames, ignore_index=True)
        sp_df.rename(columns={
            "sp_rank_order_act": "species_rank",
            "species_cd": "sp_species_cd",
            "species_age": "sp_age",
            "part_cd_act": "sp_part_cd",
            "site_class_cd": "sp_site_class",
            "height": "sp_height",
            "bhd": "sp_bhd",
            "volume": "sp_volume",
        }, inplace=True)
        sp_merge_cols = [
            "arodes_int_num", "bdl_source", "storey_cd", "species_rank",
            "sp_species_cd", "sp_age", "sp_part_cd", "sp_site_class",
            "sp_height", "sp_bhd", "sp_volume",
        ]
        sp_merge = sp_df[[c for c in sp_merge_cols
                          if c in sp_df.columns]].drop_duplicates()

        result = result.merge(
            sp_merge,
            left_on=["a_i_num", "bdl_source", "storey_cd"],
            right_on=["arodes_int_num", "bdl_source", "storey_cd"],
            how="left",
        )
        result.drop(columns=["arodes_int_num"], inplace=True, errors="ignore")

    # --- 5. Enrich with f_arod_category.txt (protection categories, 1:N) ----
    print("Enriching with f_arod_category.txt (protection categories)...")
    cat_frames = []
    for src_name, tables in tables_by_source.items():
        if "f_arod_category" in tables:
            df = tables["f_arod_category"].copy()
            df["bdl_source"] = src_name
            cat_frames.append(df)

    if cat_frames:
        cat_df = pd.concat(cat_frames, ignore_index=True)
        # Aggregate multiple protection categories into a single string per subarea
        cat_agg = (
            cat_df.groupby(["arodes_int_num", "bdl_source"])["prot_category_cd"]
            .apply(lambda x: "; ".join(sorted(x.dropna().unique())))
            .reset_index()
            .rename(columns={"prot_category_cd": "prot_categories_all"})
        )
        result = result.merge(
            cat_agg,
            left_on=["a_i_num", "bdl_source"],
            right_on=["arodes_int_num", "bdl_source"],
            how="left",
        )
        result.drop(columns=["arodes_int_num"], inplace=True, errors="ignore")

    # --- 6. Resolve coded values via dictionary tables -----------------------
    print("Resolving coded values via dictionaries...")

    # We only need one copy of each dictionary (they're identical across districts)
    first_source = next(iter(tables_by_source.values()))

    # Tree species dictionary
    if "f_tree_species_dic" in first_source:
        sp_dic = first_source["f_tree_species_dic"]
        species_lookup = build_dict_lookup(sp_dic, "species_cd", "latin_name")
        species_name_lookup = build_dict_lookup(sp_dic, "species_cd", "species_name")
        wood_kind_lookup = build_dict_lookup(sp_dic, "species_cd", "wood_kind_fl")

        # Resolve the dominant species from the shapefile
        result["species_cd_latin"] = result["species_cd"].map(species_lookup)
        result["species_cd_name"] = result["species_cd"].map(species_name_lookup)

        # Resolve the per-storey species
        if "sp_species_cd" in result.columns:
            result["sp_species_latin"] = result["sp_species_cd"].map(species_lookup)
            result["sp_species_name"] = result["sp_species_cd"].map(species_name_lookup)
            result["sp_wood_kind"] = result["sp_species_cd"].map(wood_kind_lookup)

    # Site type dictionary
    if "f_site_type_dic" in first_source:
        st_dic = first_source["f_site_type_dic"]
        site_lookup = build_dict_lookup(st_dic, "site_type_cd", "site_type_name")
        result["site_type_name"] = result["site_type"].map(site_lookup)

    # Area type dictionary
    if "f_area_type_dic" in first_source:
        at_dic = first_source["f_area_type_dic"]
        area_lookup = build_dict_lookup(at_dic, "area_type_cd", "area_type_name")
        result["area_type_name"] = result["area_type"].map(area_lookup)

    # Forest function dictionary
    if "f_forest_func_dic" in first_source:
        ff_dic = first_source["f_forest_func_dic"]
        func_lookup = build_dict_lookup(ff_dic, "forest_func_cd", "forest_func_name")
        result["forest_func_name"] = result["forest_fun"].map(func_lookup)

    # Density dictionary
    if "f_density_dic" in first_source:
        dens_dic = first_source["f_density_dic"]
        dens_lookup = build_dict_lookup(dens_dic, "density_cd", "density_name")
        if "storey_density_cd" in result.columns:
            result["storey_density_name"] = result["storey_density_cd"].map(dens_lookup)

    # Storey dictionary
    if "f_storey_dic" in first_source:
        stor_dic = first_source["f_storey_dic"]
        stor_lookup = build_dict_lookup(stor_dic, "storey_cd", "storey_name")
        if "storey_cd" in result.columns:
            result["storey_name"] = result["storey_cd"].map(stor_lookup)

    # Mixture dictionary
    if "f_mixture_dic" in first_source:
        mix_dic = first_source["f_mixture_dic"]
        mix_lookup = build_dict_lookup(mix_dic, "mixture_cd", "mixture_name")
        if "storey_mixture_cd" in result.columns:
            result["storey_mixture_name"] = result["storey_mixture_cd"].map(mix_lookup)

    # Part (share proportion) dictionary
    if "f_part_dic" in first_source:
        part_dic = first_source["f_part_dic"]
        part_lookup = build_dict_lookup(part_dic, "part_cd", "part_name")
        if "sp_part_cd" in result.columns:
            result["sp_part_name"] = result["sp_part_cd"].map(part_lookup)

    # Protection category dictionary
    if "f_prot_categ_dic" in first_source:
        prot_dic = first_source["f_prot_categ_dic"]
        prot_lookup = build_dict_lookup(prot_dic, "prot_category_cd", "prot_category_name")
        if "prot_categ" in result.columns:
            result["prot_categ_name"] = result["prot_categ"].map(prot_lookup)

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fuse TLS plot locations with BDL forest subdivision attributes."
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DATA_DIR / "plots_bdl_fused.csv"),
        help="Output CSV path (default: data/plots_bdl_fused.csv)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BDL × TLS Plot Data Fusion")
    print("=" * 60)

    # Load spatial data
    print("\nLoading subarea polygons...")
    subareas = load_all_subareas(BDL_DIR)

    print("\nLoading plot locations...")
    plots = load_plots(PLOTS_CSV)

    # Load all text tables
    print("\nLoading BDL text tables...")
    tables_by_source = load_bdl_text_tables(BDL_DIR)
    for src, tables in tables_by_source.items():
        print(f"  {src}: {len(tables)} tables")

    # Fuse
    result = fuse_data(plots, subareas, tables_by_source)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, sep=";")

    print(f"\nOutput saved to {output_path}")
    print(f"  Rows: {len(result)}")
    print(f"  Columns: {len(result.columns)}")
    print(f"  Unique plots: {result['num'].nunique()}")

    # Summary of unmatched
    unmatched = result[result["a_i_num"].isna()]
    if len(unmatched) > 0:
        print(f"  Plots without subarea match: {unmatched['num'].nunique()}")


if __name__ == "__main__":
    main()
