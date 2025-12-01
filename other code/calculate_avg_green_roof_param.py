import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd
import os

# --- CONFIGURATION ---
INPUT_GPKG = "data/hamburg_green_roofs.gpkg"
OUTPUT_GPKG = "data/hamburg_green_roofs_with_params.gpkg"

# Dictionary mapping your raster paths to the column name you want
# REPLACE 'path/to/ndvi.tif' with your actual file paths
raster_config = {
    "ndvi": "results_greening_hamburg/ndvi.tif",
    "ndbi": "results_greening_hamburg/ndbi.tif",
    "albedo": "results_greening_hamburg/albedo.tif"
}

def enrich_roofs_with_raster_data(gpkg_path, raster_dict, output_path):
    print("--- Loading Green Roofs ---")
    gdf = gpd.read_file(gpkg_path)
    print(f"Loaded {len(gdf)} buildings.")

    # Iterate through the rasters (NDVI, NDBI, Albedo)
    for param_name, raster_path in raster_dict.items():
        if not os.path.exists(raster_path):
            print(f"⚠️ Warning: Raster not found at {raster_path}. Skipping.")
            continue

        print(f"--- Processing {param_name.upper()} ---")
        
        # 1. Open Raster to check CRS (Coordinate Reference System)
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            affine = src.transform
            nodata_val = src.nodata
            
            # 2. Reproject Vector to match Raster (CRITICAL STEP)
            # If they don't match, the overlay will fail.
            if gdf.crs != raster_crs:
                print(f"Reprojecting vectors from {gdf.crs} to {raster_crs}...")
                gdf_projected = gdf.to_crs(raster_crs)
            else:
                gdf_projected = gdf

            # 3. Calculate Zonal Statistics
            # stats="mean" calculates the average pixel value within the polygon
            # all_touched=True ensures small roofs (smaller than 1 pixel) still get a value
            stats = zonal_stats(
                gdf_projected,
                raster_path,
                stats=["mean"],
                all_touched=True,
                nodata=nodata_val
            )

            # 4. Extract the 'mean' and add to the original GeoDataFrame
            # We use the index to map it back to the original gdf in case we reprojected
            column_name = f"mean_{param_name}"
            gdf[column_name] = [s['mean'] for s in stats]
            
            # Report
            valid_count = gdf[column_name].count()
            print(f"Added column '{column_name}'. Valid values found for {valid_count}/{len(gdf)} roofs.")

    # --- Final Clean up ---
    # Remove rows where we couldn't find data (optional, depends on your needs)
    # gdf = gdf.dropna(subset=[f'mean_{k}' for k in raster_dict.keys()])

    print(f"--- Saving to {output_path} ---")
    gdf.to_file(output_path, driver="GPKG")
    
    # Print Global Averages (The "Average of Averages")
    print("\n--- GLOBAL AVERAGES FOR THIS DAY ---")
    for param in raster_dict.keys():
        col = f"mean_{param}"
        if col in gdf.columns:
            global_avg = gdf[col].mean()
            print(f"Global Average {param.upper()}: {global_avg:.4f}")

# --- RUN ---
if __name__ == "__main__":
    # Ensure you have installed: pip install rasterstats rasterio geopandas
    enrich_roofs_with_raster_data(INPUT_GPKG, raster_config, OUTPUT_GPKG)