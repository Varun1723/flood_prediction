import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from tqdm import tqdm

# Input path
VALID_PAIRS_FILE = "valid_label_sar_pairs.txt"

# Output root folder
OUTPUT_DIR = "processed_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_tif_as_array(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs, src.shape

def rasterize_label(label_path, transform, shape, crs):
    gdf = gpd.read_file(label_path)
    if gdf.empty:
        return np.zeros(shape, dtype=np.uint8)
    gdf = gdf.to_crs(crs)
    mask = rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    return mask

def preprocess_all():
    with open(VALID_PAIRS_FILE, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing triplets"):
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue

        vv_path, vh_path, label_path = parts

        if not (os.path.exists(vv_path) and os.path.exists(vh_path) and os.path.exists(label_path)):
            print(f"Missing file(s) for: {vv_path}, {vh_path}, {label_path}")
            continue

        try:
            vv_array, transform, crs, shape = load_tif_as_array(vv_path)
            vh_array, _, _, _ = load_tif_as_array(vh_path)
            label_array = rasterize_label(label_path, transform, shape, crs)
        except Exception as e:
            print(f"Failed to process {vv_path}: {e}")
            continue

        # Create a folder using the VV path (e.g., sen12floods_s1_source_0023_2019_01_18)
        folder_name = os.path.basename(os.path.dirname(vv_path))
        output_path = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(output_path, exist_ok=True)

        np.save(os.path.join(output_path, "VV.npy"), vv_array)
        np.save(os.path.join(output_path, "VH.npy"), vh_array)
        np.save(os.path.join(output_path, "label.npy"), label_array)

    print(f"\n✅ All files processed and saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    preprocess_all()
