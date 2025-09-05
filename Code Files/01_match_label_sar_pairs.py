import os
import json
from tqdm import tqdm

# Paths
COLLECTION_JSON = "sen12flood/sen12floods_s1_labels/sen12floods_s1_labels/collection.json"
LABELS_ROOT = "sen12flood/sen12floods_s1_labels/sen12floods_s1_labels"
SAR_SOURCE_ROOT = "sen12flood/sen12floods_s1_source/sen12floods_s1_source"
VALID_PAIRS_TXT = "valid_label_sar_pairs.txt"

def load_collection_items(collection_path):
    with open(collection_path, 'r') as f:
        collection = json.load(f)
    return [link['href'] for link in collection['links'] if link['rel'] == 'item']

def extract_tile_and_date_from_folder(folder_name):
    # Example: sen12floods_s1_labels_0200_2019_02_23
    parts = folder_name.split('_')
    tile_id = parts[-4]
    date = '_'.join(parts[-3:])  # 2019_02_23
    return tile_id, date

def match_labels_with_sar():
    collection_items = load_collection_items(COLLECTION_JSON)
    print(f"Total STAC entries in collection: {len(collection_items)}")

    matched = []

    for item_href in tqdm(collection_items):
        stac_path = os.path.join(LABELS_ROOT, item_href)
        if not os.path.exists(stac_path):
            continue

        folder_name = os.path.basename(os.path.dirname(stac_path))
        tile_id, date = extract_tile_and_date_from_folder(folder_name)

        label_path = os.path.join(LABELS_ROOT, folder_name, "labels.geojson")

        # Match SAR folder like: sen12floods_s1_source_0200_2019_02_23
        sar_folder = f"sen12floods_s1_source_{tile_id}_{date}"
        sar_path = os.path.join(SAR_SOURCE_ROOT, sar_folder)
        vv_path = os.path.join(sar_path, "VV.tif")
        vh_path = os.path.join(sar_path, "VH.tif")

        if os.path.exists(label_path) and os.path.exists(vv_path) and os.path.exists(vh_path):
            matched.append((vv_path, vh_path, label_path))

    # Save matched pairs
    with open(VALID_PAIRS_TXT, 'w') as f:
        for vv, vh, label in matched:
            f.write(f"{vv},{vh},{label}\n")

    print(f"\n✅ Total valid (VV, VH, label) triplets found: {len(matched)}")
    print(f"Saved to: {VALID_PAIRS_TXT}")

if __name__ == "__main__":
    match_labels_with_sar()
