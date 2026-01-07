import os
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict
import shutil

# Geospatial processing logic adapted from Rasterio documentation and standard workflows

SRC_ROOT = r'/Users/williamspanswick/Desktop/University/Industry_Project'
OUT_ROOT = r'/Users/williamspanswick/Desktop/University/Industry_Project/tile_outputs'

# Test set definition
GLOBAL_TEST_FIRE_COUNT = 30
GLOBAL_TEST_BLANK_COUNT = 5

# Training Subset sizes. Increase this depending on size of dataset.
TRAINING_SUBSET_STEPS = [50, 100, 150]

# Tiling Settings
TILE_SIZE = 256
STRIDE = 256

# Noise Dilter: Minimum number of fire pixels to count as a "Fire Tile"
MIN_FIRE_PIXELS = 50

# Targets to Tile
SUBFOLDERS = ['A1', 'A2', 'B1', 'B2']
BANDS = ['NDVI.tiff', 'NDWI.tiff', 'NDMI.tiff', 'NBR.tiff']
MASKS = [
    'Ground_Truth_Mask_NBR.tiff', 
    'Ground_Truth_Mask_NDMI_Change.tiff',
    'Ground_Truth_Mask_NDMI_A2.tiff'
]

def get_event_tile_metadata(event_name, event_path):
    """
    Scans a single event to identify valid tile windows and classify them (Fire/Blank).
    Does NOT write files. Returns lists of metadata.
    """
    source_dir = os.path.join(event_path, 'Final_UNET_data')
    if not os.path.exists(source_dir):
        return [], []

    # 1. Open Reference for Grid
    ref_path = os.path.join(source_dir, MASKS[0])
    if not os.path.exists(ref_path):
        ref_path = os.path.join(source_dir, 'A2', 'NDVI.tiff')
        if not os.path.exists(ref_path):
            return [], []

    fire_meta = []
    blank_meta = []

    try:
        # Check Primary Mask
        primary_mask_path = os.path.join(source_dir, 'Ground_Truth_Mask_NBR.tiff')
        if not os.path.exists(primary_mask_path):
            return [], []

        with rasterio.open(ref_path) as ref:
            width = ref.width
            height = ref.height
        
        with rasterio.open(primary_mask_path) as mask_handle:
            tile_id_counter = 0
            
            for row_off in range(0, height, STRIDE):
                for col_off in range(0, width, STRIDE):
                    
                    win_w = min(TILE_SIZE, width - col_off)
                    win_h = min(TILE_SIZE, height - row_off)
                    window = Window(col_off, row_off, win_w, win_h)
                    
                    # Read Mask to Classify
                    mask_data = mask_handle.read(1, window=window)
                    
                    # Count fire pixels
                    fire_count = np.sum(mask_data > 0)
                    
                    # Metadata object
                    meta = {
                        'event_name': event_name,
                        'event_path': event_path,
                        'tile_id': tile_id_counter,
                        'window': window
                    }

                    # APPLY NOISE FILTER
                    if fire_count >= MIN_FIRE_PIXELS:
                        fire_meta.append(meta)
                    else:
                        blank_meta.append(meta)
                    
                    tile_id_counter += 1
                    
    except Exception as e:
        print(f"Error scanning {event_name}: {e}")
        return [], []

    return fire_meta, blank_meta

def write_tiles_for_event(event_name, event_path, tasks, out_root):
    """
    Writes the specific tiles assigned to this event to disk.
    tasks: list of {'window': window, 'tile_id': id, 'set_type': 'train_set'/'test_set'}
    """
    source_dir = os.path.join(event_path, 'Final_UNET_data')
    
    # Open Reference for Profile
    ref_path = os.path.join(source_dir, MASKS[0])
    if not os.path.exists(ref_path): 
        # Fallback
        ref_path = os.path.join(source_dir, 'A2', 'NDVI.tiff')

    with rasterio.open(ref_path) as ref:
        transform = ref.transform
        profile = ref.profile.copy()

    # Open Handles
    handles = {'masks': {}, 'folders': {f: {} for f in SUBFOLDERS}}
    try:
        for m_name in MASKS:
            p = os.path.join(source_dir, m_name)
            if os.path.exists(p): handles['masks'][m_name] = rasterio.open(p)

        for sub in SUBFOLDERS:
            for band in BANDS:
                p = os.path.join(source_dir, sub, band)
                if os.path.exists(p): handles['folders'][sub][band] = rasterio.open(p)

        # Process Tasks
        for task in tasks:
            window = task['window']
            tile_id = task['tile_id']
            subset_name = task['set_type'] # 'train_set' or 'test_set'

            # Define Output Directory
            tile_dir_name = f"{event_name}_{tile_id:03d}"
            tile_out_path = os.path.join(out_root, subset_name, tile_dir_name)
            os.makedirs(tile_out_path, exist_ok=True)

            # Update Profile
            tile_transform = window_transform(window, transform)
            tile_profile = profile.copy()
            tile_profile.update({
                "height": TILE_SIZE,
                "width": TILE_SIZE,
                "transform": tile_transform,
                "count": 1,
                "compress": "lzw"
            })

            def write_tile(src_handle, dest_folder, filename):
                data = src_handle.read(1, window=window)
                if data.shape != (TILE_SIZE, TILE_SIZE):
                    pad = np.zeros((TILE_SIZE, TILE_SIZE), dtype=data.dtype)
                    pad[:data.shape[0], :data.shape[1]] = data
                    data = pad
                
                out_p = os.path.join(dest_folder, filename)
                with rasterio.open(out_p, 'w', **tile_profile) as dst:
                    dst.write(data, 1)

            # Write Data
            for m_name, handle in handles['masks'].items():
                write_tile(handle, tile_out_path, m_name)

            for sub in SUBFOLDERS:
                sub_out_path = os.path.join(tile_out_path, sub)
                os.makedirs(sub_out_path, exist_ok=True)
                for band, handle in handles['folders'][sub].items():
                    write_tile(handle, sub_out_path, band)

    finally:
        for h in handles['masks'].values(): h.close()
        for sub in handles['folders']:
            for h in handles['folders'][sub].values(): h.close()

def create_training_subsets(train_root, output_root, steps):
    """
    Creates cumulative subsets of the training data.
    """
    print(f"\nPHASE 4: Creating Training Subsets {steps}...")
    
    if not os.path.exists(train_root):
        print(f"Error: {train_root} does not exist.")
        return

    # Get all sample folders
    all_samples = sorted([f for f in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, f))])
    total_samples = len(all_samples)
    print(f"  Total training samples available: {total_samples}")

    # Shuffle once to ensure cumulative consistency
    random.shuffle(all_samples)

    for count in steps:
        folder_name = f"train_set_{count}"
        dest_dir = os.path.join(output_root, folder_name)
        
        # Clean up if exists
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir)

        # Cap count at total samples
        actual_count = min(count, total_samples)
        subset = all_samples[:actual_count]
        
        print(f"  Generating {folder_name} with {actual_count} samples...")
        
        for sample_name in tqdm(subset, leave=False):
            src_path = os.path.join(train_root, sample_name)
            dst_path = os.path.join(dest_dir, sample_name)
            shutil.copytree(src_path, dst_path)

    # Finally, rename the original full set to 'train_set_full'
    full_set_path = os.path.join(output_root, "train_set_full")
    if os.path.exists(full_set_path):
        shutil.rmtree(full_set_path)
    
    print(f"  Renaming original 'train_set' to 'train_set_full'...")
    os.rename(train_root, full_set_path)


if __name__ == "__main__":
    if not os.path.exists(SRC_ROOT):
        print(f"Error: Source root {SRC_ROOT} does not exist.")
        exit()

    print("PHASE 1: Scanning all events to catalog tiles...")
    print(f"Filter: Keeping Fire Tiles with >= {MIN_FIRE_PIXELS} pixels")
    
    events = sorted([d for d in os.listdir(SRC_ROOT) if d.startswith('event_') and os.path.isdir(os.path.join(SRC_ROOT, d))])
    
    global_fire_tiles = []
    global_blank_tiles = []

    for event in tqdm(events):
        path = os.path.join(SRC_ROOT, event)
        f, b = get_event_tile_metadata(event, path)
        global_fire_tiles.extend(f)
        global_blank_tiles.extend(b)

    print(f"\nTotal Found: {len(global_fire_tiles)} Fire Tiles (Filtered), {len(global_blank_tiles)} Blank Tiles")

    # PHASE 2: GLOBAL SPLIT
    print("PHASE 2: Performing Global Random Split...")
    random.shuffle(global_fire_tiles)
    random.shuffle(global_blank_tiles)

    # Allocating Test Set (Strict Limit)
    test_fire = global_fire_tiles[:GLOBAL_TEST_FIRE_COUNT]
    test_blank = global_blank_tiles[:GLOBAL_TEST_BLANK_COUNT]

    # Allocating Train Set (Everything Else)
    train_fire = global_fire_tiles[GLOBAL_TEST_FIRE_COUNT:]
    train_blank = global_blank_tiles[GLOBAL_TEST_BLANK_COUNT:]

    print(f"Test Set: {len(test_fire)} Fire + {len(test_blank)} Blank = {len(test_fire)+len(test_blank)} Tiles")
    print(f"Train Set: {len(train_fire)} Fire + {len(train_blank)} Blank = {len(train_fire)+len(train_blank)} Tiles")

    # Organize tasks by event for efficient writing
    tasks_by_event = defaultdict(list)

    for t in test_fire + test_blank:
        t['set_type'] = 'test_set'
        tasks_by_event[t['event_name']].append(t)
    
    for t in train_fire + train_blank:
        t['set_type'] = 'train_set'
        tasks_by_event[t['event_name']].append(t)

    # PHASE 3: WRITING
    print("\nPHASE 3: Writing Tiled Output...")
    train_set_path = os.path.join(OUT_ROOT, "train_set")
    test_set_path = os.path.join(OUT_ROOT, "test_set")
    
    # Fresh start for main folders
    if os.path.exists(train_set_path): shutil.rmtree(train_set_path)
    if os.path.exists(test_set_path): shutil.rmtree(test_set_path)
    
    os.makedirs(train_set_path)
    os.makedirs(test_set_path)

    for event_name, tasks in tqdm(tasks_by_event.items()):
        event_path = tasks[0]['event_path'] 
        write_tiles_for_event(event_name, event_path, tasks, OUT_ROOT)

    # PHASE 4: SUBSETS
    create_training_subsets(train_set_path, OUT_ROOT, TRAINING_SUBSET_STEPS)

    print("\nGlobal Pipeline Complete.")
