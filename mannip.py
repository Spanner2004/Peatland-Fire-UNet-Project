import rasterio

from rasterio.plot import show

from matplotlib import pyplot as plt

import tifffile

import os

import numpy as np

# Geospatial processing logic adapted from Rasterio documentation and standard workflows

# MAIN FOLDER PATH

root = r'/Users/williamspanswick/Desktop/University/Industry_Project'



# Labels for folders sorted by date: Oldest -> Newest

folder_labels = ['B2', 'B1', 'A1', 'A2']



print(f"Starting search in root: {root}")



# 1. Loop through all folders in the root directory looking for 'event_'

for event_folder in os.listdir(root):

    if 'event_' not in event_folder:

        continue



    event_path = os.path.join(root, event_folder)

    if not os.path.isdir(event_path):

        continue



    # 2. Find all Sentinel SAFE folders (S2A, S2B, S2C)

    safe_folders = [f for f in os.listdir(event_path) 

                    if f.startswith(('S2A', 'S2B', 'S2C')) 

                    and os.path.isdir(os.path.join(event_path, f))]



    if not safe_folders:

        continue



    # Store tuples of (date_string, folder_name) to sort them

    dated_folders = []



    for safe_folder in safe_folders:

        try:

            # Extract date (Assuming format S2A_..._20230322T...)

            date_str = safe_folder.split('_')[2].split('T')[0] 

            dated_folders.append((date_str, safe_folder))

        except IndexError:

            print(f"Could not extract date from {safe_folder}")

            continue



    # Sort the folders by date (Oldest to Newest)

    dated_folders.sort()



    # 3. Process the folders in order and assign labels

    for i, (date_str, safe_folder) in enumerate(dated_folders):

        

        if i >= len(folder_labels):

            print(f"Skipping {safe_folder} (limit reached)")

            continue



        current_label = folder_labels[i]



        # Construct path: GRANULE -> [Subfolder] -> IMG_DATA -> R20m

        granule_path = os.path.join(event_path, safe_folder, 'GRANULE')

        

        if not os.path.exists(granule_path):

            print(f"DEBUG: GRANULE folder missing in {safe_folder}")

            continue

        

        # Ignore hidden files like .DS_Store

        sub_granules = [f for f in os.listdir(granule_path) if not f.startswith('.')]

        

        if not sub_granules:

            print(f"DEBUG: No valid subfolder inside GRANULE for {safe_folder}")

            continue

        

        target_sub_granule = sub_granules[0]

        img_data_path = os.path.join(granule_path, target_sub_granule, 'IMG_DATA', 'R20m')



        if not os.path.exists(img_data_path):

            print(f"!!! PATH ERROR in {safe_folder} !!! Path not found: {img_data_path}")

            continue



        # 4. Find the specific band files

        band_paths = {}

        target_bands = ['B03', 'B04', 'B8A', 'B11', 'B12']

        

        files_in_r20 = os.listdir(img_data_path)

        

        for file in files_in_r20:

            for band in target_bands:

                if f"_{band}_20m.jp2".lower() in file.lower():

                    band_paths[band] = os.path.join(img_data_path, file)



        if len(band_paths) < 5:

            print(f"!!! MISSING BANDS in {safe_folder} !!!")

            continue



        print(f"Processing {safe_folder} ({date_str}) -> Saving to {current_label}...")



        try:

            with rasterio.open(band_paths['B8A']) as src_b8a:

                profile = src_b8a.profile

                bandb8 = src_b8a.read(1)



            # Update profile to standard float32 GeoTIFF

            profile.update(

                driver='GTiff',

                dtype=rasterio.float32,

                count=1,

                compress='lzw'

            )



            # Read other bands

            with rasterio.open(band_paths['B03']) as src: bandb3 = src.read(1)

            with rasterio.open(band_paths['B04']) as src: bandb4 = src.read(1)

            with rasterio.open(band_paths['B11']) as src: bandb11 = src.read(1)

            with rasterio.open(band_paths['B12']) as src: bandb12 = src.read(1)



            # Calculations

            with np.errstate(divide='ignore', invalid='ignore'):

                ndvi = (bandb8.astype(float) - bandb4.astype(float)) / (bandb8.astype(float) + bandb4.astype(float))

                ndwi = (bandb3.astype(float) - bandb8.astype(float)) / (bandb8.astype(float) + bandb3.astype(float))

                ndmi = (bandb8.astype(float) - bandb11.astype(float)) / (bandb8.astype(float) + bandb11.astype(float))

                nbr = (bandb8.astype(float) - bandb12.astype(float)) / (bandb8.astype(float) + bandb12.astype(float))



            # SAVING FILES AS GEOTIFFS

            output_dir = os.path.join(event_path, current_label)

            if not os.path.exists(output_dir):

                os.makedirs(output_dir)



            # Helper to save with rasterio (retaining metadata)

            def save_geotiff(filename, data, profile):

                path = os.path.join(output_dir, filename)

                with rasterio.open(path, 'w', **profile) as dst:

                    dst.write(data.astype(rasterio.float32), 1)



            save_geotiff('NDVI.tiff', ndvi, profile)

            save_geotiff('NDWI.tiff', ndwi, profile)

            save_geotiff('NDMI.tiff', ndmi, profile)

            save_geotiff('NBR.tiff', nbr, profile)

            

        except Exception as e:

            print(f"Error processing {safe_folder}: {e}")



print("Processing Complete.")
