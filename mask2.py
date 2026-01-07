import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import os
import tifffile
import shutil

# Geospatial processing logic adapted from Rasterio documentation and standard workflows

# Project Root
root = r'/home/w/ws161/industry_project'

# Output Folder Name
output_folder_name = "Final_UNET_data"

# Thresholds for Ground Truth Masks
NBR_THRESHOLD = 0.1       # dNBR > 0.1 = Fire
NDMI_DIFF_THRESHOLD = 0.05 # dNDMI > 0.05 = Fire (Moisture Loss)
NDMI_A2_THRESHOLD = 0.05   # A2 NDMI < 0.05 = Fire (Dry/Burnt)

def align_and_crop(source_path, template_path):
    """
    Reads a source raster and reprojects/crops it to match the 
    template raster's exact grid, CRS, and dimensions.
    """
    with rasterio.open(template_path) as template_ds:
        dst_crs = template_ds.crs
        dst_transform = template_ds.transform
        dst_width = template_ds.width
        dst_height = template_ds.height
        dst_profile_data = template_ds.profile.copy()
        
    # Prepare profile for saving later
    dst_profile_data.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=None)
    
    # Create empty array for the result
    destination_array = np.zeros((dst_height, dst_width), dtype=rasterio.float32)

    with rasterio.open(source_path) as src:
        # Reproject source to match template
        reproject(
            source=rasterio.band(src, 1),
            destination=destination_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear 
        )
            
    return destination_array, dst_profile_data

# MAIN EXECUTION

print(f"Scanning {root} for events...")

for event_folder in os.listdir(root):
    if 'event_' not in event_folder:
        continue
    
    event_path = os.path.join(root, event_folder)
    
    # 1. FIND THE TEMPLATE FILE (Used for alignment)
    template_path = None
    for f in os.listdir(event_path):
        full_path = os.path.join(event_path, f)
        if os.path.isfile(full_path) and f.lower().endswith('_ndvi.tiff'):
            template_path = full_path
            break
            
    if not template_path:
        print(f"Skipping {event_folder}: No template file found.")
        continue

    print(f"\nProcessing {event_folder}...")

    # Create Main Output Directory
    final_output_dir = os.path.join(event_path, output_folder_name)
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    # ---------------------------------------------------------
    # PART 1: GENERATE MASKS (B2 - A2 ONLY)
    # ---------------------------------------------------------
    path_b2 = os.path.join(event_path, 'B2')
    path_a2 = os.path.join(event_path, 'A2')

    if os.path.exists(path_b2) and os.path.exists(path_a2):
        try:
            # --- NBR MASK (Change) ---
            src_nbr_b2 = os.path.join(path_b2, 'NBR.tiff')
            src_nbr_a2 = os.path.join(path_a2, 'NBR.tiff')

            if os.path.exists(src_nbr_b2) and os.path.exists(src_nbr_a2):
                arr_b2_nbr, profile = align_and_crop(src_nbr_b2, template_path)
                arr_a2_nbr, _ = align_and_crop(src_nbr_a2, template_path)
                
                dNBR = arr_b2_nbr - arr_a2_nbr
                mask_nbr = np.where(dNBR > NBR_THRESHOLD, 1.0, 0.0).astype(rasterio.float32)

                # Save NBR Mask
                with rasterio.open(os.path.join(final_output_dir, "Ground_Truth_Mask_NBR.tiff"), 'w', **profile) as dst:
                    dst.write(mask_nbr, 1)

            # --- NDMI MASKS (Change & A2 Threshold) ---
            src_ndmi_b2 = os.path.join(path_b2, 'NDMI.tiff')
            src_ndmi_a2 = os.path.join(path_a2, 'NDMI.tiff')

            if os.path.exists(src_ndmi_b2) and os.path.exists(src_ndmi_a2):
                arr_b2_ndmi, profile = align_and_crop(src_ndmi_b2, template_path)
                arr_a2_ndmi, _ = align_and_crop(src_ndmi_a2, template_path)

                # 1. NDMI Change Mask
                dNDMI = arr_b2_ndmi - arr_a2_ndmi
                mask_ndmi_change = np.where(dNDMI > NDMI_DIFF_THRESHOLD, 1.0, 0.0).astype(rasterio.float32)
                
                with rasterio.open(os.path.join(final_output_dir, "Ground_Truth_Mask_NDMI_Change.tiff"), 'w', **profile) as dst:
                    dst.write(mask_ndmi_change, 1)

                # 2. NDMI A2 Mask (A2 < Threshold)
                mask_ndmi_a2 = np.where(arr_a2_ndmi < NDMI_A2_THRESHOLD, 1.0, 0.0).astype(rasterio.float32)

                with rasterio.open(os.path.join(final_output_dir, "Ground_Truth_Mask_NDMI_A2.tiff"), 'w', **profile) as dst:
                    dst.write(mask_ndmi_a2, 1)

        except Exception as e:
            print(f"  Error generating masks for B2-A2: {e}")
    else:
        print("  Missing B2 or A2 folders, skipping mask generation.")


    # ---------------------------------------------------------
    # PART 2: SAVE CROPPED REGIONS (A1, A2, B1, B2)
    # ---------------------------------------------------------
    # Folders to process
    target_folders = ['A1', 'A2', 'B1', 'B2']
    # Indices to process
    target_indices = ['NDMI', 'NDVI', 'NDWI', 'NBR']

    for folder_name in target_folders:
        src_folder_path = os.path.join(event_path, folder_name)
        
        # Check if source folder exists
        if not os.path.exists(src_folder_path):
            continue

        # Create corresponding subfolder in Final_UNET_data
        dst_folder_path = os.path.join(final_output_dir, folder_name)
        if not os.path.exists(dst_folder_path):
            os.makedirs(dst_folder_path)

        # Loop through indices, crop, and save
        for index in target_indices:
            src_file = os.path.join(src_folder_path, f"{index}.tiff")
            
            if os.path.exists(src_file):
                try:
                    # Align and Crop
                    arr, profile = align_and_crop(src_file, template_path)
                    
                    # Save to subfolder
                    dst_file = os.path.join(dst_folder_path, f"{index}.tiff")
                    with rasterio.open(dst_file, 'w', **profile) as dst:
                        dst.write(arr, 1)
                except Exception as e:
                    print(f"  Error processing {folder_name}/{index}: {e}")

print("\nProcessing Complete.")
