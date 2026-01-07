import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
import csv
import random
import gc
from collections import defaultdict
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Loss function logic adapted from: 
# https://github.com/shuaizzZ/Dice-Loss-PyTorch 
# https://github.com/itakurah/focal-loss-pytorch
# https://www.geeksforgeeks.org/deep-learning/how-to-process-multiple-losses-in-pytorch/


# --- CONFIGURATION ---
SRC_ROOT = r'/home/w/ws161/industry_project'
OUT_ROOT = r'/home/w/ws161/industry_project/tile_outputs'

# --- Dataset Sizing Metrics ---
TARGET_TRAIN_SIZE = 1000        # Total images desired in Train Set (Used if Ratio = y)
TARGET_TEST_SIZE = 120          # Total images desired in Test Set
TRAIN_BLANK_RATIO = 0.5         # 0.5 = 50% Blank, 50% Fire in Train Set (Used if Ratio = y)

# Toggle for Ratio/Sizing Logic (y/n)
# y = Force balanced dataset using metrics above.
# n = Ignore metrics. Use ALL available tiles with full augmentation.
USE_BALANCED_RATIO = 'n' 

# Fixed Test Set Composition (as requested)
TEST_BLANK_COUNT = 20           # 20 Blank images for testing
TEST_FIRE_COUNT = TARGET_TEST_SIZE - TEST_BLANK_COUNT # Remaining 100 are Fire

# Tiling Settings
TILE_SIZE = 256
STRIDE = 256

# Noise Filter: Minimum number of fire pixels to count as a "Fire Tile"
MIN_FIRE_PIXELS = 50

# Targets to Tile
SUBFOLDERS = ['A1', 'A2', 'B1', 'B2']
BANDS = ['NDVI.tiff', 'NDWI.tiff', 'NDMI.tiff', 'NBR.tiff']
MASKS = [
    'Ground_Truth_Mask_NBR.tiff', 
    'Ground_Truth_Mask_NDMI_Change.tiff',
    'Ground_Truth_Mask_NDMI_A2.tiff'
]

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, padding="same", dropout=0.0):
        super().__init__()
        pad = 1 if padding == "same" else padding
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=pad)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if (dropout and dropout > 0.0) else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_chans, out_chans, layers=2, sampling_factor=2, padding="same", dropout=0.0):
        super().__init__()
        blocks = []
        blocks.append(ConvBlock(in_chans, out_chans, padding=padding, dropout=dropout))
        for _ in range(layers - 1):
            blocks.append(ConvBlock(out_chans, out_chans, padding=padding, dropout=dropout))
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.MaxPool2d(kernel_size=sampling_factor)

    def forward(self, x):
        feat = self.encoder(x)
        pooled = self.pool(feat)
        return pooled, feat

class DecoderBlock(nn.Module):
    def __init__(self, in_chans, out_chans, layers=2, sampling_factor=2, padding="same", dropout=0.0, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.tconv = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=sampling_factor, stride=sampling_factor)
        conv_in = (in_chans // 2) + (out_chans if use_skip else 0)
        blocks = []
        blocks.append(ConvBlock(conv_in, out_chans, padding=padding, dropout=dropout))
        for _ in range(layers - 1):
            blocks.append(ConvBlock(out_chans, out_chans, padding=padding, dropout=dropout))
        self.decoder = nn.Sequential(*blocks)

    def forward(self, x, skip_feat=None):
        x = self.tconv(x)
        if self.use_skip and skip_feat is not None:
            if skip_feat.size(-2) != x.size(-2) or skip_feat.size(-1) != x.size(-1):
                _, _, sh, sw = skip_feat.shape
                _, _, h, w = x.shape
                top = (sh - h) // 2
                left = (sw - w) // 2
                skip_feat = skip_feat[:, :, top:top+h, left:left+w]
            x = torch.cat([skip_feat, x], dim=1)
        x = self.decoder(x)
        return x

class SingleHeadUNet(nn.Module):
    def __init__(self,
                 in_chans=7,
                 base_filters=32,
                 depth=5,
                 layers=2,
                 sampling_factor=2,
                 padding="same",
                 dropout=0.05,
                 use_skip=True):
        super().__init__()

        self.use_skip = use_skip

        # encoder
        self.encoder_blocks = nn.ModuleList()
        in_c = in_chans
        filters = base_filters
        for _ in range(depth):
            self.encoder_blocks.append(EncoderBlock(in_c, filters, layers=layers, sampling_factor=sampling_factor, padding=padding, dropout=dropout))
            in_c = filters
            filters = filters * 2

        # bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(in_c, in_c * 2, padding=padding, dropout=dropout),
            ConvBlock(in_c * 2, in_c * 2, padding=padding, dropout=dropout)
        )
        bottleneck_channels = in_c * 2

        # decoder
        self.decoder_blocks = nn.ModuleList()
        in_ch = bottleneck_channels
        out_ch = in_ch // 2
        
        for _ in range(depth): 
            self.decoder_blocks.append(DecoderBlock(in_ch, out_ch, layers=layers, sampling_factor=sampling_factor, padding=padding, dropout=dropout, use_skip=use_skip))
            in_ch = out_ch
            out_ch = max(out_ch // 2, base_filters)

        # final conv to 1 logit channel
        self.logits = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        out = x
        for enc in self.encoder_blocks:
            out, skip = enc(out)
            skips.append(skip)
        out = self.bottleneck(out)

        # decode
        for i, db in enumerate(self.decoder_blocks):
            skip_idx = -(i + 1)
            skip_feat = skips[skip_idx] if self.use_skip else None
            out = db(out, skip_feat)

        logits = self.logits(out)
        return logits

# ==========================================
# 2. DATASET CLASS
# ==========================================

class FirePredictionDataset(Dataset):
    def __init__(self, root_dir=None, sample_list=None, target_mask_name="Ground_Truth_Mask_NBR.tiff"):
        self.target_mask_name = target_mask_name
        self.samples = []
        
        if sample_list is not None:
            self.samples = sample_list
        elif root_dir is not None and os.path.exists(root_dir):
            for folder_name in sorted(os.listdir(root_dir)):
                folder_path = os.path.join(root_dir, folder_name)
                if os.path.isdir(folder_path):
                    self.samples.append(folder_path)
        
        if len(self.samples) == 0:
            print(f"Warning: No data found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path = self.samples[idx]
        subfolders = ['B1', 'B2'] 
        bands = ['NDVI.tiff', 'NDWI.tiff', 'NDMI.tiff', 'NBR.tiff']
        
        layers = []
        try:
            for sub in subfolders:
                for band in bands:
                    band_path = os.path.join(folder_path, sub, band)
                    with rasterio.open(band_path) as src:
                        data = src.read(1)
                        data = np.nan_to_num(data, nan=0.0)
                        layers.append(data)
            
            x = np.stack(layers, axis=0)
            x = torch.from_numpy(x).float()
            
            mask_path = os.path.join(folder_path, self.target_mask_name)
            with rasterio.open(mask_path) as src:
                y = src.read(1)
                y = np.nan_to_num(y, nan=0.0)
            
            y = (y > 0).astype(np.float32)
            y = torch.from_numpy(y).unsqueeze(0) 

            return x, y

        except Exception as e:
            print(f"Error loading {folder_path}: {e}")
            return torch.zeros((8, 256, 256)), torch.zeros((1, 256, 256))
    
    @property
    def num_channels(self):
        return 8 

# ==========================================
# 3. LOSS FUNCTIONS & METRICS
# ==========================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean": return focal.mean()
        if self.reduction == "sum": return focal.sum()
        return focal

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.alpha = alpha
    def forward(self, logits, targets):
        return self.alpha * self.focal(logits, targets) + (1.0 - self.alpha) * self.dice(logits, targets)

def iou_score_single(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    inter = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets - preds * targets).sum(dim=(1,2,3))
    return (inter + 1e-6) / (union + 1e-6)

# ==========================================
# 4. TRAINING LOOP
# ==========================================

def train_singlehead(
    train_ds,
    val_ds,
    in_chans=8, 
    base_filters=32,
    depth=4,         
    layers=2,        
    epochs=60,
    batch_size=8,
    lr=1e-4,
    combo_alpha=0.5, 
    device=None,
    save_dir="unet_epochs_single"
):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        pin_memory = True
    else:
        device = torch.device("cpu")
        pin_memory = False 

    os.makedirs(save_dir, exist_ok=True)
    
    log_file_path = os.path.join(save_dir, "training_log.csv")
    with open(log_file_path, "w") as f:
        f.write("Epoch,Train_Loss,Test_Loss,Test_IoU\n")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    model = SingleHeadUNet(in_chans=in_chans, base_filters=base_filters, depth=depth, layers=layers)
    model.to(device)

    criterion = ComboLoss(alpha=combo_alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_iou = -1.0
    best_metrics = {}
    
    train_loss_history = []
    val_loss_history = []
    val_iou_history = []

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / max(1, len(train_loader))
            train_loss_history.append(avg_train_loss)

            model.eval()
            val_loss_sum = 0.0
            val_iou_sum = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    val_loss_sum += loss.item()
                    val_iou_sum += iou_score_single(logits, y).mean().item()
            
            avg_val_loss = val_loss_sum / max(1, len(val_loader))
            avg_val_iou = val_iou_sum / max(1, len(val_loader))
            
            val_loss_history.append(avg_val_loss)
            val_iou_history.append(avg_val_iou)

            print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Test Loss {avg_val_loss:.4f} | Test IoU {avg_val_iou:.4f}")

            with open(log_file_path, "a") as f:
                f.write(f"{epoch},{avg_train_loss:.5f},{avg_val_loss:.5f},{avg_val_iou:.5f}\n")

            if avg_val_iou > best_val_iou:
                best_val_iou = avg_val_iou
                best_metrics = {
                    'best_iou': best_val_iou,
                    'best_test_loss': avg_val_loss,
                    'best_train_loss': avg_train_loss,
                    'epoch': epoch
                }
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            
            # Save latest checkpoint overwriting previous to save space
            torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))

        # Plotting
        epochs_range = range(1, epochs + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, train_loss_history, label='Train Loss', color='blue')
        plt.plot(epochs_range, val_loss_history, label='Test Loss', color='orange')
        plt.title("Training vs Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "loss_curve.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, val_iou_history, label='Test IoU', color='green')
        plt.title("Test IoU Score per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("IoU Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "iou_curve.png"))
        plt.close()

        return best_metrics

    finally:
        del model
        del optimizer
        del train_loader
        del val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# ==========================================
# 5. EXPERIMENT RUNNER
# ==========================================

def get_event_tile_metadata(event_name, event_path):
    source_dir = os.path.join(event_path, 'Final_UNET_data')
    if not os.path.exists(source_dir):
        return [], []

    ref_path = os.path.join(source_dir, MASKS[0])
    if not os.path.exists(ref_path):
        ref_path = os.path.join(source_dir, 'A2', 'NDVI.tiff')
        if not os.path.exists(ref_path):
            return [], []

    fire_meta = []
    blank_meta = []

    try:
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
                    fire_count = np.sum(mask_data > 0)
                    
                    meta = {
                        'event_name': event_name,
                        'event_path': event_path,
                        'tile_id': tile_id_counter,
                        'window': window,
                        'rotation': 0, 
                        'flip_h': False 
                    }

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
    source_dir = os.path.join(event_path, 'Final_UNET_data')
    
    ref_path = os.path.join(source_dir, MASKS[0])
    if not os.path.exists(ref_path): 
        ref_path = os.path.join(source_dir, 'A2', 'NDVI.tiff')

    with rasterio.open(ref_path) as ref:
        transform = ref.transform
        profile = ref.profile.copy()

    handles = {'masks': {}, 'folders': {f: {} for f in SUBFOLDERS}}
    try:
        for m_name in MASKS:
            p = os.path.join(source_dir, m_name)
            if os.path.exists(p): handles['masks'][m_name] = rasterio.open(p)

        for sub in SUBFOLDERS:
            for band in BANDS:
                p = os.path.join(source_dir, sub, band)
                if os.path.exists(p): handles['folders'][sub][band] = rasterio.open(p)

        for i, task in enumerate(tasks):
            window = task['window']
            subset_name = task['set_type']
            rotation = task.get('rotation', 0)
            flip_h = task.get('flip_h', False)

            tile_dir_name = f"{event_name}_{i:03d}"
            tile_out_path = os.path.join(out_root, subset_name, tile_dir_name)
            os.makedirs(tile_out_path, exist_ok=True)

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
                
                if flip_h: data = np.flip(data, axis=1)
                if rotation > 0: data = np.rot90(data, k=rotation)
                
                out_p = os.path.join(dest_folder, filename)
                with rasterio.open(out_p, 'w', **tile_profile) as dst:
                    dst.write(data, 1)

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

def run_learning_curve_experiment(
    full_train_path, 
    test_path,
    output_root,
    sizes=None
):
    print("Starting Learning Curve Experiment...")
    os.makedirs(output_root, exist_ok=True)
    
    all_samples = []
    if os.path.exists(full_train_path):
        for f in os.listdir(full_train_path):
            p = os.path.join(full_train_path, f)
            if os.path.isdir(p):
                all_samples.append(p)
    else:
        print("Train path not found!")
        return

    fire_pool = []
    blank_pool = []
    
    print("  Categorizing samples...")
    for s in tqdm(all_samples):
        mask_p = os.path.join(s, "Ground_Truth_Mask_NBR.tiff")
        try:
            with rasterio.open(mask_p) as src:
                data = src.read(1)
                fire_pixels = np.sum(data > 0)
                
                
                if fire_pixels >= MIN_FIRE_PIXELS:
                    fire_pool.append(s)
                else:
                    blank_pool.append(s)
        except:
            pass

    random.seed(42)
    random.shuffle(fire_pool)
    random.shuffle(blank_pool)
    
    print(f"  Pool: {len(fire_pool)} Fire, {len(blank_pool)} Blank")

    
    if USE_BALANCED_RATIO == 'y':
        max_balanced_size = min(len(fire_pool), len(blank_pool)) * 2
        print(f"  Maximum possible balanced dataset size: {max_balanced_size}")
    else:
        # If ignoring ratio, everything can be used
        max_balanced_size = len(fire_pool) + len(blank_pool)
        print(f"  Using ALL available data: {max_balanced_size} (Imbalanced allowed)")

    if sizes is None:
        print("  No specific sizes requested. Running SINGLE experiment on MAX available dataset.")
        sizes = [max_balanced_size]

    summary_csv = os.path.join(output_root, "experiment_summary.csv")
    if not os.path.exists(summary_csv):
        with open(summary_csv, "w") as f:
            f.write("Dataset_Size,Best_IoU,Best_Test_Loss,Best_Train_Loss,Best_Epoch\n")

    summary_data = {'sizes': [], 'ious': [], 'test_losses': [], 'train_losses': []}
    has_run_max = False

    for size in sizes:
        actual_size = size
        is_max_run = False
        
        if size > max_balanced_size:
            if has_run_max:
                print(f"\n=== Size {size} exceeds max ({max_balanced_size}) and max already run. Skipping. ===")
                continue
            
            print(f"\n=== Warning: Requested {size} > Available {max_balanced_size}. Clamping to {max_balanced_size}. ===")
            actual_size = max_balanced_size
            is_max_run = True
            
        print(f"\n=== Checking Status for Size: {actual_size} ===")
        run_dir = os.path.join(output_root, f"run_{actual_size:04d}_samples")
        log_file = os.path.join(run_dir, "training_log.csv")
        
        is_complete = False
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 61: 
                    is_complete = True
        
        if is_complete:
            print(f"  Run {actual_size} seems complete. Loading metrics...")
            best_iou_found = -1.0
            best_test_loss_found = 0.0
            best_train_loss_found = 0.0
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    iou = float(row['Test_IoU'])
                    if iou > best_iou_found:
                        best_iou_found = iou
                        best_test_loss_found = float(row['Test_Loss'])
                        best_train_loss_found = float(row['Train_Loss'])
            
            summary_data['sizes'].append(actual_size)
            summary_data['ious'].append(best_iou_found)
            summary_data['test_losses'].append(best_test_loss_found)
            summary_data['train_losses'].append(best_train_loss_found)
            if is_max_run: has_run_max = True
            continue 
        
        print(f"  Starting/Restarting Training for Size: {actual_size}")
        
        if USE_BALANCED_RATIO == 'y':
            n_fire = actual_size // 2
            n_blank = actual_size - n_fire
            current_samples = fire_pool[:n_fire] + blank_pool[:n_blank]
        else:
            # Random mix of everything
            combined_pool = fire_pool + blank_pool
            random.shuffle(combined_pool) # Re-shuffle combined
            current_samples = combined_pool[:actual_size]
        
        train_ds = FirePredictionDataset(sample_list=current_samples, target_mask_name="Ground_Truth_Mask_NBR.tiff")
        val_ds = FirePredictionDataset(root_dir=test_path, target_mask_name="Ground_Truth_Mask_NBR.tiff")
        
        metrics = train_singlehead(
            train_ds=train_ds,
            val_ds=val_ds,
            in_chans=8,
            base_filters=32,
            depth=4,
            layers=2,
            batch_size=8,
            lr=1e-4,
            epochs=60,
            combo_alpha=0.5,
            save_dir=run_dir
        )
        
        del train_ds
        del val_ds
        gc.collect()
        
        print(f"  Result: Best IoU {metrics['best_iou']:.4f}")
        with open(summary_csv, "a") as f:
            f.write(f"{actual_size},{metrics['best_iou']:.4f},{metrics['best_test_loss']:.4f},{metrics['best_train_loss']:.4f},{metrics['epoch']}\n")
            
        summary_data['sizes'].append(actual_size)
        summary_data['ious'].append(metrics['best_iou'])
        summary_data['test_losses'].append(metrics['best_test_loss'])
        summary_data['train_losses'].append(metrics['best_train_loss'])
        
        if is_max_run:
            has_run_max = True

    if len(summary_data['sizes']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(summary_data['sizes'], summary_data['ious'], marker='o', color='green')
        plt.title("Model Performance vs Dataset Size")
        plt.xlabel("Number of Training Images")
        plt.ylabel("Best Test IoU")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_root, "summary_iou_vs_size.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(summary_data['sizes'], summary_data['test_losses'], marker='o', color='orange', label='Test Loss')
        plt.plot(summary_data['sizes'], summary_data['train_losses'], marker='o', color='blue', label='Train Loss')
        plt.title("Loss vs Dataset Size")
        plt.xlabel("Number of Training Images")
        plt.ylabel("Best Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_root, "summary_loss_vs_size.png"))
        plt.close()
    
    print("\nExperiment Complete. Summary saved.")

if __name__ == "__main__":
    if not os.path.exists(SRC_ROOT):
        print(f"Error: Source root {SRC_ROOT} does not exist.")
        exit()

    # PHASE 1-3: GLOBAL TILING
    
    train_set_path = os.path.join(OUT_ROOT, "train_set")
    test_set_path = os.path.join(OUT_ROOT, "test_set")

    # PRESERVE DATA IF IT EXISTS
    data_exists = False
    
    # Check if folders exist and contain data
    if os.path.exists(train_set_path) and os.path.exists(test_set_path):
        if len(os.listdir(train_set_path)) > 0:
            data_exists = True
            
    if data_exists:
        print("\n=== Existing Tiled Data Found. SKIPPING PHASES 1-3 (Tiling). ===")
        print(f"Using Train Data: {train_set_path}")
        print(f"Using Test Data:  {test_set_path}")
    else:
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

        # PHASE 2: GLOBAL SPLIT & AUGMENTATION
        print("PHASE 2: Performing Global Split & Augmentation...")
        random.shuffle(global_fire_tiles)
        random.shuffle(global_blank_tiles)

        # --- 1. Test Set Allocation ---
        if len(global_fire_tiles) < TEST_FIRE_COUNT:
            print(f"Warning: Not enough fire tiles for test set! Have {len(global_fire_tiles)}, need {TEST_FIRE_COUNT}")
            test_fire = global_fire_tiles[:]
            remaining_fire = []
        else:
            test_fire = global_fire_tiles[:TEST_FIRE_COUNT]
            remaining_fire = global_fire_tiles[TEST_FIRE_COUNT:]

        test_blank = global_blank_tiles[:TEST_BLANK_COUNT]
        remaining_blank = global_blank_tiles[TEST_BLANK_COUNT:]

        # --- 2. Train Set Allocation with Augmentation ---
        
        def fill_quota(source_tiles, quota, label="Fire"):
            tasks = []
            pool = source_tiles[:]
            random.shuffle(pool)
            
            candidates = []
            for tile in pool:
                for flip in [False, True]:
                    for rot in [0, 1, 2, 3]:
                        new_task = tile.copy()
                        new_task['rotation'] = rot
                        new_task['flip_h'] = flip
                        candidates.append(new_task)
            
            random.shuffle(candidates)
            
            if len(candidates) >= quota:
                tasks = candidates[:quota]
            else:
                print(f"Warning: Could not fill {label} quota! Requested {quota}, max unique augmented is {len(candidates)}.")
                tasks = candidates
                
            return tasks

        if USE_BALANCED_RATIO == 'y':
            target_train_blank = int(TARGET_TRAIN_SIZE * TRAIN_BLANK_RATIO)
            target_train_fire = TARGET_TRAIN_SIZE - target_train_blank
            
            train_fire_tasks = fill_quota(remaining_fire, target_train_fire, "Train Fire")
            train_blank_tasks = fill_quota(remaining_blank, target_train_blank, "Train Blank")
            print(f" Train Set: {len(train_fire_tasks)} Fire + {len(train_blank_tasks)} Blank (Target: {TARGET_TRAIN_SIZE})")
            
        else: # 'n' = PROCESS EVERYTHING
            print("  Ignoring Target Size/Ratio. Using ALL available tiles with full augmentation.")
            
            # Max possible augmentation for remaining items
            max_fire_quota = len(remaining_fire) * 8
            max_blank_quota = len(remaining_blank) * 8
            
            train_fire_tasks = fill_quota(remaining_fire, max_fire_quota, "Train Fire (All)")
            train_blank_tasks = fill_quota(remaining_blank, max_blank_quota, "Train Blank (All)")
            print(f" Train Set (All): {len(train_fire_tasks)} Fire + {len(train_blank_tasks)} Blank")

        print(f"\nFinal Splits:")
        print(f"Test Set:  {len(test_fire)} Fire + {len(test_blank)} Blank")

        tasks_by_event = defaultdict(list)

        for t in test_fire + test_blank:
            t['set_type'] = 'test_set'
            tasks_by_event[t['event_name']].append(t)
        
        for t in train_fire_tasks + train_blank_tasks:
            t['set_type'] = 'train_set'
            tasks_by_event[t['event_name']].append(t)

        print("\nPHASE 3: Writing Tiled Output...")
        
        if os.path.exists(train_set_path): shutil.rmtree(train_set_path)
        if os.path.exists(test_set_path): shutil.rmtree(test_set_path)
        
        os.makedirs(train_set_path)
        os.makedirs(test_set_path)

        for event_name, tasks in tqdm(tasks_by_event.items()):
            event_path = tasks[0]['event_path'] 
            write_tiles_for_event(event_name, event_path, tasks, OUT_ROOT)

    print("\nGlobal Tiling Pipeline Complete. Proceeding to Training Experiment...")
    
    # PHASE 4: LEARNING CURVE EXPERIMENT
    experiment_root = "/home/w/ws161/industry_project/experiments_output"
    
    run_learning_curve_experiment(
        full_train_path=train_set_path,
        test_path=test_set_path,
        output_root=experiment_root
    )
