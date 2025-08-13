import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import time
import random
import gc
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchio as tio
import warnings
warnings.filterwarnings("ignore")

# --- Enhanced Configuration ---
DATA_DIR = "/content/drive/MyDrive/InputScans_Final"
MODEL_SAVE_PATH = "/content/drive/MyDrive/best_model_3d_final_v3.pt"
PLOT_SAVE_PATH = "/content/drive/MyDrive/training_plots_v3.png"

# --- GPU & Memory Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CACHED_SCANS = 150  # Cache nearly all training scans to eliminate cache misses
NUM_WORKERS = 2     # can try 2 if the code is too slow - decreased from 4

# --- Enhanced Training Hyperparameters ---
EPOCHS = 300  # Reduced from 300 - larger patches converge faster
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2 # Effective batch size = 2 (balanced performance/memory)
INITIAL_LR = 1e-4
WEIGHT_DECAY = 1e-6
EARLY_STOPPING_PATIENCE = 30
PATCHES_PER_SCAN_TRAIN = 8  # Reduced from 8 (50% fewer batches)
PATCHES_PER_SCAN_VAL = 4    # Reduced from 4 
# --- FIX: Increased the minimum number of scans per epoch for better generalization ---
MIN_SCANS_PER_EPOCH = 64

# --- Model Settings ---
IN_CHANNELS = 5
OUT_CHANNELS = 5
PATCH_SIZE = (144, 144, 144)  # UPDATED: Increased from (112, 112, 112)
DROPOUT_RATE = 0.2
GRADIENT_CLIP_VAL = 1.0

# --- FIX: Replaced manual augmentation with a more powerful torchio pipeline ---
def get_torchio_transform():
    """Defines the advanced augmentation pipeline using torchio."""
    return tio.Compose([
        tio.RandomAffine(scales=(0.9, 1.2), degrees=15, translation=10, p=0.75),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=7.5, p=0.5),
        tio.RandomAnisotropy(p=0.2),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),
        tio.RandomNoise(p=0.25),
        tio.RandomFlip(axes=(0, 1, 2), p=0.75),
    ])

class ImprovedVolumetricDataset(Dataset):
    def __init__(self, file_paths, patch_size, patches_per_scan=8, is_training=True, use_fixed_validation=False):
        self.file_paths = file_paths
        self.patch_size = patch_size
        self.patches_per_scan = patches_per_scan
        self.is_training = is_training
        self.scan_cache = {}
        self.cache_order = []
        self.transform = get_torchio_transform() if is_training else None
        
        if use_fixed_validation:
            self.scans_to_process = file_paths
        else:
            # --- FIXED: Use a consistent subset to maximize cache hits ---
            scans_to_use = max(MIN_SCANS_PER_EPOCH, int(len(file_paths) * 0.75))
            # Use the FIRST N scans consistently instead of random sampling
            self.scans_to_process = file_paths[:min(len(file_paths), scans_to_use)]

    def _load_scan(self, path):
        if path in self.scan_cache:
            # Move to end for LRU management
            if path in self.cache_order:  # Safety check
                self.cache_order.remove(path)
            self.cache_order.append(path)
            return self.scan_cache[path]
        try:
            data = torch.load(path, map_location='cpu')
            
            # Validate loaded data
            if not isinstance(data, dict) or 'image' not in data or 'label' not in data:
                print(f"Warning: Invalid data format in {path}")
                return None
                
            # Validate tensor shapes
            image, label = data['image'], data['label']
            if len(image.shape) != 4 or len(label.shape) != 3:
                print(f"Warning: Invalid tensor shapes in {path}: image {image.shape}, label {label.shape}")
                return None
            
            # More aggressive cache management
            while len(self.scan_cache) >= MAX_CACHED_SCANS:
                if self.cache_order:  # Safety check
                    oldest_path = self.cache_order.pop(0)
                    if oldest_path in self.scan_cache:
                        del self.scan_cache[oldest_path]
                else:
                    break
            
            self.scan_cache[path] = data
            self.cache_order.append(path)
            return data
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def __len__(self):
        return len(self.scans_to_process) * self.patches_per_scan

    def __getitem__(self, idx):
        scan_idx = idx // self.patches_per_scan
        path = self.scans_to_process[scan_idx]
        scan_data = self._load_scan(path)
        if scan_data is None:
            return torch.zeros(IN_CHANNELS, *self.patch_size), torch.zeros(*self.patch_size, dtype=torch.long)

        image, label = scan_data['image'].float(), scan_data['label'].long().unsqueeze(0) # Add channel dim for torchio

        # FIXED: Check and pad volumes smaller than patch size
        image_shape = image.shape[1:]  # Get spatial dimensions (C, H, W, D)
        label_shape = label.shape[1:]  # Get spatial dimensions (1, H, W, D)
        
        # Calculate padding needed for each dimension
        pad_needed = []
        for i, (img_dim, patch_dim) in enumerate(zip(image_shape, self.patch_size)):
            if img_dim < patch_dim:
                pad_before = (patch_dim - img_dim) // 2
                pad_after = patch_dim - img_dim - pad_before
                pad_needed.extend([pad_before, pad_after])
            else:
                pad_needed.extend([0, 0])
        
        # Apply padding if needed (pad format: [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back])
        if any(p > 0 for p in pad_needed):
            # Reverse order for F.pad (it expects last dim first)
            pad_reversed = [pad_needed[4], pad_needed[5], pad_needed[2], pad_needed[3], pad_needed[0], pad_needed[1]]
            image = F.pad(image, pad_reversed, mode='constant', value=0)
            label = F.pad(label, pad_reversed, mode='constant', value=0)

        # Create a torchio Subject
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image),
            label=tio.LabelMap(tensor=label)
        )

        # Use torchio's patch sampler - now guaranteed to work
        sampler = tio.data.UniformSampler(self.patch_size)
        patch = next(iter(sampler(subject)))

        # Apply augmentations if in training mode
        if self.is_training and self.transform:
            patch = self.transform(patch)

        image_patch = patch['image'].data
        label_patch = patch['label'].data.squeeze(0)

        # Normalize the final patch
        for c in range(image_patch.shape[0]):
            modality = image_patch[c]
            p99 = torch.quantile(modality, 0.99)
            p1 = torch.quantile(modality, 0.01)
            image_patch[c] = torch.clamp((modality - p1) / (p99 - p1 + 1e-6), 0, 1)

        return image_patch, label_patch.long()

# --- Enhanced Architecture Components ---
class AdaptiveLayerNorm3d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        dims = [2, 3, 4]
        mean = x.mean(dims, keepdim=True)
        var = x.var(dims, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight.view(1, -1, 1, 1, 1) * x + self.bias.view(1, -1, 1, 1, 1)

class AdvancedConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.norm1 = AdaptiveLayerNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = AdaptiveLayerNorm3d(out_channels)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Conv3d(out_channels, out_channels//8, 1),
            nn.GELU(), nn.Conv3d(out_channels//8, out_channels, 1), nn.Sigmoid()
        )
        
        self.residual = nn.Conv3d(in_channels, out_channels, 1, stride) if in_channels != out_channels or stride != 1 else nn.Identity()
        self.dropout = nn.Dropout3d(DROPOUT_RATE)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = self.residual(x)
        out = self.gelu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out = out * self.se(out)
        out = out + residual
        return self.gelu(out)

class ImprovedAttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv3d(F_g, F_int, 1, bias=False)
        self.W_x = nn.Conv3d(F_l, F_int, 1, bias=False)
        self.norm_g = AdaptiveLayerNorm3d(F_int)
        self.norm_x = AdaptiveLayerNorm3d(F_int)
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, 1), nn.Sigmoid())
        self.gelu = nn.GELU()

    def forward(self, g, x):
        g1 = self.norm_g(self.W_g(g))
        x1 = self.norm_x(self.W_x(x))
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        psi = self.psi(self.gelu(g1 + x1))
        return x * psi

# --- FIXED: Multi-Scale Feature Pyramid Network ---
class MultiScaleFusionModule(nn.Module):
    def __init__(self, encoder_channels=[32, 64, 128, 256, 384]):
        super().__init__()
        # Use a common fusion dimension that's reasonable for all scales
        self.fusion_dim = 64
        
        # Lateral convolutions to bring all features to the same dimension
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, self.fusion_dim, 1, bias=False),
                AdaptiveLayerNorm3d(self.fusion_dim),
                nn.GELU()
            ) for ch in encoder_channels
        ])
        
        # Output convolutions to restore original channel dimensions
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(self.fusion_dim, ch, 1, bias=False),
                AdaptiveLayerNorm3d(ch),
                nn.GELU()
            ) for ch in encoder_channels
        ])
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(encoder_channels) - 1))  # -1 because bottleneck doesn't get fused
        
    def forward(self, encoder_features):
        # encoder_features = [e1, e2, e3, e4, bottleneck] from smallest to largest feature maps
        # Convert all features to common dimension
        laterals = []
        for feat, lateral_conv in zip(encoder_features, self.lateral_convs):
            laterals.append(lateral_conv(feat))
        
        # Top-down pathway starting from bottleneck
        enhanced_laterals = [laterals[-1]]  # Start with bottleneck (no fusion needed)
        
        # Fuse features from top to bottom
        for i in range(len(laterals) - 2, -1, -1):  # Go from 4th to 1st feature
            # Get the higher-level feature and upsample it
            higher_level = enhanced_laterals[0]
            current_level = laterals[i]
            
            # Upsample higher level to match current level spatial dimensions
            if higher_level.shape[2:] != current_level.shape[2:]:
                upsampled = F.interpolate(higher_level, size=current_level.shape[2:], 
                                        mode='trilinear', align_corners=False)
            else:
                upsampled = higher_level
            
            # Weighted fusion
            weight = torch.sigmoid(self.fusion_weights[i])
            fused = weight * current_level + (1 - weight) * upsampled
            enhanced_laterals.insert(0, fused)
        
        # Convert back to original channel dimensions
        enhanced_features = []
        for enhanced_lateral, output_conv in zip(enhanced_laterals, self.output_convs):
            enhanced_features.append(output_conv(enhanced_lateral))
        
        return enhanced_features

class EnhancedDenseTrans3D(nn.Module):
    def __init__(self, in_channels, out_channels, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        filters = [32, 64, 128, 256, 384]
        
        self.down = nn.MaxPool3d(2, stride=2)
        
        self.enc1 = AdvancedConvBlock3D(in_channels, filters[0])
        self.enc2 = AdvancedConvBlock3D(filters[0], filters[1])
        self.enc3 = AdvancedConvBlock3D(filters[1], filters[2])
        self.enc4 = AdvancedConvBlock3D(filters[2], filters[3])
        self.bottleneck = AdvancedConvBlock3D(filters[3], filters[4])
        
        # FIXED: Multi-scale feature pyramid with proper channel handling
        self.fpn = MultiScaleFusionModule(filters)
        
        self.up4 = nn.ConvTranspose3d(filters[4], filters[3], 2, 2)
        self.att4 = ImprovedAttentionGate3D(filters[3], filters[3], filters[3]//2)
        self.dec4 = AdvancedConvBlock3D(filters[3] * 2, filters[3])
        
        self.up3 = nn.ConvTranspose3d(filters[3], filters[2], 2, 2)
        self.att3 = ImprovedAttentionGate3D(filters[2], filters[2], filters[2]//2)
        self.dec3 = AdvancedConvBlock3D(filters[2] * 2, filters[2])
        
        self.up2 = nn.ConvTranspose3d(filters[2], filters[1], 2, 2)
        self.att2 = ImprovedAttentionGate3D(filters[1], filters[1], filters[1]//2)
        self.dec2 = AdvancedConvBlock3D(filters[1] * 2, filters[1])
        
        self.up1 = nn.ConvTranspose3d(filters[1], filters[0], 2, 2)
        self.att1 = ImprovedAttentionGate3D(filters[0], filters[0], filters[0]//2)
        self.dec1 = AdvancedConvBlock3D(filters[0] * 2, filters[0])
        
        if deep_supervision:
            self.out4 = nn.Conv3d(filters[3], out_channels, 1)
            self.out3 = nn.Conv3d(filters[2], out_channels, 1)
            self.out2 = nn.Conv3d(filters[1], out_channels, 1)
        
        self.final_out = nn.Conv3d(filters[0], out_channels, 1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        e4 = self.enc4(self.down(e3))
        bottleneck = self.bottleneck(self.down(e4))
        
        # FIXED: Multi-scale feature enhancement with proper channel dimensions
        encoder_features = [e1, e2, e3, e4, bottleneck]
        enhanced_features = self.fpn(encoder_features)
        e1, e2, e3, e4, bottleneck = enhanced_features
        
        # Decoder path with enhanced features
        d4 = self.up4(bottleneck)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))
        
        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))
        
        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))
        
        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))
        
        outputs = [self.final_out(d1)]
        
        if self.deep_supervision:
            out4 = F.interpolate(self.out4(d4), size=d1.shape[2:], mode='trilinear', align_corners=False)
            out3 = F.interpolate(self.out3(d3), size=d1.shape[2:], mode='trilinear', align_corners=False)
            out2 = F.interpolate(self.out2(d2), size=d1.shape[2:], mode='trilinear', align_corners=False)
            outputs = [out4, out3, out2, outputs[0]]
        
        return outputs

class AsymmetricFocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs_soft = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=OUT_CHANNELS).permute(0, 4, 1, 2, 3).float()
        
        dims = (0, 2, 3, 4)
        
        tp = torch.sum(inputs_soft * targets_one_hot, dims)
        fp = torch.sum(inputs_soft * (1 - targets_one_hot), dims)
        fn = torch.sum((1 - inputs_soft) * targets_one_hot, dims)
        
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow(1 - tversky_index, self.gamma)
        
        return focal_tversky.mean()

def calculate_metrics(pred, target, num_classes=OUT_CHANNELS):
    pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
    dice_scores, iou_scores = [], []
    
    for i in range(1, num_classes):
        pred_i, target_i = (pred == i).float(), (target == i).float()
        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum()
        
        if union > 0:
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            iou = intersection / (union - intersection + 1e-6)
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
    
    return {'dice': np.mean(dice_scores) if dice_scores else 0.0, 'iou': np.mean(iou_scores) if iou_scores else 0.0}

def save_training_plots(train_history, save_path):
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(train_history['loss'], label='Train Loss')
        axes[0].plot(train_history['val_loss'], label='Val Loss')
        axes[0].set_title('Loss Curves'); axes[0].legend(); axes[0].grid(True)
        
        axes[1].plot(train_history['val_dice'], label='Val Dice')
        axes[1].plot(train_history['val_iou'], label='Val IoU')
        axes[1].set_title('Validation Metrics'); axes[1].legend(); axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"Training plots saved to {save_path}")
    except ImportError:
        print("Matplotlib not found. Skipping plot generation.")

def validate_data_files(file_paths, patch_size, num_to_check=5):
    """Validate a sample of data files to check for common issues."""
    print(f"Validating {num_to_check} sample files...")
    
    for i, path in enumerate(file_paths[:num_to_check]):
        try:
            data = torch.load(path, map_location='cpu')
            if not isinstance(data, dict) or 'image' not in data or 'label' not in data:
                print(f"Invalid format: {path}")
                continue
                
            image, label = data['image'], data['label']
            print(f"File {i+1}: {path}")
            print(f"   Image shape: {image.shape}, Label shape: {label.shape}")
            
            # Check if any dimension is smaller than patch size
            spatial_dims = image.shape[1:]  # Skip channel dimension
            small_dims = [dim < patch_dim for dim, patch_dim in zip(spatial_dims, patch_size)]
            if any(small_dims):
                print(f"   Small dimensions detected: {spatial_dims} < {patch_size}")
                print(f"   Will be automatically padded during training.")
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    print("Data validation complete.\n")

def main():
    print("Starting Enhanced 3D DenseTrans Training with FPN and Larger Patches")
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found!")
        return
    
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt")))
    if len(all_files) == 0:
        print(f"Error: No .pt files found in {DATA_DIR}")
        return
    
    # Validate sample data files
    validate_data_files(all_files, PATCH_SIZE)
    
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    print(f"Train: {len(train_files)} scans, Val: {len(val_files)} scans")
    print(f"Using patch size: {PATCH_SIZE}")

    # Initialize model and check for CUDA availability
    print(f"Using device: {DEVICE}")
    model = EnhancedDenseTrans3D(IN_CHANNELS, OUT_CHANNELS).to(DEVICE)
    print(f"Model: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M params")

    # Initialize training components
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-7)
    loss_fn = AsymmetricFocalTverskyLoss().to(DEVICE)
    scaler = GradScaler()
    
    best_val_dice = 0.0
    patience_counter = 0
    train_history = {'loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
    
    # Create validation dataset
    val_dataset = ImprovedVolumetricDataset(val_files, PATCH_SIZE, patches_per_scan=PATCHES_PER_SCAN_VAL, is_training=False, use_fixed_validation=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Training phase
        model.train()
        train_dataset = ImprovedVolumetricDataset(train_files, PATCH_SIZE, patches_per_scan=PATCHES_PER_SCAN_TRAIN, is_training=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        
        train_loss, num_batches = 0.0, 0
        pbar = tqdm(train_loader, desc="Training")
        optimizer.zero_grad()
        
        for i, (inputs, targets) in enumerate(pbar):
            try:
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                
                with autocast(device_type=DEVICE.type):
                    outputs = model(inputs)
                    weights = [0.6, 0.8, 0.9, 1.0]
                    loss = sum(w * loss_fn(o, targets) for w, o in zip(weights, outputs)) / sum(weights)
                    loss = loss / GRADIENT_ACCUMULATION_STEPS

                if torch.isfinite(loss):
                    scaler.scale(loss).backward()
                    if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                    num_batches += 1
                else:
                    print(f"Warning: Non-finite loss detected at batch {i}, skipping...")
                    optimizer.zero_grad()
                
                pbar.set_postfix({'loss': f'{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}'})
                
            except RuntimeError as e:
                print(f"Training error at batch {i}: {e}")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
        
        if num_batches == 0:
            print("Warning: No valid batches processed in this epoch!")
            continue
            
        avg_train_loss = train_loss / num_batches

        # Validation phase
        model.eval()
        val_loss, val_metrics_agg = 0.0, {'dice': [], 'iou': []}
        val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                try:
                    inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                    with autocast(device_type=DEVICE.type):
                        outputs = model(inputs)
                        val_loss += loss_fn(outputs[-1], targets).item()
                    
                    metrics = calculate_metrics(outputs[-1], targets)
                    for k in val_metrics_agg: 
                        val_metrics_agg[k].append(metrics[k])
                    val_batches += 1
                    
                except RuntimeError as e:
                    print(f"Validation error: {e}")
                    torch.cuda.empty_cache()
                    continue
        
        if val_batches == 0:
            print("Warning: No valid validation batches processed!")
            continue
            
        avg_val_loss = val_loss / val_batches
        avg_val_metrics = {k: np.mean(v) for k, v in val_metrics_agg.items()}
        
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        print(f'Val Dice: {avg_val_metrics["dice"]:.4f} | Val IoU: {avg_val_metrics["iou"]:.4f} | LR: {current_lr:.2e}')
        
        # Save training history
        train_history['loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        train_history['val_dice'].append(avg_val_metrics['dice'])
        train_history['val_iou'].append(avg_val_metrics['iou'])

        # Model checkpointing and early stopping
        if avg_val_metrics['dice'] > best_val_dice:
            best_val_dice = avg_val_metrics['dice']
            patience_counter = 0
            try:
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"Model saved with Dice: {best_val_dice:.4f}")
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch+1} epochs.")
                break
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\nTraining completed! Best validation Dice: {best_val_dice:.4f}")
    save_training_plots(train_history, PLOT_SAVE_PATH)

if __name__ == '__main__':
    main()
