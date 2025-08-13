import torch
import torch.nn as nn
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
import warnings
warnings.filterwarnings("ignore")

# --- Enhanced Configuration ---
DATA_DIR = r"G:\My Drive\InputScans_Final"
MODEL_SAVE_PATH = "./best_model_3d_final_v3.pt"
PLOT_SAVE_PATH = "./training_plots_v3.png"

# --- GPU & Memory Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CACHED_SCANS = 16
# --- FIX: Set NUM_WORKERS to 0 to prevent multiprocessing issues on Windows ---
NUM_WORKERS = 0

# --- Enhanced Training Hyperparameters ---
EPOCHS = 200
# --- FIX: Use larger batch size with gradient accumulation ---
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1 # Effective batch size = 16
INITIAL_LR = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 30
PATCHES_PER_SCAN_TRAIN = 4
PATCHES_PER_SCAN_VAL = 4
MIN_SCANS_PER_EPOCH = 32

# --- Model Settings ---
IN_CHANNELS = 5
OUT_CHANNELS = 5
# --- FIX: Increased patch size to utilize VRAM ---
PATCH_SIZE = (112, 112, 112)
DROPOUT_RATE = 0.2
GRADIENT_CLIP_VAL = 1.0

class ImprovedVolumetricDataset(Dataset):
    # --- FIX: Added the missing 'use_fixed_validation' parameter ---
    def __init__(self, file_paths, patch_size, patches_per_scan=8, is_training=True, use_fixed_validation=False):
        self.file_paths = file_paths
        self.patch_size = patch_size
        self.patches_per_scan = patches_per_scan
        self.is_training = is_training
        self.scan_cache = {}
        self.cache_order = []
        
        if use_fixed_validation:
            self.scans_to_process = file_paths
        else:
            scans_to_use = max(MIN_SCANS_PER_EPOCH, len(file_paths) // 2)
            self.scans_to_process = random.sample(file_paths, min(len(file_paths), scans_to_use))

    def _enhanced_augmentation(self, image, label):
        if not self.is_training:
            return image, label
            
        if random.random() > 0.5:
            k = random.randint(1, 3)
            axes = random.choice([(1,2), (1,3), (2,3)])
            image = torch.rot90(image, k, axes)
            label = torch.rot90(label, k, [ax-1 for ax in axes])
        
        for dim in [1, 2, 3]:
            if random.random() > 0.5:
                image = torch.flip(image, dims=[dim])
                label = torch.flip(label, dims=[dim-1])
        
        if random.random() > 0.5:
            gamma = 0.7 + random.random() * 0.6
            image = torch.pow(image.clamp(min=1e-7), gamma)
        
        if random.random() > 0.5:
            noise = torch.randn_like(image) * 0.05
            image = torch.clamp(image + noise, 0, 1)
            
        return image, label

    def _smart_crop(self, image, label):
        pd, ph, pw = self.patch_size
        d, h, w = image.shape[1:]
        
        if d < pd or h < ph or w < pw:
            return image[:, :pd, :ph, :pw], label[:pd, :ph, :pw]
        
        if self.is_training and label.sum() > 100 and random.random() > 0.2:
            label_coords = torch.nonzero(label > 0)
            center_idx = random.randint(0, len(label_coords) - 1)
            center_z, center_y, center_x = label_coords[center_idx]
            
            z = max(0, min(d - pd, center_z - pd // 2))
            y = max(0, min(h - ph, center_y - ph // 2))
            x = max(0, min(w - pw, center_x - pw // 2))
        else:
            z = random.randint(0, d - pd)
            y = random.randint(0, h - ph)
            x = random.randint(0, w - pw)
        
        return image[:, z:z+pd, y:y+ph, x:x+pw], label[z:z+pd, y:y+ph, x:x+pw]

    def _load_scan(self, path):
        if path in self.scan_cache:
            return self.scan_cache[path]
        try:
            data = torch.load(path, map_location='cpu')
            if len(self.scan_cache) >= MAX_CACHED_SCANS:
                oldest_path = self.cache_order.pop(0)
                del self.scan_cache[oldest_path]
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

        image, label = scan_data['image'].float(), scan_data['label'].long()
        
        for c in range(image.shape[0]):
            modality = image[c]
            p99 = torch.quantile(modality, 0.99)
            p1 = torch.quantile(modality, 0.01)
            image[c] = torch.clamp((modality - p1) / (p99 - p1 + 1e-6), 0, 1)

        pd, ph, pw = self.patch_size
        d, h, w = image.shape[1:]
        pad_d, pad_h, pad_w = max(0, pd - d), max(0, ph - h), max(0, pw - w)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image = F.pad(image, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2, pad_d//2, pad_d-pad_d//2))
            label = F.pad(label, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2, pad_d//2, pad_d-pad_d//2))
        
        image_patch, label_patch = self._smart_crop(image, label)
        image_patch, label_patch = self._enhanced_augmentation(image_patch, label_patch)

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
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        e4 = self.enc4(self.down(e3))
        
        bottleneck = self.bottleneck(self.down(e4))
        
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

# --- FIX: Implemented Asymmetric Focal Tversky Loss ---
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
        print(f"📊 Training plots saved to {save_path}")
    except ImportError:
        print("Matplotlib not found. Skipping plot generation.")

def main():
    print("🚀 Starting Enhanced 3D DenseTrans Training 🚀")
    
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt")))
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    print(f"📊 Train: {len(train_files)} scans, Val: {len(val_files)} scans")

    model = EnhancedDenseTrans3D(IN_CHANNELS, OUT_CHANNELS).to(DEVICE)
    print(f"🧠 Model: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M params")

    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-7)
    loss_fn = AsymmetricFocalTverskyLoss().to(DEVICE)
    scaler = GradScaler()
    
    best_val_dice = 0.0
    patience_counter = 0
    train_history = {'loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
    
    val_dataset = ImprovedVolumetricDataset(val_files, PATCH_SIZE, patches_per_scan=PATCHES_PER_SCAN_VAL, is_training=False, use_fixed_validation=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    for epoch in range(EPOCHS):
        print(f"\n🔄 Epoch {epoch+1}/{EPOCHS}")
        
        model.train()
        train_dataset = ImprovedVolumetricDataset(train_files, PATCH_SIZE, patches_per_scan=PATCHES_PER_SCAN_TRAIN, is_training=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        
        train_loss, num_batches = 0.0, 0
        pbar = tqdm(train_loader, desc="🏋️ Training")
        optimizer.zero_grad()
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
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
            
            pbar.set_postfix({'loss': f'{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}'})
        
        avg_train_loss = train_loss / max(num_batches, 1)

        model.eval()
        val_loss, val_metrics_agg = 0.0, {'dice': [], 'iou': []}
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="🔍 Validation"):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                with autocast(device_type=DEVICE.type):
                    outputs = model(inputs)
                    val_loss += loss_fn(outputs[-1], targets).item()
                metrics = calculate_metrics(outputs[-1], targets)
                for k in val_metrics_agg: val_metrics_agg[k].append(metrics[k])
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_metrics = {k: np.mean(v) for k, v in val_metrics_agg.items()}
        
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'📈 Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        print(f'🎯 Val Dice: {avg_val_metrics["dice"]:.4f} | Val IoU: {avg_val_metrics["iou"]:.4f} | LR: {current_lr:.2e}')
        
        train_history['loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        train_history['val_dice'].append(avg_val_metrics['dice'])
        train_history['val_iou'].append(avg_val_metrics['iou'])

        if avg_val_metrics['dice'] > best_val_dice:
            best_val_dice = avg_val_metrics['dice']
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Model saved with Dice: {best_val_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"🛑 Early stopping after {epoch+1} epochs.")
                break
    
    print(f"\n🎉 Training completed! Best validation Dice: {best_val_dice:.4f}")
    save_training_plots(train_history, PLOT_SAVE_PATH)

if __name__ == '__main__':
    main()
