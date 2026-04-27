# ============================================================================
# FALCON HACKATHON TRAINING - SAFE STRONGER V2
# Copy this entire file into a single Colab cell.
# Preserves the current inference contract:
# - DINOv2 backbone family unchanged
# - Same SegmentationHeadConvNeXt architecture
# - Same 10-class mapping and checkpoint format
# - Saves classifier.state_dict() as best_model.pth / final_model.pth
# ============================================================================

import copy
import json
import math
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

print("Installing required packages...")
os.system("pip install -q albumentations")

import albumentations as A
from albumentations.pytorch import ToTensorV2


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)
torch.backends.cudnn.benchmark = True


# ============================================================================
# CONFIGURATION
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

CONFIG = {
    "backbone_size": "base",          # DINOv2 ViT-B/14
    "batch_size": 2,
    "effective_batch_size": 4,        # via gradient accumulation
    "image_size": 518,
    "num_epochs": 40,
    "learning_rate": 7.5e-4,          # slightly calmer than 1e-3
    "warmup_epochs": 4,
    "weight_decay": 0.01,
    "focal_gamma": 2,
    "dice_weight": 0.75,              # push IoU objective harder
    "num_classes": 10,
    "num_workers": 2,
    "ema_decay": 0.995,               # evaluate/save EMA weights
    "use_amp": True,
    "max_grad_norm": 1.0,
    "min_lr": 1e-6,
    "ignore_index": 255,
}

# Update this path in Colab if needed.
DATA_ROOT = "/content/drive/MyDrive/TRACK2/Offroad_Segmentation_Training_Dataset"

CONFIG["grad_accum_steps"] = max(1, CONFIG["effective_batch_size"] // CONFIG["batch_size"])

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")


# ============================================================================
# DATASET VALIDATION
# ============================================================================

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def require_dir(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Required directory not found: {path}")


def list_image_files(folder):
    return sorted(
        file_name
        for file_name in os.listdir(folder)
        if file_name.lower().endswith(IMAGE_EXTS)
    )


def validate_dataset_layout(data_root):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    require_dir(data_root)
    require_dir(train_dir)
    require_dir(val_dir)

    required_dirs = [
        os.path.join(train_dir, "Color_Images"),
        os.path.join(train_dir, "Segmentation"),
        os.path.join(val_dir, "Color_Images"),
        os.path.join(val_dir, "Segmentation"),
    ]
    for path in required_dirs:
        require_dir(path)

    for split_name, split_dir in [("train", train_dir), ("val", val_dir)]:
        image_dir = os.path.join(split_dir, "Color_Images")
        mask_dir = os.path.join(split_dir, "Segmentation")

        image_ids = list_image_files(image_dir)
        mask_ids = list_image_files(mask_dir)

        if not image_ids:
            raise RuntimeError(f"No images found in {image_dir}")
        if not mask_ids:
            raise RuntimeError(f"No masks found in {mask_dir}")

        image_set = set(image_ids)
        mask_set = set(mask_ids)
        missing_masks = sorted(image_set - mask_set)
        missing_images = sorted(mask_set - image_set)

        if missing_masks:
            raise RuntimeError(
                f"{split_name}: missing masks for {len(missing_masks)} files. "
                f"Example: {missing_masks[:3]}"
            )
        if missing_images:
            raise RuntimeError(
                f"{split_name}: masks without matching images: {len(missing_images)} files. "
                f"Example: {missing_images[:3]}"
            )

        print(f"{split_name}: {len(image_ids)} image/mask pairs verified")


# Raw dataset values mapped to model class IDs 0..9
value_map = {
    100: 0,    # Trees
    200: 1,    # Lush Bushes
    300: 2,    # Dry Grass
    500: 3,    # Dry Bushes
    550: 4,    # Ground Clutter
    600: 5,    # Flowers
    700: 6,    # Logs
    800: 7,    # Rocks
    7100: 8,   # Landscape
    10000: 9,  # Sky
}


def inspect_mask_values(mask_path):
    raw_mask = np.array(Image.open(mask_path))
    unique_values = sorted(np.unique(raw_mask).tolist())
    unknown_values = [value for value in unique_values if value not in value_map]
    return unique_values, unknown_values


validate_dataset_layout(DATA_ROOT)

sample_mask_path = os.path.join(DATA_ROOT, "train", "Segmentation", list_image_files(os.path.join(DATA_ROOT, "train", "Segmentation"))[0])
sample_values, sample_unknown = inspect_mask_values(sample_mask_path)
print(f"Sample raw mask values: {sample_values[:12]}")
if sample_unknown:
    raise RuntimeError(f"Found unknown raw mask values not in value_map: {sample_unknown}")


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            weight=self.alpha,
            ignore_index=CONFIG["ignore_index"],
        )
        valid_mask = targets != CONFIG["ignore_index"]
        if not valid_mask.any():
            return inputs.new_tensor(0.0)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss[valid_mask].mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets, num_classes=10):
        valid_mask = targets != CONFIG["ignore_index"]
        if not valid_mask.any():
            return inputs.new_tensor(0.0)

        inputs = F.softmax(inputs, dim=1)
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0
        targets_one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1)
        inputs = inputs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights, focal_gamma=2, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets, num_classes=CONFIG["num_classes"])
        return focal_loss + self.dice_weight * dice_loss


class_weights = torch.tensor([
    6.0,   # Trees
    4.0,   # Lush Bushes
    1.3,   # Dry Grass
    16.0,  # Dry Bushes
    5.0,   # Ground Clutter
    7.0,   # Flowers
    40.0,  # Logs
    14.0,  # Rocks
    1.0,   # Landscape
    0.7,   # Sky
], dtype=torch.float32).to(device)

print("\nClass weights:")
class_names = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]
for i, (name, weight) in enumerate(zip(class_names, class_weights)):
    print(f"  Class {i+1} - {name:15s}: {weight:6.2f}")


# ============================================================================
# AUGMENTATION
# ============================================================================

train_transform = A.Compose([
    # Preserve aspect ratio instead of warping scenes into a square.
    A.LongestMaxSize(max_size=CONFIG["image_size"]),
    A.PadIfNeeded(
        min_height=CONFIG["image_size"],
        min_width=CONFIG["image_size"],
        border_mode=cv2.BORDER_CONSTANT,
        mask_value=CONFIG["ignore_index"],
    ),
    A.HorizontalFlip(p=0.5),
    A.Affine(
        scale=(0.95, 1.08),
        translate_percent=(-0.04, 0.04),
        rotate=(-10, 10),
        shear=(-3, 3),
        p=0.35,
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.12,
        contrast_limit=0.12,
        p=0.25,
    ),
    A.HueSaturationValue(
        hue_shift_limit=8,
        sat_shift_limit=12,
        val_shift_limit=8,
        p=0.15,
    ),
    A.GaussianBlur(blur_limit=(3, 3), p=0.08),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.LongestMaxSize(max_size=CONFIG["image_size"]),
    A.PadIfNeeded(
        min_height=CONFIG["image_size"],
        min_width=CONFIG["image_size"],
        border_mode=cv2.BORDER_CONSTANT,
        mask_value=CONFIG["ignore_index"],
    ),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

print("\nData augmentation configured.")


# ============================================================================
# DATASET
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.masks_dir = os.path.join(data_dir, "Segmentation")
        self.transform = transform
        self.data_ids = list_image_files(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        mask = self.convert_mask(mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask

    def convert_mask(self, mask):
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        for raw_value, new_value in value_map.items():
            new_mask[mask == raw_value] = new_value
        return new_mask


TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")

trainset = MaskDataset(TRAIN_DIR, transform=train_transform)
valset = MaskDataset(VAL_DIR, transform=val_transform)

pin_memory = device.type == "cuda"

train_loader = DataLoader(
    trainset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG["num_workers"],
    pin_memory=pin_memory,
)

val_loader = DataLoader(
    valset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=pin_memory,
)

print("\nDataset loaded:")
print(f"  Training samples: {len(trainset)}")
print(f"  Validation samples: {len(valset)}")
print(f"  Batches per epoch: {len(train_loader)}")


# ============================================================================
# MODEL
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU(),
        )

        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        batch_size, num_tokens, channels = x.shape
        x = x.reshape(batch_size, self.H, self.W, channels).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


print("\nLoading DINOv2 backbone...")
backbone_archs = {
    "small": "vits14",
    "base": "vitb14_reg",
    "large": "vitl14_reg",
    "giant": "vitg14_reg",
}
backbone_arch = backbone_archs[CONFIG["backbone_size"]]
backbone_name = f"dinov2_{backbone_arch}"

backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
backbone_model.eval()
backbone_model.to(device)

print(f"Loaded {backbone_name}")

imgs, _ = next(iter(train_loader))
imgs = imgs.to(device)
with torch.no_grad():
    sample_output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
n_embedding = sample_output.shape[2]

print(f"  Embedding dimension: {n_embedding}")
print(f"  Patch tokens shape: {sample_output.shape}")

classifier = SegmentationHeadConvNeXt(
    in_channels=n_embedding,
    out_channels=CONFIG["num_classes"],
    tokenW=CONFIG["image_size"] // 14,
    tokenH=CONFIG["image_size"] // 14,
).to(device)

ema_classifier = copy.deepcopy(classifier).to(device)
ema_classifier.eval()
for parameter in ema_classifier.parameters():
    parameter.requires_grad_(False)

print(f"  Head parameters: {sum(p.numel() for p in classifier.parameters()):,}")


# ============================================================================
# TRAINING SETUP
# ============================================================================

loss_fct = CombinedLoss(
    class_weights=class_weights,
    focal_gamma=CONFIG["focal_gamma"],
    dice_weight=CONFIG["dice_weight"],
)

optimizer = optim.AdamW(
    classifier.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=max(1, CONFIG["num_epochs"] - CONFIG["warmup_epochs"]),
    eta_min=CONFIG["min_lr"],
)

scaler = GradScaler(enabled=CONFIG["use_amp"] and device.type == "cuda")

print("\nTraining setup complete.")
print(f"  Loss: Focal (gamma={CONFIG['focal_gamma']}) + Dice (weight={CONFIG['dice_weight']})")
print(f"  Optimizer: AdamW (lr={CONFIG['learning_rate']}, wd={CONFIG['weight_decay']})")
print(f"  Effective batch size: {CONFIG['effective_batch_size']}")
print(f"  EMA decay: {CONFIG['ema_decay']}")
print(f"  AMP enabled: {CONFIG['use_amp'] and device.type == 'cuda'}")


# ============================================================================
# METRICS
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1)
    pred = pred.view(-1)
    target = target.view(-1)
    valid_mask = target != CONFIG["ignore_index"]
    if not valid_mask.any():
        return float("nan"), [float("nan")] * num_classes
    pred = pred[valid_mask]
    target = target[valid_mask]

    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float("nan"))
        else:
            iou_per_class.append((intersection / union).item())

    return float(np.nanmean(iou_per_class)), iou_per_class


def update_ema_model(ema_model, model, decay):
    with torch.no_grad():
        ema_state = ema_model.state_dict()
        model_state = model.state_dict()
        for key in ema_state.keys():
            ema_state[key].mul_(decay).add_(model_state[key], alpha=1.0 - decay)


def evaluate_model(model, backbone, data_loader, device):
    model.eval()
    all_ious = []
    all_losses = []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=CONFIG["use_amp"] and device.type == "cuda"):
                features = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(features)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                loss = loss_fct(outputs, labels)

            all_losses.append(loss.item())
            mean_iou, _ = compute_iou(outputs, labels, num_classes=CONFIG["num_classes"])
            all_ious.append(mean_iou)

    model.train()
    return float(np.mean(all_losses)), float(np.mean(all_ious))


# ============================================================================
# TRAINING LOOP
# ============================================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_iou": [],
    "val_iou": [],
    "learning_rate": [],
}

best_val_iou = 0.0

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80 + "\n")

for epoch in range(CONFIG["num_epochs"]):
    if epoch < CONFIG["warmup_epochs"]:
        lr = CONFIG["learning_rate"] * (epoch + 1) / CONFIG["warmup_epochs"]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    current_lr = optimizer.param_groups[0]["lr"]

    classifier.train()
    train_losses = []
    train_ious = []

    optimizer.zero_grad(set_to_none=True)

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['num_epochs']} [Train]")
    for step, (imgs, labels) in enumerate(train_pbar, start=1):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            with autocast(enabled=CONFIG["use_amp"] and device.type == "cuda"):
                features = backbone_model.forward_features(imgs)["x_norm_patchtokens"]

        with autocast(enabled=CONFIG["use_amp"] and device.type == "cuda"):
            logits = classifier(features)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            loss = loss_fct(outputs, labels)

        raw_loss = loss.item()
        train_losses.append(raw_loss)

        mean_iou, _ = compute_iou(outputs.detach(), labels, num_classes=CONFIG["num_classes"])
        train_ious.append(mean_iou)

        scaled_loss = loss / CONFIG["grad_accum_steps"]
        scaler.scale(scaled_loss).backward()

        if step % CONFIG["grad_accum_steps"] == 0 or step == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), CONFIG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_ema_model(ema_classifier, classifier, CONFIG["ema_decay"])

        train_pbar.set_postfix(
            loss=f"{raw_loss:.4f}",
            iou=f"{mean_iou:.4f}",
            lr=f"{current_lr:.2e}",
        )

    val_loss, val_iou = evaluate_model(ema_classifier, backbone_model, val_loader, device)

    epoch_train_loss = float(np.mean(train_losses))
    epoch_train_iou = float(np.mean(train_ious))

    history["train_loss"].append(epoch_train_loss)
    history["val_loss"].append(val_loss)
    history["train_iou"].append(epoch_train_iou)
    history["val_iou"].append(val_iou)
    history["learning_rate"].append(current_lr)

    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(ema_classifier.state_dict(), "best_model.pth")
        print(f"\nNew best model! Val IoU: {val_iou:.4f}")

    print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']} Summary:")
    print(f"  Train Loss: {epoch_train_loss:.4f} | Train IoU: {epoch_train_iou:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f}")
    print(f"  LR: {current_lr:.2e} | Best Val IoU: {best_val_iou:.4f}")
    print("-" * 80)

    if epoch >= CONFIG["warmup_epochs"]:
        scheduler.step()

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nBest Validation IoU: {best_val_iou:.4f}")
print(f"Final Validation IoU: {history['val_iou'][-1]:.4f}")


# ============================================================================
# PLOTS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].plot(history["train_loss"], label="Train", marker="o")
axes[0, 0].plot(history["val_loss"], label="Val", marker="s")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Training and Validation Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history["train_iou"], label="Train", marker="o")
axes[0, 1].plot(history["val_iou"], label="Val", marker="s")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("IoU")
axes[0, 1].set_title("Training and Validation IoU")
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(history["learning_rate"], marker="o")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Learning Rate")
axes[1, 0].set_title("Learning Rate Schedule")
axes[1, 0].set_yscale("log")
axes[1, 0].grid(True)

axes[1, 1].plot(np.array(history["val_iou"]) - history["val_iou"][0], marker="s", color="green")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("IoU Improvement")
axes[1, 1].set_title("Validation IoU Improvement")
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150, bbox_inches="tight")
plt.show()

torch.save(ema_classifier.state_dict(), "final_model.pth")

with open("training_history.json", "w") as file:
    json.dump(history, file, indent=2)

run_metadata = {
    "data_root": DATA_ROOT,
    "config": CONFIG,
    "class_names": class_names,
    "class_weights": [float(weight) for weight in class_weights.detach().cpu().tolist()],
    "value_map": value_map,
    "best_val_iou": best_val_iou,
}
with open("run_metadata.json", "w") as file:
    json.dump(run_metadata, file, indent=2)

print("\nSaved:")
print("  best_model.pth")
print("  final_model.pth")
print("  training_results.png")
print("  training_history.json")
print("  run_metadata.json")
