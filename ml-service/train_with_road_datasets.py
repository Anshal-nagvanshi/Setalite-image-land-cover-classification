"""
Road & Highway-Optimized Land Cover Classification Training Pipeline
Target: Resolving Road/Highway misclassifications against dominant background classes (Forest, Water, Crops)

Features:
1. Dynamic High-Resolution Tiling & Patch Extraction:
   - Slices large aerial orthophotos (e.g. LandCover.ai or road corridor images) into 224x224 patches.
   - Dynamic road detection filter (using HSV asphalt/edge density proxy) to oversample highway patches.
2. Albumentations Linear Feature Augmentation:
   - A.Perspective, A.ShiftScaleRotate, A.HorizontalFlip, A.VerticalFlip, A.RandomRotate90
   - Prevents linear orientation bias (diagonal, curved, horizontal, vertical roads).
3. Class-Weighted Cross-Entropy Loss:
   - Assigns 2.5x - 3.0x higher penalty weight to "Highway / Road" errors to counter foliage dominance.
4. ResNet-50 Two-Phase Transfer Learning:
   - Head warmup -> Deep fine-tuning with CosineAnnealingLR.
   - Persists best model to 'road_optimized_resnet50.pth'.
"""

import os
import sys
import time
import copy
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
from torchvision.models import ResNet50_Weights

# Albumentations for robust spatial transforms
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("⚠️ 'albumentations' is required. Please install via: pip install albumentations")
    sys.exit(1)


# ================= CONFIGURATION ================= #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "road_optimized_resnet50.pth")

NUM_CLASSES = 10
BATCH_SIZE = 64
PATCH_SIZE = 224
STRIDE = 160  # Stride < PATCH_SIZE provides overlapping coverage along road corridors
NUM_WORKERS = 4 if os.name != 'nt' else 0

# Standard 10 Land Cover Classes
CLASSES = [
    "Annual Crop",
    "Forest",
    "Herbaceous Vegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "Residential",
    "River",
    "Sea/Lake",
    "Permanent Crop"
]

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
HIGHWAY_IDX = CLASS_TO_IDX["Highway"]

# Epoch & Learning Rates
PHASE1_EPOCHS = 3
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 12
PHASE2_LR = 1e-5


# ================= 1. DYNAMIC ROAD SLICING & PATCH EXTRACTION ================= #

def contains_linear_road_feature(patch_bgr: np.ndarray) -> bool:
    """
    Fast computer-vision heuristic to verify if an aerial patch contains asphalt/concrete road pixels:
    - Neutral grayscale chroma (abs(R - B) is low)
    - High directional gradient / Sobel edge response
    """
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    # Sobel gradient along X and Y
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    edge_density = np.mean(magnitude)

    # Asphalt color profile in BGR: balanced channels, low saturation
    b, g, r = patch_bgr[:, :, 0], patch_bgr[:, :, 1], patch_bgr[:, :, 2]
    chroma_diff = np.mean(np.abs(r.astype(float) - b.astype(float)))

    return bool(edge_density > 18.0 and chroma_diff < 20.0)


def slice_large_aerial_image(
    image_path: str,
    default_label_idx: int = HIGHWAY_IDX,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    oversample_roads: bool = True
) -> List[Tuple[np.ndarray, int]]:
    """
    Loads large aerial orthophoto (e.g., from LandCover.ai or drone mapping)
    and slices it into 224x224 patches with stride overlap.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return []

    h, w, _ = img_bgr.shape
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            crop = img_bgr[y:y + patch_size, x:x + patch_size]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            if default_label_idx == HIGHWAY_IDX and oversample_roads:
                # Keep all patches meeting the road heuristic
                if contains_linear_road_feature(crop):
                    patches.append((crop_rgb, HIGHWAY_IDX))
                else:
                    # Non-road background portion of the aerial corridor
                    patches.append((crop_rgb, CLASS_TO_IDX["Forest"]))
            else:
                patches.append((crop_rgb, default_label_idx))

    return patches


# ================= 2. ALBUMENTATIONS LINEAR FEATURE AUGMENTATION ================= #

def get_road_augmentations(image_size: int = PATCH_SIZE) -> A.Compose:
    """
    Data augmentation pipeline tailored specifically for linear path & road recognition:
    - Full 360-degree rotation & 90-degree flips prevent orientation bias.
    - Perspective transforms simulate non-overhead aerial camera perspectives.
    - Shift/Scale simulates varying drone altitudes and offset lanes.
    """
    return A.Compose([
        A.Resize(image_size, image_size),

        # 1. Rotational & Linear Symmetry Invariance
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.12,
            scale_limit=0.15,
            rotate_limit=90,
            border_mode=cv2.BORDER_REFLECT,
            p=0.7
        ),

        # 2. Camera Pitch & Perspective Distortion (Drone / Oblique aerial views)
        A.Perspective(scale=(0.05, 0.15), keep_size=True, pad_mode=cv2.BORDER_REFLECT, p=0.6),

        # 3. Environmental Variations (Sun glare, shadows, pavement age)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.5),

        # 4. Standard ImageNet Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def get_val_augmentations(image_size: int = PATCH_SIZE) -> A.Compose:
    """Validation pipeline: deterministic resize and normalization."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


class RoadPatchDataset(Dataset):
    """PyTorch Dataset that wraps extracted image patches with Albumentations."""
    def __init__(self, samples: List[Tuple[np.ndarray, int]], transform: Optional[A.Compose] = None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rgb_img, label = self.samples[idx]
        if self.transform is not None:
            augmented = self.transform(image=rgb_img)
            tensor = augmented["image"]
        else:
            tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
        return tensor, label


# ================= 3. CLASS-WEIGHTED LOSS & RESNET-50 SETUP ================= #

def build_model(num_classes: int = 10) -> nn.Module:
    """Initializes ResNet-50 with custom Dropout classification head."""
    print("🔧 Initializing ResNet-50 with pretrained ImageNet weights...")
    try:
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    except Exception:
        model = models.resnet50(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


def configure_phases(model: nn.Module, phase: int):
    """
    Two-Phase Transfer Learning:
      - Phase 1: Freeze backbone, train only classification head.
      - Phase 2: Unfreeze layer4 + head for fine-tuning.
    """
    if phase == 1:
        print("\n🔒 [Phase 1 Configuration] Freezing backbone. Training ONLY classification head.")
        for name, param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif phase == 2:
        print("\n🔓 [Phase 2 Configuration] Unfreezing 'layer4' convolutional block + classification head.")
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable Parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


def compute_class_weights(samples: List[Tuple[np.ndarray, int]], highway_boost: float = 2.5) -> torch.Tensor:
    """
    Computes inverse frequency class weights to prevent majority background classes
    (e.g. Forest, Crops) from drowning out minority linear highway networks.
    """
    class_counts = np.zeros(NUM_CLASSES, dtype=np.float32)
    for _, lbl in samples:
        class_counts[lbl] += 1

    total_samples = len(samples)
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)

    for i in range(NUM_CLASSES):
        count = class_counts[i]
        if count > 0:
            # Standard balanced inverse frequency formula
            weights[i] = total_samples / (NUM_CLASSES * count)
        else:
            weights[i] = 1.0

    # Apply explicit penalty multiplier to the Highway class
    weights[HIGHWAY_IDX] *= highway_boost
    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)

    print("\n⚖️  Class-Weighted Loss Multipliers:")
    for name, w, cnt in zip(CLASSES, weights, class_counts):
        tag = " ⭐ (Road Boosted)" if name == "Highway" else ""
        print(f"   - {name:<24}: Weight = {w:.3f} | Sample Count = {int(cnt)}{tag}")

    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# ================= 4. TRAINING & VALIDATION LOOPS ================= #

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"})

    return running_loss / total, (correct / total) * 100.0


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    highway_correct = 0
    highway_total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

            # Highway class specific tracking
            highway_mask = (labels == HIGHWAY_IDX)
            highway_total += torch.sum(highway_mask).item()
            highway_correct += torch.sum((preds == HIGHWAY_IDX) & highway_mask).item()

    overall_acc = (correct / total) * 100.0
    highway_acc = (highway_correct / max(1, highway_total)) * 100.0
    return running_loss / total, overall_acc, highway_acc


# ================= 5. MASTER EXECUTION PIPELINE ================= #

def load_or_generate_dataset() -> List[Tuple[np.ndarray, int]]:
    """
    Collects road patches and background samples from local folders,
    or generates realistic synthetic aerial tiles for dry-run verification.
    """
    samples: List[Tuple[np.ndarray, int]] = []
    road_dirs = ["./data/road_patches", "./data/landcover_ai", "./data/highways"]

    for d in road_dirs:
        if os.path.exists(d):
            print(f"📂 Scanning road dataset directory: {d}")
            for fname in os.listdir(d):
                fpath = os.path.join(d, fname)
                if fname.lower().endswith((".jpg", ".png", ".tif")):
                    # Slice or read patch
                    extracted = slice_large_aerial_image(fpath, default_label_idx=HIGHWAY_IDX)
                    samples.extend(extracted)

    # If no local raw aerial dataset found, generate balanced synthetic samples
    if len(samples) < 50:
        print("ℹ️ Note: No local raw orthophoto directory found. Creating synthetic road & background dataset for pipeline verification...")
        for c_idx in range(NUM_CLASSES):
            num_samples = 40 if c_idx == HIGHWAY_IDX else 25
            for _ in range(num_samples):
                if c_idx == HIGHWAY_IDX:
                    # Synthetic road: dark gray asphalt corridor with white/yellow markings
                    img = np.full((PATCH_SIZE, PATCH_SIZE, 3), 50, dtype=np.uint8)
                    cv2.line(img, (0, PATCH_SIZE // 2), (PATCH_SIZE, PATCH_SIZE // 2), (200, 200, 200), 16)
                    cv2.line(img, (0, PATCH_SIZE // 2), (PATCH_SIZE, PATCH_SIZE // 2), (0, 220, 220), 2)
                elif c_idx == CLASS_TO_IDX["Forest"]:
                    # Green dominant texture
                    img = np.random.randint(20, 90, (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
                    img[:, :, 1] = np.random.randint(70, 160, (PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                else:
                    img = np.random.randint(30, 220, (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)

                samples.append((img, c_idx))

    return samples


def main():
    print("=" * 70)
    print("🛣️  Road & Highway-Optimized ResNet-50 Training Pipeline")
    print(f"⚡ Device: {DEVICE} | Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 70)

    # 1. Load samples
    samples = load_or_generate_dataset()
    print(f"📊 Total Dataset Patches: {len(samples)} across {NUM_CLASSES} classes")

    # Split into 80% train / 20% validation
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    split_idx = int(0.8 * len(samples))

    train_samples = [samples[i] for i in indices[:split_idx]]
    val_samples = [samples[i] for i in indices[split_idx:]]

    print(f"   - Training subset:   {len(train_samples)} patches")
    print(f"   - Validation subset: {len(val_samples)} patches")

    # Datasets and Loaders
    train_dataset = RoadPatchDataset(train_samples, transform=get_road_augmentations())
    val_dataset = RoadPatchDataset(val_samples, transform=get_val_augmentations())

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(BATCH_SIZE, len(train_samples)),
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(BATCH_SIZE, len(val_samples)),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    # 2. Build Model & Weighted Loss
    model = build_model(num_classes=NUM_CLASSES)
    model.to(DEVICE)

    class_weights = compute_class_weights(train_samples, highway_boost=2.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0

    # ==================== PHASE 1: HEAD WARMUP ==================== #
    configure_phases(model, phase=1)
    optimizer_p1 = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=PHASE1_LR,
        weight_decay=1e-2
    )

    print("\n" + "=" * 45)
    print(f"🚀 Phase 1: Classification Head Warmup ({PHASE1_EPOCHS} Epochs @ lr={PHASE1_LR})")
    print("=" * 45)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_p1, DEVICE)
        val_loss, val_acc, hwy_acc = validate(model, val_loader, criterion, DEVICE)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{PHASE1_EPOCHS:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | Highway Recall: {hwy_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "phase": 1,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": CLASSES
            }, CHECKPOINT_PATH)
            print(f"   ⭐ New best validation accuracy: {val_acc:.2f}%. Saved to '{CHECKPOINT_PATH}'")

    # ==================== PHASE 2: DEEP FINE-TUNING ==================== #
    configure_phases(model, phase=2)
    optimizer_p2 = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=PHASE2_LR,
        weight_decay=1e-4
    )
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=PHASE2_EPOCHS, eta_min=1e-7)

    print("\n" + "=" * 45)
    print(f"🚀 Phase 2: Layer4 + Head Fine-Tuning ({PHASE2_EPOCHS} Epochs @ lr={PHASE2_LR})")
    print("=" * 45)

    for epoch in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_p2, DEVICE)
        val_loss, val_acc, hwy_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler_p2.step()
        elapsed = time.time() - t0

        current_lr = optimizer_p2.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{PHASE2_EPOCHS:02d}] ({elapsed:.1f}s, lr: {current_lr:.2e}) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | Highway Recall: {hwy_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "phase": 2,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": CLASSES
            }, CHECKPOINT_PATH)
            print(f"   ⭐ New best validation accuracy: {val_acc:.2f}%. Saved to '{CHECKPOINT_PATH}'")

    print("\n" + "=" * 70)
    print(f"✅ Road-Optimized Training Complete! Best Accuracy: {best_val_acc:.2f}%")
    print(f"📦 Final Production Checkpoint: {os.path.abspath(CHECKPOINT_PATH)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
