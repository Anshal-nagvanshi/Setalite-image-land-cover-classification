"""
EuroSAT & Drone/Oblique Land Cover Classification - Perspective Invariant Training
Major Project: Resolving Top-Down Satellite vs. Drone vs. Oblique Perspective Domain Gap

Features:
1. Albumentations Data Pipeline:
   - Geometric/Perspective: A.Perspective, A.ShiftScaleRotate, A.HorizontalFlip, A.VerticalFlip, A.RandomRotate90
   - Photometric: A.RandomBrightnessContrast, A.HueSaturationValue
   - Normalization & Tensor conversion: A.Normalize, ToTensorV2
2. Custom Dataset Class:
   - Reads image files using OpenCV (cv2.imread -> BGR to RGB)
   - Supports dataset folder structure (root/class_name/xxx.jpg) or sample list
3. ResNet-50 Two-Phase Transfer Learning:
   - Phase 1: Freeze backbone, train classification head (3 epochs, lr=1e-3, AdamW)
   - Phase 2: Unfreeze layer4 + head, train with lower learning rate (12 epochs, lr=1e-5, AdamW)
4. Robust Evaluation & Best Model Checkpointing:
   - Epoch-by-epoch loss & accuracy tracking
   - Saves best model to 'mixed_perspective_resnet50.pth'
"""

import os
import sys
import glob
import time
import copy
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.models as models
from torchvision.models import ResNet50_Weights

# Albumentations for robust perspective and distortion transforms
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("⚠️ 'albumentations' is required. Please install via: pip install albumentations")
    sys.exit(1)


# ================= CONFIGURATION ================= #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.getenv("DATA_DIR", "./data/EuroSAT_RGB")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "mixed_perspective_resnet50.pth")

NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_WORKERS = 4 if os.name != 'nt' else 0  # Safe default on Windows

# Phase 1: Head Warmup
PHASE1_EPOCHS = 3
PHASE1_LR = 1e-3

# Phase 2: Layer4 + Head Fine-Tuning
PHASE2_EPOCHS = 12
PHASE2_LR = 1e-5

# Standard 10 Land Cover Classes
CLASS_NAMES = [
    "Annual Crop",
    "Forest",
    "Herbaceous Vegetation",
    "Highway",
    "Industrial Buildings",
    "Pasture",
    "Permanent Crop",
    "Residential Buildings",
    "River",
    "Sea & Lake"
]


# ================= 1. ALBUMENTATIONS DATA PIPELINE ================= #

def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Data augmentation pipeline tailored for bridging the domain gap between
    flat top-down satellite imagery, low-altitude drone tiles, and oblique angle shots.
    """
    return A.Compose([
        # Standard size input
        A.Resize(image_size, image_size),

        # Geometric & Perspective Invariance
        # 1. Perspective transformation simulates camera tilt / oblique views
        A.Perspective(scale=(0.05, 0.15), keep_size=True, pad_mode=cv2.BORDER_REFLECT, p=0.6),

        # 2. Shift, scale, and arbitrary rotation for drone maneuvers
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, border_mode=cv2.BORDER_REFLECT, p=0.7),

        # 3. Flips and 90-degree rotations for rotational invariance
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Lighting & Atmospheric Variations (haze, shadow, sun angle)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.5),

        # Normalization with standard ImageNet statistics
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Deterministic validation pipeline: standard resize and ImageNet normalization.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


# ================= 2. CUSTOM DATASET CLASS ================= #

class MixedPerspectiveDataset(Dataset):
    """
    PyTorch Dataset that loads images via OpenCV (BGR -> RGB)
    and applies an Albumentations transformation pipeline.
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[A.Compose] = None
    ):
        """
        Args:
            samples: List of (image_file_path, integer_label) tuples
            transform: Albumentations transformation pipeline
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]

        # Load image via OpenCV
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            # Fallback for corrupted or unreadable images
            raise FileNotFoundError(f"Unable to read image at: {image_path}")

        # Convert OpenCV BGR to standard RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply Albumentations pipeline
        if self.transform is not None:
            augmented = self.transform(image=img_rgb)
            img_tensor = augmented["image"]
        else:
            # Basic fallback tensor conversion
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        return img_tensor, label


def scan_dataset(root_dir: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    """
    Scans a standard ImageFolder directory structure (root/class_name/*.jpg|png)
    and returns (samples, class_names).
    """
    if not os.path.exists(root_dir):
        return [], []

    classes = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
    ])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    samples = []

    for cls_name in classes:
        cls_dir = os.path.join(root_dir, cls_name)
        for fname in os.listdir(cls_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                samples.append((os.path.join(cls_dir, fname), class_to_idx[cls_name]))

    return samples, classes


# ================= 3. RESNET-50 SETUP & TWO-PHASE FINE-TUNING ================= #

def build_model(num_classes: int) -> nn.Module:
    """
    Loads ResNet-50 with default pretrained ImageNet weights
    and replaces the fc head with a custom Dropout + Linear head.
    """
    print("🔧 Initializing ResNet-50 with pretrained ImageNet weights...")
    try:
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    except Exception:
        # Fallback for older torchvision versions
        model = models.resnet50(pretrained=True)

    in_features = model.fc.in_features
    # Robust classification head with Dropout to avoid overfitting
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


def set_trainable_layers(model: nn.Module, phase: int):
    """
    Implements Two-Phase Transfer Learning:
      - Phase 1: Freeze all convolutional layers (conv1 through layer4),
                 only train the new classification head (model.fc).
      - Phase 2: Unfreeze layer4 + classification head,
                 keep earlier lower-level feature extractors (conv1, layer1, layer2, layer3) frozen.
    """
    if phase == 1:
        print("\n🔒 [Phase 1 Configuration] Freezing all convolutional layers. Training ONLY classification head.")
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

    # Summary of trainable vs frozen parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Trainable Parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")


# ================= 4. TRAINING & VALIDATION ENGINE ================= #

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single training epoch and returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
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

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}%"
        })

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluates the model on validation data and returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = (correct / total) * 100.0
    return val_loss, val_acc


def main():
    print("=" * 70)
    print("🌍 EuroSAT / Drone Mixed-Perspective ResNet-50 Training Pipeline")
    print(f"⚡ Device: {DEVICE} | Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 70)

    # 1. Discover or create mock dataset for verification
    samples, detected_classes = scan_dataset(DATA_DIR)
    num_classes = len(detected_classes) if detected_classes else NUM_CLASSES
    class_names = detected_classes if detected_classes else CLASS_NAMES

    if len(samples) == 0:
        print(f"⚠️ No dataset found at '{DATA_DIR}'. Generating synthetic sample cache for dry-run verification...")
        mock_dir = "./data/synthetic_perspective_samples"
        os.makedirs(mock_dir, exist_ok=True)
        samples = []
        for c_idx, c_name in enumerate(class_names):
            c_dir = os.path.join(mock_dir, c_name.replace(" ", "_"))
            os.makedirs(c_dir, exist_ok=True)
            for i in range(8):
                img_path = os.path.join(c_dir, f"sample_{i}.jpg")
                if not os.path.exists(img_path):
                    # Random texture image
                    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    cv2.imwrite(img_path, dummy)
                samples.append((img_path, c_idx))

    print(f"📊 Dataset total samples: {len(samples)} across {num_classes} classes")

    # Split into Train (80%) and Validation (20%)
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    split_idx = int(0.8 * len(samples))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    print(f"   - Training set:   {len(train_samples)} images")
    print(f"   - Validation set: {len(val_samples)} images")

    # Datasets and DataLoaders
    train_dataset = MixedPerspectiveDataset(train_samples, transform=get_train_transforms())
    val_dataset = MixedPerspectiveDataset(val_samples, transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(BATCH_SIZE, max(2, len(train_samples))),
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(BATCH_SIZE, max(2, len(val_samples))),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    # 2. Build Model & Loss
    model = build_model(num_classes=num_classes)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    # ==================== PHASE 1: WARMUP CLASSIFICATION HEAD ==================== #
    set_trainable_layers(model, phase=1)
    optimizer_p1 = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=PHASE1_LR,
        weight_decay=1e-2
    )

    print("\n" + "=" * 40)
    print(f"🚀 Starting Phase 1 ({PHASE1_EPOCHS} Epochs @ lr={PHASE1_LR})")
    print("=" * 40)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_p1, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{PHASE1_EPOCHS:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch": epoch,
                "phase": 1,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": class_names
            }, CHECKPOINT_PATH)
            print(f"   ⭐ New best validation accuracy: {val_acc:.2f}%. Saved to '{CHECKPOINT_PATH}'")

    # ==================== PHASE 2: FINE-TUNE LAYER4 + HEAD ==================== #
    set_trainable_layers(model, phase=2)
    optimizer_p2 = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=PHASE2_LR,
        weight_decay=1e-4
    )
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=PHASE2_EPOCHS, eta_min=1e-7)

    print("\n" + "=" * 40)
    print(f"🚀 Starting Phase 2 ({PHASE2_EPOCHS} Epochs @ lr={PHASE2_LR})")
    print("=" * 40)

    for epoch in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_p2, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler_p2.step()
        elapsed = time.time() - t0

        current_lr = optimizer_p2.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{PHASE2_EPOCHS:02d}] ({elapsed:.1f}s, lr: {current_lr:.2e}) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch": epoch,
                "phase": 2,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": class_names
            }, CHECKPOINT_PATH)
            print(f"   ⭐ New best validation accuracy: {val_acc:.2f}%. Saved to '{CHECKPOINT_PATH}'")

    print("\n" + "=" * 70)
    print(f"✅ Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"📦 Final Production Checkpoint Saved: {os.path.abspath(CHECKPOINT_PATH)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
