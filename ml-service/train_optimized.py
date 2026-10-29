"""
EuroSAT Satellite Land Cover Classification - Optimized ResNet-50 Training Pipeline
Major Project: Resolving linear feature confusions (e.g. Highways vs Rivers)

Features:
1. Two-Phase Transfer Learning (Head warmup -> Layer4 + Head fine-tuning)
2. Mixup Regularization (Smooths linear boundaries between overlapping classes)
3. Learning Rate Schedulers (CosineAnnealingLR) + Weight Decay
4. Validation & Test Confusion Matrix Generation using Scikit-Learn + Seaborn
5. Checkpoint export to 'eurosat_resnet50_model.pth'
"""

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms, datasets
from torchvision.models import ResNet50_Weights

# ================= CONFIGURATION ================= #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_WORKERS = 4
CHECKPOINT_PATH = "eurosat_resnet50_model.pth"
CONFUSION_MATRIX_PATH = "confusion_matrix.png"

# Standard EuroSAT 10 Classes
EUROSAT_CLASSES = [
    "Annual Crop",
    "Forest",
    "Herbaceous Vegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "Permanent Crop",
    "Residential",
    "River",
    "Sea/Lake"
]

# Phase 1: Classification Head Warmup
PHASE1_EPOCHS = 3
PHASE1_LR = 1e-3

# Phase 2: Layer4 + Head Deep Fine-Tuning
PHASE2_EPOCHS = 12
PHASE2_LR = 1e-5

# Mixup parameter (beta distribution)
MIXUP_ALPHA = 0.2


# ================= DATA AUGMENTATION & MULTI-SCALE CROPPING ================= #
def get_transforms():
    """
    Returns train and validation transforms.
    Uses RandomResizedCrop (scale 0.4 to 1.0) to zoom into sub-regions,
    forcing the model to detect linear road/highway features rather than
    relying on 90%+ dominant background trees or grass.
    """
    train_transform = transforms.Compose([
        # Focus on local spatial details: zooms in anywhere from 40% to 100% of the patch area
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.4, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


# ================= MIXUP AUGMENTATION ================= #
def mixup_data(x, y, alpha=0.2):
    """
    Applies linear Mixup augmentation:
    x_mixed = lambda * x + (1 - lambda) * x_shuffled
    Returns mixed inputs, pairs of targets, and lambda mixing ratio.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(DEVICE)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Computes blended cross-entropy loss for Mixup targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ================= DATASET PREPARATION ================= #
def load_eurosat_data(data_dir="./data/EuroSAT_RGB"):
    """
    Prepares train and validation DataLoaders.
    Attempts torchvision.datasets.EuroSAT first, with ImageFolder and HuggingFace fallbacks.
    """
    train_tf, val_tf = get_transforms()

    dataset_loaded = False
    full_dataset = None

    # Option A: Local ImageFolder (if user downloaded EuroSAT folder)
    if os.path.exists(data_dir):
        print(f"📁 Loading EuroSAT from local directory: {data_dir}")
        full_dataset = datasets.ImageFolder(root=data_dir, transform=train_tf)
        dataset_loaded = True
    else:
        # Option B: Automatic download via torchvision
        try:
            print("🌐 Downloading EuroSAT via torchvision.datasets.EuroSAT...")
            full_dataset = datasets.EuroSAT(root="./data", download=True, transform=train_tf)
            dataset_loaded = True
        except Exception as e:
            print(f"⚠️ Torchvision EuroSAT download notice: {e}")

    # Option C: Hugging Face datasets fallback if torchvision fails
    if not dataset_loaded or full_dataset is None:
        print("🌐 Loading EuroSAT from Hugging Face 'blanchon/EuroSAT_RGB'...")
        from datasets import load_dataset
        hf_dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")

        class HuggingFaceEuroSATDataset(torch.utils.data.Dataset):
            def __init__(self, hf_ds, transform=None):
                self.ds = hf_ds
                self.transform = transform

            def __len__(self):
                return len(self.ds)

            def __getitem__(self, idx):
                item = self.ds[idx]
                img = item["image"].convert("RGB")
                label = item["label"]
                if self.transform:
                    img = self.transform(img)
                return img, label

        full_dataset = HuggingFaceEuroSATDataset(hf_dataset, transform=train_tf)

    # 80/20 Train/Validation Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply validation transforms to val_subset
    val_dataset = copy.deepcopy(val_subset)
    if hasattr(val_dataset.dataset, "transform"):
        val_dataset.dataset.transform = val_tf

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"✅ Dataset split ready: {train_size} training samples, {val_size} validation samples.")
    return train_loader, val_loader


# ================= MODEL INITIALIZATION ================= #
def build_resnet50_model():
    """
    Initializes ResNet-50 with pretrained ImageNet weights.
    Customizes classification head with Dropout (p=0.4) for EuroSAT 10 classes.
    """
    print("🧠 Initializing ResNet-50 backbone with ImageNet weights...")
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Replace final linear projection with regularization
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, NUM_CLASSES)
    )

    model.to(DEVICE)
    return model


# ================= TRAINING & EVALUATION LOOP ================= #
def train_one_epoch(model, loader, criterion, optimizer, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train Batch", leave=False)
    for images, targets in pbar:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()

        if use_mixup:
            mixed_images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=MIXUP_ALPHA)
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            # Mixed accuracy metric
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += targets.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="  Val Batch", leave=False):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_targets)


# ================= CONFUSION MATRIX PLOT ================= #
def plot_and_save_confusion_matrix(y_true, y_pred, class_names, output_path="confusion_matrix.png"):
    """
    Computes and plots normalized confusion matrix using Seaborn heatmap.
    Specifically useful for checking linear confusion (e.g. Highway vs River).
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Classification Accuracy (%)"}
    )
    plt.title("EuroSAT ResNet-50 Normalized Confusion Matrix (%)", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Ground Truth Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📊 Confusion matrix saved to: {output_path}")


# ================= MAIN EXECUTION PIPELINE ================= #
def main():
    print(f"🚀 Running EuroSAT ResNet-50 Training Pipeline on: {DEVICE}")
    train_loader, val_loader = load_eurosat_data()
    model = build_resnet50_model()

    # Class-Weighted Cross-Entropy Loss
    # Assign higher weights to linear/fine-grained classes ("Highway", "Residential", "Industrial")
    # and lower weights to broad background classes ("Forest", "Herbaceous Vegetation")
    # EuroSAT classes: ["Annual Crop", "Forest", "Herbaceous Vegetation", "Highway", "Industrial", "Pasture", "Permanent Crop", "Residential", "River", "Sea/Lake"]
    class_weights = torch.tensor([
        1.0,  # Annual Crop
        0.6,  # Forest (down-weighted so surrounding canopy doesn't overpower linear road cues)
        0.8,  # Herbaceous Vegetation
        2.5,  # Highway (boosted 2.5x to penalize background suppression)
        1.8,  # Industrial (boosted)
        1.0,  # Pasture
        1.0,  # Permanent Crop
        1.6,  # Residential (boosted)
        2.0,  # River (boosted 2.0x for linear discrimination)
        1.0   # Sea/Lake
    ], dtype=torch.float).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"⚖️  Applied Class-Weighted Cross-Entropy Loss (Highway weight: {class_weights[3]}x, Forest weight: {class_weights[1]}x)")

    best_val_acc = 0.0
    best_preds = None
    best_targets = None

    # -------------------------------------------------------------
    # PHASE 1: FREEZE EARLY LAYERS & TRAIN CLASSIFICATION HEAD
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"📌 PHASE 1: Head Warmup ({PHASE1_EPOCHS} epochs, LR = {PHASE1_LR})")
    print("   Freezing convolutional backbone (conv1 through layer4)...")
    print("=" * 65)

    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer_phase1 = optim.AdamW(model.fc.parameters(), lr=PHASE1_LR, weight_decay=1e-4)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_phase1, use_mixup=True)
        val_loss, val_acc, preds, targets = validate(model, val_loader, criterion)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch}/{PHASE1_EPOCHS}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_preds = preds
            best_targets = targets
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    # -------------------------------------------------------------
    # PHASE 2: UNFREEZE LAYER4 + HEAD & DEEP FINE-TUNE
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"📌 PHASE 2: Unfreezing Layer4 + Head ({PHASE2_EPOCHS} epochs, LR = {PHASE2_LR})")
    print("   Resolving linear class ambiguities (Highways vs. Rivers)...")
    print("=" * 65)

    # Unfreeze layer4 and classification head
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Parameter groups with differential learning rates
    optimizer_phase2 = optim.AdamW([
        {"params": model.layer4.parameters(), "lr": PHASE2_LR * 0.5},
        {"params": model.fc.parameters(), "lr": PHASE2_LR}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=PHASE2_EPOCHS, eta_min=1e-7)

    for epoch in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_phase2, use_mixup=True)
        val_loss, val_acc, preds, targets = validate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch [{epoch}/{PHASE2_EPOCHS}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
              f"LR: {scheduler.get_last_lr()[-1]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_preds = preds
            best_targets = targets
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  ⭐ New best model saved ({best_val_acc*100:.2f}%) -> {CHECKPOINT_PATH}")

    # -------------------------------------------------------------
    # EVALUATION & CONFUSION MATRIX GENERATION
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("📈 FINAL EVALUATION & METRICS SUMMARY")
    print("=" * 65)
    print(f"🏆 Best Validation Accuracy: {best_val_acc * 100:.2f}%\n")

    # Classification Report
    print("Detailed Classification Report across 10 EuroSAT Classes:")
    print(classification_report(best_targets, best_preds, target_names=EUROSAT_CLASSES, digits=4))

    # Generate Confusion Matrix Plot
    plot_and_save_confusion_matrix(best_targets, best_preds, EUROSAT_CLASSES, CONFUSION_MATRIX_PATH)

    # Check Highway vs River confusion specifically
    cm = confusion_matrix(best_targets, best_preds)
    highway_idx = EUROSAT_CLASSES.index("Highway")
    river_idx = EUROSAT_CLASSES.index("River")
    print(f"🔍 Linear Confusion Diagnostic:")
    print(f"   - Highway samples classified as River: {cm[highway_idx][river_idx]}")
    print(f"   - River samples classified as Highway: {cm[river_idx][highway_idx]}")
    print(f"   - Highway Accuracy: {cm[highway_idx][highway_idx] / cm[highway_idx].sum() * 100:.2f}%")
    print(f"   - River Accuracy:   {cm[river_idx][river_idx] / cm[river_idx].sum() * 100:.2f}%")
    print("\n✅ Training complete! Checkpoint ready for deployment.")


if __name__ == "__main__":
    main()
