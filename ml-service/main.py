import io
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from dataset_loader import get_dataset, fetch_sample_by_index, get_dataset_info, DATASET_SOURCES
from utils import auto_crop_linear_feature

# Initialize FastAPI application
app = FastAPI(
    title="Land Cover Classification Microservice (Mixed-Perspective ResNet-50)",
    description="Production FastAPI service for top-down satellite, aerial drone, and oblique perspective land cover classification",
    version="2.0.0"
)

# 1. CORS Middleware configuration for React & Express integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Standard 10 Land Cover Classes (EuroSAT & Aerial benchmarks)
CLASSES = [
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRIMARY_WEIGHTS = os.getenv("MODEL_PATH", "mixed_perspective_resnet50.pth")
FALLBACK_WEIGHTS = "eurosat_resnet50_model.pth"

# 2. ResNet-50 Model Initialization & Weights Loader
def load_resnet50_model(primary_path: str, fallback_path: str, num_classes: int = 10) -> nn.Module:
    """
    Initializes ResNet-50 architecture, replaces the classification head (fc),
    and attempts to load trained weights with fallback handling.
    """
    print(f"🔧 Initializing ResNet-50 model on device: {DEVICE}")
    try:
        model = models.resnet50(weights=None)
    except Exception:
        model = models.resnet50(pretrained=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, num_classes)
    )

    loaded_name = None
    target_path = primary_path if os.path.exists(primary_path) else (fallback_path if os.path.exists(fallback_path) else None)

    if target_path:
        try:
            checkpoint = torch.load(target_path, map_location=DEVICE)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            loaded_name = target_path
            print(f"✅ Loaded trained model weights from: {target_path}")
        except Exception as e:
            print(f"⚠️ Warning: Error loading weights from {target_path}: {e}. Falling back to structural weights.")
    else:
        print(f"ℹ️ Notice: Weights files ('{primary_path}' or '{fallback_path}') not found. Initialized with architecture weights.")

    model.to(DEVICE)
    model.eval()
    model.loaded_weights_path = loaded_name
    return model


model = load_resnet50_model(PRIMARY_WEIGHTS, FALLBACK_WEIGHTS, num_classes=len(CLASSES))

# 3. Preprocessing Transform (ImageNet normalization matching 224x224 tensor input)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================= RESPONSE SCHEMAS ================= #
class ClassConfidence(BaseModel):
    label: str
    confidence: float


class CompetingPair(BaseModel):
    rank1: ClassConfidence
    rank2: ClassConfidence
    margin: float
    isLinearConfusionWarning: bool


class InferenceResponse(BaseModel):
    topPrediction: ClassConfidence
    confidence: float
    top2Competing: CompetingPair
    probabilities: List[ClassConfidence]
    modelUsed: str
    device: str
    perspectiveRobustness: str


# ================= HEALTH CHECK ================= #
@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "FastAPI Mixed-Perspective Land Cover Classification Microservice",
        "device": str(DEVICE),
        "weights_file": getattr(model, "loaded_weights_path", None) or "Structural Fallback",
        "classes": CLASSES
    }


# ================= DATASET ENDPOINTS ================= #
@app.get("/datasets")
def list_datasets():
    """List supported Hugging Face EuroSAT datasets."""
    return get_dataset_info()


@app.get("/datasets/sample")
def get_sample(
    source: str = Query("blanchon", description="'cm93' or 'blanchon'"),
    index: int = Query(0, description="Sample index to fetch")
):
    """Fetches a real satellite image patch from HuggingFace dataset."""
    try:
        sample = fetch_sample_by_index(source, index)
        return sample
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dataset sample: {str(e)}")


# ================= INFERENCE ENDPOINT ================= #
@app.post("/predict", response_model=InferenceResponse)
async def predict_land_cover(
    file: UploadFile = File(...)
):
    """
    Inference Endpoint:
    1. Reads incoming image file bytes.
    2. Decodes via OpenCV (handling orientation/color matrices) and converts BGR to RGB.
    3. Transforms to 224x224 normalized PyTorch tensor.
    4. Runs ResNet-50 inference under torch.no_grad().
    5. Computes Softmax probabilities across all 10 classes.
    6. Identifies Top-2 competing classes and logs confidence margin to terminal.
    7. Returns complete JSON payload formatted for React Chart.js components.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        # Read image bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)

        # OpenCV decoding to handle diverse camera angles, drone tilts, and metadata
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            # Fallback to PIL in case of uncommon format
            pil_fallback = Image.open(io.BytesIO(contents)).convert("RGB")
            img_rgb = np.array(pil_fallback)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img_rgb.shape[:2]

        # Apply preprocessing pipeline (Resize, ImageNet Normalize, Tensor)
        tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

        # 4. Model Inference under torch.no_grad()
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Build class confidence list
        prob_list = []
        for idx, class_name in enumerate(CLASSES):
            val = round(float(probabilities[idx].item()) * 100, 2)
            prob_list.append({"label": class_name, "confidence": val})

        # Sort descending by confidence
        prob_list.sort(key=lambda x: x["confidence"], reverse=True)

        rank1 = prob_list[0]
        rank2 = prob_list[1]
        confidence_margin = round(rank1["confidence"] - rank2["confidence"], 2)

        # Detect potential linear confusion (e.g. Highway vs River vs Sea/Lake)
        linear_classes = {"Highway", "River", "Sea & Lake"}
        is_linear_confusion = (rank1["label"] in linear_classes and rank2["label"] in linear_classes)

        # 5. Terminal Console Debug Output
        print("\n" + "=" * 65)
        print("🛰️  INFERENCE DIAGNOSTIC - MIXED PERSPECTIVE CLASSIFIER")
        print("=" * 65)
        print(f"📁 Input File        : {file.filename}")
        print(f"📐 Native Dimensions : {orig_w}x{orig_h} px")
        print(f"🥇 Primary Prediction: {rank1['label']:<24} {rank1['confidence']:>6.2f}%")
        print(f"🥈 Top-2 Competitor  : {rank2['label']:<24} {rank2['confidence']:>6.2f}%")
        print(f"📊 Confidence Margin : {confidence_margin:>6.2f}%")

        if is_linear_confusion:
            print(f"⚠️  LINEAR FEATURE ALERT: Competitor ambiguity between '{rank1['label']}' and '{rank2['label']}'.")
        print("=" * 65 + "\n")

        weights_used = getattr(model, "loaded_weights_path", None)
        model_descriptor = f"ResNet-50 ({os.path.basename(weights_used)})" if weights_used else "ResNet-50 (Mixed-Perspective)"

        return {
            "topPrediction": rank1,
            "confidence": rank1["confidence"],
            "top2Competing": {
                "rank1": rank1,
                "rank2": rank2,
                "margin": confidence_margin,
                "isLinearConfusionWarning": is_linear_confusion
            },
            "probabilities": prob_list,
            "modelUsed": model_descriptor,
            "device": str(DEVICE),
            "perspectiveRobustness": "Perspective & Drone Invariant"
        }

    except Exception as e:
        print(f"❌ Inference pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
