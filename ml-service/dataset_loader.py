import io
import os
import base64
from datasets import load_dataset
from PIL import Image

# Multi-Source Remote Sensing & Aerial Dataset Repositories on Hugging Face
DATASET_SOURCES = {
    # 1. EuroSAT Sentinel-2 Multi-spectral & RGB (Top-Down Satellite Benchmark)
    "cm93": "cm93/eurosat",
    "blanchon": "blanchon/EuroSAT_RGB",
    "tanganke": "tanganke/eurosat",

    # 2. Aerial Image Dataset (Multi-angle, Drone & Oblique Aerial Scenes)
    "aid": "blanchon/AID",

    # 3. Gaofen Image Dataset (Fine-grained Infrastructure, Roads & Built-up Land)
    "gid": "jonathan-roberts1/GID",

    # 4. UC Merced High-Resolution Land Use (Aerial Orthophotos)
    "ucmerced": "Project-AgML/ucm_land_use_classification"
}

# Cache loaded datasets in memory so they are only downloaded once
_cached_datasets = {}

# Unified 10-Class Taxonomy
UNIFIED_CLASSES = [
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

def map_label_to_unified(raw_label, source_key):
    """Maps raw label strings or integers from different datasets into the unified 10-class scheme."""
    raw_str = str(raw_label).lower().strip().replace("-", " ").replace("_", " ")

    eurosat_names = [
        "annual crop", "forest", "herbaceous vegetation", "highway",
        "industrial", "pasture", "permanent crop", "residential", "river", "sea lake"
    ]
    if isinstance(raw_label, int) and source_key in ["cm93", "blanchon", "tanganke"]:
        if 0 <= raw_label < len(eurosat_names):
            raw_str = eurosat_names[raw_label]

    if any(k in raw_str for k in ["annual crop", "annualcrop", "crop", "farmland", "field", "paddy field"]):
        return "Annual Crop"
    elif any(k in raw_str for k in ["permanent crop", "permanentcrop", "orchard", "garden land"]):
        return "Permanent Crop"
    elif any(k in raw_str for k in ["forest", "trees", "woods", "arbor forest"]):
        return "Forest"
    elif any(k in raw_str for k in ["herbaceous vegetation", "herbaceous", "meadow", "shrub", "park"]):
        return "Herbaceous Vegetation"
    elif any(k in raw_str for k in ["highway", "freeway", "road", "viaduct", "bridge", "traffic land", "runway"]):
        return "Highway"
    elif any(k in raw_str for k in ["industrial", "factory", "storage tanks", "port", "commercial"]):
        return "Industrial"
    elif any(k in raw_str for k in ["pasture", "grass", "bareland", "stadium", "playground", "baseball field"]):
        return "Pasture"
    elif any(k in raw_str for k in ["residential", "houses", "buildings", "center", "school"]):
        return "Residential"
    elif any(k in raw_str for k in ["river", "canal", "stream"]):
        return "River"
    elif any(k in raw_str for k in ["sea", "lake", "ocean", "water", "beach", "pond", "resort"]):
        return "Sea/Lake"

    return "Forest"


def get_dataset(source_key="blanchon"):
    """Loads and caches the Hugging Face dataset."""
    repo_id = DATASET_SOURCES.get(source_key, "blanchon/EuroSAT_RGB")
    if repo_id not in _cached_datasets:
        print(f"🔄 Loading dataset from Hugging Face: {repo_id}...")
        try:
            _cached_datasets[repo_id] = load_dataset(repo_id, split="train")
            print(f" Loaded {repo_id} with {len(_cached_datasets[repo_id])} samples.")
        except Exception as e:
            print(f" Streaming mode fallback for {repo_id} due to: {e}")
            _cached_datasets[repo_id] = load_dataset(repo_id, split="train", streaming=True)
    return _cached_datasets[repo_id]


def fetch_sample_by_index(source_key="blanchon", index=0):
    """
    Fetches an image and its unified label from the requested dataset.
    Returns:
       dict: { 'index': ..., 'source': ..., 'label': ..., 'unifiedLabel': ..., 'imageBase64': ... }
    """
    ds = get_dataset(source_key)

    # If dataset is in streaming mode, advance to index
    if hasattr(ds, "_iterator") or type(ds).__name__ == "IterableDataset":
        iterator = iter(ds)
        item = None
        for i in range(index + 1):
            try:
                item = next(iterator)
            except StopIteration:
                break
        if item is None:
            raise IndexError("Index out of range for streaming dataset")
    else:
        clamped_idx = index % len(ds)
        item = ds[clamped_idx]

    img_data = item.get("image") or item.get("img") or item.get("patch")
    raw_label = item.get("label") if "label" in item else item.get("class", "Unknown")

    if not isinstance(img_data, Image.Image):
        img_data = Image.open(io.BytesIO(img_data)).convert("RGB")
    else:
        img_data = img_data.convert("RGB")

    buffered = io.BytesIO()
    img_data.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    unified = map_label_to_unified(raw_label, source_key)

    return {
        "index": index,
        "source": DATASET_SOURCES.get(source_key, source_key),
        "sourceKey": source_key,
        "rawLabel": str(raw_label),
        "label": unified,
        "imageBase64": f"data:image/jpeg;base64,{img_b64}"
    }


def get_dataset_info():
    """Returns all available dataset sources and their metadata."""
    return [
        {
            "id": "cm93",
            "name": "cm93/eurosat",
            "hfUrl": "https://huggingface.co/datasets/cm93/eurosat",
            "type": "EuroSAT Sentinel-2 Multi-spectral",
            "badge": "Top-Down Satellite",
            "description": "Standard EuroSAT benchmark containing 27,000 Sentinel-2 satellite patches across 10 land cover classes."
        },
        {
            "id": "blanchon",
            "name": "blanchon/EuroSAT_RGB",
            "hfUrl": "https://huggingface.co/datasets/blanchon/EuroSAT_RGB",
            "type": "EuroSAT RGB Optical Bands",
            "badge": "Satellite RGB",
            "description": "Curated RGB optical satellite patches from Sentinel-2 designed specifically for computer vision models."
        },
        {
            "id": "aid",
            "name": "blanchon/AID",
            "hfUrl": "https://huggingface.co/datasets/blanchon/AID",
            "type": "Aerial Image Dataset (AID)",
            "badge": "Multi-Angle Aerial",
            "description": "10,000 aerial scene patches captured with multi-resolution sensors across nadir and oblique drone angles."
        },
        {
            "id": "gid",
            "name": "jonathan-roberts1/GID",
            "hfUrl": "https://huggingface.co/datasets/jonathan-roberts1/GID",
            "type": "Gaofen Image Dataset (GID)",
            "badge": "Road & Traffic Land",
            "description": "High-resolution Chinese Gaofen satellite imagery with fine-grained linear road networks and urban infrastructure."
        },
        {
            "id": "ucmerced",
            "name": "Project-AgML/ucm_land_use_classification",
            "hfUrl": "https://huggingface.co/datasets/Project-AgML/ucm_land_use_classification",
            "type": "UC Merced Land Use",
            "badge": "Urban Orthophotos",
            "description": "USGS high-resolution urban and rural aerial orthophotos with 1-foot spatial resolution."
        }
    ]
