import os
import sys
import subprocess
import time

TRAINING_SCRIPTS = [
    {
        "name": "Multi-Dataset Harmonizer & Loader (AID + GID)",
        "file": "load_multi_datasets.py",
        "checkpoint": None,
        "description": "Loads and unifies remote sensing datasets into 10 classes"
    },
    {
        "name": "Road & Linear Infrastructure Pipeline",
        "file": "train_with_road_datasets.py",
        "checkpoint": "road_optimized_resnet50.pth",
        "description": "Tiling, road oversampling, and class-weighted loss for highway recall"
    },
    {
        "name": "Mixed-Perspective Invariant Pipeline",
        "file": "train_mixed_perspective.py",
        "checkpoint": "mixed_perspective_resnet50.pth",
        "description": "Albumentations perspective transforms (drone, nadir, oblique views)"
    },
    {
        "name": "Terrain & Arid Land Pipeline",
        "file": "train_terrain_model.py",
        "checkpoint": "terrain_optimized_resnet50.pth",
        "description": "12-class topography training for Mountain/Hill and Sand/Barren land"
    },
    {
        "name": "Master Multi-Source Mixed Pipeline",
        "file": "train_master_mixed.py",
        "checkpoint": "mixed_perspective_resnet50.pth",
        "description": "Ingests EuroSAT + UCMerced with 10-class Seaborn confusion matrix"
    }
]

def run_script(script_info, index, total):
    print("\n" + "=" * 75)
    print(f"[{index}/{total}] STARTING: {script_info['name']}")
    print(f"📄 Script File: {script_info['file']}")
    print(f"🎯 Objective  : {script_info['description']}")
    if script_info['checkpoint']:
        print(f"💾 Checkpoint : {script_info['checkpoint']}")
    print("=" * 75 + "\n")

    cmd = [sys.executable, script_info['file']]
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n✅ [{index}/{total}] COMPLETED: {script_info['name']} in {elapsed:.1f}s")
    else:
        print(f"\n❌ [{index}/{total}] FAILED: {script_info['name']} (Exit Code: {result.returncode})")

    return result.returncode == 0

def main():
    print("=" * 75)
    print("🛰️  MASTER TRAINING RUNNER - ALL SATELLITE & TERRAIN MODELS")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    total = len(TRAINING_SCRIPTS)
    passed = 0

    for idx, script in enumerate(TRAINING_SCRIPTS, 1):
        success = run_script(script, idx, total)
        if success:
            passed += 1

    print("\n" + "=" * 75)
    print(f"🏁 ALL TRAININGS FINISHED: {passed}/{total} Pipelines Completed")
    print("=" * 75)

if __name__ == "__main__":
    main()
