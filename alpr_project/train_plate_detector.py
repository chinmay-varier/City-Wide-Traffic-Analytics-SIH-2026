"""
Downloads the Roboflow plate-detection dataset and fine-tunes YOLOv8n on it.

Run this ONCE from the alpr_project root folder (with your venv activated):
    python train_plate_detector.py

This will:
    1. Download the dataset into ./dataset/ (skipped if already present)
    2. Fine-tune yolov8n.pt on it using your GPU
    3. Save the best weights to runs/detect/plate_train/weights/best.pt
    4. Copy that file to models/plate_detector.pt so it's ready to use
       with your existing pipeline (src/config.py already points there)

Expected runtime on an RTX 4050 Laptop GPU with these settings: ~30-60 minutes
for 25 epochs at imgsz=512 on ~5,750 images. This is a reduced-time config for
a quick demo-ready model — accuracy will be lower than a full 60+ epoch run,
but should be enough to prove the detector works meaningfully better than the
generic COCO model. Increase EPOCHS back up later if you have time to spare
and want to improve accuracy further.

NOTE: everything is wrapped inside main() and guarded by
`if __name__ == "__main__":` at the bottom. This is REQUIRED on Windows --
PyTorch's DataLoader spawns worker processes, and Windows needs this guard
to avoid each worker re-executing the whole script from the top (which
would otherwise cause infinite recursion / a RuntimeError).
"""

import os
import shutil
from roboflow import Roboflow
from ultralytics import YOLO


def main():
    # -----------------------------------------------------------------------
    # 1. Download dataset (skips re-download if the folder already exists)
    # -----------------------------------------------------------------------
    DATASET_DIR = "dataset"

    if not os.path.isdir(DATASET_DIR):
        print("[train] Downloading dataset from Roboflow...")
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ROBOFLOW_API_KEY environment variable not set. "
                "Set it before running, e.g.:\n"
                "  Windows (PowerShell): $env:ROBOFLOW_API_KEY = 'your_key_here'\n"
                "  Mac/Linux:            export ROBOFLOW_API_KEY=your_key_here"
            )
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("blazingflames").project("yolov8-number-plate-detection-w5fo8")
        version = project.version(1)
        dataset = version.download("yolov8", location=DATASET_DIR)
        print(f"[train] Dataset downloaded to: {dataset.location}")
    else:
        print(f"[train] Dataset folder '{DATASET_DIR}' already exists, skipping download.")

    DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")

    if not os.path.isfile(DATA_YAML_PATH):
        raise FileNotFoundError(
            f"Couldn't find {DATA_YAML_PATH}. Check the dataset downloaded correctly "
            f"and that 'data.yaml' sits directly inside the '{DATASET_DIR}' folder."
        )

    # -----------------------------------------------------------------------
    # 2. Fine-tune YOLOv8n on the dataset
    # -----------------------------------------------------------------------
    EPOCHS = 25          # scaled down for a ~30-60 min run on RTX 4050
    IMG_SIZE = 512         # smaller than 640 = faster per-epoch, still solid for plates
    BATCH_SIZE = 16       # RTX 4050 (6GB) should comfortably handle this at imgsz=512
    DEVICE = 0             # GPU index 0 = your RTX 4050; use "cpu" as fallback if needed
    WORKERS = 4            # reduced from default 8 -- fewer spawned processes, more stable on Windows

    print("[train] Loading base model yolov8n.pt (pretrained on COCO)...")
    model = YOLO("yolov8n.pt")

    print(f"[train] Starting training: {EPOCHS} epochs, imgsz={IMG_SIZE}, batch={BATCH_SIZE}, device={DEVICE}")
    model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        patience=8,          # stop early if validation loss plateaus for 8 epochs
        project="runs/detect",
        name="plate_train",
        exist_ok=True,       # allow re-running without erroring on folder collision
    )

    # -----------------------------------------------------------------------
    # 3. Copy best weights into models/ so the main pipeline can use them
    # -----------------------------------------------------------------------
    best_weights_path = os.path.join("runs", "detect", "plate_train", "weights", "best.pt")
    target_path = os.path.join("models", "plate_detector.pt")

    os.makedirs("models", exist_ok=True)

    if os.path.isfile(best_weights_path):
        shutil.copy(best_weights_path, target_path)
        print(f"[train] Training complete. Best weights copied to: {target_path}")
        print("[train] Your src/config.py already points PLATE_DETECTOR_MODEL_PATH here.")
        print("[train] Just make sure config.py says: PLATE_DETECTOR_MODEL_PATH = \"models/plate_detector.pt\"")
    else:
        print(f"[train] WARNING: expected weights at {best_weights_path} but didn't find them. "
              f"Check the runs/detect/plate_train/weights/ folder manually.")


if __name__ == "__main__":
    main()