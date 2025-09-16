import os
import multiprocessing
import torch
from ultralytics import YOLO


def main() -> None:
    """Entrypoint for training YOLOv11 model on custom dataset."""

    # Path to your dataset config
    data_yaml_path = "data.yaml"

    if not os.path.exists(data_yaml_path):
        print(f"❌ Error: '{data_yaml_path}' not found. "
              f"Please make sure it's inside this folder: {os.getcwd()}")
        return

    # Load pretrained YOLOv11 model (Ultralytics auto-downloads if not present)
    weights = "yolo11n.pt"
    print(f"📥 Loading YOLO model: {weights}")
    model = YOLO(weights)

    # Choose device: GPU (0) if available, else CPU
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Using device: {device}")

    # Start training
    print("🚀 Starting training...")
    results = model.train(
        data=data_yaml_path,  # dataset config
        epochs=300,           # number of training epochs
        imgsz=640,            # image size
        device=device,        # device selection
        workers=2,            # safer on Windows (reduce workers)
        batch=16              # batch size (adjust if OOM)
    )

    print("✅ Training complete!")
    print(results)


if __name__ == "__main__":
    # Windows requires this guard for multiprocessing
    multiprocessing.freeze_support()
    main()
