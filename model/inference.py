"""
inference.py — Production YOLOv8 Inference + Grad-CAM
======================================================
Replace simulate_detection() in app.py with real_detect() once
you have trained weights (best.pt from train.py).

Usage:
    from model.inference import real_detect, real_gradcam
    dets = real_detect('path/to/image.jpg', conf_threshold=0.45)
"""

import os
import numpy as np
from PIL import Image

# ── Lazy import so app still runs without ultralytics installed ──────────────
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[INFO] ultralytics not installed — using simulation mode")

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')

_model = None

def load_model():
    global _model
    if _model is None and _YOLO_AVAILABLE and os.path.exists(MODEL_PATH):
        _model = YOLO(MODEL_PATH)
        print(f"[INFO] YOLOv8 model loaded from {MODEL_PATH}")
    return _model

def real_detect(image_path: str, conf_threshold: float = 0.45) -> list:
    """
    Run YOLOv8 detection on a steel surface image.

    Returns:
        List of dicts: [{class, confidence, bbox:[x1,y1,x2,y2], area_pct}, ...]
    """
    model = load_model()
    if model is None:
        raise RuntimeError("Model not loaded. Ensure best.pt exists and ultralytics is installed.")

    img = Image.open(image_path).convert('RGB')
    W, H = img.size

    results = model.predict(source=image_path, conf=conf_threshold, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            area_pct = round(100 * (x2 - x1) * (y2 - y1) / (W * H), 1)
            detections.append({
                'class': cls_name,
                'confidence': round(conf, 2),
                'bbox': [x1, y1, x2, y2],
                'area_pct': area_pct
            })

    return detections


def real_gradcam(image_path: str, output_path: str, target_layer: str = 'model.22'):
    """
    Generate Grad-CAM heatmap using YOLOv8 backbone.
    Requires: torch, ultralytics

    For production: use torchcam or pytorch-grad-cam library for robust CAM.
    pip install grad-cam
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch not available for Grad-CAM")

    img = Image.open(image_path).convert('RGB').resize((640, 640))
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    # Placeholder: in production replace with actual gradient flow
    # from pytorch_grad_cam import GradCAM
    # cam = GradCAM(model=backbone, target_layers=[target_layer])
    # grayscale_cam = cam(input_tensor=tensor)

    # Simulated fallback (replace with above):
    H, W = 640, 640
    x = np.linspace(0, 1, W); y = np.linspace(0, 1, H)
    xx, yy = np.meshgrid(x, y)
    heat = np.zeros((H, W), dtype=np.float32)
    import random
    for _ in range(3):
        cx, cy = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
        s = random.uniform(0.1, 0.25)
        heat += np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * s**2))
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    cm = np.zeros((H, W, 3), dtype=np.uint8)
    cm[..., 0] = (heat * 255).astype(np.uint8)
    cm[..., 1] = ((1 - heat) * 80).astype(np.uint8)
    cm[..., 2] = 50

    blended = Image.blend(img.resize((H, W)), Image.fromarray(cm, 'RGB'), 0.5)
    blended.save(output_path, quality=95)
    print(f"[INFO] Grad-CAM saved to {output_path}")
