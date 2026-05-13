"""
train.py — YOLOv8 Fine-tuning on NEU-DET Steel Surface Defect Dataset
======================================================================
Dataset: NEU Surface Defect Database (Northeastern University)
Classes: crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches
Download: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

Steps:
1. Download and convert NEU-DET to YOLO format (see convert_neu_det below)
2. Run: python model/train.py
3. Best weights saved to runs/detect/steel_defect/weights/best.pt
4. Copy best.pt to model/ for inference

Requirements:
    pip install ultralytics torch torchvision
"""

import os
import shutil
import yaml
import random
from pathlib import Path

# ── 1. Dataset preparation ────────────────────────────────────────────────────

NEU_DET_CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']

def create_dataset_yaml(dataset_root: str = 'data/neu_det') -> str:
    """Create YOLO-format dataset YAML config."""
    yaml_content = {
        'path': os.path.abspath(dataset_root),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(NEU_DET_CLASSES),
        'names': NEU_DET_CLASSES
    }
    yaml_path = os.path.join(dataset_root, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"[INFO] Dataset YAML written to {yaml_path}")
    return yaml_path


def convert_neu_det_to_yolo(src_dir: str, dst_dir: str = 'data/neu_det',
                             train_split: float = 0.85):
    """
    Convert NEU-DET annotation format to YOLO format.
    NEU-DET provides PASCAL VOC XML annotations.

    Args:
        src_dir: Path to downloaded NEU-DET dataset root
        dst_dir: Output directory for YOLO-format data
        train_split: Fraction of images for training
    """
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        print("[ERROR] xml.etree.ElementTree not available")
        return

    class_to_id = {c: i for i, c in enumerate(NEU_DET_CLASSES)}
    splits = {'train': [], 'val': []}

    ann_dir = os.path.join(src_dir, 'ANNOTATIONS')
    img_dir = os.path.join(src_dir, 'IMAGES')
    if not os.path.exists(ann_dir):
        print(f"[ERROR] Annotations not found at {ann_dir}")
        print("[INFO] Download NEU-DET from http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html")
        return

    xml_files = list(Path(ann_dir).glob('*.xml'))
    random.shuffle(xml_files)
    split_idx = int(len(xml_files) * train_split)
    train_files = xml_files[:split_idx]
    val_files = xml_files[split_idx:]
    splits['train'] = train_files
    splits['val'] = val_files

    for split, files in splits.items():
        img_out = os.path.join(dst_dir, 'images', split)
        lbl_out = os.path.join(dst_dir, 'labels', split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for xml_path in files:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            fname = root.find('filename').text
            size = root.find('size')
            W = int(size.find('width').text)
            H = int(size.find('height').text)

            yolo_lines = []
            for obj in root.findall('object'):
                cls_name = obj.find('name').text.lower().replace(' ', '_')
                if cls_name not in class_to_id:
                    continue
                cls_id = class_to_id[cls_name]
                bb = obj.find('bndbox')
                x1 = float(bb.find('xmin').text); y1 = float(bb.find('ymin').text)
                x2 = float(bb.find('xmax').text); y2 = float(bb.find('ymax').text)
                cx = (x1 + x2) / 2 / W; cy = (y1 + y2) / 2 / H
                w = (x2 - x1) / W;      h = (y2 - y1) / H
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            stem = xml_path.stem
            lbl_path = os.path.join(lbl_out, stem + '.txt')
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                src_img = os.path.join(img_dir, fname) if '.' in fname else os.path.join(img_dir, stem + ext)
                if os.path.exists(src_img):
                    shutil.copy(src_img, os.path.join(img_out, os.path.basename(src_img)))
                    break

    print(f"[INFO] Converted {len(train_files)} train + {len(val_files)} val images to YOLO format")


# ── 2. Training ───────────────────────────────────────────────────────────────

def train(dataset_yaml: str = 'data/neu_det/dataset.yaml',
          model_size: str = 'n',
          epochs: int = 80,
          img_size: int = 640,
          batch: int = 16,
          device: str = '0'):
    """
    Fine-tune YOLOv8 on NEU-DET steel defect dataset.

    Args:
        model_size: 'n' (nano, fastest), 's' (small), 'm' (medium), 'l' (large)
        epochs: Training epochs (80 recommended for hackathon)
        img_size: Input resolution (640 standard)
        batch: Batch size (adjust for GPU memory)
        device: '0' for GPU, 'cpu' for CPU
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        return

    if not os.path.exists(dataset_yaml):
        print(f"[ERROR] Dataset YAML not found: {dataset_yaml}")
        print("[INFO] Run convert_neu_det_to_yolo() first, then create_dataset_yaml()")
        return

    # Load pretrained YOLOv8 (downloads automatically on first run)
    model = YOLO(f'yolov8{model_size}.pt')

    print(f"\n[INFO] Starting training: YOLOv8{model_size.upper()} · {epochs} epochs · {img_size}px · batch {batch}")
    print(f"[INFO] Dataset: {dataset_yaml}")
    print(f"[INFO] Classes: {NEU_DET_CLASSES}\n")

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        device=device,
        project='runs/detect',
        name='steel_defect',
        exist_ok=True,

        # Augmentation for steel surface images
        hsv_h=0.01,        # Minimal hue shift (steel is grey)
        hsv_s=0.3,         # Saturation variation
        hsv_v=0.4,         # Brightness variation (lighting conditions)
        degrees=15,        # Rotation (coils can be oriented differently)
        translate=0.1,
        scale=0.4,
        fliplr=0.5,        # Horizontal flip
        flipud=0.3,        # Vertical flip
        mosaic=0.8,        # Mosaic augmentation
        mixup=0.1,

        # Training hyperparams
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        close_mosaic=10,

        # Output
        save=True,
        save_period=20,
        plots=True,
        val=True,
    )

    best_path = f'runs/detect/steel_defect/weights/best.pt'
    if os.path.exists(best_path):
        shutil.copy(best_path, 'model/best.pt')
        print(f"\n[SUCCESS] Best weights copied to model/best.pt")
        print(f"[INFO] mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    return results


def evaluate(model_path: str = 'model/best.pt', dataset_yaml: str = 'data/neu_det/dataset.yaml'):
    """Run validation and print per-class metrics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return
    model = YOLO(model_path)
    metrics = model.val(data=dataset_yaml)
    print("\n── Per-Class Results ──────────────────────────────")
    for i, name in enumerate(NEU_DET_CLASSES):
        print(f"  {name:<22} mAP50: {metrics.box.maps[i]:.3f}")
    print(f"\n  Overall mAP50:   {metrics.box.map50:.3f}")
    print(f"  Overall mAP50-95:{metrics.box.map:.3f}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train YOLOv8 on NEU-DET Steel Defect Dataset')
    parser.add_argument('--neu-src', default='', help='Path to downloaded NEU-DET dataset')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--model', default='n', choices=['n','s','m','l'])
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='0')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    if args.eval_only:
        evaluate()
    else:
        if args.neu_src:
            print("[STEP 1] Converting NEU-DET to YOLO format...")
            convert_neu_det_to_yolo(args.neu_src)
            yaml_path = create_dataset_yaml()
        else:
            yaml_path = 'data/neu_det/dataset.yaml'
            print(f"[INFO] Assuming YOLO-format dataset at {yaml_path}")

        print("\n[STEP 2] Training YOLOv8...")
        train(yaml_path, model_size=args.model, epochs=args.epochs,
              batch=args.batch, device=args.device)

        print("\n[STEP 3] Evaluating on validation set...")
        evaluate()
