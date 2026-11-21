# train_yolo_barcode.py
"""
Entrenamiento de YOLO (v8s) para detección de códigos de barras.
Sin CBAM. Optimizado para objetos pequeños (imgsz=1024).
"""

import json
import shutil
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ===========================================================================
# CONFIGURACIÓN — EDITAR AQUÍ
# ===========================================================================
@dataclass
class Config:
    # --- Rutas ---
    source_dir: str = "dataset_1024"                 # entrada: images/ + labels/
    prepared_dir: str = "dataset_prepared"      # salida del preprocesamiento
    project_dir: str = "runs/yolo_barcode"  # salida del entrenamiento

    # --- Dataset ---
    classes: List[str] = field(default_factory=lambda: ["barcode"])
    val_split: float = 0.2
    seed: int = 42
    force_prepare: bool = True

    # --- Entrenamiento general ---
    # Usamos 1024 para no perder los códigos pequeños
    imgsz: int = 1024 
    batch: int = 8
    device: str = "0" # En Colab usa "0"

    # --- Fase 1: Modelo Base ---
    run_base: bool = True
    epochs_base: int = 150  # 150 épocas suele ser suficiente para empezar
    name_base: str = "yolov8s_base"

    # --- Hiperparámetros Fase 1 ---
    lr0_base: float = 0.001
    lrf_base: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs_base: float = 3.0
    patience_base: int = 50

    # --- Augmentation (Aumento de datos) ---
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 10.0     # Rotación leve
    translate: float = 0.1
    scale: float = 0.5        # Escala agresiva para simular zoom (objetos pequeños)
    shear: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0       # Mosaic ayuda a ver objetos pequeños en contextos nuevos
    mixup: float = 0.1

    deterministic: bool = True
    verbose: bool = True

CFG = Config()

# ===========================================================================
# PARSEO DE LABELS Y VERIFICACIÓN (Sin cambios, se mantiene igual)
# ===========================================================================
def _parse_label_line(parts: List[str]) -> Tuple[int, float, float, float, float]:
    cls = int(float(parts[0]))
    coords = [float(v) for v in parts[1:]]

    if len(coords) == 4:
        xc, yc, w, h = coords
        return cls, xc, yc, w, h, "bbox"

    if len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        xc = (x_min + x_max) / 2.0
        yc = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min
        return cls, xc, yc, w, h, "polygon"

    raise ValueError(f"Formato no reconocido: {len(coords)} coords")

def verify_dataset(images_dir: Path, labels_dir: Path, classes: List[str]):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in exts])
    label_files = {f.stem: f for f in labels_dir.glob("*.txt")}

    report = {
        "n_images": len(image_files), "n_labels": len(label_files),
        "images_without_label": [], "labels_without_image": [],
        "invalid_lines": [], "empty_labels": [], "out_of_range": [],
        "class_distribution": {}, "polygon_labels": 0, "bbox_labels": 0,
    }
    labels_by_stem = {}

    for img_path in image_files:
        stem = img_path.stem
        if stem not in label_files:
            report["images_without_label"].append(img_path.name)
            labels_by_stem[stem] = []
            continue

        lines = [ln.strip() for ln in label_files[stem].read_text().splitlines() if ln.strip()]
        if not lines:
            report["empty_labels"].append(stem)
            labels_by_stem[stem] = []
            continue

        parsed = []
        for i, ln in enumerate(lines):
            parts = ln.split()
            if len(parts) < 5:
                report["invalid_lines"].append((stem, i, ln))
                continue
            try:
                cls, xc, yc, w, h, fmt = _parse_label_line(parts)
            except (ValueError, IndexError):
                report["invalid_lines"].append((stem, i, ln))
                continue

            if not all(-0.01 <= v <= 1.01 for v in (xc, yc, w, h)):
                report["out_of_range"].append((stem, i, ln))
                continue

            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)

            parsed.append([cls, xc, yc, w, h])
            report["class_distribution"][cls] = report["class_distribution"].get(cls, 0) + 1
            if fmt == "polygon": report["polygon_labels"] += 1
            else: report["bbox_labels"] += 1

        labels_by_stem[stem] = parsed

    image_stems = {f.stem for f in image_files}
    for stem in label_files:
        if stem not in image_stems:
            report["labels_without_image"].append(stem)

    return [f.name for f in image_files], labels_by_stem, report

def print_report(report: Dict, classes: List[str]):
    print("\n" + "=" * 60)
    print("VERIFICACIÓN DEL DATASET")
    print("=" * 60)
    print(f"  Imágenes:              {report['n_images']}")
    print(f"  Labels:                {report['n_labels']}")
    print(f"  Imágenes sin label:    {len(report['images_without_label'])}")
    print(f"  Labels bbox:           {report.get('bbox_labels', 0)}")
    print(f"  Labels polígono:       {report.get('polygon_labels', 0)}")
    print("\n  Distribución de clases:")
    for cls, count in sorted(report["class_distribution"].items()):
        name = classes[cls] if cls < len(classes) else f"class_{cls}"
        print(f"    {cls} ({name}): {count}")

# ===========================================================================
# SPLIT POR FUENTE (Sin cambios)
# ===========================================================================
def _source_prefix(filename: str) -> str:
    stem = Path(filename).stem
    if ".rf." in stem:
        return stem.split(".rf.")[0]
    return stem

def split_train_val(files, labels_by_stem, val_split, seed):
    by_source = {}
    for f in files:
        by_source.setdefault(_source_prefix(f), []).append(f)

    sources = list(by_source.keys())
    rng = random.Random(seed)
    rng.shuffle(sources)

    n_val = max(1, int(len(sources) * val_split))
    val_sources = set(sources[:n_val])
    train_sources = set(sources[n_val:])

    train_files = [f for f in files if _source_prefix(f) in train_sources]
    val_files = [f for f in files if _source_prefix(f) in val_sources]

    print(f"\n  Split por fuente:")
    print(f"    Fuentes train: {len(train_sources)} → {len(train_files)} archivos")
    print(f"    Fuentes val:   {len(val_sources)} → {len(val_files)} archivos")
    return train_files, val_files

# ===========================================================================
# ESCRITURA DE DATASET (Sin cambios)
# ===========================================================================
def write_yolo_split(images_dir, labels_by_stem, output_dir, train_files, val_files):
    for split, files in [("train", train_files), ("val", val_files)]:
        img_out = output_dir / "yolo" / "images" / split
        lbl_out = output_dir / "yolo" / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for fname in files:
            src_img = images_dir / fname
            if not src_img.exists(): continue
            shutil.copy2(src_img, img_out / fname)
            stem = Path(fname).stem
            boxes = labels_by_stem.get(stem, [])
            lbl_path = lbl_out / (stem + ".txt")
            with open(lbl_path, "w") as f:
                for box in boxes:
                    cls, xc, yc, w, h = box
                    f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        print(f"  YOLO {split}: {len(files)} imágenes")

def write_dataset_yaml(output_dir: Path, classes: List[str]):
    yaml_path = output_dir / "yolo" / "dataset.yaml"
    content = f"""# Dataset de códigos de barras (auto-generado)
path: {output_dir.resolve() / 'yolo'}
train: images/train
val: images/val
nc: {len(classes)}
names:
"""
    for i, c in enumerate(classes):
        content += f"  {i}: {c}\n"
    yaml_path.write_text(content)
    print(f"  dataset.yaml → {yaml_path}")

def prepare_dataset(cfg: Config) -> Path:
    source = Path(cfg.source_dir)
    output = Path(cfg.prepared_dir)
    images_dir = source / "images"
    labels_dir = source / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Se esperaba {images_dir} y {labels_dir}")

    output.mkdir(parents=True, exist_ok=True)
    print("\nPreparando dataset...")
    image_files, labels_by_stem, report = verify_dataset(images_dir, labels_dir, cfg.classes)
    print_report(report, cfg.classes)

    train_files, val_files = split_train_val(image_files, labels_by_stem, cfg.val_split, cfg.seed)

    print("\nEscribiendo YOLO...")
    write_yolo_split(images_dir, labels_by_stem, output, train_files, val_files)
    write_dataset_yaml(output, cfg.classes)
    return output / "yolo" / "dataset.yaml"

def ensure_dataset(cfg: Config) -> Path:
    prepared = Path(cfg.prepared_dir)
    yaml_path = prepared / "yolo" / "dataset.yaml"
    if yaml_path.exists() and not cfg.force_prepare:
        print(f"✓ Dataset ya preparado: {yaml_path}")
        return yaml_path
    return prepare_dataset(cfg)

# ===========================================================================
# ENTRENAMIENTO
# ===========================================================================
def train_yolov8_base(cfg: Config, data_yaml: Path) -> str:
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO: YOLOv8s (Optimizado para objetos pequeños)")
    print("=" * 60)

    # Usamos yolov8s.pt (es más moderno y preciso que v5)
    model = YOLO("yolov8s.pt") 
    
    model.train(
        data=str(data_yaml),
        epochs=cfg.epochs_base,
        imgsz=cfg.imgsz, # 1024 para no perder los códigos pequeños
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.project_dir,
        name=cfg.name_base,
        lr0=cfg.lr0_base,
        lrf=cfg.lrf_base,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs_base,
        hsv_h=cfg.hsv_h, hsv_s=cfg.hsv_s, hsv_v=cfg.hsv_v,
        degrees=cfg.degrees, translate=cfg.translate, scale=cfg.scale,
        shear=cfg.shear, flipud=cfg.flipud, fliplr=cfg.fliplr,
        mosaic=cfg.mosaic, mixup=cfg.mixup,
        save=True, save_period=10,
        patience=cfg.patience_base,
        deterministic=cfg.deterministic,
        seed=cfg.seed,
        verbose=cfg.verbose,
    )
    
    real_save_dir = Path(model.trainer.save_dir)
    best = real_save_dir / "weights" / "best.pt"
    print(f"\n✓ Ruta real del run: {real_save_dir}")
    print(f"✓ Mejores pesos: {best}")
    return str(best)

# ===========================================================================
# MAIN
# ===========================================================================
def print_config(cfg: Config):
    print("\n" + "=" * 60)
    print("CONFIGURACIÓN")
    print("=" * 60)
    for k, v in cfg.__dict__.items():
        print(f"  {k:24s} = {v}")
    print("=" * 60)

def main(cfg: Config = CFG):
    print_config(cfg)
    data_yaml = ensure_dataset(cfg)
    
    if cfg.run_base:
        train_yolov8_base(cfg, data_yaml)

    print("\n✓ Entrenamiento completado.")

if __name__ == "__main__":
    main()