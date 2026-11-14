# train_yolo_cbam.py
"""
Entrenamiento de YOLOv5s para detección de códigos de barras.

Los parámetros están definidos en la clase `Config` al inicio del archivo..

Soporta:
    - Labels en formato bbox YOLO estándar: [cls, xc, yc, w, h]
    - Labels en formato polígono (segmentación): [cls, x1, y1, x2, y2, ...]
      → se convierten automáticamente a bbox (min/max de coordenadas).

Estructura:
    - Config: parámetros editables.
    - verify_dataset(): parsea labels (bbox o polígono).
    - split_train_val(): split por fuente (evita leakage por augmentation).
    - prepare_dataset(): orquesta verify + split + write YOLO/COCO.
    - train_yolov5_base(): Fase 1.
    - train_yolov5_cbam(): Fase 2 (opcional).
    - main(): orquesta el flujo.
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
    source_dir: str = "dataset"                 # entrada: images/ + labels/
    prepared_dir: str = "dataset_prepared"      # salida del preprocesamiento
    project_dir: str = "runs/detect/runs/barcode"           # donde se guardan los runs

    # --- Dataset ---
    classes: List[str] = field(default_factory=lambda: ["barcode"])
    val_split: float = 0.2
    seed: int = 42
    force_prepare: bool = False                  # True → regenerar aunque exista

    # --- Entrenamiento general ---
    imgsz: int = 640
    batch: int = -1
    device: str = "0"                         # "0", "0,1", "cpu"

    # --- Fase 1: YOLOv5s base ---
    run_base: bool = True
    epochs_base: int = 300
    name_base: str = "yolov5s_base"

    # --- Fase 2: fine-tuning con CBAM ---
    run_cbam: bool = False
    epochs_cbam: int = 15
    name_cbam: str = "yolov5s_cbam"
    base_weights: str = "runs/detect/runs/barcode/yolov5s_base/weights/best.pt"                      # vacío → usar el de Fase 1

    # --- Hiperparámetros Fase 1 ---
    lr0_base: float = 0.001
    lrf_base: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs_base: float = 3.0
    patience_base: int = 50

    # --- Hiperparámetros Fase 2 (CBAM) ---
    lr0_cbam: float = 0.001
    lrf_cbam: float = 0.01
    warmup_epochs_cbam: float = 1.0
    patience_cbam: int = 10

    # --- Augmentation ---
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 5.0
    translate: float = 0.1
    scale: float = 0.3
    shear: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 0.5
    mixup: float = 0.0

    # --- Reproducibilidad ---
    deterministic: bool = True

    # --- Logging ---
    verbose: bool = True


# Instancia global editable.
CFG = Config()


# ===========================================================================
# PARSEO DE LABELS
# ===========================================================================
def _parse_label_line(parts: List[str]) -> Tuple[int, float, float, float, float]:
    """
    Parsea una línea de label y devuelve (cls, xc, yc, w, h) normalizado.

    Soporta:
        - bbox YOLO estándar: [cls, xc, yc, w, h]  (5 tokens)
        - polígono: [cls, x1, y1, x2, y2, ...]     (≥7 tokens, par)

    Raises:
        ValueError si el formato no es reconocible.
    """
    cls = int(float(parts[0]))
    coords = [float(v) for v in parts[1:]]

    if len(coords) == 4:
        # bbox estándar.
        xc, yc, w, h = coords
        return cls, xc, yc, w, h, "bbox"

    if len(coords) >= 6 and len(coords) % 2 == 0:
        # Polígono → bbox (min/max).
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


# ===========================================================================
# VERIFICACIÓN DEL DATASET
# ===========================================================================
def verify_dataset(images_dir: Path, labels_dir: Path, classes: List[str]):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in exts])
    label_files = {f.stem: f for f in labels_dir.glob("*.txt")}

    report = {
        "n_images": len(image_files),
        "n_labels": len(label_files),
        "images_without_label": [],
        "labels_without_image": [],
        "invalid_lines": [],
        "empty_labels": [],
        "out_of_range": [],
        "class_distribution": {},
        "polygon_labels": 0,
        "bbox_labels": 0,
    }
    labels_by_stem = {}

    for img_path in image_files:
        stem = img_path.stem
        if stem not in label_files:
            report["images_without_label"].append(img_path.name)
            labels_by_stem[stem] = []
            continue

        lines = [
            ln.strip() for ln in label_files[stem].read_text().splitlines() if ln.strip()
        ]
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

            # Validar rangos.
            if not all(-0.01 <= v <= 1.01 for v in (xc, yc, w, h)):
                report["out_of_range"].append((stem, i, ln))
                continue

            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)

            parsed.append([cls, xc, yc, w, h])
            report["class_distribution"][cls] = (
                report["class_distribution"].get(cls, 0) + 1
            )
            if fmt == "polygon":
                report["polygon_labels"] += 1
            else:
                report["bbox_labels"] += 1

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
    print(f"  Imágenes sin label:    {len(report['images_without_label'])} (serán negativos)")
    print(f"  Labels sin imagen:     {len(report['labels_without_image'])}")
    print(f"  Labels vacíos:         {len(report['empty_labels'])}")
    print(f"  Líneas inválidas:      {len(report['invalid_lines'])}")
    print(f"  Valores fuera rango:   {len(report['out_of_range'])}")
    print(f"  Labels bbox:           {report.get('bbox_labels', 0)}")
    print(f"  Labels polígono:       {report.get('polygon_labels', 0)} (convertidos a bbox)")
    print(f"\n  Distribución de clases:")
    for cls, count in sorted(report["class_distribution"].items()):
        name = classes[cls] if cls < len(classes) else f"class_{cls}"
        print(f"    {cls} ({name}): {count}")


# ===========================================================================
# SPLIT POR FUENTE
# ===========================================================================
def _source_prefix(filename: str) -> str:
    """
    Extrae el prefijo de fuente de un nombre Roboflow.
    Ej: '-P-ISBN_18_0079_jpg.rf.b5a2f089.jpg' → '-P-ISBN_18_0079_jpg'
    Todas las variantes aumentadas comparten este prefijo.
    """
    stem = Path(filename).stem
    if ".rf." in stem:
        return stem.split(".rf.")[0]
    return stem


def split_train_val(files, labels_by_stem, val_split, seed):
    """Split agrupado por fuente para evitar leakage de augmentation."""
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

    def stats(lst):
        pos = sum(1 for f in lst if labels_by_stem.get(Path(f).stem, []))
        return len(lst), pos

    n_tr, n_pos_tr = stats(train_files)
    n_va, n_pos_va = stats(val_files)

    print(f"\n  Split por fuente (evita leakage de augmentation):")
    print(f"    Fuentes train: {len(train_sources)} → {n_tr} archivos ({n_pos_tr} con barcode)")
    print(f"    Fuentes val:   {len(val_sources)} → {n_va} archivos ({n_pos_va} con barcode)")

    return train_files, val_files


# ===========================================================================
# ESCRITURA DE DATASET
# ===========================================================================
def write_yolo_split(
    images_dir: Path,
    labels_by_stem: Dict[str, List[List[float]]],
    output_dir: Path,
    train_files: List[str],
    val_files: List[str],
):
    """
    Copia imágenes y escribe labels en formato YOLO bbox estándar.
    IMPORTANTE: no copia las labels originales (que pueden ser polígonos);
    las reescribe desde `labels_by_stem` ya convertidas a bbox.
    """
    for split, files in [("train", train_files), ("val", val_files)]:
        img_out = output_dir / "yolo" / "images" / split
        lbl_out = output_dir / "yolo" / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for fname in files:
            src_img = images_dir / fname
            if not src_img.exists():
                continue

            # Copiar imagen.
            shutil.copy2(src_img, img_out / fname)

            # Escribir label (bbox YOLO) desde el parser.
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


def yolo_to_coco_bbox(yolo_bbox, img_w, img_h):
    xc, yc, w, h = yolo_bbox
    return [(xc - w / 2) * img_w, (yc - h / 2) * img_h, w * img_w, h * img_h]


def write_coco_split(images_dir, labels_by_stem, output_dir, files, split_name, classes):
    img_out = output_dir / "coco" / split_name / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": n, "supercategory": "object"}
            for i, n in enumerate(classes)
        ],
    }
    ann_id = 1
    for img_id, fname in enumerate(sorted(files), start=1):
        src = images_dir / fname
        if not src.exists():
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        shutil.copy2(src, img_out / fname)
        coco["images"].append({
            "id": img_id, "file_name": fname, "width": w, "height": h,
        })

        for yolo_box in labels_by_stem.get(Path(fname).stem, []):
            cls = int(yolo_box[0])
            if cls >= len(classes):
                continue
            bbox = yolo_to_coco_bbox(yolo_box[1:], w, h)
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cls,
                "bbox": [round(v, 2) for v in bbox],
                "area": round(bbox[2] * bbox[3], 2),
                "iscrowd": 0,
            })
            ann_id += 1

    ann_path = output_dir / "coco" / split_name / "annotations.json"
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"  COCO {split_name}: {len(coco['images'])} imgs, {len(coco['annotations'])} anns")


def prepare_dataset(cfg: Config) -> Path:
    """Verifica, hace split y escribe YOLO + COCO. Devuelve dataset.yaml."""
    source = Path(cfg.source_dir)
    output = Path(cfg.prepared_dir)
    images_dir = source / "images"
    labels_dir = source / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Se esperaba {images_dir} y {labels_dir}")

    output.mkdir(parents=True, exist_ok=True)
    print("\nPreparando dataset...")
    image_files, labels_by_stem, report = verify_dataset(
        images_dir, labels_dir, cfg.classes
    )
    print_report(report, cfg.classes)

    train_files, val_files = split_train_val(
        image_files, labels_by_stem, cfg.val_split, cfg.seed
    )

    print("\nEscribiendo YOLO...")
    write_yolo_split(images_dir, labels_by_stem, output, train_files, val_files)
    write_dataset_yaml(output, cfg.classes)

    print("\nEscribiendo COCO...")
    write_coco_split(images_dir, labels_by_stem, output, train_files, "train", cfg.classes)
    write_coco_split(images_dir, labels_by_stem, output, val_files, "val", cfg.classes)

    (output / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n✓ Dataset preparado en {output}")
    return output / "yolo" / "dataset.yaml"


def ensure_dataset(cfg: Config) -> Path:
    """Prepara el dataset si no existe o si force_prepare=False."""
    prepared = Path(cfg.prepared_dir)
    yaml_path = prepared / "yolo" / "dataset.yaml"

    if yaml_path.exists() and not cfg.force_prepare:
        print(f"✓ Dataset ya preparado: {yaml_path}")
        print("  (usá force_prepare=True en Config para regenerar)")
        return yaml_path

    return prepare_dataset(cfg)


# ===========================================================================
# ENTRENAMIENTO
# ===========================================================================
def train_yolov5_base(cfg: Config, data_yaml: Path) -> str:
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO: YOLOv5s base")
    print("=" * 60)

    model = YOLO("yolov5su.pt")
    model.train(
        data=str(data_yaml),
        epochs=cfg.epochs_base,
        imgsz=cfg.imgsz,
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
    # Ultralytics puede reescribir la ruta (prefix 'runs/detect/',
    # sufijo '-2' si el nombre ya existe). Usar el save_dir real.
    real_save_dir = Path(model.trainer.save_dir)
    best = real_save_dir / "weights" / "best.pt"
    print(f"\n✓ Ruta real del run: {real_save_dir}")
    print(f"✓ Mejores pesos: {best}")
    return str(best)


def train_yolov5_cbam(cfg: Config, data_yaml: Path, base_weights: str) -> str:
    import sys
    import torch
    from ultralytics import YOLO

    # Import robusto: agregar la carpeta 'modules' del proyecto al sys.path.
    project_root = Path(__file__).resolve().parent
    modules_dir = project_root / "modules"
    if str(modules_dir) not in sys.path:
        sys.path.insert(0, str(modules_dir))

    from yolo_cbam import attach_cbam_hooks

    print("\n" + "=" * 60)
    print("FINE-TUNING: YOLOv5s-CBAM")
    print("=" * 60)

    model = YOLO(base_weights)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cbam_modules = attach_cbam_hooks(model.model, device=dev)

    for n, p in model.model.named_parameters():
        if "cbam" not in n.lower() and "head" not in n.lower():
            p.requires_grad = False

    model.train(
        data=str(data_yaml),
        epochs=cfg.epochs_cbam,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.project_dir,
        name=cfg.name_cbam,
        lr0=cfg.lr0_cbam,
        lrf=cfg.lrf_cbam,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs_cbam,
        hsv_h=cfg.hsv_h, hsv_s=cfg.hsv_s, hsv_v=cfg.hsv_v,
        degrees=cfg.degrees, translate=cfg.translate, scale=cfg.scale,
        shear=cfg.shear, flipud=cfg.flipud, fliplr=cfg.fliplr,
        mosaic=cfg.mosaic, mixup=cfg.mixup,
        save=True, save_period=5,
        patience=cfg.patience_cbam,
        deterministic=cfg.deterministic,
        seed=cfg.seed,
        verbose=cfg.verbose,
    )

    # Ruta real donde ultralytics guardó (puede tener prefijos/sufijos).
    real_save_dir = Path(model.trainer.save_dir)
    real_save_dir.mkdir(parents=True, exist_ok=True)

    cbam_sd = cbam_modules.state_dict()
    cbam_path = real_save_dir / "cbam_weights.pt"
    torch.save(cbam_sd, cbam_path)

    best = real_save_dir / "weights" / "best.pt"
    print(f"\n✓ Ruta real del run: {real_save_dir}")
    print(f"✓ Pesos CBAM: {cbam_path}")
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

    base_weights = cfg.base_weights or None
    if cfg.run_base:
        base_weights = train_yolov5_base(cfg, data_yaml)

    if cfg.run_cbam:
        if base_weights is None:
            raise ValueError(
                "run_cbam=True pero no hay base_weights. "
                "Activá run_base=True o poné base_weights en Config."
            )
        train_yolov5_cbam(cfg, data_yaml, base_weights)

    print("\n✓ Entrenamiento completado.")


if __name__ == "__main__":
    main()