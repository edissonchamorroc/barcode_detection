# train_swin_fcos.py
"""
Entrenamiento del FCOS head de SwinROIProposer.

Dataset: dataset_prepared/coco/train/ y val/ (generados por train_yolo_cbam.py).
Los parámetros están en la clase Config al inicio. Editá y ejecutá.

Uso:
    python train_swin_fcos.py
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import del modelo.
_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root / "modules"))
from swin_transformer import SwinROIProposer
from fcos_losses import FCOSLoss


# ===========================================================================
# CONFIGURACIÓN — EDITAR AQUÍ
# ===========================================================================
@dataclass
class Config:
    # --- Rutas ---
    train_images: str = "dataset_prepared/coco/train/images"
    train_ann: str = "dataset_prepared/coco/train/annotations.json"
    val_images: str = "dataset_prepared/coco/val/images"
    val_ann: str = "dataset_prepared/coco/val/annotations.json"
    output_dir: str = "runs/swin_fcos"

    # --- Dataset ---
    imgsz: int = 384                    # múltiplo de 32, coincide con el dataset
    num_classes: int = 2                # 0=fondo, 1=barcode

    # --- Entrenamiento ---
    epochs: int = 30
    batch_size: int = 4
    num_workers: int = 2                # 0 para Windows
    device: str = "0"                 # "0" si hay GPU


    # --- Optimizador ---
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    freeze_backbone_epochs: int = 0   # ← NUEVO. Congelar Swin por N epochs iniciales


    # --- Loss ---
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    loss_reg_weight: float = 1.0
    loss_cnt_weight: float = 1.0

    # --- Logging / Checkpoints ---
    save_every: int = 5                 # guardar checkpoint cada N epochs
    val_every: int = 2                  # validar cada N epochs
    print_every: int = 20               # imprimir batch cada N

    # --- Reproducibilidad ---
    seed: int = 42
    deterministic: bool = True


CFG = Config()


# ===========================================================================
# DATASET
# ===========================================================================
class CocoBarcodeDataset(Dataset):
    """
    Dataset que carga imágenes + anotaciones COCO.
    Devuelve: (imagen_tensor, gt_boxes [N,4], gt_labels [N]).
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        images_dir: str,
        ann_path: str,
        imgsz: int = 384,
        augment: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.imgsz = imgsz
        self.augment = augment

        with open(ann_path) as f:
            coco = json.load(f)

        # Mapear image_id → file_name y dimensiones.
        self.id_to_info = {
            img["id"]: {
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
            }
            for img in coco["images"]
        }

        # Agrupar anotaciones por imagen.
        self.anns_by_img = {}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0) == 1:
                continue
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = sorted(self.id_to_info.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.id_to_info[img_id]

        # Cargar imagen.
        img_path = self.images_dir / info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)

        h, w = img.shape[:2]

        # Resize si es necesario (mantener aspect con padding).
        if h != self.imgsz or w != self.imgsz:
            img, scale, pad = self._resize_keep_aspect(img, self.imgsz)
        else:
            scale = 1.0
            pad = (0, 0)

        # Anotaciones → [N, 4] con [x1, y1, x2, y2].
        anns = self.anns_by_img.get(img_id, [])
        gt_boxes = []
        gt_labels = []
        for a in anns:
            x, y, bw, bh = a["bbox"]  # COCO: x, y, w, h
            # Aplicar resize + pad.
            x1 = x * scale + pad[0]
            y1 = y * scale + pad[1]
            x2 = (x + bw) * scale + pad[0]
            y2 = (y + bh) * scale + pad[1]
            gt_boxes.append([x1, y1, x2, y2])
            gt_labels.append(int(a["category_id"]))

        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4)
        gt_labels = torch.tensor(gt_labels, dtype=torch.long)

        # Augmentation: horizontal flip.
        if self.augment and torch.rand(1).item() < 0.5:
            img = cv2.flip(img, 1)
            if gt_boxes.numel() > 0:
                x1_new = self.imgsz - gt_boxes[:, 2]
                x2_new = self.imgsz - gt_boxes[:, 0]
                gt_boxes = torch.stack(
                    [x1_new, gt_boxes[:, 1], x2_new, gt_boxes[:, 3]], dim=-1
                )

        # Normalizar y convertir a tensor CHW.
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = (x - self.IMAGENET_MEAN) / self.IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))
        tensor = torch.from_numpy(x).contiguous()

        return tensor, gt_boxes, gt_labels

    def _resize_keep_aspect(self, img, target_size):
        """Resize con letterbox (padding gris)."""
        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = target_size - new_h
        pad_w = target_size - new_w
        top = pad_h // 2
        left = pad_w // 2

        out = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        out[top:top + new_h, left:left + new_w] = resized

        return out, scale, (left, top)


def collate_fn(batch):
    """Collate personalizado: NO apila GT boxes (tamaños variables)."""
    images = torch.stack([b[0] for b in batch], dim=0)
    boxes_list = [b[1] for b in batch]
    labels_list = [b[2] for b in batch]
    return images, boxes_list, labels_list


# ===========================================================================
# Scheduler: warmup lineal + cosine
# ===========================================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ===========================================================================
# Validación
# ===========================================================================
@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for images, boxes_list, _ in val_loader:
        images = images.to(device)
        with torch.amp.autocast("cuda", enabled=(device.startswith("cuda"))):
            features = model.backbone(images)
            features = model._to_nchw(features, model.out_channels)
            fpn_out = model.fpn({str(i): f for i, f in enumerate(features)})
            fpn_features = list(fpn_out.values())
            cls_logits, reg_preds, cnt_preds = model.head(fpn_features)
            loss, _ = criterion(cls_logits, reg_preds, cnt_preds, boxes_list)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


# ===========================================================================
# Main
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

    # Reproducibilidad.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = cfg.device
    device_str = f"cuda:{device}" if device.isdigit() else device
    print(f"\nDevice: {device_str}")

    # Dataset.
    print("\nCargando datasets...")
    train_ds = CocoBarcodeDataset(
        cfg.train_images, cfg.train_ann, cfg.imgsz, augment=True,
    )
    val_ds = CocoBarcodeDataset(
        cfg.val_images, cfg.val_ann, cfg.imgsz, augment=False,
    )
    print(f"  Train: {len(train_ds)} imágenes")
    print(f"  Val:   {len(val_ds)} imágenes")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=False,
    )

    # Modelo.
    print("\nCargando SwinROIProposer...")
    model = SwinROIProposer(
        pretrained=True,
        img_size=cfg.imgsz,
        num_classes=cfg.num_classes,
        score_thresh=0.05,
    ).to(device_str)
    model.train()
    
    # Congelar backbone inicialmente.
    def set_backbone_frozen(frozen: bool):
        for p in model.backbone.parameters():
            p.requires_grad = not frozen
        state = "congelado" if frozen else "descongelado"
        print(f"  → Backbone {state}")

    if cfg.freeze_backbone_epochs > 0:
        set_backbone_frozen(True)

    # Pérdida.
    criterion = FCOSLoss(
        strides=[8, 16, 32],
        num_classes=cfg.num_classes,
        ranges=[(0, 128), (128, 256), (256, 1e10)],
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
        reg_weight=cfg.loss_reg_weight,
        cnt_weight=cfg.loss_cnt_weight,
    ).to(device_str)

    # Optimizador.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer, cfg.warmup_epochs, cfg.epochs, cfg.lr,
    )

    # Directorio de salida.
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loop.
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO")
    print("=" * 60)

    best_val_loss = float("inf")
    history = []

    for epoch in range(cfg.epochs):
        # Descongelar backbone después de N epochs.
        if epoch == cfg.freeze_backbone_epochs and cfg.freeze_backbone_epochs > 0:
            set_backbone_frozen(False)
            # Recrear optimizador con LR reducido.
            new_lr = cfg.lr * 0.5
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=new_lr, weight_decay=cfg.weight_decay,
            )
            # CLAVE: actualizar base_lr del scheduler para que respete el nuevo valor.
            scheduler.optimizer = optimizer
            scheduler.base_lr = new_lr
            print(f"  → Backbone descongelado. LR base ajustado a {new_lr:.2e}")

        lr = scheduler.step(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_losses = {"loss_cls": 0.0, "loss_reg": 0.0, "loss_cnt": 0.0}
        n_batches = 0
        t0 = time.time()

        for batch_idx, (images, boxes_list, _) in enumerate(train_loader):
            images = images.to(device_str)
            boxes_list = [b.to(device_str) for b in boxes_list]

            # Forward.
            features = model.backbone(images)
            features = model._to_nchw(features, model.out_channels)
            fpn_out = model.fpn({str(i): f for i, f in enumerate(features)})
            fpn_features = list(fpn_out.values())
            cls_logits, reg_preds, cnt_preds = model.head(fpn_features)

            # Loss.
            loss, loss_dict = criterion(cls_logits, reg_preds, cnt_preds, boxes_list)

            # Backward.
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            n_batches += 1

            if (batch_idx + 1) % cfg.print_every == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{cfg.epochs} "
                    f"[{batch_idx+1:3d}/{len(train_loader)}] "
                    f"loss={loss.item():.4f} "
                    f"(cls={loss_dict['loss_cls']:.3f} "
                    f"reg={loss_dict['loss_reg']:.3f} "
                    f"cnt={loss_dict['loss_cnt']:.3f}) "
                    f"lr={lr:.2e}"
                )

        avg_loss = epoch_loss / n_batches
        avg_parts = {k: v / n_batches for k, v in epoch_losses.items()}
        elapsed = time.time() - t0

        # Validación.
        val_loss = None
        if (epoch + 1) % cfg.val_every == 0 or epoch == cfg.epochs - 1:
            val_loss = validate(model, val_loader, criterion, device_str)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_dir / "best.pt")
                print(f"  → Nuevo mejor modelo (val_loss={val_loss:.4f})")

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_parts": avg_parts,
            "val_loss": val_loss,
            "lr": lr,
            "time_s": elapsed,
        })

        msg = (
            f"\n[Epoch {epoch+1:3d}/{cfg.epochs}] "
            f"train_loss={avg_loss:.4f} "
            f"(cls={avg_parts['loss_cls']:.3f} "
            f"reg={avg_parts['loss_reg']:.3f} "
            f"cnt={avg_parts['loss_cnt']:.3f})"
        )
        if val_loss is not None:
            msg += f" | val_loss={val_loss:.4f}"
        msg += f" | {elapsed:.1f}s"
        print(msg + "\n")

        # Checkpoint.
        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, ckpt_path)
            print(f"  → Checkpoint: {ckpt_path}")

        # Guardar history.
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # Guardar modelo final.
    torch.save(model.state_dict(), output_dir / "last.pt")
    print("\n" + "=" * 60)
    print(f"Entrenamiento completado")
    print(f"  Mejor val_loss: {best_val_loss:.4f}")
    print(f"  Pesos: {output_dir}/best.pt")
    print(f"         {output_dir}/last.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()