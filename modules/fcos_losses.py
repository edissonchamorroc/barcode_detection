# modules/fcos_losses.py
"""
Pérdidas y asignación de targets para entrenar FCOSHead.

Componentes:
    - FCOSAssigner: asigna puntos del feature map a GT boxes.
    - sigmoid_focal_loss: pérdida de clasificación (maneja desbalance).
    - giou_loss: pérdida de regresión (IoU generalizada).
    - FCOSLoss: módulo completo que combina las 3.

Traducción para no expertos:
    Este archivo es el "profesor" del modelo SwinROIProposer. Le dice:
      1. "Estos puntos del mapa deberían predecir 'código de barras'."
      2. "Estas distancias a los bordes deberían ser las correctas."
      3. "Estas zonas centrales merecen más confianza que las de los bordes."

    Solo se usa DURANTE EL ENTRENAMIENTO. En producción (inferencia) se
    ignora, porque el modelo ya aprendió y no necesita que le corrijan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ===========================================================================
# ASIGNACIÓN DE TARGETS (FCOS)
# ===========================================================================
# "Target" = la respuesta correcta que el modelo DEBERÍA dar en cada punto.
# "Asignar" = decidir qué punto del feature map es responsable de qué objeto.
#
# Analogía: imagina que cada celda del feature map es un alumno y cada
# caja GT (ground truth) es una pregunta del examen. El asignador decide
# QUÉ alumno se encarga de QUÉ pregunta.
# ===========================================================================
class FCOSAssigner:
    """
    Asigna cada punto del feature map a una GT box (o fondo).

    Reglas (FCOS estándar):
        1. Un punto es positivo si está DENTRO de una GT box.
        2. Si cae en varias GT boxes, se asigna a la de MENOR área.
        3. El tamaño máximo de la GT (max(w,h)) debe estar en el RANGO del nivel.
           Esto distribuye objetos de distintos tamaños a niveles apropiados.

    Args:
        strides: strides por nivel. Ej: [8, 16, 32].
        ranges: rangos de tamaño (min, max) por nivel.

    Traducción para no expertos:
        Cada punto del feature map es un "detector potencial". Algunos
        verán objetos chicos (nivel 0, stride 8), otros medianos (nivel 1)
        y otros grandes (nivel 2). Esta clase decide, punto por punto:
            - "¿Estás DENTRO de alguna caja real?" → positivo (1).
            - "¿Qué caja te corresponde?" → la de menor área (para no
              confundir objetos que se solapan).
            - "¿Eres del tamaño correcto para tu nivel?" → si no, eres fondo.
    """

    # `__init__` = constructor. Guarda la configuración en `self`.
    def __init__(
        self,
        strides: List[int] = [8, 16, 32],
        # `1e10` = 10,000,000,000 (un "infinito" práctico) para que el
        # último nivel acepte cualquier objeto grande.
        ranges: List[Tuple[float, float]] = [
            (0, 128),       # nivel 0 (stride 8):  barcodes chicos
            (128, 256),     # nivel 1 (stride 16): barcodes medianos
            (256, 1e10),    # nivel 2 (stride 32): barcodes grandes
        ],
    ):
        self.strides = strides
        self.ranges = ranges

    def assign_single_image(
        self,
        gt_boxes: torch.Tensor,      # [N, 4] en [x1, y1, x2, y2]
        feat_sizes: List[Tuple[int, int]],  # [(H0,W0), (H1,W1), (H2,W2)]
        device: torch.device,
    ) -> Tuple[
        List[torch.Tensor],          # reg_targets por nivel [M_i, 4]
        List[torch.Tensor],          # cls_targets por nivel [M_i] (0/1)
        List[torch.Tensor],          # cnt_targets por nivel [M_i] (0-1)
        List[torch.Tensor],          # points por nivel [M_i, 2] (coords absolutas)
    ]:
        """
        Asigna targets a UNA imagen.

        Returns:
            reg_targets: lista (por nivel) de [M_i, 4] con (l, t, r, b) en PÍXELES / stride.
            cls_targets: lista de [M_i] con 1 si positivo, 0 si fondo.
            cnt_targets: lista de [M_i] con centerness target (0-1).
            points: lista de [M_i, 2] con (x, y) absolutos de cada punto.
        """
        # Listas que iremos llenando: una entrada por nivel del FPN.
        reg_targets_per_level = []
        cls_targets_per_level = []
        cnt_targets_per_level = []
        points_per_level = []

        # CASO ESPECIAL: no hay objetos en esta imagen.
        # `.numel()` = "número de elementos del tensor". Si es 0, no hay GT.
        if gt_boxes.numel() == 0:
            # Todos los puntos serán fondo (targets = 0 en todo).
            for (H, W), stride in zip(feat_sizes, self.strides):
                M = H * W  # total de puntos en este nivel

                reg_targets_per_level.append(torch.zeros(M, 4, device=device))
                cls_targets_per_level.append(torch.zeros(M, device=device))
                cnt_targets_per_level.append(torch.zeros(M, device=device))

                # Generamos los puntos absolutos (uno por celda del feature map).
                # `arange(H)*stride + stride/2` = coordenada del CENTRO de
                # cada celda, en píxeles de la imagen original.
                ys, xs = torch.meshgrid(
                    torch.arange(H, device=device) * stride + stride / 2,
                    torch.arange(W, device=device) * stride + stride / 2,
                    indexing="ij",
                )
                # `stack(..., dim=-1)` apila x e y por pares → [M, 2].
                points = torch.stack([xs.flatten(), ys.flatten()], dim=-1)
                points_per_level.append(points)
            return (
                reg_targets_per_level,
                cls_targets_per_level,
                cnt_targets_per_level,
                points_per_level,
            )

        # ---- CASO NORMAL: hay al menos un objeto ----

        # Área de cada GT: ancho × alto.
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

        # Tamaño "representativo" de cada GT: el mayor de ancho/alto.
        # Se usa para decidir a qué nivel del FPN va cada objeto.
        gt_sizes = torch.max(
            gt_boxes[:, 2] - gt_boxes[:, 0],
            gt_boxes[:, 3] - gt_boxes[:, 1],
        )  # [N]

        # Recorremos los niveles del FPN con su stride y su rango de tamaños.
        for lvl_idx, ((H, W), stride) in enumerate(zip(feat_sizes, self.strides)):
            lo, hi = self.ranges[lvl_idx]

            # Filtro por rango: solo nos quedamos con las GT cuyo tamaño
            # encaja con este nivel. `&` = AND lógico entre tensores.
            in_range = (gt_sizes >= lo) & (gt_sizes < hi)  # [N]
            lvl_boxes = gt_boxes[in_range]                # [N_i, 4]
            lvl_areas = gt_areas[in_range]

            M = H * W

            # Grid de puntos absolutos (igual que en el caso vacío).
            ys, xs = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32) * stride + stride / 2,
                torch.arange(W, device=device, dtype=torch.float32) * stride + stride / 2,
                indexing="ij",
            )
            points = torch.stack([xs.flatten(), ys.flatten()], dim=-1)  # [M, 2]

            # Si en este nivel no hay ninguna GT que encaje, todo es fondo.
            if lvl_boxes.shape[0] == 0:
                reg_targets_per_level.append(torch.zeros(M, 4, device=device))
                cls_targets_per_level.append(torch.zeros(M, device=device))
                cnt_targets_per_level.append(torch.zeros(M, device=device))
                points_per_level.append(points)
                continue

            # ---- Distancias de cada punto a cada caja ----
            # `points[:, 0:1]` = x de cada punto, con forma [M, 1].
            # `lvl_boxes[:, 0].unsqueeze(0)` = x1 de cada caja, forma [1, N_i].
            # Al restar, se hace "broadcasting" → resultado [M, N_i]:
            # para CADA punto y CADA caja, la distancia horizontal izquierda.
            l = points[:, 0:1] - lvl_boxes[:, 0].unsqueeze(0)  # [M, N_i]
            t = points[:, 1:2] - lvl_boxes[:, 1].unsqueeze(0)  # arriba
            r = lvl_boxes[:, 2].unsqueeze(0) - points[:, 0:1]  # derecha
            b = lvl_boxes[:, 3].unsqueeze(0) - points[:, 1:2]  # abajo

            # Un punto está DENTRO de una caja si las 4 distancias son > 0.
            inside = (l > 0) & (t > 0) & (r > 0) & (b > 0)  # [M, N_i]

            # Para cada punto, ¿qué área tiene la caja que lo contiene?
            # `torch.where(cond, A, B)` = "donde cond es True, pon A; si no, B".
            # Si el punto NO está dentro de una caja, le ponemos un área
            # gigante (1e10) para que nunca sea elegida como "la menor".
            areas_per_point = torch.where(
                inside, lvl_areas.unsqueeze(0).expand_as(inside),
                torch.full_like(inside, 1e10, dtype=torch.float32),
            )

            # `.min(dim=1)` = "el menor de cada fila". Devuelve el valor
            # mínimo Y el índice de la caja que lo produjo.
            min_area, best_idx = areas_per_point.min(dim=1)  # [M], [M]

            # Es positivo si encontró alguna caja (el área NO es el "infinito").
            is_pos = min_area < 1e9  # [M]

            # Preparamos tensores de salida (todo empieza en 0 = fondo).
            reg_targets = torch.zeros(M, 4, device=device)
            cls_targets = torch.zeros(M, device=device)
            cnt_targets = torch.zeros(M, device=device)

            # Si hay al menos un positivo, calculamos sus targets.
            # `.any()` = "¿hay al menos un True?".
            if is_pos.any():
                # `best_idx[is_pos]` = para cada punto positivo, el índice
                # de la caja elegida. Luego indexamos las distancias
                # correspondientes.
                pos_idx = best_idx[is_pos]  # índice de la GT elegida
                l_pos = l[is_pos, pos_idx]
                t_pos = t[is_pos, pos_idx]
                r_pos = r[is_pos, pos_idx]
                b_pos = b[is_pos, pos_idx]

                # La red predice distancias EN UNIDADES DE STRIDE (no en
                # píxeles brutos). Por eso dividimos.
                reg_targets[is_pos, 0] = l_pos / stride
                reg_targets[is_pos, 1] = t_pos / stride
                reg_targets[is_pos, 2] = r_pos / stride
                reg_targets[is_pos, 3] = b_pos / stride

                # Marca los positivos con clase 1 (barcode).
                cls_targets[is_pos] = 1.0

                # ---- Centerness target ----
                # Fórmula FCOS: sqrt((min(l,r)/max(l,r)) * (min(t,b)/max(t,b)))
                # Da 1 si el punto está perfectamente centrado en la caja,
                # y se acerca a 0 si está pegado a un borde.
                # `+ 1e-6` evita división entre cero.
                lr_min = torch.min(l_pos, r_pos)
                lr_max = torch.max(l_pos, r_pos)
                tb_min = torch.min(t_pos, b_pos)
                tb_max = torch.max(t_pos, b_pos)
                centerness = torch.sqrt(
                    (lr_min / (lr_max + 1e-6)) * (tb_min / (tb_max + 1e-6))
                )
                cnt_targets[is_pos] = centerness

            # Guardamos los targets de este nivel.
            reg_targets_per_level.append(reg_targets)
            cls_targets_per_level.append(cls_targets)
            cnt_targets_per_level.append(cnt_targets)
            points_per_level.append(points)

        return (
            reg_targets_per_level,
            cls_targets_per_level,
            cnt_targets_per_level,
            points_per_level,
        )


# ===========================================================================
# PÉRDIDAS
# ===========================================================================
# "Pérdida" (loss) = un número que mide cuán equivocado está el modelo.
# Durante el entrenamiento, el objetivo es HACER ESA PÉRDIDA LO MÁS
# PEQUEÑA POSIBLE. Es el "profesor" que corrige al alumno.
# ===========================================================================
def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal Loss (Lin et al., 2017). Maneja el desbalance fondo/objeto.

    L = -alpha * (1-p)^gamma * log(p)  si target=1
        -(1-alpha) * p^gamma * log(1-p)  si target=0

    Traducción para no expertos:
        En una imagen hay MUCHOS más píxeles de "fondo" que de "objeto"
        (puede haber 99% fondo, 1% código). Si usáramos la pérdida normal
        (cross-entropy), el modelo aprendería a decir "todo es fondo" y
        acertaría el 99%, pero sería inútil.

        Focal Loss SOLUCIONA eso: reduce el peso de los ejemplos FÁCILES
        (los que el modelo ya acierta con confianza) y concentra el
        aprendizaje en los DIFÍCILES (bordes, objetos pequeños, ruido).

        `(1-p)^gamma` es el "factor de enfoque": si el modelo acierta
        con p=0.99, el factor es (0.01)^2 = 0.0001 → casi no contribuye.
        Si está dudando con p=0.5, el factor es 0.25 → sí contribuye.
    """
    # sigmoid convierte logits (números reales) a probabilidad [0,1].
    p = torch.sigmoid(logits)

    # `binary_cross_entropy_with_logits` calcula la pérdida CE estándar
    # por elemento (sin reducir). Por dentro aplica sigmoid + log.
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # `p_t` = probabilidad que el modelo le asignó a la clase CORRECTA.
    # Si target=1 → p_t = p. Si target=0 → p_t = 1-p.
    p_t = p * targets + (1 - p) * (1 - targets)

    # Factor de enfoque: reduce la contribución de los ejemplos fáciles.
    loss = ce_loss * ((1 - p_t) ** gamma)

    # `alpha` = balance entre positivos y negativos.
    # alpha=0.25 significa "un positivo pesa 4 veces más que un negativo".
    # Se compensa así el desbalance porque suele haber muchos más negativos.
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Devolvemos el promedio sobre todos los elementos.
    return loss.mean()


def giou_loss(
    pred_boxes: torch.Tensor,      # [N, 4] en [x1, y1, x2, y2]
    target_boxes: torch.Tensor,    # [N, 4]
) -> torch.Tensor:
    """
    Generalized IoU loss (Rezatofighi et al., 2019).

    Traducción para no expertos:
        "IoU" (Intersection over Union) mide cuánto se solapan dos cajas:
            IoU = área(intersección) / área(unión)
        Va de 0 (no se tocan) a 1 (idénticas).

        GIoU añade un castigo extra: si las cajas NO se solapan, además
        de IoU=0, considera cuánto espacio de sobra hay entre ellas.
        Así el modelo aprende a ACERCARSE aunque al principio no se toque.
        Es como un profesor que no solo dice "mal", sino "estás lejos,
        muévete hacia la izquierda".
    """
    # ---- IoU: intersección ----
    # Esquinas de la intersección: max de x1/y1, min de x2/y2.
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    # Área de la intersección. `.clamp(min=0)` evita áreas negativas
    # (si las cajas no se tocan, x2-x1 < 0 → se queda en 0).
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # Área de cada caja.
    area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * \
             (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    area_t = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0) * \
             (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0)

    # Unión = suma de áreas - intersección (+ epsilon para no dividir por 0).
    union = area_p + area_t - inter + 1e-6
    iou = inter / union

    # ---- GIoU: caja convexa envolvente ----
    # La caja más pequeña que CONTIENE a las dos.
    x1c = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1c = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2c = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2c = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    area_c = (x2c - x1c).clamp(min=0) * (y2c - y1c).clamp(min=0) + 1e-6

    # GIoU = IoU - (espacio no usado por las cajas / área de la envolvente)
    giou = iou - (area_c - union) / area_c

    # La pérdida es (1 - GIoU) para que se minimice cuando GIoU → 1.
    return (1 - giou).mean()


class FCOSLoss(nn.Module):
    """
    Módulo completo de pérdida para FCOS.
    Combina focal + GIoU + BCE centerness.

    Traducción para no expertos:
        Es el "profesor titular" que combina las 3 correcciones:
          1. ¿Acertaste la CLASE (barcode vs fondo)?   → focal loss
          2. ¿Acertaste las DISTANCIAS a los bordes?   → GIoU loss
          3. ¿Aprendiste a dar más confianza en el CENTRO?  → BCE
        Suma ponderada de las 3 → una sola cifra a minimizar.
    """

    def __init__(
        self,
        strides: List[int] = [8, 16, 32],
        num_classes: int = 2,       # 0=fondo, 1=barcode
        ranges: List[Tuple[float, float]] = [
            (0, 128), (128, 256), (256, 1e10),
        ],
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        reg_weight: float = 1.0,
        cnt_weight: float = 1.0,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Creamos el asignador con la misma configuración del modelo.
        self.assigner = FCOSAssigner(strides=strides, ranges=ranges)

        # Hiperparámetros de focal loss.
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Pesos para combinar las pérdidas. Si quieres que el modelo se
        # centre más en la regresión, sube `reg_weight`.
        self.reg_weight = reg_weight
        self.cnt_weight = cnt_weight

    def forward(
        self,
        cls_logits: List[torch.Tensor],   # [cls_preds por nivel]
        reg_preds: List[torch.Tensor],    # [reg_preds por nivel]
        cnt_preds: List[torch.Tensor],    # [cnt_preds por nivel]
        gt_boxes_list: List[torch.Tensor],   # lista de [N_i, 4] por imagen del batch
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calcula la pérdida total y devuelve un dict con componentes por separado.

        `forward` se ejecuta cuando llamas `loss_fn(predicciones, gt)`.
        """
        # B = tamaño del batch (cuántas imágenes a la vez).
        B = cls_logits[0].shape[0]
        device = cls_logits[0].device

        # Tamaños espaciales de cada nivel (H, W).
        feat_sizes = [(f.shape[2], f.shape[3]) for f in cls_logits]

        # Acumuladores (empezamos en 0 y sumamos por cada imagen y nivel).
        total_cls = 0.0
        total_reg = 0.0
        total_cnt = 0.0
        n_reg_batches = 0  # para normalizar correctamente

        # Recorremos cada imagen del batch.
        for b in range(B):
            # GT de esta imagen, movidas al dispositivo correcto.
            gt_boxes = gt_boxes_list[b].to(device)

            # Asignar targets: qué punto es responsable de qué caja.
            (
                reg_tgts_per_lvl,
                cls_tgts_per_lvl,
                cnt_tgts_per_lvl,
                pts_per_lvl,          # ← capturar para reusar
            ) = self.assigner.assign_single_image(gt_boxes, feat_sizes, device)

            # Recorremos cada nivel del FPN.
            for lvl in range(len(cls_logits)):
                # ----------------------------------------------------
                # 1. Pérdida de CLASIFICACIÓN (focal)
                # ----------------------------------------------------
                # `cls_logits[lvl][b]` = predicción de la imagen b en nivel lvl.
                cls_lvl = cls_logits[lvl][b]          # [C, H, W]
                # `[1:]` descarta el fondo (índice 0). Solo nos interesa
                # la clase "barcode" (índice 1).
                # `.flatten()` = colapsar a un vector [H*W].
                cls_fg_logits = cls_lvl[1:].flatten()
                cls_tgt = cls_tgts_per_lvl[lvl]        # [H*W]

                total_cls += sigmoid_focal_loss(
                    cls_fg_logits, cls_tgt,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                )

                # ----------------------------------------------------
                # 2. Pérdida de REGRESIÓN (solo positivos)
                # ----------------------------------------------------
                # Reorganizamos las predicciones a [M, 4].
                reg_lvl = reg_preds[lvl][b].permute(1, 2, 0).reshape(-1, 4)  # [M, 4]
                reg_tgt = reg_tgts_per_lvl[lvl]                              # [M, 4]

                # Máscara de positivos: puntos que son objeto (no fondo).
                pos_mask = cls_tgt > 0.5

                if pos_mask.any():
                    stride = self.assigner.strides[lvl]
                    pts = pts_per_lvl[lvl]

                    # Predicciones y targets de los puntos positivos,
                    # pasados de unidades-stride a píxeles.
                    pred_ltrb = reg_lvl[pos_mask] * stride  # [N_pos, 4]
                    pred_pts = pts[pos_mask]                # [N_pos, 2]

                    # Convertimos (l, t, r, b) a (x1, y1, x2, y2).
                    pred_x1 = pred_pts[:, 0] - pred_ltrb[:, 0]
                    pred_y1 = pred_pts[:, 1] - pred_ltrb[:, 1]
                    pred_x2 = pred_pts[:, 0] + pred_ltrb[:, 2]
                    pred_y2 = pred_pts[:, 1] + pred_ltrb[:, 3]

                    # Hacemos lo mismo con los targets.
                    tgt_ltrb = reg_tgt[pos_mask] * stride
                    tgt_x1 = pred_pts[:, 0] - tgt_ltrb[:, 0]
                    tgt_y1 = pred_pts[:, 1] - tgt_ltrb[:, 1]
                    tgt_x2 = pred_pts[:, 0] + tgt_ltrb[:, 2]
                    tgt_y2 = pred_pts[:, 1] + tgt_ltrb[:, 3]

                    # GIoU loss entre cajas predichas y target.
                    total_reg += giou_loss(
                        torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1),
                        torch.stack([tgt_x1, tgt_y1, tgt_x2, tgt_y2], dim=-1),
                    )
                    n_reg_batches += 1

                # ----------------------------------------------------
                # 3. Pérdida de CENTERNESS (solo positivos)
                # ----------------------------------------------------
                cnt_lvl = cnt_preds[lvl][b].flatten()  # [M]
                cnt_tgt = cnt_tgts_per_lvl[lvl]
                if pos_mask.any():
                    # BCE = Binary Cross-Entropy, para valores [0,1].
                    total_cnt += F.binary_cross_entropy_with_logits(
                        cnt_lvl[pos_mask], cnt_tgt[pos_mask],
                    )

        # ---- Normalizar ----
        # Sumamos por imagen y nivel; dividimos para obtener el promedio
        # (evita que el valor de la pérdida dependa del tamaño del batch).
        n_levels = len(cls_logits)
        total_cls = total_cls / (B * n_levels)
        total_reg = total_reg / max(n_reg_batches, 1)
        total_cnt = total_cnt / max(n_reg_batches, 1)

        # Pérdida total combinada.
        total_loss = total_cls + self.reg_weight * total_reg + self.cnt_weight * total_cnt

        # Devolvemos la pérdida total Y un dict con los componentes
        # desglosados (útil para logging y depuración).
        return total_loss, {
            "loss_cls": total_cls.item(),
            "loss_reg": total_reg.item() if isinstance(total_reg, torch.Tensor) else total_reg,
            "loss_cnt": total_cnt.item() if isinstance(total_cnt, torch.Tensor) else total_cnt,
        }