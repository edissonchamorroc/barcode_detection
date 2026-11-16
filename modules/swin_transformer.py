# swin_transformer.py
"""
Etapa 1 del pipeline: Swin Transformer como PROPUESOR de ROI (alta recall).

Su objetivo NO es dar bboxes perfectas, sino proponer todas las regiones que
PODRÍAN contener un código de barras, aunque haya falsos positivos. El filtrado
y refinamiento se delega a la etapa 2 (YOLOv5s-CBAM).

Diseño:
    Imagen → Swin (timm, features_only) → FPN → FCOSHead → decode + NMS → propuestas
"""


import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.ops import FeaturePyramidNetwork, nms


# ============================================================================
# CABEZAL FCOS: anchor-free, alta recall
# ============================================================================
# Un "cabezal" (head) en visión artificial es la parte FINAL de la red que
# convierte los "mapas de características" (representaciones internas) en
# predicciones útiles: "aquí hay un objeto", "en esta caja", "de esta clase".
# Se ejecuta en pipeline 
# ============================================================================
class FCOSHead(nn.Module):
    """
    Cabezal compartido entre niveles (FCOS-style). Para cada posición del
    feature map predice:
        - logits de clase (num_classes)
        - distancias (l, t, r, b) al borde de la caja
        - centerness (calidad de la predicción)

    Se elige FCOS porque:
        - Es anchor-free → no hay que afinar anchor boxes a mano.
        - Es rápido y ligero (pocas convoluciones 3x3).
        - Es fácil de entrenar con focal loss + GIoU.
        - Da recall alto incluso en objetos pequeños.

    Traducción para no expertos:
        - "anchor-free" = en vez de tener "cajas plantilla" predefinidas
          y ajustarlas, el modelo predice directamente las distancias
          desde cada punto hasta los 4 bordes de la caja.
        - "recall alto" = preferimos que sobren detecciones a que falten.
          Es un "buscador pesimista": propone mucho y luego se filtra.
    """

    # `__init__` = constructor. Se ejecuta al crear el objeto.
    # Los parámetros con `=` tienen valores por defecto.
    #   in_channels: cuántos "canales" (capas de información) llegan por píxel.
    #   num_classes: cuántas clases hay. 2 = fondo (0) + código de barras (1).
    #   num_convs: cuántas convoluciones seguidas antes de la predicción final.
    def __init__(
        self, in_channels: int = 256, num_classes: int = 2, num_convs: int = 4
    ):
        # `super().__init__()` inicializa la parte "Module" de PyTorch.
        # SIEMPRE hay que llamarlo al definir una capa nueva; si no, PyTorch
        # no sabe que esto es una red y no registra los parámetros.
        super().__init__()
        self.num_classes = num_classes

        # --- Rama de clasificación ---
        # "Rama" = una sub-red pequeña con un propósito concreto.
        # Esta decide QUÉ hay en cada posición (fondo vs código).
        cls_layers = []  # lista vacía donde iremos metiendo capas
        for _ in range(num_convs):
            # El guion bajo `_` es una variable "descartable": no nos
            # interesa el número de iteración, solo repetir `num_convs` veces.
            cls_layers += [
                # Conv2d: capa convolucional. Aplica un filtro de 3x3 que
                # "barre" la imagen y detecta patrones locales (bordes,
                # esquinas, texturas...). `bias=False` porque a continuación
                # hay una normalización que ya añade su propio sesgo.
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                # GroupNorm: normaliza las activaciones para que los valores
                # no se disparen ni se apaguen. Ayuda a estabilizar el
                # entrenamiento. `32` = número de grupos.
                nn.GroupNorm(32, in_channels),
                # ReLU: función de activación. Convierte números negativos
                # en 0. Es una "no-linealidad": sin ella, apilar capas
                # equivaldría a una sola capa lineal (inútil).
                nn.ReLU(inplace=True),
            ]
        # Capa final: 1x1 implícito por el padding=1 + kernel 3. Produce
        # `num_classes` valores por posición: uno por clase (fondo, código...).
        cls_layers.append(nn.Conv2d(in_channels, num_classes, 3, padding=1))
        # `nn.Sequential` encadena las capas: la salida de una entra en la
        # siguiente, en orden. Se usa luego con `self.cls_branch(x)`.
        self.cls_branch = nn.Sequential(*cls_layers)

        # --- Rama de regresión (l, t, r, b) ---
        # Esta predice la CAJA: distancia desde el punto actual hasta los
        # bordes izquierdo (l), superior (t), derecho (r) e inferior (b).
        # Es la "filosofía FCOS": cada píxel del feature map se pregunta
        # "¿cuánto tengo que crecer para encerrar el objeto que veo?".
        reg_layers = []
        for _ in range(num_convs):
            reg_layers += [
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
            ]
        # Salida de 4 valores por posición (l, t, r, b).
        reg_layers.append(nn.Conv2d(in_channels, 4, 3, padding=1))
        self.reg_branch = nn.Sequential(*reg_layers)

        # --- Rama de centerness ---
        # "Centerness" es un score que dice CUÁNTO se parece el punto actual
        # al centro del objeto. Vale entre 0 y 1. Sirve para descartar
        # predicciones hechas desde puntos muy al borde de la caja (que
        # suelen ser peores). Es como preguntar: "¿esta detección la hice
        # desde el medio o desde un borde?".
        cnt_layers = []
        for _ in range(num_convs):
            cnt_layers += [
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
            ]
        # Salida de 1 valor por posición (el score de centerness).
        cnt_layers.append(nn.Conv2d(in_channels, 1, 3, padding=1))
        self.cnt_branch = nn.Sequential(*cnt_layers)

        # --- Inicialización ---
        # Antes de entrenar, los pesos tienen que empezar en valores sensatos.
        # `self.modules()` recorre todas las sub-capas.
        # Pesos pequeños (estándar RetinaNet/FCOS).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Distribución normal con desviación típica pequeña (0.01).
                # Evita que la red empiece "gritando" números enormes.
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Sesgo de clasificación negativo: el modelo empieza prediciendo
        # "fondo". ¿Por qué? Porque en la imagen hay MUCHOS más píxeles
        # de fondo que de código de barras. Si empezara diciendo "objeto",
        # el entrenamiento se desestabilizaría (desbalance de clases).
        # sigmoid(-4) ≈ 0.018 → casi 0 de probabilidad de objeto.
        # `[-1]` = última capa del Sequential (la de salida).
        nn.init.constant_(self.cls_branch[-1].bias, -4.0)

        # Centerness empieza neutro (0).
        nn.init.constant_(self.cnt_branch[-1].bias, 0.0)

        # Ajuste en el regresor: empezar con sesgo 1.0 (distancias
        # positivas iniciales). Después, `F.relu` garantiza que no haya
        # distancias negativas.
        nn.init.constant_(self.reg_branch[-1].bias, 1.0)

    def forward(self, features):
        """
        `forward` define qué pasa cuando llamas al objeto como función:
        `head(features)`. Es el "cuerpo" del cálculo.

        Args:
            features: lista de tensores [B, C, H_i, W_i] del FPN.
                B = tamaño del batch (cuántas imágenes a la vez).
                C = canales (256 por defecto).
                H_i, W_i = alto y ancho del feature map en ese nivel.

        Returns:
            cls_logits: lista por nivel [B, num_classes, H_i, W_i]
                Logits = scores SIN sigmoid (números reales, no probs).
            reg_preds:  lista por nivel [B, 4, H_i, W_i] (valores ≥ 0)
                Distancias (l, t, r, b) ya con ReLU aplicado (no negativas).
            cnt_preds:  lista por nivel [B, 1, H_i, W_i]
                Logits de centerness.
        """
        # Tres listas donde iremos guardando la salida de cada rama por
        # cada nivel de la pirámide (por eso es una lista: hay varios
        # niveles con distintas resoluciones).
        cls_logits, reg_preds, cnt_preds = [], [], []

        # Recorremos cada nivel del FPN. `feat` es un tensor [B, C, H, W].
        for feat in features:
            # Cada rama procesa su feat y guardamos el resultado.
            cls_logits.append(self.cls_branch(feat))
            # `F.relu` fuerza que las distancias sean ≥ 0 (no puede haber
            # una caja con "distancia negativa" al borde).
            reg_preds.append(F.relu(self.reg_branch(feat)))
            cnt_preds.append(self.cnt_branch(feat))
        return cls_logits, reg_preds, cnt_preds


# ============================================================================
# BACKBONE SWIN + FPN + FCOSHEAD
# Se ejectua en pipeline
# ============================================================================
class SwinROIProposer(nn.Module):
    """
    Propuesta de ROI para la etapa 1 del pipeline.

    Traducción para no expertos:
        - "Backbone" = la red "troncal" que extrae características de la
          imagen. Aquí es un Swin Transformer.
        - "ROI" = Region Of Interest (región de interés). El modelo propone
          posibles zonas donde podría haber un código de barras.
        - El objetivo NO es clasificar perfectamente, sino NO DEJARSE
          NINGÚN código sin proponer (alta recall). Los falsos positivos
          los filtra la etapa 2 (YOLO).

    Args:
        model_name: nombre del modelo en timm.
        pretrained: usar pesos ImageNet. (ImageNet es un dataset gigante
            de imágenes normales; empezar con esos pesos ahorra mucho
            entrenamiento.)
        num_classes: 2 por defecto (0=fondo, 1=código de barras).
        fpn_channels: canales unificados tras la FPN.
        score_thresh: umbral bajo (0.05) → alta recall, muchos falsos
            positivos. Es intencional.
        nms_thresh: IoU de NMS. 0.5 estándar; subir para permitir cajas
            solapadas (barcodes muy juntos en una misma caja).
        topk_per_level: límite de cajas por nivel antes de NMS.
        max_dets: límite final de propuestas por imagen.
    """

    # Swin exige H, W múltiplos de 32. ¿Por qué? Porque el Swin Transformer
    # reduce la imagen en varias etapas (patch de 4 y 3 downsamples de 2):
    # 4 × 2 × 2 × 2 = 32. Si la imagen no es múltiplo de 32, la red falla
    # al "reducir" (al final sobrarían píxeles o faltarían).
    PATCH_MULTIPLE = 32

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        img_size: int = 512,
        num_classes: int = 2,
        fpn_channels: int = 256,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        topk_per_level: int = 1000,
        max_dets: int = 300,
    ):
        super().__init__()

        # Guardamos la configuración en `self.*` para usarla luego.
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.topk_per_level = topk_per_level
        self.max_dets = max_dets

        # ------------------------------------------------------------------
        # 1. Backbone Swin sin cabeza de clasificación, salida multiescala.
        # ------------------------------------------------------------------
        # `timm.create_model` "fabrica" un modelo listo para usar.
        # `features_only=True` → en vez de clasificar, devuelve los "mapas
        # de características" intermedios. Como quitar el último bloque
        # (el que dice "esto es un perro") y quedarnos con la parte que
        # "entiende" la imagen.
        #
        # `out_indices=(1,2,3)` → qué etapas devolver. Swin tiene 4 etapas;
        # nos quedamos con las 2ª, 3ª y 4ª (resoluciones cada vez más
        # pequeñas). Dan distintos "niveles de detalle":
        #   - Etapa 2: stride 8 (más resolución, bueno para objetos pequeños)
        #   - Etapa 3: stride 16 (equilibrado)
        #   - Etapa 4: stride 32 (más contexto, bueno para objetos grandes)
        #
        # `img_size=img_size` es una pista que le damos a timm para que
        # ajuste internamente los tamaños de ventana/posición.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),
            img_size=img_size,
            strict_img_size=False,
        )
        # `feature_info.channels()` devuelve una lista con los canales de
        # cada nivel que elegimos. Ejemplo típico: [192, 384, 768].
        self.out_channels = self.backbone.feature_info.channels()

        # ------------------------------------------------------------------
        # 2. FPN: unifica a fpn_channels y mezcla información entre escalas.
        # ------------------------------------------------------------------
        # FPN (Feature Pyramid Network) es un bloque que:
        #   1. Reduce todas las salidas del backbone a `fpn_channels` (256)
        #      con capas 1x1 → así todas hablan "el mismo idioma".
        #   2. Mezcla información de los niveles "grandes" (poca resolución,
        #      mucho contexto) hacia los "pequeños" (mucha resolución,
        #      detalle fino), con suma + upsampling.
        # Resultado: cada nivel ve tanto el detalle local como el contexto
        # global. Mejora mucho la detección de objetos de distintos tamaños.
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.out_channels,
            out_channels=fpn_channels,
        )

        # ------------------------------------------------------------------
        # 3. Cabezal FCOS compartido entre niveles.
        # ------------------------------------------------------------------
        # El MISMO cabezal se aplica a cada nivel del FPN (esto es lo
        # "compartido"). Ahí está la gracia: un solo juego de pesos aprende
        # a predecir cajas en cualquier resolución.
        self.head = FCOSHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
        )

    # ======================================================================
    # UTILIDADES
    # ======================================================================
    def _validate_input(self, x: torch.Tensor):
        """
        Verifica que H y W cumplan la restricción del Swin (múltiplos de 32).
        Si no, lanza un error claro en vez de dejar que la red falle de
        forma incomprensible más adelante.
        """
        # `x.shape[-2:]` = "las dos últimas dimensiones" de x. En [B, C, H, W]
        # esas son H y W. `-1` = última, `-2` = penúltima.
        h, w = x.shape[-2:]

        # Si el resto de dividir entre 32 NO es cero → error.
        if h % self.PATCH_MULTIPLE or w % self.PATCH_MULTIPLE:
            raise ValueError(
                f"Swin requiere H y W múltiplos de {self.PATCH_MULTIPLE}. "
                f"Recibido: ({h}, {w}). Redimensiona antes de pasar al modelo."
            )

    @staticmethod
    def _to_nchw(features, expected_channels):
        """
        Convierte una lista de feature maps a NCHW.
        Swin (timm) devuelve NHWC: (B, H, W, C).
        Detectamos el formato comparando la última dim con los canales esperados.

        ¿Por qué importa el orden? Porque el FPN y el resto de PyTorch
        esperan NCHW (batch, canales, alto, ancho), pero Swin devuelve
        NHWC. Es como pasar de "fila por fila" a "columna por columna":
        los mismos datos, distinto orden.
        """
        out = []
        # `zip` empareja cada feature con el canal esperado de ese nivel.
        for feat, exp_c in zip(features, expected_channels):
            # Si la ÚLTIMA dimensión coincide con los canales esperados
            # y la segunda NO → efectivamente es NHWC y hay que reordenar.
            if feat.shape[-1] == exp_c and feat.shape[1] != exp_c:
                # `permute(0, 3, 1, 2)` reordena los ejes:
                # de (B, H, W, C) → (B, C, H, W).
                # `contiguous()` garantiza que el tensor esté en memoria
                # de forma ordenada (necesario para algunas operaciones).
                feat = feat.permute(0, 3, 1, 2).contiguous()
            out.append(feat)
        return out

    @staticmethod
    def _infer_strides(features, input_hw):
        """
        Stride efectivo de cada nivel, derivado del tamaño real del feature map.

        "Stride" = cuántos píxeles de la imagen original representa cada
        celda del feature map. Ejemplo: si la imagen es 512x512 y el feature
        map es 64x64, el stride es 512/64 = 8. Cada "píxel" del feature map
        resume una zona de 8x8 de la imagen.
        """
        strides = []
        for feat in features:
            # `_` = descartamos B; necesitamos solo la dimensión H (índice 2).
            _, _, h, _ = feat.shape
            # División entera: 512 // 64 = 8.
            strides.append(input_hw[0] // h)
        return strides

    # ======================================================================
    # DECODE FCOS → (boxes, scores, labels) en coordenadas absolutas
    # ======================================================================
    # `@torch.no_grad()` desactiva el cálculo de gradientes. En inferencia
    # no vamos a entrenar, así que ahorramos memoria y tiempo. Buena
    # práctica siempre que no haya backprop.
    @torch.no_grad()
    def _decode(self, cls_logits, reg_preds, cnt_preds, strides, img_hw):
        """
        Convierte las predicciones "crudas" en cajas reales.

        Devuelve listas de tamaño B: cada elemento es (boxes[N,4], scores[N], labels[N]).
        Las clases se asumen: 0=fondo, 1..C-1=objeto. Se reporta la clase argmax > 0.
        """
        # B = tamaño del batch (cuántas imágenes).
        B = cls_logits[0].shape[0]
        # `device` = dónde están los datos (CPU o GPU). Lo usamos para crear
        # tensores nuevos en el mismo sitio.
        device = cls_logits[0].device

        # Listas por imagen: cada una tendrá la info de esa imagen.
        per_image_boxes = [[] for _ in range(B)]
        per_image_scores = [[] for _ in range(B)]
        per_image_labels = [[] for _ in range(B)]

        # Recorremos cada NIVEL del FPN con su stride asociado.
        for cls_lvl, reg_lvl, cnt_lvl, stride in zip(
            cls_logits, reg_preds, cnt_preds, strides
        ):
            # B_l = tamaño batch (por si acaso), C = clases, H, W = tamaño del feature map.
            B_l, C, H, W = cls_lvl.shape

            # ---------------------------------------------------------
            # Construir la "rejilla" de puntos centrales en coordenadas
            # de la imagen original.
            # ---------------------------------------------------------
            # Para cada celda (i, j) del feature map, el punto central
            # en la imagen original está en (i+0.5)*stride, (j+0.5)*stride.
            # El +0.5 es porque estamos en el centro de la celda.
            ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * stride
            xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * stride

            # `meshgrid` construye dos matrices [H, W]: una con la Y de cada
            # punto y otra con la X. Así cada celda del feature map sabe
            # sus coordenadas en la imagen real.
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")

            # ---------------------------------------------------------
            # Convertir distancias relativas en cajas absolutas.
            # ---------------------------------------------------------
            # La red predijo distancias relativas al "tamaño de celda"
            # (stride). Multiplicamos por stride para pasarlas a píxeles.
            reg = reg_lvl * stride  # [B, 4, H, W]

            # Desempaquetamos las 4 distancias.
            l = reg[:, 0]  # izquierda
            t = reg[:, 1]  # arriba (top)
            r = reg[:, 2]  # derecha
            b = reg[:, 3]  # abajo (bottom)

            # Coordenadas de la caja: centro ± distancias.
            # `unsqueeze(0)` añade una dimensión al principio para poder
            # restar (broadcasting) contra los tensores [B, H, W].
            x1 = xx.unsqueeze(0) - l
            y1 = yy.unsqueeze(0) - t
            x2 = xx.unsqueeze(0) + r
            y2 = yy.unsqueeze(0) + b

            # Apilamos las 4 coordenadas → [B, H, W, 4].
            boxes_lvl = torch.stack([x1, y1, x2, y2], dim=-1)

            # ---------------------------------------------------------
            # Calcular score final: clase × sqrt(centerness).
            # ---------------------------------------------------------
            # sigmoid aplasta los logits a [0, 1] para convertirlos en probs.
            cls_sig = cls_lvl.sigmoid()
            cnt_sig = cnt_lvl.sigmoid()

            # Nos interesan solo las clases de objeto (índices 1..C-1).
            # `[:, 1:]` descarta el canal 0 (fondo).
            cls_fg = cls_sig[:, 1:]

            # Multiplicamos por sqrt(centerness). ¿Por qué sqrt y no
            # directamente? Porque centerness ya está en [0,1]; usar sqrt
            # lo hace menos agresivo (no penaliza tanto las detecciones
            # decentes pero no perfectas). Convención FCOS.
            cls_fg = cls_fg * cnt_sig.sqrt()

            # ---------------------------------------------------------
            # Aplanar dimensiones espaciales (H×W → una sola dimensión).
            # ---------------------------------------------------------
            # Permutamos a [B, H, W, C-1] y luego reshape a [B, H*W, C-1].
            # Cada fila = una celda del feature map; cada columna = una clase.
            cls_fg = cls_fg.permute(0, 2, 3, 1).reshape(B, -1, C - 1)
            boxes_flat = boxes_lvl.reshape(B, -1, 4)  # [B, H*W, 4]

            # Mejor clase y mejor score por cada posición.
            # `max(dim=-1)` devuelve (valores, índices).
            best_score, best_label = cls_fg.max(dim=-1)  # [B, HW], [B, HW]

            # Guardar las predicciones de este nivel para cada imagen.
            for bi in range(B):
                per_image_boxes[bi].append(boxes_flat[bi])
                per_image_scores[bi].append(best_score[bi])
                per_image_labels[bi].append(best_label[bi])

        # ------------------------------------------------------------------
        # Consolidar niveles, filtrar por umbral, NMS.
        # ------------------------------------------------------------------
        out_boxes, out_scores, out_labels = [], [], []

        for bi in range(B):
            # Juntar las predicciones de todos los niveles en un solo tensor.
            # `cat(dim=0)` = concatenar por la primera dimensión.
            boxes = torch.cat(per_image_boxes[bi], dim=0)  # [N, 4]
            scores = torch.cat(per_image_scores[bi], dim=0)  # [N]
            labels = torch.cat(per_image_labels[bi], dim=0) + 1  # +1: 1-based
            # (+1 porque antes descartamos el fondo con [:, 1:], así que
            # los índices empezaban en 0. Los pasamos a 1-based para que
            # "1" = primer objeto, tal como espera el pipeline.)

            # Filtrar por umbral bajo → alta recall.
            # `keep` es una máscara booleana (True/False por elemento).
            keep = scores > self.score_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # Caso borde: no hay detecciones. Devolver vacío.
            if boxes.numel() == 0:
                out_boxes.append(boxes)
                out_scores.append(scores)
                out_labels.append(labels)
                continue

            # Recortar cajas a los límites de la imagen.
            # `clamp(0, max)` = "no dejes que se salga de [0, max]".
            # `0::2` = índices pares (x1, x2); `1::2` = impares (y1, y2).
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_hw[1])
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_hw[0])

            # Top-K antes de NMS: si hay demasiadas cajas, quedarnos con
            # las de mayor score. Acelera el NMS.
            max_before_nms = self.topk_per_level * len(strides)
            if scores.numel() > max_before_nms:
                # `topk(k).indices` = los índices de las K puntuaciones más altas.
                top_idx = scores.topk(max_before_nms).indices
                boxes, scores, labels = boxes[top_idx], scores[top_idx], labels[top_idx]

            # NMS (Non-Maximum Suppression).
            # ¿Qué hace? Si dos cajas se solapan mucho (IoU > nms_thresh),
            # se queda solo con la de mayor score. Evita duplicados.
            # `[: self.max_dets]` limita el número final de propuestas.
            keep_idx = nms(boxes, scores, self.nms_thresh)[: self.max_dets]

            out_boxes.append(boxes[keep_idx])
            out_scores.append(scores[keep_idx])
            out_labels.append(labels[keep_idx])

        return out_boxes, out_scores, out_labels

    # ======================================================================
    # FORWARD
    # ======================================================================
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x: [B, 3, H, W] con H, W múltiplos de 32.
                (B imágenes, 3 canales RGB, H de alto, W de ancho).
            return_features:
                False → devuelve (boxes, scores, labels) como listas de tamaño B.
                True  → devuelve la lista de features FPN (para el YOLOv5s-CBAM
                        o para debug).

        Nota: en modo detección, `boxes`, `scores`, `labels` son listas de
        tensores (una entrada por imagen del batch). Para B=1, se indexa [0].
        """
        profile = not self.training
        timings = []

        def start_timer():
            if not profile:
                return None
            if x.device.type == "cuda":
                torch.cuda.synchronize(x.device)
            return time.perf_counter()

        def record_timing(label, start):
            if profile:
                if x.device.type == "cuda":
                    torch.cuda.synchronize(x.device)
                timings.append((label, time.perf_counter() - start))

        def print_timings(total_start):
            if not profile:
                return
            print("[Tiempos SwinROIProposer.forward]")
            for label, elapsed in timings:
                print(f"  {label}: {elapsed:.3f} s")
            print(
                f"  Total SwinROIProposer.forward: "
                f"{time.perf_counter() - total_start:.3f} s"
            )

        total_start = start_timer()
        stage_start = start_timer()
        self._validate_input(x)
        record_timing("Validación de entrada", stage_start)

        # ---- 1. Backbone ----
        # `self.backbone(x)` devuelve una lista de feature maps, uno por
        # cada nivel que pedimos (out_indices=(1,2,3)).
        stage_start = start_timer()
        features = self.backbone(x)  # lista [B, C_i, H_i, W_i] (o NHWC)
        record_timing("Backbone Swin", stage_start)

        # ---- 2. Normalizar a NCHW ----
        # Swin devuelve NHWC; lo pasamos a NCHW (lo que espera el FPN).
        stage_start = start_timer()
        features = self._to_nchw(features, self.out_channels)

        # ---- 3. FPN ----
        # El FPN espera un DICCIONARIO {"0": feat0, "1": feat1, ...},
        # por eso lo construimos con un dict comprehension.
        # `enumerate(features)` da pares (índice, feature).
        feat_dict = {str(i): f for i, f in enumerate(features)}
        record_timing("Conversión de features y preparación para FPN", stage_start)

        # Aplicamos el FPN. `fpn_out` es otro diccionario con la misma
        # estructura pero canales unificados (256) y con información
        # mezclada entre niveles.
        stage_start = start_timer()
        fpn_out = self.fpn(feat_dict)
        record_timing("FPN", stage_start)

        # Volvemos a lista (mismo orden que las claves 0,1,2...).
        fpn_features = list(fpn_out.values())

        # Si el usuario solo quería las features (para el YOLO o debug),
        # devolvemos aquí.
        if return_features:
            print_timings(total_start)
            return fpn_features

        # ---- 4. Head + decode ----
        # El cabezal FCOS predice logits, distancias y centerness por nivel.
        stage_start = start_timer()
        cls_logits, reg_preds, cnt_preds = self.head(fpn_features)
        record_timing("Cabezal FCOS", stage_start)

        # Calculamos el stride efectivo de cada nivel a partir del tamaño
        # real del feature map (no asumimos uno fijo).
        stage_start = start_timer()
        strides = self._infer_strides(fpn_features, x.shape[-2:])
        record_timing("Cálculo de strides", stage_start)

        # Convertimos todo a cajas reales.
        stage_start = start_timer()
        boxes, scores, labels = self._decode(
            cls_logits, reg_preds, cnt_preds, strides, x.shape[-2:]
        )
        record_timing("Decodificación FCOS y NMS", stage_start)
        print_timings(total_start)
        return boxes, scores, labels

