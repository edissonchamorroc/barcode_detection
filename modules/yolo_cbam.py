# yolo_cbam.py
"""
Etapa 2 del pipeline: YOLOv5s con CBAM (vía forward hooks) para refinamiento de ROI.

Flujo:
    Propuestas Swin → crops → YOLOv5s-CBAM → bbox refinada → decisión SAM/pyzbar
"""

# ----------------------------------------------------------------------------
# IMPORTACIONES: traemos herramientas ya hechas de otras librerías.
# ----------------------------------------------------------------------------

# torch = PyTorch. La "caja de herramientas" estándar para redes neuronales.
import time

import torch

# `torch.nn` = bloques prefabricados (capas convolucionales, etc.).
# `as nn` es un alias para escribir menos.
import torch.nn as nn

# numpy = calculadora para tablas grandes de números. Una imagen es
# una tabla de píxeles; numpy la maneja eficientemente.
import numpy as np

# `typing` = etiquetas de tipos (List, Tuple, etc.). No cambia la
# ejecución; solo documenta qué entra y qué sale.
from typing import List, Tuple, Optional, Dict


# ============================================================================
# 1. MÓDULO CBAM
# ============================================================================
# CBAM = "Convolutional Block Attention Module". Es un "filtro inteligente"
# que se coloca entre capas de una red. Su trabajo: resaltar lo importante
# de la imagen y atenuar lo que no lo es. Tiene dos atenciones:
#
#   1. Atención de canal (channel attention): "¿qué TIPOS de información
#      son relevantes?" (colores, texturas, bordes, etc.).
#   2. Atención espacial (spatial attention): "¿en qué ZONAS de la imagen
#      hay que fijarse?".
#
# Analogía: imagina que estudias para un examen. La atención de canal
# sería "¿qué asignaturas priorizo?"; la espacial, "¿qué páginas del
# libro son las importantes?".
# ============================================================================
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (channel + spatial attention).
    """

    # channels: cuántos "canales" (capas de información) tiene la entrada.
    # reduction: cuánto comprimir en el MLP intermedio (16 → divide entre 16).
    # kernel_size: tamaño del filtro espacial (7 → ventana de 7x7 píxeles).
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()

        # Tamaño de la capa intermedia. `max(..., 8)` evita que en canales
        # pequeños (por ejemplo 32 // 16 = 2) la capa se vuelva demasiado
        # pequeña (mínimo 8). Es como "no comprimas demasiado".
        hidden = max(channels // reduction, 8)

        # --- Módulo de atención de CANAL ---
        # Es un "cuello de botella": comprime (channels → hidden) y
        # expande (hidden → channels). Así aprende relaciones entre canales
        # con pocos parámetros. Es un mini-MLP (perceptrón multicapa).
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),    # comprime
            nn.ReLU(inplace=True),                         # activación
            nn.Conv2d(hidden, channels, 1, bias=False),    # expande
        )

        # Sigmoid aplasta la salida a [0, 1]. Esos valores serán "pesos"
        # para multiplicar cada canal: 0 = ignóralo, 1 = préstale toda
        # la atención.
        self.channel_sigmoid = nn.Sigmoid()

        # --- Módulo de atención ESPACIAL ---
        # Recibe 2 mapas (promedio y máximo a lo largo de canales) y los
        # combina en 1 mapa de atención con una convolución 7x7 (contexto
        # amplio). El 2 → 1 significa "fusión de dos mapas en uno".
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        `forward` = qué hace el módulo al aplicarlo: `cbam(x)`.
        Recibe un tensor [B, C, H, W] y devuelve otro del mismo tamaño
        pero con la información "filtrada" por las dos atenciones.
        """
        # ---- ATENCIÓN DE CANAL ----
        # Comprimimos espacialmente con dos estadísticos por canal:
        #   - promedio (mean): valor "típico" de ese canal.
        #   - máximo (amax): valor "más intenso" de ese canal.
        # `dim=[2, 3]` = colapsar las dimensiones espaciales (alto y ancho).
        # `keepdim=True` conserva esas dimensiones como tamaño 1 (para que
        # los tensores sigan siendo [B, C, 1, 1] y se pueda multiplicar
        # luego sin problemas).
        avg_pool = x.mean(dim=[2, 3], keepdim=True)
        max_pool = x.amax(dim=[2, 3], keepdim=True)

        # Pasamos ambos estadísticos por el mismo MLP y sumamos.
        # Resultado: un vector [B, C, 1, 1] con un "peso" por canal.
        # Sigmoid final → valores en [0, 1].
        channel_att = self.channel_sigmoid(
            self.channel_mlp(avg_pool) + self.channel_mlp(max_pool)
        )

        # Multiplicamos la entrada por esos pesos: canales importantes
        # se refuerzan; los irrelevantes se atenúan.
        x = x * channel_att

        # ---- ATENCIÓN ESPACIAL ----
        # Ahora colapsamos los canales (no las dimensiones espaciales).
        # `dim=1` = colapsar canales. Obtenemos dos mapas [B, 1, H, W].
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)

        # Los pegamos como dos "bandas" (2 canales) → [B, 2, H, W].
        spatial_cat = torch.cat([avg_out, max_out], dim=1)

        # Convolución 7x7 sobre esas 2 bandas → 1 mapa de atención espacial
        # [B, 1, H, W], con valores en [0, 1] tras sigmoid.
        spatial_att = self.spatial_sigmoid(self.spatial_conv(spatial_cat))

        # Multiplicamos: zonas con atención alta se refuerzan; zonas con
        # atención baja se atenúan.
        x = x * spatial_att
        return x


# ============================================================================
# 2. HELPERS (NIVEL DE MÓDULO, sin indentación dentro de clase)
# ============================================================================
# Estas funciones NO pertenecen a ninguna clase. Son utilidades "sueltas"
# que se llaman desde el resto del archivo.
# ============================================================================
def _infer_out_channels(module: nn.Module) -> int:
    """
    Infiere canales de salida de un módulo, según su tipo.

    Traducción para no expertos:
        Las capas de YOLOv5 están envueltas en módulos con nombres (C3,
        SPPF...). Cada módulo esconde varias subcapas internas y NO
        siempre exponen directamente cuántos canales tienen. Esta función
        "espía" dentro para averiguarlo, sea cual sea la variante.

    Reglas:
        - C3  : la salida real es cv3 (cv2 es interno con c2/2 canales).
        - SPPF: la salida real es cv2.
        - Otro: último Conv2d registrado.
    """
    # `type(module).__name__` nos da el NOMBRE de la clase del módulo como
    # texto: "C3", "SPPF", "Conv", etc. Es la forma de preguntar
    # "¿qué tipo de capa es esto?".
    cls_name = type(module).__name__

    # --- C3: la convolución de salida es cv3 ---
    if cls_name == "C3":
        # `hasattr` pregunta: "¿tiene este objeto un atributo llamado X?".
        # C3 en YOLOv5 tiene un atributo `cv3` que es la última conv.
        if hasattr(module, "cv3"):
            sub = module.cv3
            # Verificamos que esa subcapa tenga `conv` y que sea Conv2d.
            if hasattr(sub, "conv") and isinstance(sub.conv, nn.Conv2d):
                return sub.conv.out_channels

        # Fallback para variantes raras de C3:
        # `reversed(list(module.modules()))` recorre los submódulos en orden
        # INVERSO (de la salida hacia la entrada). Nos quedamos con el
        # último Conv2d, que es el que produce la salida.
        for m in reversed(list(module.modules())):
            if isinstance(m, nn.Conv2d):
                return m.out_channels
        raise ValueError("No se pudo inferir canales para C3")

    # --- SPPF: la convolución de salida es cv2 ---
    # SPPF ("Spatial Pyramid Pooling - Fast") es un bloque que mezcla
    # información a varias escalas. Su salida está en `cv2`.
    if cls_name == "SPPF":
        if hasattr(module, "cv2"):
            sub = module.cv2
            if hasattr(sub, "conv") and isinstance(sub.conv, nn.Conv2d):
                return sub.conv.out_channels
        for m in reversed(list(module.modules())):
            if isinstance(m, nn.Conv2d):
                return m.out_channels
        raise ValueError("No se pudo inferir canales para SPPF")

    # --- Genérico (por si añades más tipos) ---
    # Último recurso: el último Conv2d del módulo.
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Conv2d):
            return m.out_channels
    raise ValueError(f"No se pudo inferir canales para {cls_name}")


def attach_cbam_hooks(
    detection_model: nn.Module,
    device: str = "cpu",
    reduction: int = 16,
    kernel_size: int = 7,
) -> nn.ModuleDict:
    """
    Adjunta CBAM a las capas C3 y SPPF de YOLOv5 vía forward hooks.
    Devuelve un ModuleDict con los CBAM para que aparezcan en state_dict.

    Traducción para no expertos:
        "Inyectar CBAM en YOLO" = insertar el filtro de atención que
        definimos arriba DESPUÉS de ciertas capas del modelo YOLO, sin
        modificar el archivo original de YOLO. Lo hacemos con "hooks":
        PyTorch ofrece una forma de decir "cuando esta capa termine de
        calcular su salida, pásala por esta función antes de continuar".

    Por qué C3 y SPPF:
        - C3: bloques que procesan características en cada etapa del
          backbone. Ahí CBAM puede "limpiar" la información.
        - SPPF: cuello entre backbone y cabeza. Ahí CBAM puede
          "priorizar" qué información pasa a la fase de detección.
    """
    # `nn.ModuleDict` es un diccionario que guarda módulos PyTorch.
    # Usarlo (en vez de un dict normal) hace que `state_dict()` de PyTorch
    # los incluya automáticamente → se pueden guardar/cargar con el modelo.
    cbam_modules = nn.ModuleDict()

    # Aquí guardaremos los "handles" (identificadores) de los hooks, para
    # poder eliminarlos si hiciera falta.
    hook_handles = []

    # `getattr(obj, "model", None)` intenta leer `obj.model`; si no
    # existe, devuelve None en lugar de fallar.
    inner_seq = getattr(detection_model, "model", None)
    if inner_seq is None:
        raise ValueError(
            "El modelo no tiene atributo .model. ¿Es un DetectionModel de ultralytics?"
        )

    # `enumerate(inner_seq)` recorre la secuencia de capas del modelo
    # (que es una `nn.Sequential`) devolviendo (índice, capa).
    for idx, layer in enumerate(inner_seq):
        cls_name = type(layer).__name__

        # Solo nos interesan capas C3 y SPPF. Cualquier otra la saltamos.
        if cls_name not in ("C3", "SPPF"):
            continue

        # Averiguamos cuántos canales tiene esta capa de salida.
        try:
            channels = _infer_out_channels(layer)
        except ValueError as e:
            # `try/except` = "intenta esto; si falla, muestra un aviso
            # y sigue". No queremos que un fallo en UNA capa tumbe todo.
            print(f"  [skip] capa {idx} ({cls_name}): {e}")
            continue

        # Creamos el CBAM con los canales detectados y lo movemos al
        # dispositivo (CPU o GPU).
        cbam = CBAM(channels, reduction=reduction, kernel_size=kernel_size).to(device)

        # Le ponemos una clave única: "cbam_<índice>_<tipo>".
        key = f"cbam_{idx}_{cls_name.lower()}"
        cbam_modules[key] = cbam

        # ----- FORWARD HOOK -----
        # Un hook = una función que PyTorch llama automáticamente justo
        # después de que la capa haga su forward.
        # Recibe (module, inputs, output); lo que DEVUELVA reemplaza a
        # `output` para las capas siguientes. Aprovechamos eso: el hook
        # devuelve el output "pasado por CBAM".
        #
        # `make_hook` es una "fábrica de hooks": como estamos en un bucle,
        # si pasáramos `cbam` directamente, TODOS los hooks acabarían
        # apuntando al mismo `cbam`. Al crear una función por iteración
        # capturamos el `cbam` correcto en cada una.
        def make_hook(cbam_ref):
            def hook(module, inputs, output):
                return cbam_ref(output)
            return hook

        # Registramos el hook en la capa. Guardamos el handle para luego.
        handle = layer.register_forward_hook(make_hook(cbam))
        hook_handles.append(handle)
        print(f"  CBAM adjunto a capa {idx} ({cls_name}, {channels} canales)")

    if len(cbam_modules) == 0:
        print("  [warning] No se adjuntó ningún CBAM.")

    # Guardamos los CBAM y los handles como atributos del modelo YOLO para
    # que el refiner (más abajo) pueda acceder a ellos.
    detection_model._cbam_modules = cbam_modules
    detection_model._cbam_hook_handles = hook_handles

    return cbam_modules


# ============================================================================
# 3. REFINADOR YOLOv5s-CBAM
# ============================================================================
class YOLOv5CBAMRefiner:
    """
    Refinador de ROI basado en YOLOv5s + CBAM.

    Traducción para no expertos:
        Esta es la etapa 2 del pipeline. Recibe las "propuestas" que dio
        Swin (posibles zonas con código de barras) y hace dos cosas:
            1. Recorta cada propuesta y pasa el crop por YOLOv5s (con CBAM
               inyectado) para refinar la caja.
            2. Decide si la detección es lo bastante segura o si necesita
               un paso extra con SAM (Segment Anything Model).

    Args:
        model_path: ruta a los pesos YOLO. None = "yolov5s.pt" por defecto.
        conf_thresh: umbral mínimo de confianza (0.25 por defecto).
        iou_thresh: umbral de IoU para NMS (0.45).
        device: "cpu" o "cuda". None = automático.
        inject_cbam: si inyectar CBAM en YOLO (True por defecto).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: Optional[str] = None,
        inject_cbam: bool = True,
    ):
        # Guardamos la configuración.
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Elección automática de dispositivo:
        # `device or (...)` = "usa `device` si no es None; si es None,
        # usa el resultado del paréntesis".
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Importamos YOLO de ultralytics AQUÍ (no al principio del archivo)
        # para no cargar la librería si no se usa la clase. Ahorra tiempo
        # de importación.
        from ultralytics import YOLO

        # Si no nos dan ruta, usamos los pesos oficiales preentrenados.
        weights = model_path if model_path else "yolov5s.pt"
        print(f"Cargando YOLO desde: {weights}")

        # `YOLO(weights)` carga el modelo con esos pesos.
        self.model = YOLO(weights)

        # `.model.model` es la red "cruda" (sin la capa que envuelve
        # Ultralytics). La movemos al dispositivo (CPU o GPU).
        self.model.model.to(self.device)

        if inject_cbam:
            print("Inyectando CBAM:")
            # `attach_cbam_hooks` inserta los CBAM en las capas C3/SPPF.
            self.cbam_modules = attach_cbam_hooks(
                self.model.model, device=self.device
            )
        else:
            self.cbam_modules = None

        # `.eval()` = modo evaluación. Desactiva el dropout (que es
        # aleatorio) y fija las estadísticas de normalización. En
        # inferencia queremos resultados reproducibles.
        self.model.model.eval()
        print(f"YOLOv5s-CBAM listo en {self.device}")

    # ======================================================================
    # UTILIDAD: extraer un crop alrededor de una caja
    # ======================================================================
    def _extract_crop(
        self,
        image: np.ndarray,
        box: torch.Tensor,
        expand_ratio: float = 0.1,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Recorta la imagen alrededor de una caja con un margen extra.
        El margen (10%) evita cortar barras o zona de silencio.
        """
        # h = alto, w = ancho de la imagen completa.
        h, w = image.shape[:2]

        # `box.tolist()` convierte el tensor [x1, y1, x2, y2] a una lista
        # Python de números.
        x1, y1, x2, y2 = box.tolist()

        # Ancho y alto de la caja.
        bw, bh = x2 - x1, y2 - y1

        # Expandimos proporcionalmente. `max(0, ...)` / `min(w, ...)` evitan
        # salir de la imagen. `int()` convierte a entero (los píxeles son
        # enteros).
        x1 = max(0, int(x1 - bw * expand_ratio))
        y1 = max(0, int(y1 - bh * expand_ratio))
        x2 = min(w, int(x2 + bw * expand_ratio))
        y2 = min(h, int(y2 + bh * expand_ratio))

        # `.copy()` para que el crop sea independiente del original
        # (no queremos modificar la imagen original por accidente).
        crop = image[y1:y2, x1:x2].copy()

        # Devolvemos dos cosas: el crop y las coordenadas del crop en la
        # imagen original (para poder "traducir" las detecciones de vuelta).
        return crop, (x1, y1, x2, y2)

    # ======================================================================
    # REFINAMIENTO PRINCIPAL
    # ======================================================================
    @torch.no_grad()  # desactiva gradientes: estamos en inferencia
    def refine(
        self,
        image: np.ndarray,
        proposals: torch.Tensor,
        proposal_scores: torch.Tensor,
    ) -> List[Dict]:
        """
        Refina una lista de propuestas de Swin.

        Args:
            image: imagen original HWC BGR.
            proposals: tensor [N, 4] con cajas (x1, y1, x2, y2).
            proposal_scores: tensor [N] con scores de Swin.

        Returns:
            Lista de dicts con: box refinada (en coordenadas de la imagen
            ORIGINAL), score de YOLO, score de Swin, needs_sam.
        """
        timings = []

        def synchronize_device():
            if (
                torch.cuda.is_available()
                and str(self.device).startswith("cuda")
            ):
                torch.cuda.synchronize()

        def start_timer():
            synchronize_device()
            return time.perf_counter()

        def record_timing(label, start):
            synchronize_device()
            timings.append((label, time.perf_counter() - start))

        def print_timings(total_start):
            for label, elapsed in timings:
                print(f"  {label}: {elapsed:.3f} s")
            print(
                f"  Total YOLOv5CBAMRefiner.refine: "
                f"{time.perf_counter() - total_start:.3f} s"
            )

        total_start = start_timer()
        # Si no hay propuestas, no hay nada que hacer.
        if len(proposals) == 0:
            print("[Tiempos YOLO-CBAM]")
            print_timings(total_start)
            return []

        results: List[Dict] = []

        # `zip(proposals, proposal_scores)` empareja cada caja con su score.
        # `i` = índice de la propuesta.
        for i, (box, swin_score) in enumerate(zip(proposals, proposal_scores)):
            # Recortamos la zona de la propuesta (con margen extra).
            crop_start = start_timer()
            crop, (cx1, cy1, cx2, cy2) = self._extract_crop(image, box)
            record_timing(f"Recorte propuesta {i + 1}", crop_start)

            # Si el crop quedó vacío (caja inválida), saltamos.
            if crop.size == 0:
                continue

            # Pasamos el CROP por YOLO. Ojo: YOLO predice coordenadas
            # RELATIVAS al crop, no a la imagen original.
            predict_start = start_timer()
            detections = self.model.predict(
                source=crop,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                verbose=False,  # silenciar logs
            )
            record_timing(f"Inferencia YOLO propuesta {i + 1}", predict_start)

            # `detections` es una lista (una por imagen del batch; aquí
            # batch=1). Recorremos cada resultado.
            postprocess_start = start_timer()
            for det in detections:
                # `det.boxes` contiene las cajas detectadas.
                boxes = det.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                # `.cpu().numpy()` = trae los tensores a CPU y los
                # convierte a arrays de numpy para manipularlos como
                # simples tablas de números.
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                # Iteramos cada detección dentro del crop.
                for (x1c, y1c, x2c, y2c), conf in zip(xyxy, confs):
                    # sumamos el offset del crop (cx1, cy1)
                    # para volver a coordenadas de la imagen original.
                    x1o = cx1 + float(x1c)
                    y1o = cy1 + float(y1c)
                    x2o = cx1 + float(x2c)
                    y2o = cy1 + float(y2c)
                    
                    # ========================================================
                    # FILTRO: Múltiples barcodes en una sola propuesta
                    # ========================================================
                    # Si YOLO detectó más de un código en el mismo recorte,
                    # significa que Swin abarcó el espacio entre dos códigos.
                    # Descartamos la propuesta completa para evitar procesar cortes.
                    if len(xyxy) > 1:
                        continue

                    # NUEVA ESTRATEGIA: 
                    # Si YOLO confirma que es un código de barras (conf > umbral),
                    # usamos la caja ORIGINAL de Swin (que es más generosa)
                    # en lugar de la caja apretada de YOLO.
                    # La variable `box` contiene las coordenadas absolutas de Swin.
                    
                    results.append({
                        "box": box.tolist(), # <--- Retornamos la caja de Swin
                        "score": float(conf), # Mantenemos la confianza de YOLO
                        "swin_score": float(swin_score),
                        "source_proposal_idx": i,
                    })
            record_timing(
                f"Postprocesamiento propuesta {i + 1}", postprocess_start
            )

        # NMS global: como pueden venir cajas solapadas (de propuestas
        # distintas), las limpiamos.
        if results:
            nms_start = start_timer()
            results = self._global_nms(results)
            record_timing("NMS global", nms_start)

        print("[Tiempos YOLO-CBAM]")
        print_timings(total_start)
        return results

    # ======================================================================
    # BASELINE: YOLO sobre la imagen completa (sin Swin)
    # ======================================================================
    @torch.no_grad()
    def refine_full_image(
        self,
        image: np.ndarray,
        imgsz: int = 640,
    ) -> List[Dict]:
        """
        Corre YOLO directamente sobre la imagen completa, sin propuestas de Swin.
        Útil para evaluación YOLO-only y como baseline.

        Args:
            image: imagen HWC BGR.
            imgsz: resolución de inferencia (subir a 960/1280 para barcodes
                pequeños). YOLO redimensiona la imagen a este tamaño
                internamente antes de pasarla por la red.

        Returns:
            Lista de dicts con la misma estructura que `refine()`.
        """
        detections = self.model.predict(
            source=image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=imgsz,
            verbose=False,
        )

        results: List[Dict] = []
        for det in detections:
            boxes = det.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                results.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(conf),
                    # No hay Swin aquí, así que el score asociado es 0.
                    "swin_score": 0.0,
                    # Sin propuesta de origen → -1.
                    "source_proposal_idx": -1,
                })

        if results:
            results = self._global_nms(results)

        return results

    # ======================================================================
    # NMS global sobre los resultados
    # ======================================================================
    def _global_nms(self, results: List[Dict]) -> List[Dict]:
        """
        Aplica NMS a una lista de resultados ya en formato dict.

        NMS = "Non-Maximum Suppression". Si dos cajas se solapan mucho,
        nos quedamos solo con la de mayor confianza. Evita tener la misma
        detección repetida varias veces.
        """
        # Con 0 o 1 resultado no hay nada que suprimir.
        if len(results) <= 1:
            return results

        # Importamos nms de torchvision aquí (no al inicio del archivo).
        from torchvision.ops import nms

        # Construimos tensores desde los dicts.
        boxes = torch.tensor([r["box"] for r in results], dtype=torch.float32)
        scores = torch.tensor([r["score"] for r in results], dtype=torch.float32)

        # `nms` devuelve los ÍNDICES a conservar.
        keep = nms(boxes, scores, self.iou_thresh)

        # Reconstruimos la lista usando solo esos índices.
        # `.tolist()` convierte tensor de índices a lista Python.
        return [results[i] for i in keep.tolist()]

    
    # ======================================================================
    # GUARDADO / CARGA DE PESOS CBAM
    # ======================================================================
    def state_dict_cbam(self) -> Dict[str, torch.Tensor]:
        """
        Devuelve los pesos de los CBAM inyectados.
        Útil para guardarlos junto con el modelo o por separado.
        """
        if self.cbam_modules is None:
            return {}
        return self.cbam_modules.state_dict()

    def load_state_dict_cbam(self, state_dict: Dict[str, torch.Tensor]):
        """
        Carga pesos preentrenados de los CBAM.
        """
        if self.cbam_modules is None:
            raise RuntimeError("CBAM no fue inyectado.")
        self.cbam_modules.load_state_dict(state_dict)

