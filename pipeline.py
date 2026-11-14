# test_pipeline.py

# ----------------------------------------------------------------------------
# IMPORTACIONES
# ----------------------------------------------------------------------------
import sys  # Para manipular el "path" de búsqueda de módulos.
from pathlib import Path  # Manejo moderno de rutas de archivos.
import shutil  # Para borrar carpetas recursivamente.
import time  # Medición del tiempo de procesamiento.
import cv2  # OpenCV: leer, dibujar y guardar imágenes.
import numpy as np  # Cálculos con tablas de números (imágenes).
import torch  # PyTorch: tensores y redes neuronales.

# --- Configuración ---
# Rutas a los archivos de entrada. Si cambias de imagen o de pesos,
# edita estas líneas.
IMAGE_PATH = "dataset_prepared/yolo/images/test/prueba4.jpg"
SWIN_WEIGHTS = "runs/swin_fcos/best.pt"
YOLO_WEIGHTS = "runs/detect/runs/barcode/yolov5s_base/weights/best.pt"
CBAM_WEIGHTS = "runs/detect/runs/barcode/yolov5s_cbam/weights/best.pt"
CHAR_YOLO_WEIGHTS = "runs/char_detector/best.pt"
OUTPUT_DIR = "test_cascade_out"  # Carpeta donde se guardarán las imágenes.
DEVICE = "cpu"  # "0" si tenés GPU local. (En GPU sería "0" o "cuda".)
PROCESS_VIDEO = False  # True: procesa video; False: procesa una sola imagen.
VIDEO_PATH = "video.mp4"
VIDEO_OUTPUT_PATH = "video_resultado.mp4"

# Imports del proyecto.
# `sys.path.insert(0, "modules")` añade la carpeta "modules" al "buscador
# de módulos" de Python. Sirve para poder hacer `from swin_transformer import...`
# sin instalar nada, aunque los archivos estén dentro de "modules/".
sys.path.insert(0, "modules")

# Traemos las 3 piezas principales del pipeline (esta versión NO usa SAM).
from swin_transformer import SwinROIProposer  # Etapa 1
from yolo_cbam import YOLOv5CBAMRefiner  # Etapa 2
from barcode_decoder import BarcodeDecoder  # Etapa 3 (lectura)
from yolo_ocr import SwinCharOCRDecoder


# ===========================================================================
# UTILIDADES DE VISUALIZACIÓN
# ---------------------------------------------------------------------------
# Estas funciones NO forman parte del pipeline en sí. Solo sirven para
# DIBUJAR las cajas sobre la imagen y guardar un PNG que puedas abrir con
# el visor de fotos para "ver" lo que hizo el modelo.
# ===========================================================================
def draw_stage1(image, boxes, scores, title="Swin: propuestas"):
    """
    Panel 1: cajas de Swin (azul).

    `boxes` es una lista de cajas [x1, y1, x2, y2] y `scores` una lista
    con un número por caja (la confianza). Dibujamos un rectángulo por
    cada una y le ponemos el score como texto encima.
    """
    # `.copy()` = copia la imagen para no modificar la original.
    viz = image.copy()

    # `zip(boxes, scores)` empareja cada caja con su score.
    for box, score in zip(boxes, scores):
        # Convertimos las coordenadas a enteros (los píxeles lo son).
        x1, y1, x2, y2 = [int(v) for v in box]

        # `cv2.rectangle(img, p1, p2, color_BGR, grosor)` dibuja un rectángulo.
        # (255, 100, 0) en BGR es un azul-anaranjado; grosor 10 px.
        cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 100, 0), 10)

        # `cv2.putText` escribe texto sobre la imagen.
        # `max(12, y1 - 4)` evita que el texto se salga por arriba.
        cv2.putText(
            viz,
            f"{score:.2f}",  # score con 2 decimales
            (x1, max(12, y1 - 4)),  # posición
            cv2.FONT_HERSHEY_SIMPLEX,  # fuente
            0.4,  # tamaño
            (255, 100, 0),  # color
            1,  # grosor
        )

    # Añadimos una barra superior con el título y el número de cajas.
    return add_title(viz, f"{title} ({len(boxes)})")


def draw_stage2(image, refined):
    """
    Panel 2: cajas refinadas por YOLO (verde).

    Igual que `draw_stage1`, pero `refined` son dicts con campos
    "box" y "score" en lugar de listas separadas.
    """
    viz = image.copy()
    for r in refined:
        x1, y1, x2, y2 = [int(v) for v in r["box"]]
        # (0, 200, 0) en BGR = verde.
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 200, 0), 10)
        cv2.putText(
            viz,
            f"{r['score']:.2f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 200, 0),
            1,
        )
    return add_title(viz, f"YOLO-CBAM: refinadas ({len(refined)})")


def draw_stage3(image, decoded):
    """
    Panel 3: texto decodificado.
    Verde brillante = OK, rojo oscuro = no decode.

    Para cada caja, si `decoded` tiene texto, lo mostramos; si no,
    mostramos "no decode" y pintamos el recuadro en rojo.
    """
    viz = image.copy()
    for r in decoded:
        x1, y1, x2, y2 = [int(v) for v in r["source_box"]]

        # ¿Decodificó o no?
        ok = r["decoded"] is not None

        # Verde si ok, rojo si no.
        color = (0, 255, 0) if ok else (0, 0, 200)
        label = r["decoded"] if ok else "no decode"

        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 10)

        # `cv2.getTextSize` mide cuánto ocupará el texto, para ponerle un
        # fondo del tamaño justo (así se lee aunque la imagen sea clara).
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Rectángulo relleno (grosor -1) detrás del texto.
        cv2.rectangle(viz, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)

        # Texto blanco encima del fondo de color.
        cv2.putText(
            viz,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # Título: cuántas decodificaron del total.
    # `sum(1 for r in decoded if r['decoded'])` = contador de éxitos.
    # ⚠ Residuo: "pyzbar" ya no se usa (el decoder real es zxingcpp).
    return add_title(
        viz, f"pyzbar: decodificadas ({sum(1 for r in decoded if r['decoded'])})"
    )


def add_title(img, text):
    """
    Añade una barra negra superior con el título en blanco.
    Uso típico: armar una imagen "compuesta" con varias etapas.
    """
    # Creamos una barra negra de 30 px de alto, mismo ancho que `img`.
    bar = np.zeros((30, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    # `np.vstack` apila la barra ARRIBA de la imagen (verticalmente).
    return np.vstack([bar, img])


# ===========================================================================
# MAIN: función principal
# ===========================================================================
def process_video_frame(img, frame_number, swin, refiner, decoder, ocr_fallback):
    h, w = img.shape[:2]
    pipeline_start = time.perf_counter()
    print(f"\n--- Fotograma {frame_number} ({w}×{h}) ---")

    proc_size = 512
    scale = min(proc_size / max(h, w), 1.0)
    new_w = max(32, (int(w * scale) // 32) * 32)
    new_h = max(32, (int(h * scale) // 32) * 32)
    resized = cv2.resize(img, (new_w, new_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    tensor = (
        torch.from_numpy(rgb.transpose(2, 0, 1))
        .contiguous()
        .unsqueeze(0)
        .to(DEVICE)
        .float()
    )

    with torch.no_grad():
        boxes, scores, _ = swin(tensor)
    boxes = boxes[0].cpu().numpy()
    scores = scores[0].cpu().numpy()
    if new_w != w:
        boxes[:, 0::2] *= w / new_w
    if new_h != h:
        boxes[:, 1::2] *= h / new_h

    stage1 = draw_stage1(img, boxes, scores)
    if len(boxes) == 0:
        print("Swin no propuso regiones.")
        stage2 = draw_stage2(img, [])
        stage3 = draw_stage3(img, [])
    else:
        refined = refiner.refine(img, torch.from_numpy(boxes), torch.from_numpy(scores))
        stage2 = draw_stage2(img, refined)
        decoded = []
        for i, result in enumerate(refined):
            x1, y1, x2, y2 = result["box"]
            margin_x = int((x2 - x1) * 0.20)
            margin_y = int((y2 - y1) * 0.20)
            crop = img[
                max(0, int(y1) - margin_y) : min(h, int(y2) + margin_y),
                max(0, int(x1) - margin_x) : min(w, int(x2) + margin_x),
            ]
            hits = decoder.decode(crop, i=i)
            decoded_text = hits[0]["data"] if hits else None
            if decoded_text is None:
                ocr_result = ocr_fallback.process_barcode_region(
                    image=img
                )
                if ocr_result and ocr_result["texto"]:
                    decoded_text = ocr_result["texto"]
            decoded.append(
                {
                    "source_box": result["box"],
                    "decoded": decoded_text,
                }
            )
        stage3 = draw_stage3(img, decoded)

    composite = np.hstack(
        [add_title(img.copy(), f"Original ({w}×{h})"), stage1, stage2, stage3]
    )
    elapsed = time.perf_counter() - pipeline_start
    print(f"Tiempo de procesamiento del fotograma: {elapsed:.2f} s")
    return composite


def main_image():
    # ------------------------------------------------------------------
    # LIMPIEZA Y CARGA
    # ------------------------------------------------------------------
    # Borra la carpeta de salida si existía. `ignore_errors=True` evita
    # error si no existe.
    shutil.rmtree("test_cascade_out", ignore_errors=True)

    output_dir = Path(OUTPUT_DIR)
    # `parents=True` crea carpetas intermedias; `exist_ok=True` no falla
    # si ya existía.
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar imagen.
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        # `imread` devuelve None si el archivo no existe o está corrupto.
        raise FileNotFoundError(IMAGE_PATH)

    # h = alto, w = ancho. `[:2]` = "las dos primeras dimensiones" (H, W).
    h, w = img.shape[:2]
    print(f"Imagen: {IMAGE_PATH} ({w}×{h})")
   
    # ------------------------------------------------------------------
    # 2. Cargar Swin (etapa 1)
    # ------------------------------------------------------------------
    print(f"\n[1/3] Cargando Swin: {SWIN_WEIGHTS}")

    # `SwinROIProposer(pretrained=False, ...)`:
    #   - pretrained=False porque vamos a cargar NUESTROS pesos en la
    #     línea siguiente. Si fuera True, timm intentaría descargar
    #     pesos ImageNet que luego sobrescribiríamos (pérdida de tiempo).
    # `.to(DEVICE)` = mover a CPU (o GPU).
    # `.eval()` = modo inferencia (desactiva dropout, etc.).
    swin = (
        SwinROIProposer(
            pretrained=False,
            img_size=512,
            num_classes=2,
            score_thresh=0.05,  # umbral BAJO → alta recall
        )
        .to(DEVICE)
        .eval()
    )

    # `torch.load` carga el archivo de pesos.
    # `map_location=DEVICE` asegura que se cargue en el mismo dispositivo
    # en que está el modelo (si no, error).
    swin.load_state_dict(torch.load(SWIN_WEIGHTS, map_location=DEVICE))
    print("      ✓ Swin cargado")

    # ------------------------------------------------------------------
    # 3. Cargar YOLO-CBAM (etapa 2)
    # ------------------------------------------------------------------
    print(f"\n[2/3] Cargando YOLO-CBAM: {YOLO_WEIGHTS}")

    # ⚠ CONTRADICCIÓN: `inject_cbam=True` (el código SÍ inyecta CBAM),
    # pero el comentario dice "usar base (sin CBAM, porque fue mejor)".
    # Si querías la base sin CBAM, debería ser `inject_cbam=False`.
    # Revisa cuál es la intención real.
    refiner = YOLOv5CBAMRefiner(
        model_path=YOLO_WEIGHTS,
        conf_thresh=0.25,
        iou_thresh=0.45,
        device=DEVICE,
        inject_cbam=False,
    )
    # Si cbam es true
    # cbam_sd = torch.load(CBAM_WEIGHTS, map_location=DEVICE, weights_only=True)
    # refiner.load_state_dict_cbam(cbam_sd)
    # print(f"      ✓ CBAM cargado desde {CBAM_WEIGHTS} ({len(cbam_sd)} tensores)")

    print("      ✓ YOLO-CBAM cargado")

    # ------------------------------------------------------------------
    # 4. Cargar decodificador (etapa 3)
    # ------------------------------------------------------------------
    print("\n[3/3] Cargando decodificador")  # Con zxingcpp,

    decoder = BarcodeDecoder(
        min_crop_size=200,  # solo agranda si el crop mide <100 px
        upscale_factor=3.0,  # ×3 si es pequeño
        quiet_zone_ratio=0.30,  # 30% de margen blanco
    )
    print("      ✓ Decodificador cargado")
    # ------------------------------------------------------------------
    # 4. Cargar OCR (fallback)
    # ------------------------------------------------------------------
    print("\n[4/4] Cargando decodificador OCR (Fallback)")
    ocr_fallback = SwinCharOCRDecoder(
         char_yolo_path=CHAR_YOLO_WEIGHTS, use_gpu=(DEVICE != "cpu")
    )
    print("      ✓ Decodificador cargado")

    print("\n[4/4] Cargando decodificador OCR (Fallback)")

    # ===================================================================
    # ETAPA 1: Swin propone ROI
    # ===================================================================
    pipeline_start = time.perf_counter()
    print("\n" + "=" * 60)
    print("ETAPA 1: Swin propone ROI")
    print("=" * 60)

    # ---- Preprocesar la imagen para el Swin ----
    # El Swin exige H y W múltiplos de 32. Calculamos una escala que
    # mantenga la proporción, y luego redondeamos al múltiplo de 32.
    proc_size = 512
    scale = min(proc_size / max(h, w), 1.0)  # nunca agrandar, solo reducir
    new_w, new_h = int(w * scale), int(h * scale)

    # `// 32 * 32` = "redondea hacia abajo al múltiplo de 32".
    # `max(32, ...)` evita que quede en 0 (imágenes muy pequeñas).
    new_w = max(32, (new_w // 32) * 32)
    new_h = max(32, (new_h // 32) * 32)

    resized = cv2.resize(img, (new_w, new_h))

    # ---- Normalización ImageNet ----
    # El Swin fue preentrenado con imágenes normalizadas con la media y
    # desviación estándar de ImageNet. Si no replicamos esa normalización,
    # las predicciones serán basura.
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    rgb = rgb.astype(np.float32)  # (redundante pero inofensivo)

    # Convertir a tensor de PyTorch con forma [B, C, H, W]:
    #   - `transpose(2, 0, 1)`: de HWC a CHW.
    #   - `unsqueeze(0)`: añade la dimensión de batch (B=1).
    #   - `.to(DEVICE)`: al dispositivo correcto.
    #   - `.float()`: garantiza tipo float32 (por seguridad).
    tensor = (
        torch.from_numpy(rgb.transpose(2, 0, 1))
        .contiguous()
        .unsqueeze(0)
        .to(DEVICE)
        .float()
    )

    # ---- Ejecutar Swin ----
    # `with torch.no_grad():` desactiva gradientes (inferencia, más rápido).
    with torch.no_grad():
        boxes, scores, _ = swin(tensor)
    # `_` descarta `labels` (no lo necesitamos aquí).

    # `boxes` y `scores` son listas de tamaño B. Como B=1, tomamos [0].
    boxes = boxes[0].cpu().numpy()
    scores = scores[0].cpu().numpy()

    # ---- Reescalar coordenadas del tensor procesado → imagen original ----
    # Si redujimos la imagen para el Swin, las cajas están en las
    # coordenadas de la versión reducida. Hay que multiplicarlas por el
    # factor inverso para que encajen en la imagen original.
    # `0::2` = índices pares (x1, x2); `1::2` = impares (y1, y2).
    if new_w != w:
        boxes[:, 0::2] *= w / new_w
    if new_h != h:
        boxes[:, 1::2] *= h / new_h

    print(f"Propuestas de Swin: {len(boxes)}")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(f"  [{i}] box={[round(v) for v in box]} score={score:.3f}")

    # Si Swin no propone nada, abortar (no tiene sentido seguir).
    if len(boxes) == 0:
        print("\n⚠ Swin no propuso nada. Revisar el umbral o el entrenamiento.")
        elapsed = time.perf_counter() - pipeline_start
        print(f"Tiempo de procesamiento (etapa 1 hasta finalizar): {elapsed:.2f} s")
        return

    # Guardar el panel visual de la etapa 1.
    #stage1 = draw_stage1(img, boxes, scores)
    # log
    #cv2.imwrite(str(output_dir / "01_swin.png"), stage1)

    # ===================================================================
    # ETAPA 2: YOLO-CBAM refina cada propuesta
    # ===================================================================
    print("\n" + "=" * 60)
    print("ETAPA 2: YOLO-CBAM refina")
    print("=" * 60)

    # `refiner.refine` recibe la imagen original + las cajas y scores de
    # Swin. Devuelve una lista de dicts con las cajas refinadas.
    # Nota: `torch.from_numpy(...)` convierte arrays numpy a tensores.
    refined = refiner.refine(
        img,
        torch.from_numpy(boxes),
        torch.from_numpy(scores),
    )
    print(f"Refinadas por YOLO-CBAM: {len(refined)}")
    for i, r in enumerate(refined):
        print(
            f"  [{i}] box={[round(v) for v in r['box']]} "
            f"yolo={r['score']:.3f} swin={r['swin_score']:.3f} "
        )

    #stage2 = draw_stage2(img, refined)
    #log
    #cv2.imwrite(str(output_dir / "02_yolo.png"), stage2)

    # ===================================================================
    # ETAPA 3: decodifica
    # ===================================================================
    print("\n" + "=" * 60)
    print("ETAPA 3:  decodifica")  # ← Residuo: "pyzbar" ya no se usa, es zxingcpp.
    print("=" * 60)

    decoded = []
    n_ok = 0
    n_ocr_fallback = 0

    for i, r in enumerate(refined):

        x1, y1, x2, y2 = r["box"]

        # ---- Expandir la caja un 20% para dar "quiet zone" ----
        # ZBar/zxing necesitan margen blanco alrededor del código.
        # Sin ese margen, el decodificador falla aunque la caja sea perfecta.
        w_box = x2 - x1
        h_box = y2 - y1
        margin_x = int(w_box * 0.20)
        margin_y = int(h_box * 0.20)

        # `max(0, ...)` y `min(w/h, ...)` para no salir de la imagen.
        exp_x1 = max(0, int(x1) - margin_x)
        exp_y1 = max(0, int(y1) - margin_y)
        exp_x2 = min(w, int(x2) + margin_x)
        exp_y2 = min(h, int(y2) + margin_y)

        # ---- Recorte de ALTA RESOLUCIÓN ----
        # A diferencia del decoder, aquí NO hacemos resize: usamos la
        # imagen original. Más píxeles = mejor lectura de códigos
        # pequeños.
        crop_hires = img[exp_y1:exp_y2, exp_x1:exp_x2]
        #log
        cv2.imwrite(str(output_dir / f"crop_hires_{i}.png"), crop_hires)

        # ---- Intento 1: decodificar el crop nativo ----
        hits = decoder.decode(crop_hires, i=i)

        # Preparar el dict de resultado (con o sin decode).
        res_dict = {
            "source_box": r["box"],
            "score": r["score"],
            "swin_score": r["swin_score"],
            "decoded": None,
            "type": None,
            "quality": 0.0,
            "method": None,
        }

        if hits:
            # ---- Éxito directo ----
            best = hits[0]  # hits viene ordenado por calidad; [0] = mejor.
            res_dict["decoded"] = best["data"]
            res_dict["type"] = best["type"]
            res_dict["quality"] = best["quality"]
            res_dict["method"] = f"zxingcpp:{best.get('_variant', 'original')}"
            n_ok += 1
            print(
                f"  [{i}] ✓ {res_dict['decoded']!r} ({res_dict['type']}, "
                f"{res_dict['method']})"
            )
        else:
            # --- FALLBACK: INTENTO CON OCR ---
            print(f"  [{i}] ✗ Código de barras falló. Intentando OCR...")

            # Llamamos a la función de tu yolo_ocr.py
            ocr_res = ocr_fallback.process_barcode_region(
                image=crop_hires
            )

            if ocr_res and ocr_res["texto"]:
                # --- ÉXITO CON OCR ---
                res_dict["decoded"] = ocr_res["texto"]
                res_dict["type"] = "OCR_Text"
                res_dict["quality"] = 1.0  # Confianza simulada para el OCR
                res_dict["method"] = "ocr"
                n_ok += 1
                n_ocr_fallback += 1
                print(
                    f"  [{i}] ✓ OCR: {res_dict['decoded']!r} ({ocr_res['num_caracteres']} chars)"
                )
            else:
                print(f"  [{i}] ✗ Decodificador tradicional y OCR fallaron.")

        decoded.append(res_dict)
        print(f"Decodificadas exitosamente: {n_ok}/{len(decoded)}")
        print(f"  (De las cuales {n_ocr_fallback} fueron rescatadas por OCR)")

    # ---- Resumen de la etapa 3 ----
    print(f"Decodificadas: {n_ok}/{len(decoded)}")
    for i, r in enumerate(decoded):
        if r["decoded"]:
            print(f"  [{i}] ✓ {r['decoded']!r} ({r['type']}, q={r['quality']:.0f})")
        else:
            print(f"  [{i}] ✗ no decode")

    #stage3 = draw_stage3(img, decoded)
    #log
    #cv2.imwrite(str(output_dir / "03_decoded.png"), stage3)


    # ===================================================================
    # Veredicto
    # ===================================================================
    print("\n" + "=" * 60)
    print("VEREDICTO")
    print("=" * 60)
    print(f"  Propuestas Swin:    {len(boxes)}")
    print(f"  Refinadas YOLO:     {len(refined)}")
    print(f"  Decodificadas:      {n_ok}")

    if n_ok > 0:
        print("\n  ✅ EL PIPELINE FUNCIONA: la cascada detectó y decodificó.")
    elif len(refined) > 0:
        print("\n  ⚠ La cascada detectó, pero decoder no pudo decodificar.")
        print("     Posible causa: bbox ajustado (quiet zone), rotación, blur.")
    else:
        print("\n  ❌ La cascada no detectó nada. Revisar pesos y umbrales.")

    elapsed = time.perf_counter() - pipeline_start
    print(f"\nTiempo de procesamiento (etapa 1 hasta finalizar): {elapsed:.2f} s")


def main_video():
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"No se pudo abrir el video: {VIDEO_PATH}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    print(f"Cargando modelos para procesar el video: {VIDEO_PATH}")
    swin = (
        SwinROIProposer(
            pretrained=False,
            img_size=512,
            num_classes=2,
            score_thresh=0.05,
        )
        .to(DEVICE)
        .eval()
    )
    swin.load_state_dict(torch.load(SWIN_WEIGHTS, map_location=DEVICE))
    refiner = YOLOv5CBAMRefiner(
        model_path=YOLO_WEIGHTS,
        conf_thresh=0.25,
        iou_thresh=0.45,
        device=DEVICE,
        inject_cbam=False,
    )
    decoder = BarcodeDecoder(
        min_crop_size=200,
        upscale_factor=3.0,
        quiet_zone_ratio=0.30,
    )
    ocr_fallback = SwinCharOCRDecoder(
        char_yolo_path=CHAR_YOLO_WEIGHTS,
        use_gpu=(DEVICE != "cpu"),
    )

    output_path = Path(VIDEO_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frame_count = 0
    processing_start = time.perf_counter()
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            composite = process_video_frame(
                frame, frame_count + 1, swin, refiner, decoder, ocr_fallback
            )
            if writer is None:
                height, width = composite.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise OSError(f"No se pudo crear el video de salida: {output_path}")

            writer.write(composite)
            frame_count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise ValueError(f"El video no contiene fotogramas legibles: {VIDEO_PATH}")

    elapsed = time.perf_counter() - processing_start
    print(f"\nFotogramas procesados: {frame_count}")
    print(f"Video de salida: {output_path}")
    print(f"Tiempo total de procesamiento del video: {elapsed:.2f} s")


def main():
    if PROCESS_VIDEO:
        main_video()
    else:
        main_image()


# Punto de entrada: solo se ejecuta `main()` si corres este archivo directamente.
if __name__ == "__main__":
    main()
