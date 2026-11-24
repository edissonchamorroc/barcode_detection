# pipeline.py

import sys
from pathlib import Path
import shutil
import time
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import cv2
import numpy as np
import torch

# --- Configuración ---
IMAGE_PATH = "test/Foto75cm.png"
YOLO_WEIGHTS = "runs/yolo_barcode/best.pt"
CHAR_YOLO_WEIGHTS = "runs/char_detector/best.pt"
OUTPUT_DIR = "test_cascade_out"
DEVICE = "cpu"  # Usamos CPU por ahora
PROCESS_VIDEO = False
VIDEO_PATH = "Video_prueba50cm.mp4"
VIDEO_OUTPUT_PATH = "Video_resultado.mp4"
LOGGER = logging.getLogger(__name__)

_DECODER_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_OCR_EXECUTOR = ThreadPoolExecutor(max_workers=1)

sys.path.insert(0, "modules")

# Importamos solo lo que necesitamos ahora
from yolo_barcode import YOLOv8BarcodeDetector 
from barcode_decoder import BarcodeDecoder
from yolo_ocr import SwinCharOCRDecoder

def _sync_device():
    if DEVICE != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

# ===========================================================================
# UTILIDADES DE VISUALIZACIÓN
# ===========================================================================
def draw_detections(image, detections, title="YOLO: detectados"):
    """
    Dibuja las cajas detectadas por YOLO (verde).
    """
    viz = image.copy()
    for r in detections:
        x1, y1, x2, y2 = [int(v) for v in r["box"]]
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 200, 0), 5)
        cv2.putText(
            viz, f"{r['score']:.2f}", (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1,
        )
    return add_title(viz, f"{title} ({len(detections)})")

def draw_decoded(image, decoded):
    """
    Dibuja las cajas con el texto decodificado. Verde = OK, Rojo = Falló.
    """
    viz = image.copy()
    for r in decoded:
        x1, y1, x2, y2 = [int(v) for v in r["source_box"]]
        ok = r["decoded"] is not None
        color = (0, 255, 0) if ok else (0, 0, 200)
        label = r["decoded"] if ok else "no decode"

        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 5)
        font_scale = 1.6
        font_thickness = 4
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        cv2.rectangle(
            viz,
            (x1, y1 - th - 20),
            (x1 + tw + 16, y1),
            color,
            -1,
        )
        cv2.putText(
            viz, label, (x1 + 4, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness,
        )
        
    return add_title(
        viz, f"Decodificadas ({sum(1 for r in decoded if r['decoded'])})"
    )

def add_title(img, text):
    bar = np.zeros((30, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([bar, img])


def _report_background_failure(future, task_name):
    if future.cancelled():
        return

    error = future.exception()
    if error is not None:
        LOGGER.error(
            "La tarea paralela %s falló después de finalizar el crop.",
            task_name,
            exc_info=(type(error), error, error.__traceback__),
        )


def _decode_crop_race(crop, decoder, ocr_fallback, index):
    if decoder.detectar_color_codigo(crop) == "Verde":
        decoder_future = _DECODER_EXECUTOR.submit(decoder.decode, crop, i=index)
        hits, _ = decoder_future.result()
        if hits:
            return hits[0]["data"], "decoder", True
        return None, None, True

    futures = {
        _DECODER_EXECUTOR.submit(decoder.decode, crop, i=index): "decoder",
        _OCR_EXECUTOR.submit(ocr_fallback.process_barcode_region, image=crop): "YOLO-OCR",
    }

    try:
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                task_name = futures.pop(future)
                result = future.result()

                if task_name == "decoder":
                    hits, _ = result
                    if hits:
                        return hits[0]["data"], task_name, False
                elif result and result["texto"]:
                    return result["texto"], task_name, False

        return None, None, False
    finally:
        for future, task_name in futures.items():
            if not future.cancel():
                future.add_done_callback(
                    lambda completed, name=task_name: _report_background_failure(
                        completed, name
                    )
                )


# ===========================================================================
# LÓGICA DE PROCESAMIENTO (Compartida entre imagen y video)
# ===========================================================================
def process_and_decode(img, detector, decoder, ocr_fallback, tile_size=1024):
    """
    Recibe una imagen, corre YOLO, recorta y decodifica/OCR.
    """
    pipeline_start = time.perf_counter()
    h_orig, w_orig = img.shape[:2]

    # Reducir la imagen 4K para que YOLO para acelerar el proceso
    MAX_DIM = 640
    scale = 1.0
    if max(h_orig, w_orig) > MAX_DIM:
        scale = MAX_DIM / max(h_orig, w_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        img_proc = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"📸 Imagen reducida de {w_orig}x{h_orig} a {new_w}x{new_h} para YOLO.")
    else:
        img_proc = img

    # 1. Detección con YOLO
    stage1_start = time.perf_counter()
    detections = detector.detect(img_proc, tile_size=640)
    _sync_device()
    print(f"Tiempo detección YOLO: {time.perf_counter() - stage1_start:.2f} s")

    # Las coordenadas devueltas por YOLO corresponden a img_proc.
    # Convertirlas a la resolución original antes de mostrar o guardar cajas
    # sobre la imagen 4K.
    detections_original = [
        {
            "box": [v / scale for v in r["box"]],
            "score": r["score"],
        }
        for r in detections
    ]
    
    print(f"Códigos detectados: {len(detections_original)}")
    for i, r in enumerate(detections_original):
        print(f"  [{i}] box={[round(v) for v in r['box']]} yolo={r['score']:.3f}")

    # 2. Decodificación y OCR
    print("\n--- Iniciando Decodificación/OCR ---")
    decoded = []
    n_ok = 0
    
    stage2_start = time.perf_counter()
    for i, r in enumerate(detections):
        #Dividimos por 'scale' para volver a las coordenadas originales
        x1 = r["box"][0] / scale
        y1 = r["box"][1] / scale
        x2 = r["box"][2] / scale
        y2 = r["box"][3] / scale
        
        # Margen del 10%
        margin_x = int((x2 - x1) * 0.0)
        margin_y = int((y2 - y1) * 0.0)
        
        exp_x1 = max(0, int(x1) - margin_x)
        exp_y1 = max(0, int(y1) - margin_y)
        exp_x2 = min(w_orig, int(x2) + margin_x)
        exp_y2 = min(h_orig, int(y2) + margin_y)

        crop_hires = img[exp_y1:exp_y2, exp_x1:exp_x2]
        cv2.imwrite(f"{OUTPUT_DIR}/crop_{i}.jpg", crop_hires)
        decoded_text, decoder_name, ocr_skipped = _decode_crop_race(
            crop_hires, decoder, ocr_fallback, i
        )

        if decoded_text:
            n_ok += 1
            print(f"✓ Código decodificado con {decoder_name}: {decoded_text!r}")
        else:
            if ocr_skipped:
                print("✗ El decodificador no logró leer el código verde.")
            else:
                print("✗ El decodificador y YOLO-OCR no lograron leer el código.")

        decoded.append({
            "source_box":  [x1, y1, x2, y2],
            "decoded": decoded_text,
        })

    _sync_device()
    print(f"Tiempo decodificación/OCR: {time.perf_counter() - stage2_start:.2f} s")
    print(f"Decodificados exitosamente: {n_ok}/{len(decoded)}")
    
    elapsed = time.perf_counter() - pipeline_start
    print(f"Tiempo TOTAL del frame: {elapsed:.2f} s\n")
    
    return detections_original, decoded, n_ok

# ===========================================================================
# MAIN: Imagen
# ===========================================================================
def main_image():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(IMAGE_PATH)

    h, w = img.shape[:2]
    print(f"Imagen: {IMAGE_PATH} ({w}×{h})")

    # 1. Cargar Modelos
    print(f"\n[1/3] Cargando Detector YOLO: {YOLO_WEIGHTS}")
    detector = YOLOv8BarcodeDetector(
        model_path=YOLO_WEIGHTS,
        conf_thresh=0.25,
        iou_thresh=0.30,
        device=DEVICE,
    )

    print("\n[2/3] Cargando decodificador ZXing")
    decoder = BarcodeDecoder(min_crop_size=200, upscale_factor=3.0, quiet_zone_ratio=0.30)

    print("\n[3/3] Cargando OCR (Fallback)")
    ocr_fallback = SwinCharOCRDecoder(char_yolo_path=CHAR_YOLO_WEIGHTS, use_gpu=False)

    # 2. Warmup (Calentamiento)
    print("\nCalentando modelos (ignorar el tiempo de este paso)...")
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    detector.detect(dummy_img, tile_size=640) 
    print("Modelos listos.")

    # 3. Ejecutar Pipeline
    print("\n" + "=" * 60)
    print("EJECUTANDO PIPELINE")
    print("=" * 60)
    
    detections, decoded, n_ok = process_and_decode(img, detector, decoder, ocr_fallback)

    # 4. Guardar imágenes finales
    final_img = draw_decoded(img, decoded)
    cv2.imwrite(str(output_dir / "resultado_final.png"), final_img)
    print(f"Imagen con resultados guardada en: {output_dir / 'resultado_final.png'}")

# ===========================================================================
# MAIN: Video
# ===========================================================================
def main_video():
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {VIDEO_PATH}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0

    output_path = Path(VIDEO_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frame_count = 0

    # Cargar modelos
    print("Cargando modelos para procesar el video...")
    detector = YOLOv8BarcodeDetector(YOLO_WEIGHTS, device=DEVICE)
    decoder = BarcodeDecoder(min_crop_size=200, upscale_factor=3.0, quiet_zone_ratio=0.30)
    ocr_fallback = SwinCharOCRDecoder(char_yolo_path=CHAR_YOLO_WEIGHTS, use_gpu=False)

    window_name = "Pipeline en vivo (q para salir)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 900)

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_number = frame_count + 1
            print(f"\n--- Fotograma {frame_number} ---")
            
            _, decoded, _ = process_and_decode(frame, detector, decoder, ocr_fallback)
            composite = draw_decoded(frame, decoded)

            # Ajustar tamaño para el visor
            frame_h, frame_w = composite.shape[:2]
            scale = min(900 / frame_w, 900 / frame_h)
            display_frame = cv2.resize(composite, (int(frame_w * scale), int(frame_h * scale)))
            
            # Guardar en video
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, 
                    (display_frame.shape[1], display_frame.shape[0])
                )
            writer.write(display_frame)
            frame_count += 1

            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        capture.release()
        if writer: writer.release()
        cv2.destroyAllWindows()

    print(f"\nVideo guardado en: {output_path}")

def main():
    if PROCESS_VIDEO:
        main_video()
    else:
        main_image()

if __name__ == "__main__":
    main()