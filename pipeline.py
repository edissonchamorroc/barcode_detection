# pipeline.py

import sys
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
import torch

# --- Configuración ---
IMAGE_PATH = "dataset_prepared/yolo/images/test/Foto2_50cm.jpg"
YOLO_WEIGHTS = "runs/yolo_barcode/best.pt"
CHAR_YOLO_WEIGHTS = "runs/char_detector/best.pt"
OUTPUT_DIR = "test_cascade_out"
DEVICE = "cpu"  # Usamos CPU por ahora
PROCESS_VIDEO = False
VIDEO_PATH = "Video_prueba50cm.mp4"
VIDEO_OUTPUT_PATH = "Video_resultado.mp4"

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
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(viz, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            viz, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        
    return add_title(
        viz, f"Decodificadas ({sum(1 for r in decoded if r['decoded'])})"
    )

def add_title(img, text):
    bar = np.zeros((30, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([bar, img])

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
    MAX_DIM = 1280 
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
    detections = detector.detect(img_proc, tile_size=1280)
    _sync_device()
    print(f"Tiempo detección YOLO: {time.perf_counter() - stage1_start:.2f} s")
    
    print(f"Códigos detectados: {len(detections)}")
    for i, r in enumerate(detections):
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
        margin_x = int((x2 - x1) * 0.10)
        margin_y = int((y2 - y1) * 0.10)
        
        exp_x1 = max(0, int(x1) - margin_x)
        exp_y1 = max(0, int(y1) - margin_y)
        exp_x2 = min(w_orig, int(x2) + margin_x)
        exp_y2 = min(h_orig, int(y2) + margin_y)

        crop_hires = img[exp_y1:exp_y2, exp_x1:exp_x2]
        
        # Intento decodificar
        hits, variants = decoder.decode(crop_hires, i=i)
        decoded_text = hits[0]["data"] if hits else None

        if hits:
            n_ok += 1
            print(f"✓ Código decodificado: {decoded_text!r}")
        else:
            print(f"✗ Código de barras falló. Intentando OCR...")
            for variant_name, variant_image in variants.items():
                if variant_image.ndim == 2:
                    variant_image = cv2.cvtColor(variant_image, cv2.COLOR_GRAY2BGR)

                ocr_res = ocr_fallback.process_barcode_region(image=variant_image)
                if ocr_res and ocr_res["texto"] and ocr_res["num_caracteres"] >= 10:
                    decoded_text = ocr_res["texto"]
                    n_ok += 1
                    print(f"✓ OCR ({variant_name}): {decoded_text!r}")
                    break
            else:
                print(f"✗ Decodificador y OCR fallaron.")

        decoded.append({
            "source_box":  [x1, y1, x2, y2],
            "decoded": decoded_text,
        })

    _sync_device()
    print(f"Tiempo decodificación/OCR: {time.perf_counter() - stage2_start:.2f} s")
    print(f"Decodificados exitosamente: {n_ok}/{len(decoded)}")
    
    elapsed = time.perf_counter() - pipeline_start
    print(f"Tiempo TOTAL del frame: {elapsed:.2f} s\n")
    
    return detections, decoded, n_ok

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

    # ===================================================================
    # NUEVO: Guardar imagen con las cajas que detectó YOLO (antes de decodificar)
    # ===================================================================
    img_yolo = draw_detections(img, detections, title="YOLO: Barcodes Detectados")
    path_yolo = output_dir / "01_yolo_detecciones.png"
    cv2.imwrite(str(path_yolo), img_yolo)
    print(f"🖼️  Imagen de detecciones de YOLO guardada en: {path_yolo}")

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