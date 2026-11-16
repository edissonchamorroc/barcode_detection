# modules/yolo_ocr.py
import time

import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
from pathlib import Path

OUTPUT_DIR = "test_cascade_out"

class SwinCharOCRDecoder:
    def __init__(self, char_yolo_path, use_gpu=True):
        self.char_yolo = YOLO(char_yolo_path)
        self.names = self.char_yolo.names
        # Inicializamos PaddleOCR en modo solo reconocimiento (sin detector de cajas)
        # Los parámetros use_doc... se ponen en False para que sea más rápido y no procese la imagen como documento
        self.ocr_reader = PaddleOCR(
            lang='en',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False 
        )
        self._debug_region_counter = 0

    def _sort_boxes_left_to_right(self, boxes):
        return sorted(boxes, key=lambda b: (b[0] + b[2]) / 2.0)
    
    def _deduplicate_char_boxes(self, boxes, iou_threshold=0.95):
        """
        Elimina cajas duplicadas basándose en un umbral de IoU alto.
        'boxes' es una lista de tuplas (x1, y1, x2, y2, conf).
        """
        if not boxes:
            return []

        # Ordenar las cajas por confianza de forma descendente
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        filtered_boxes = []

        while boxes:
            # Tomar la caja con mayor confianza
            best_box = boxes.pop(0)
            filtered_boxes.append(best_box)

            # Comparar con el resto de cajas y descartar las que se solapen demasiado
            boxes_to_remove = []
            for other_box in boxes:
                # Calcular la IoU entre la mejor caja y la caja 'other_box'
                iou = self._calculate_iou(best_box, other_box)
                if iou > iou_threshold:
                    boxes_to_remove.append(other_box)
            
            # Eliminar las cajas duplicadas de la lista principal
            for box in boxes_to_remove:
                boxes.remove(box)

        return filtered_boxes

    def _calculate_iou(self, box_a, box_b):
        """Calcula la IoU (Intersection over Union) entre dos cajas."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter_area = max(0, xb - xa) * max(0, yb - ya)
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        iou = inter_area / float(box_a_area + box_b_area - inter_area)
        return iou

    def process_barcode_region(self, image, bottom_ratio=0.32):
        total_start = time.perf_counter()
        timings = []
        region_id = self._debug_region_counter

        def record_timing(label, start):
            timings.append((label, time.perf_counter() - start))

        def print_timings():
            print(f"\n[Tiempos OCR región {region_id:04d}]")
            for label, elapsed in timings:
                print(f"  {label}: {elapsed:.3f} s")
            print(
                f"  Total process_barcode_region: "
                f"{time.perf_counter() - total_start:.3f} s"
            )

        setup_start = time.perf_counter()
        output_dir = Path(OUTPUT_DIR)
        h, w = image.shape[:2]
        record_timing("Preparación de ruta y dimensiones", setup_start)

        roi_start_time = time.perf_counter()
        roi_start = int(h * (1.0 - bottom_ratio))
        roi = image[roi_start:h, :w]
        record_timing("Extracción de ROI", roi_start_time)

        if roi.size == 0:
            print_timings()
            return None

        setup_start = time.perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._debug_region_counter += 1
        region_id = self._debug_region_counter
        record_timing("Preparación de salida e ID de región", setup_start)

        yolo_start = time.perf_counter()
        char_results = self.char_yolo(
            roi, conf=0.15, iou=0.2, imgsz=1024, verbose=False
        )[0]
        record_timing("Detección de caracteres (YOLO)", yolo_start)

        boxes_start = time.perf_counter()
        char_boxes = []
        for box in char_results.boxes:
            cx1, cy1, cx2, cy2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            class_letter = self.names[class_id]
            char_boxes.append(
                (cx1, cy1, cx2, cy2, float(box.conf[0]), class_letter)
            )
        record_timing("Conversión de detecciones a cajas/clases", boxes_start)

        # Elimina cajas duplicadas
        deduplicate_start = time.perf_counter()
        char_boxes = self._deduplicate_char_boxes(char_boxes, iou_threshold=0.95)
        record_timing("Eliminación de cajas duplicadas", deduplicate_start)

        # Organiza las cajas de izquierda a derecha
        sort_start = time.perf_counter()
        sorted_boxes = self._sort_boxes_left_to_right(char_boxes)
        record_timing("Ordenamiento de caracteres", sort_start)

        """  
        draw_start = time.perf_counter()
        roi_debug = roi.copy()
        for idx, (
            cx1,
            cy1,
            cx2,
            cy2,
            yolo_conf,
            class_letter,
        ) in enumerate(sorted_boxes):
            color = (0, 255, 0)
            cv2.rectangle(roi_debug, (cx1, cy1), (cx2, cy2), color, 2)
            label = class_letter
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            text_y = max(0, cy1 - text_h - 6)
            cv2.rectangle(
                roi_debug,
                (cx1, text_y),
                (cx1 + text_w + 4, text_y + text_h + 4),
                color,
                -1,
            )
            cv2.putText(
                roi_debug,
                label,
                (cx1 + 2, text_y + text_h + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
        record_timing("Dibujo de cajas y etiquetas", draw_start)

        roi_debug_path = output_dir / f"roi_chars_region_{region_id:04d}.png"
        save_start = time.perf_counter()
        if not cv2.imwrite(str(roi_debug_path), roi_debug):
            raise OSError(f"No se pudo guardar la imagen de depuracion: {roi_debug_path}")
        record_timing("Guardado de imagen de depuración", save_start)
        """
        decode_start = time.perf_counter()
        decoded_chars = [box[5] for box in sorted_boxes]
        full_text = "".join(decoded_chars)
        record_timing("Construcción del texto", decode_start)

        print_timings()
        return {
            "texto": full_text,
            "num_caracteres": len(decoded_chars)
        }

    