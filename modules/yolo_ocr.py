# modules/yolo_ocr.py
import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
from pathlib import Path

OUTPUT_DIR = "test_cascade_out"

class SwinCharOCRDecoder:
    def __init__(self, char_yolo_path, use_gpu=True):
        self.char_yolo = YOLO(char_yolo_path)
        
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

    def process_barcode_region(self, image, bottom_ratio=0.40):
        
        output_dir = Path(OUTPUT_DIR)
        h, w = image.shape[:2]
        
        roi_start = int(h * (1.0 - bottom_ratio))
        roi = image[roi_start:h, :w]
       
        if roi.size == 0:
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        self._debug_region_counter += 1
        region_id = self._debug_region_counter

        # YOLO detecta caracteres
        char_results = self.char_yolo(roi, conf=0.15, iou=0.7, imgsz=1024, verbose=False)[0]
        
        
        char_boxes = []
        for box in char_results.boxes:
            cx1, cy1, cx2, cy2 = map(int, box.xyxy[0].tolist())
            char_boxes.append((cx1, cy1, cx2, cy2, float(box.conf[0])))
        # Elimina cajas duplicadas
        char_boxes = self._deduplicate_char_boxes(char_boxes, iou_threshold=0.95)
        # Organiza las cajas de izquierda a derecha
        sorted_boxes = self._sort_boxes_left_to_right(char_boxes)
        
        roi_debug = roi.copy()
        for idx, (cx1, cy1, cx2, cy2, yolo_conf) in enumerate(sorted_boxes):
            color = (0, 255, 0)
            cv2.rectangle(roi_debug, (cx1, cy1), (cx2, cy2), color, 2)
            label = str(idx)
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

        roi_debug_path = output_dir / f"roi_chars_region_{region_id:04d}.png"
        if not cv2.imwrite(str(roi_debug_path), roi_debug):
            raise OSError(f"No se pudo guardar la imagen de depuracion: {roi_debug_path}")
  
        if not char_boxes:
            return None
       
        decoded_chars = []
        
        for idx, cb in enumerate(sorted_boxes):
            lx1, ly1, lx2, ly2, yolo_conf = cb

            # Expandimos la caja un 15% para darle contexto a PaddleOCR
            box_w = lx2 - lx1
            box_h = ly2 - ly1
            pad_x = int(box_w * 0.15)
            pad_y = int(box_h * 0.15)
            # Aseguramos que no nos salgamos de la imagen ROI
            px1 = max(0, lx1 - pad_x)
            py1 = max(0, ly1 - pad_y)
            px2 = min(w, lx2 + pad_x)
            py2 = min(h, ly2 + pad_y)

            char_crop = roi[py1:py2, px1:px2]
            
            if char_crop.size == 0:
                continue
            # ========================================================
            # ENDEREZAR CARÁCTER (DESKEW)
            # ========================================================
            # Convertimos a escala de grises para detectar el ángulo de la letra
            gray = cv2.cvtColor(char_crop, cv2.COLOR_BGR2GRAY)
            
            # Umbralizamos para aislar la letra del fondo
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Encontramos los contornos (la forma de la letra)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Tomamos el contorno más grande (asumiendo que es la letra)
                c = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                angle = rect[-1]
                
                # Ajustar el ángulo para que quede entre -45 y 45 grados
                if angle < -45:
                    angle = 90 + angle
                elif angle > 45:
                    angle = angle - 90
                
                # Si la letra está rotada más de 5 grados, la enderezamos
                if abs(angle) > 5:
                    (h_crop, w_crop) = char_crop.shape[:2]
                    center = (w_crop // 2, h_crop // 2)
                    # Creamos la matriz de rotación
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    # Rotamos la imagen
                    char_crop = cv2.warpAffine(
                        char_crop, M, (w_crop, h_crop), 
                        flags=cv2.INTER_CUBIC, 
                        borderMode=cv2.BORDER_REPLICATE # Rellena los bordes con el color del borde
                    )
            # ========================================================
            target_h = 48
            scale = target_h / max(char_crop.shape[0], 1)
            target_w = max(1, int(char_crop.shape[1] * scale))
            
            # Redimensionar
            char_resized = cv2.resize(char_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
            #log
            cv2.imwrite(str(output_dir / f"char_resized_{region_id:04d}_{idx:02d}.png"), char_resized)    
            # Inferencia con PaddleOCR v3.x
            # El método predict devuelve una lista de diccionarios con los resultados
            rec_result = self.ocr_reader.predict(char_resized)
            
            if rec_result and len(rec_result) > 0:
                # En la versión 3.x, los textos y confianzas vienen en listas
                res = rec_result[0]
                if 'rec_texts' in res and len(res['rec_texts']) > 0:
                    text = res['rec_texts'][0]
                    ocr_conf = res['rec_scores'][0]
                    
                    if text.strip() and ocr_conf > 0.4:
                        decoded_chars.append(text.strip())

        full_text = "".join(decoded_chars)
        
        return {
            "texto": full_text,
            "num_caracteres": len(decoded_chars)
        }

    