# modules/yolo_ocr.py
import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
from pathlib import Path

OUTPUT_DIR = "test_cascade_out"

class SwinCharOCRDecoder:
    def __init__(self, swin_proposer, char_yolo_path, use_gpu=True):
        self.swin = swin_proposer
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

    def process_barcode_region(self, image, swin_bbox, swin_conf, swin_class, bottom_ratio=0.40):
        x1, y1, x2, y2 = map(int, swin_bbox)
        output_dir = Path(OUTPUT_DIR)
        h, w = image.shape[:2]
        
        # Margen horizontal extra para no cortar la primera/última letra
        margin_x = int((x2 - x1) * 0.05)
        x1, x2 = max(0, x1 - margin_x), min(w, x2 + margin_x)
        y1, y2 = max(0, y1), min(h, y2)
        
        box_h = y2 - y1
        new_y1 = y1 + int(box_h * (1.0 - bottom_ratio))
        
        roi = image[new_y1:y2, x1:x2]
        #log 
        #cv2.imwrite(str(output_dir / f"roi_{x1}_{new_y1}_{x2}_{y2}.png"), roi)
        
        if roi.size == 0:
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        self._debug_region_counter += 1
        region_id = self._debug_region_counter

        # 1. YOLO detecta caracteres (bajamos conf a 0.25 y subimos imgsz a 1024)
        char_results = self.char_yolo(roi, conf=0.25, imgsz=1024, verbose=False)[0]
        
        char_boxes = []
        for box in char_results.boxes:
            cx1, cy1, cx2, cy2 = map(int, box.xyxy[0].tolist())
            char_boxes.append((cx1, cy1, cx2, cy2, float(box.conf[0])))

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
            char_crop = roi[ly1:ly2, lx1:lx2]
            
            if char_crop.size == 0:
                continue
            
            # PaddleOCR necesita que la imagen tenga un tamaño mínimo para leer bien.
            # Si el recorte de la letra es muy pequeño, lo agrandamos.
            ch, cw = char_crop.shape[:2]
            if ch < 32 or cw < 32:
                char_crop = cv2.resize(char_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            #log
            char_crop_gray = cv2.cvtColor(char_crop, cv2.COLOR_BGR2GRAY)
            char_crop_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(char_crop_gray)
            char_crop_otsu = cv2.threshold(char_crop_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            char_crop_path = output_dir / (
                f"char_yolo{yolo_conf:.2f}_region_{region_id:04d}_idx{idx}.png"
            )
            if not cv2.imwrite(str(char_crop_path), char_crop_otsu):
                raise OSError(f"No se pudo guardar el recorte: {char_crop_path}")
            
            # 2. Inferencia con PaddleOCR v3.x
            # El método predict devuelve una lista de diccionarios con los resultados
            char_crop_otsu_bgr = cv2.cvtColor(char_crop_otsu, cv2.COLOR_GRAY2BGR)
            rec_result = self.ocr_reader.predict(char_crop_otsu_bgr)
            
            if rec_result and len(rec_result) > 0:
                # En la versión 3.x, los textos y confianzas vienen en listas
                res = rec_result[0]
                if 'rec_texts' in res and len(res['rec_texts']) > 0:
                    text = res['rec_texts'][0]
                    ocr_conf = res['rec_scores'][0]
                    
                    if text.strip() and ocr_conf > 0.1:
                        decoded_chars.append(text.strip())

        full_text = "".join(decoded_chars)
        
        return {
            "texto": full_text,
            "bbox_global": [x1, y1, x2, y2],
            "confianza_swin": swin_conf,
            "clase_swin": swin_class,
            "num_caracteres": len(decoded_chars)
        }

    def forward(self, image_tensor, original_image):
        # ... (Este método se mantiene igual)
        swin_boxes, swin_scores, swin_labels = self.swin(image_tensor)
        resultados = []
        boxes = swin_boxes[0].cpu().numpy()
        scores = swin_scores[0].cpu().numpy()
        labels = swin_labels[0].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if label == 0 or score < 0.1:
                continue
                
            res = self.process_barcode_region(original_image, box, score, label)
            if res and len(res["texto"]) > 0:
                resultados.append(res)

        return resultados