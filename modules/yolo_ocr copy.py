# modules/yolo_ocr.py
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from pathlib import Path  # Manejo moderno de rutas de archivos.
OUTPUT_DIR = "test_cascade_out"  # Carpeta donde se guardarán las imágenes.

class SwinCharOCRDecoder:
    def __init__(self, swin_proposer, char_yolo_path, lang='en', use_gpu=True):
        self.swin = swin_proposer
        self.char_yolo = YOLO(char_yolo_path)
        
        # Inicializamos EasyOCR
        self.ocr_reader = easyocr.Reader(
            [lang], 
            gpu=use_gpu,
            verbose=False
        )

    def _sort_boxes_left_to_right(self, boxes):
        """Ordena las cajas de caracteres de izquierda a derecha basándose en el centro X."""
        return sorted(boxes, key=lambda b: (b[0] + b[2]) / 2.0)

    def process_barcode_region(self, image, swin_bbox, swin_conf, swin_class):
        x1, y1, x2, y2 = map(int, swin_bbox)
        output_dir = Path(OUTPUT_DIR)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        # Calculamos la altura total de la caja propuesta por Swin
        box_h = y2 - y1
        # Calculamos el nuevo Y1 para que empiece desde el porcentaje deseado hacia abajo
        # Ejemplo: si box_h=100 y bottom_ratio=0.35, new_y1 estará 35 píxeles más abajo de y1.
        new_y1 = y1 + int(box_h * (1.0 - 0.35))
        
        # Recortamos solo la franja inferior
        
        roi = image[new_y1:y2, x1:x2]
        cv2.imwrite(str(output_dir / f"roi_{x1}_{new_y1}_{x2}_{y2}.png"), roi)
        if roi.size == 0:
            return None

        # 2. Detección carácter por carácter con YOLO
        char_results = self.char_yolo(roi, conf=0.25, imgsz=1024, verbose=False)[0]
        
        char_boxes = []
        for box in char_results.boxes:
            cx1, cy1, cx2, cy2 = map(int, box.xyxy[0].tolist())
            char_boxes.append((cx1, cy1, cx2, cy2, float(box.conf[0])))

        if not char_boxes:
            return None

        sorted_boxes = self._sort_boxes_left_to_right(char_boxes)

        # 4. Reconocimiento con EasyOCR
        decoded_chars = []
        for idx, cb in enumerate(sorted_boxes):
            lx1, ly1, lx2, ly2, yolo_conf = cb
            char_crop = roi[ly1:ly2, lx1:lx2]
            
            if char_crop.size == 0:
                continue
            
            cv2.imwrite(str(output_dir / f"char_{idx}_yolo{yolo_conf:.2f}.png"), char_crop)
            # EasyOCR devuelve: [(bbox, texto, confianza)]
            rec_result = self.ocr_reader.readtext(char_crop)
            
            if rec_result:
                _, text, ocr_conf = rec_result[0]
                if text.strip() and ocr_conf > 0.3:
                    decoded_chars.append(text.strip())

        # 5. Retornar el texto final concatenado
        full_text = "".join(decoded_chars)
        
        return {
            "texto": full_text,
            "bbox_global": [x1, y1, x2, y2],
            "confianza_swin": swin_conf,
            "clase_swin": swin_class,
            "num_caracteres": len(decoded_chars)
        }

    def forward(self, image_tensor, original_image):
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