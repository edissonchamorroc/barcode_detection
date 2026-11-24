# yolo_barcode.py
"""
Etapa de detección: YOLOv8s con Tiling para imágenes de alta resolución (4K).
"""

import torch
import numpy as np
import cv2
from typing import List, Dict, Optional
from torchvision.ops import nms

class YOLOv8BarcodeDetector:
    """
    Detector de códigos de barras usando YOLOv8s.
    Incluye lógica de Tiling para dividir imágenes 4K en parches de 1024x1024.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: Optional[str] = None,
    ):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        from ultralytics import YOLO
        weights = model_path if model_path else "yolov8s.pt"
        print(f"Cargando YOLO desde: {weights}")
        
        self.model = YOLO(weights)
        self.model.model.to(self.device)
        self.model.model.eval()
        print(f"Detector YOLOv8 listo en {self.device}")

    @torch.no_grad()
    def detect(self, image: np.ndarray, tile_size: int = 640, overlap: float = 0.2) -> List[Dict]:
        h_img, w_img = image.shape[:2]
        step = int(tile_size * (1 - overlap))
        
        crops = []
        offsets = []

        if h_img <= tile_size and w_img <= tile_size:
            crops.append(image)
            offsets.append((0, 0))
        else:
            for y in range(0, h_img, step):
                for x in range(0, w_img, step):
                    x2 = min(x + tile_size, w_img)
                    y2 = min(y + tile_size, h_img)
                    x1 = max(0, x2 - tile_size)
                    y1 = max(0, y2 - tile_size)

                    crop = image[y1:y2, x1:x2]
                    crops.append(crop)
                    offsets.append((x1, y1))

        # Procesar en lotes de 4 para acelerar
        all_detections = self.model.predict(
            source=crops,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            batch=4,
            imgsz=640,
            verbose=False,
        )

        all_boxes = []
        all_scores = []

        for det, (offset_x, offset_y) in zip(all_detections, offsets):
            boxes = det.boxes
            if boxes is None or len(boxes) == 0:
                continue
                
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            
            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                all_boxes.append([float(x1 + offset_x), float(y1 + offset_y), 
                                float(x2 + offset_x), float(y2 + offset_y)])
                all_scores.append(float(conf))

        if not all_boxes:
            return []

        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
        
        keep = nms(boxes_tensor, scores_tensor, self.iou_thresh)
        
        # --- FILTRO ANTI-CONTENCIÓN (Elimina cajas que están dentro de otras) ---
        final_keep = []
        for i in keep:
            keep_i = True
            for j in final_keep:
                box_i = boxes_tensor[i]
                box_j = boxes_tensor[j]
                
                # Calcular área de intersección
                xx1 = max(box_i[0], box_j[0])
                yy1 = max(box_i[1], box_j[1])
                xx2 = min(box_i[2], box_j[2])
                yy2 = min(box_i[3], box_j[3])
                
                w_inter = max(0, xx2 - xx1)
                h_inter = max(0, yy2 - yy1)
                inter_area = w_inter * h_inter
                
                area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
                min_area = min(area_i, area_j)
                
                # Si la intersección cubre más del 70% de la caja más pequeña
                if min_area > 0 and (inter_area / min_area) > 0.70:
                    keep_i = False 
                    break
            
            if keep_i:
                final_keep.append(i)

        final_results = []
        for idx in final_keep:
            final_results.append({
                "box": all_boxes[idx],
                "score": float(all_scores[idx])
            })
            
        return final_results

    def _predict_single(self, img_patch, offset_x, offset_y):
        """Infiere en un solo parche y traduce las coordenadas al mapa original."""
        detections = self.model.predict(
            source=img_patch,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )
        
        results = []
        for det in detections:
            boxes = det.boxes
            if boxes is None or len(boxes) == 0:
                continue
                
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            
            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                # Sumar el offset (x1, y1) para devolver coordenadas de la imagen 4K original
                results.append({
                    "box": [float(x1 + offset_x), float(y1 + offset_y), 
                            float(x2 + offset_x), float(y2 + offset_y)],
                    "score": float(conf),
                })
        return results