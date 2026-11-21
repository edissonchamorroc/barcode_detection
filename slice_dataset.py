# slice_dataset.py
import cv2
from pathlib import Path

# --- CONFIGURACIÓN ---
INPUT_IMAGES_DIR = Path("dataset/images") # Tus fotos 4K
INPUT_LABELS_DIR = Path("dataset/labels") # Tus .txt
OUTPUT_DIR = Path("dataset_1024")         # Carpeta nueva resultante
TILE_SIZE = 1024
OVERLAP = 0.2 # 20% de superposición

def slice_dataset():
    out_img_dir = OUTPUT_DIR / "images"
    out_lbl_dir = OUTPUT_DIR / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    step = int(TILE_SIZE * (1 - OVERLAP))
    exts = {".jpg", ".jpeg", ".png"}
    
    count_imgs = 0
    count_tiles = 0
    
    print("Iniciando el corte de imágenes 4K... (esto puede tardar un poco)")
    
    for img_path in INPUT_IMAGES_DIR.iterdir():
        if img_path.suffix.lower() not in exts:
            continue
            
        count_imgs += 1
        lbl_path = INPUT_LABELS_DIR / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        h_img, w_img = img.shape[:2]

        # Leer etiquetas originales (formato YOLO: cls xc yc w h)
        with open(lbl_path, 'r') as f:
            lines = f.read().strip().splitlines()
            boxes = [list(map(float, ln.split())) for ln in lines if ln.strip()]

        # Deslizar la ventana (Tiling)
        for y in range(0, h_img, step):
            for x in range(0, w_img, step):
                # Calcular límites del parche
                x2 = min(x + TILE_SIZE, w_img)
                y2 = min(y + TILE_SIZE, h_img)
                x1 = max(0, x2 - TILE_SIZE)
                y1 = max(0, y2 - TILE_SIZE)

                # Recortar imagen
                crop = img[y1:y2, x1:x2]
                if crop.shape[0] == 0 or crop.shape[1] == 0:
                    continue

                # Buscar qué cajas caen dentro de este parche
                crop_boxes = []
                for box in boxes:
                    cls, xc, yc, w, h = box
                    
                    # Convertir YOLO normales a píxeles absolutos en la imagen 4K
                    abs_xc, abs_yc = xc * w_img, yc * h_img
                    abs_w, abs_h = w * w_img, h * h_img

                    # Verificar si el centro del barcode está dentro del parche actual
                    if x1 <= abs_xc < x2 and y1 <= abs_yc < y2:
                        # Recalcular YOLO normalizado relativo al PARCHE
                        new_xc = (abs_xc - x1) / TILE_SIZE
                        new_yc = (abs_yc - y1) / TILE_SIZE
                        new_w = abs_w / TILE_SIZE
                        new_h = abs_h / TILE_SIZE
                        crop_boxes.append(f"{int(cls)} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}")

                # Solo guardar parches que contengan al menos un código
                if crop_boxes:
                    name = f"{img_path.stem}_x{x}_y{y}"
                    cv2.imwrite(str(out_img_dir / f"{name}.jpg"), crop)
                    with open(out_lbl_dir / f"{name}.txt", "w") as f:
                        f.write("\n".join(crop_boxes))
                    count_tiles += 1

    print(f"\n✅ ¡Corte terminado!")
    print(f"Se procesaron {count_imgs} imágenes originales.")
    print(f"Se generaron {count_tiles} parches de {TILE_SIZE}x{TILE_SIZE} en la carpeta '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    slice_dataset()