import os
import shutil
import random

# --- CONFIGURACIÓN ---
SYNTHETIC_DIR = "dataset_sintetico"
ROBOFLOW_DIR ="C:/Users/marce/Downloads/barcode-ocr.v1i.yolov8"  # <-- CAMBIA ESTO
# Proporción: Por cada 1 imagen real, ¿cuántas sintéticas queremos? 
# Si tienes 150 reales y pones 3.0, tomará 450 sintéticas. (Total 600)
MAX_SYNTHETIC_PER_REAL = 3.0 

MAPEO_CLASES = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
    10: 10, 11: 12, 12: 13, 13: 20, 14: 23, 15: 25, 16: 28
}

def copiar_y_mapear(split_origen, split_destino, prefix="real_"):
    # ... (Esta función se mantiene igual que en la respuesta anterior) ...
    src_img_dir = os.path.join(ROBOFLOW_DIR, split_origen, "images")
    src_lbl_dir = os.path.join(ROBOFLOW_DIR, split_origen, "labels")
    dst_img_dir = os.path.join(SYNTHETIC_DIR, "images", split_destino)
    dst_lbl_dir = os.path.join(SYNTHETIC_DIR, "labels", split_destino)
    
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    
    if not os.path.exists(src_img_dir):
        return 0 # Retornamos 0 si no hay imágenes

    count = 0
    for img_name in os.listdir(src_img_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        new_img_name = prefix + img_name
        shutil.copy(os.path.join(src_img_dir, img_name), os.path.join(dst_img_dir, new_img_name))
        
        lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
        src_lbl_path = os.path.join(src_lbl_dir, lbl_name)
        dst_lbl_path = os.path.join(dst_lbl_dir, prefix + lbl_name)
        
        if os.path.exists(src_lbl_path):
            with open(src_lbl_path, 'r') as f_in, open(dst_lbl_path, 'w') as f_out:
                for linea in f_in:
                    partes = linea.strip().split()
                    if partes:
                        clase_roboflow = int(partes[0])
                        if clase_roboflow in MAPEO_CLASES:
                            nueva_clase = MAPEO_CLASES[clase_roboflow]
                            f_out.write(f"{nueva_clase} " + " ".join(partes[1:]) + "\n")
        count += 1
    print(f"  -> Copiadas {count} imágenes reales de {split_origen} a {split_destino}")
    return count

def limpiar_sinteticas_y_balancear(num_reales):
    """
    Elimina imágenes sintéticas si hay demasiadas en comparación con las reales.
    """
    if num_reales == 0:
        print("⚠️ No hay imágenes reales. Manteniendo todas las sintéticas.")
        return

    limite_sinteticas = int(num_reales * MAX_SYNTHETIC_PER_REAL)
    print(f"\n⚖️ Balanceando: {num_reales} reales -> Límite de {limite_sinteticas} sintéticas.")

    train_img_dir = os.path.join(SYNTHETIC_DIR, "images/train")
    train_lbl_dir = os.path.join(SYNTHETIC_DIR, "labels/train")

    # Buscamos solo las sintéticas (las que empiezan con 'img_')
    sinteticas = [f for f in os.listdir(train_img_dir) if f.startswith("img_")]
    
    if len(sinteticas) > limite_sinteticas:
        # Mezclamos y seleccionamos solo las que vamos a conservar
        random.shuffle(sinteticas)
        a_borrar = sinteticas[limite_sinteticas:]
        
        print(f"  -> Borrando {len(a_borrar)} imágenes sintéticas para no sesgar el modelo...")
        for img_name in a_borrar:
            os.remove(os.path.join(train_img_dir, img_name))
            lbl_name = img_name.replace('.jpg', '.txt')
            lbl_path = os.path.join(train_lbl_dir, lbl_name)
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
        print(f"  -> Se conservaron {limite_sinteticas} imágenes sintéticas.")
    else:
        print("  -> Ya había una cantidad balanceada de imágenes sintéticas.")

import random # Asegúrate de tener esto importado arriba

def main():
    print("🚀 Iniciando fusión y balanceo...")
    
    # 1. Copiar reales de TRAIN y VAL
    num_train_reales = copiar_y_mapear("train", "train", prefix="real_train_")
    copiar_y_mapear("valid", "val", prefix="real_val_")
    
    # 2. ¡NUEVO! En lugar de copiar test a test, lo repartimos 50/50
    # entre train y val para aprovechar las imágenes.
    test_img_dir = os.path.join(ROBOFLOW_DIR, "test", "images")
    if os.path.exists(test_img_dir):
        print("\n♻️ Reciclando imágenes de TEST hacia TRAIN y VAL...")
        # Tomamos las imágenes y las mezclamos
        test_images = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(test_images)
        
        # Las dividimos a la mitad
        mitad = len(test_images) // 2
        test_para_train = test_images[:mitad]
        test_para_val = test_images[mitad:]
        
        # Función auxiliar rápida para copiar y mapear
        def copiar_lista_especifica(lista_imgs, split_destino, prefix):
            src_img_dir = os.path.join(ROBOFLOW_DIR, "test", "images")
            src_lbl_dir = os.path.join(ROBOFLOW_DIR, "test", "labels")
            dst_img_dir = os.path.join(SYNTHETIC_DIR, "images", split_destino)
            dst_lbl_dir = os.path.join(SYNTHETIC_DIR, "labels", split_destino)
            
            for img_name in lista_imgs:
                shutil.copy(os.path.join(src_img_dir, img_name), os.path.join(dst_img_dir, prefix + img_name))
                lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
                src_lbl_path = os.path.join(src_lbl_dir, lbl_name)
                dst_lbl_path = os.path.join(dst_lbl_dir, prefix + lbl_name)
                
                if os.path.exists(src_lbl_path):
                    with open(src_lbl_path, 'r') as f_in, open(dst_lbl_path, 'w') as f_out:
                        for linea in f_in:
                            partes = linea.strip().split()
                            if partes:
                                clase_roboflow = int(partes[0])
                                if clase_roboflow in MAPEO_CLASES:
                                    f_out.write(f"{MAPEO_CLASES[clase_roboflow]} " + " ".join(partes[1:]) + "\n")
            print(f"  -> {len(lista_imgs)} imágenes de test movidas a {split_destino}")

        copiar_lista_especifica(test_para_train, "train", "real_test_a_train_")
        copiar_lista_especifica(test_para_val, "val", "real_test_a_val_")
        
        # Sumamos estas nuevas imágenes reales al contador para el balanceo
        num_train_reales += len(test_para_train)
    else:
        print("\n⚠️ No se encontró carpeta 'test' en Roboflow. Saltando este paso.")
        
    # 3. Balancear el dataset sintético en base al nuevo total de reales en train
    limpiar_sinteticas_y_balancear(num_train_reales)
    
    # 4. Actualizar data.yaml (eliminamos la línea de test porque ya no hay carpeta test)
    yaml_path = os.path.join(SYNTHETIC_DIR, 'data.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            lineas = f.readlines()
        
        # Reescribimos el yaml pero quitando cualquier línea que diga "test:"
        with open(yaml_path, 'w') as f:
            for line in lineas:
                if not line.startswith('test:'):
                    f.write(line)
        print("✅ data.yaml actualizado (se eliminó la referencia a 'test').")

    print("✅ ¡Fusión y balanceo completados!")
    
if __name__ == "__main__":
    main()