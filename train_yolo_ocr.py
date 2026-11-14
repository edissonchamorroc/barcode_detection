# train.py
import os
import yaml
from ultralytics import YOLO

# ============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# ============================================================================
BASE_DATASET_PATH = 'dataset_sintetico'
PROJECT_SAVE_PATH = 'runs/detect/char_detector'

# ============================================================================
# 2. GENERACIÓN DEL ARCHIVO DE CONFIGURACIÓN dataset.yaml
# ============================================================================
def create_dataset_yaml():
    class_names = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]
    
    data_config = {
        'path': os.path.abspath(BASE_DATASET_PATH),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = os.path.join(BASE_DATASET_PATH, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path

# ============================================================================
# 3. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ============================================================================
def main():
    print("Preparando archivo de configuración...")
    # NOTA: Omitimos split_train_val() porque el script de fusión ya organizó 
    # los datos (sintéticos + reales en train, reales en val).
    yaml_path = create_dataset_yaml()
    
    print("\n🚀 Cargando modelo base YOLOv8n...")
    model = YOLO('yolov8n.pt')
    
    print("\n🏋️ Iniciando entrenamiento (Dataset Combinado)...")
    results = model.train(
        data=yaml_path,
        epochs=100,
        
        # CRÍTICO: Subimos a 640 para captar las letras pequeñas a 1 metro de distancia
        imgsz=640, 
        
        # Si tienes error de memoria (Out of Memory) en tu GPU, baja esto a 8 o 4
        batch=16,          
        
        name='char_detector_combinado_v1',
        project=PROJECT_SAVE_PATH,
        patience=20,
        
        # --- MAGIA PARA TEXTO ---
        mosaic=0.0,        # Desactivado: no cortar letras
        mixup=0.0,         
        copy_paste=0.0,    
        degrees=2.0,       # Rotaciones muy leves
        translate=0.05,    
        scale=0.2,         
        fliplr=0.0,        # PROHIBIDO voltear
        flipud=0.0         
    )
    
    print("\n✅ Entrenamiento completado.")
    print(f"Mejores pesos guardados en: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()