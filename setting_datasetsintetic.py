"""
fusion_auto.py
Fusiona Roboflow -> dataset_sintetico con mapeo automático de clases por nombre.
"""
import os
import sys
import shutil
import random
import unicodedata
import yaml

# ==================== CONFIGURACIÓN ====================
SYNTHETIC_DIR = "dataset_sintetico"
ROBOFLOW_DIR  = "C:/Users/marce/Downloads/barcode-ocr.v4i.yolov8"  # <-- CAMBIA
MAX_SYNTHETIC_PER_REAL = 3.0
CONFIRMAR_SI_HAY_SIN_MATCH = True   # pide confirmación si hay clases sin mapear
# =======================================================


# ---------- Utilidades ----------
def normalizar(n):
    """Normaliza el nombre para comparar: minúsculas, sin tildes, sin espacios extra."""
    s = str(n).strip().lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return s


def cargar_names(yaml_path):
    """Devuelve la lista de nombres de clase de un data.yaml."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"No existe: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    names = data.get('names')
    if names is None:
        raise ValueError(f"{yaml_path} no tiene la clave 'names'")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    return [str(n) for n in names]


def construir_mapeo(names_src, names_dst):
    """Empareja por nombre. Devuelve (mapeo, sin_match, duplicados)."""
    idx_dst = {}
    for i, n in enumerate(names_dst):
        idx_dst.setdefault(normalizar(n), []).append(i)

    mapeo, sin_match, duplicados = {}, [], []
    for i, n in enumerate(names_src):
        cands = idx_dst.get(normalizar(n), [])
        if not cands:
            sin_match.append((i, n))
        elif len(cands) == 1:
            mapeo[i] = cands[0]
        else:
            duplicados.append((i, n, cands))
            mapeo[i] = cands[0]   # usa el primero
    return mapeo, sin_match, duplicados


def mostrar_mapeo(names_src, names_dst, mapeo, sin_match, duplicados):
    print("\n" + "=" * 72)
    print("  MAPEO DE CLASES (auto, por nombre)")
    print("=" * 72)
    for i, n in enumerate(names_src):
        if i in mapeo:
            j = mapeo[i]
            print(f"  {i:>2}  '{n}'  ->  {j:>2}  '{names_dst[j]}'   OK")
        else:
            print(f"  {i:>2}  '{n}'  ->  SIN MATCH (se descartará)")

    if duplicados:
        print("\n  AVISO: nombres duplicados en el destino:")
        for i, n, cands in duplicados:
            print(f"     '{n}' (origen {i}) aparece en IDs {cands}. Usando {cands[0]}.")

    usados = set(mapeo.values())
    vacios = [i for i in range(len(names_dst)) if i not in usados]
    if vacios:
        print("\n  INFO: clases del destino que NO reciben datos de Roboflow:")
        for i in vacios:
            print(f"     {i}: '{names_dst[i]}'")

    print("=" * 72)
    print(f"  {len(mapeo)} clases mapeadas, {len(sin_match)} sin match")
    print("=" * 72)


# ---------- Copia y conversión ----------
def copiar_y_mapear(split_origen, split_destino, mapeo, prefix, nombres_archivos=None):
    src_img = os.path.join(ROBOFLOW_DIR, split_origen, "images")
    src_lbl = os.path.join(ROBOFLOW_DIR, split_origen, "labels")
    dst_img = os.path.join(SYNTHETIC_DIR, "images", split_destino)
    dst_lbl = os.path.join(SYNTHETIC_DIR, "labels", split_destino)

    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    if not os.path.exists(src_img):
        print(f"  AVISO: no existe {src_img}")
        return 0

    if nombres_archivos is None:
        nombres_archivos = [f for f in os.listdir(src_img)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    count = 0
    for img_name in nombres_archivos:
        shutil.copy(os.path.join(src_img, img_name),
                    os.path.join(dst_img, prefix + img_name))

        lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
        src_lbl_path = os.path.join(src_lbl, lbl_name)
        dst_lbl_path = os.path.join(dst_lbl, prefix + lbl_name)

        if os.path.exists(src_lbl_path):
            with open(src_lbl_path, 'r', encoding='utf-8') as f_in, \
                 open(dst_lbl_path, 'w', encoding='utf-8') as f_out:
                for linea in f_in:
                    partes = linea.strip().split()
                    if partes:
                        c = int(partes[0])
                        if c in mapeo:
                            f_out.write(f"{mapeo[c]} " + " ".join(partes[1:]) + "\n")
        count += 1

    print(f"  {count} imágenes '{split_origen}' -> '{split_destino}' (prefix '{prefix}')")
    return count


def limpiar_sinteticas_y_balancear(num_reales):
    if num_reales == 0:
        print("  AVISO: sin imágenes reales, no se balancea.")
        return
    limite = int(num_reales * MAX_SYNTHETIC_PER_REAL)
    print(f"\n Balanceando: {num_reales} reales -> máximo {limite} sintéticas")

    train_img = os.path.join(SYNTHETIC_DIR, "images", "train")
    train_lbl = os.path.join(SYNTHETIC_DIR, "labels", "train")
    if not os.path.isdir(train_img):
        return

    sinteticas = [f for f in os.listdir(train_img) if f.startswith("img_")]
    if len(sinteticas) > limite:
        random.shuffle(sinteticas)
        a_borrar = sinteticas[limite:]
        print(f"  Borrando {len(a_borrar)} sintéticas")
        for img in a_borrar:
            os.remove(os.path.join(train_img, img))
            lbl = img.rsplit('.', 1)[0] + '.txt'
            lp = os.path.join(train_lbl, lbl)
            if os.path.exists(lp):
                os.remove(lp)
        print(f"  Conservadas {limite} sintéticas")
    else:
        print(f"  Ya hay {len(sinteticas)} sintéticas (<= {limite}). OK.")


def actualizar_yaml(yaml_path, names):
    """Quita la línea 'test:' y avisa si nc no cuadra. No toca el resto del formato."""
    if not os.path.exists(yaml_path):
        return
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    nc = data.get('nc')
    if nc is not None and nc != len(names):
        print(f"  AVISO: nc={nc} en yaml, pero names tiene {len(names)}. Ajústalo a mano.")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        lineas = f.readlines()
    with open(yaml_path, 'w', encoding='utf-8') as f:
        for line in lineas:
            if not line.strip().startswith('test:'):
                f.write(line)
    print(f"  {yaml_path} sin línea 'test:'")


# ---------- MAIN ----------
def main():
    print("Fusión Roboflow -> dataset_sintetico (mapeo automático)\n")

    src_yaml = os.path.join(ROBOFLOW_DIR, 'data.yaml')
    dst_yaml = os.path.join(SYNTHETIC_DIR, 'dataset.yaml')

    try:
        names_src = cargar_names(src_yaml)
        print(f"Roboflow: {len(names_src)} clases")
    except Exception as e:
        print(f"ERROR leyendo {src_yaml}: {e}")
        return 1

    try:
        names_dst = cargar_names(dst_yaml)
        print(f"Destino:  {len(names_dst)} clases")
    except Exception as e:
        print(f"ERROR leyendo {dst_yaml}: {e}")
        print("Crea tu data.yaml de destino con TODAS tus clases antes de ejecutar.")
        return 1

    mapeo, sin_match, duplicados = construir_mapeo(names_src, names_dst)
    mostrar_mapeo(names_src, names_dst, mapeo, sin_match, duplicados)

    if sin_match and CONFIRMAR_SI_HAY_SIN_MATCH:
        print(f"\n{len(sin_match)} clase(s) de Roboflow NO se usarán.")
        r = input("¿Continuar? [s/N]: ").strip().lower()
        if r not in ('s', 'si', 'sí', 'y', 'yes'):
            print("Cancelado.")
            return 0

    if not mapeo:
        print("No hay nada que mapear. Abortando.")
        return 1

    # 1. Copiar train y valid
    print("\nCopiando splits reales...")
    num_train_reales = copiar_y_mapear("train", "train", mapeo, prefix="real_train_")
    copiar_y_mapear("valid", "val", mapeo, prefix="real_val_")

    # 2. Reciclar test 50/50
    test_img_dir = os.path.join(ROBOFLOW_DIR, "test", "images")
    if os.path.exists(test_img_dir):
        print("\nReciclando test -> train/val (50/50)...")
        test_imgs = [f for f in os.listdir(test_img_dir)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(test_imgs)
        mitad = len(test_imgs) // 2
        a_train = test_imgs[:mitad]
        a_val = test_imgs[mitad:]

        num_train_reales += copiar_y_mapear("test", "train", mapeo,
                                            prefix="real_test_a_train_",
                                            nombres_archivos=a_train)
        copiar_y_mapear("test", "val", mapeo,
                        prefix="real_test_a_val_",
                        nombres_archivos=a_val)
    else:
        print("\nAVISO: no existe split 'test' en Roboflow.")

    # 3. Balancear sintéticas
    limpiar_sinteticas_y_balancear(num_train_reales)

    # 4. Actualizar yaml destino
    print()
    actualizar_yaml(dst_yaml, names_dst)

    print("\nFusión completada.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())