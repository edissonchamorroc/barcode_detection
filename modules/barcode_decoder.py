# barcode_decoder.py
"""
decodificación algorítmica.

Recibe las bboxes refinadas por YOLOv5s-CBAM y extrae el contenido
alfanumérico de los códigos de barras 1D/2D.

Estrategias de robustez (basadas en limitaciones conocidas de ZBar):
    1. Conversión a escala de grises.
    2. Padding alrededor del crop → decodificador necesita "quiet zones" limpias.
    3. Upscaling si el crop es pequeño → ZBar pierde precisión < 64px.
    4. Múltiples preprocesamientos: Otsu, adaptive threshold, CLAHE, sharpen.
    5. Rotaciones: 0°, 90°, 180°, 270°, y ángulos arbitrarios cada N grados.
    6. Filtro por formato de código (evita falsos positivos de otros patrones).


"""

import logging
from pathlib import Path  # Manejo moderno de rutas de archivos.

# cv2 = OpenCV. Librería especializada en procesar imágenes (leer, recortar,
# rotar, cambiar color, filtrar, etc.). Piensa en ella como "Photoshop
# programable".
import cv2

# numpy = librería para manejar listas gigantes de números de forma eficiente.
# Una imagen, al final, no es más que una gran tabla de números (cada píxel
# es un número que representa un color). numpy es la calculadora que hace
# cuentas con esa tabla rapidísimo.
import numpy as np

# zxingcpp = motor que "lee" los códigos de barras. Le das una imagen y te
# devuelve el texto que contiene (por ejemplo "5901234123457").
import zxingcpp

# `typing` sirve para anotar QUÉ TIPO de datos recibe o devuelve una función.
# No cambia cómo funciona el código; es como poner etiquetas para que quien
# lea sepa qué esperar (List = lista, Dict = diccionario, Optional = "puede
# venir o no", Tuple = grupo fijo de valores).
from typing import List, Dict, Optional, Tuple

OUTPUT_DIR = "test_cascade_out"  # Carpeta donde se guardarán las imágenes.
LOGGER = logging.getLogger(__name__)
# ----------------------------------------------------------------------------
# CLASE PRINCIPAL
# En Python una "clase" es una plantilla que agrupa funciones (métodos) y
# datos (atributos) relacionados. Aquí agrupamos todo lo necesario para
# "leer un código de barras".
# ----------------------------------------------------------------------------


class BarcodeDecoder:

    # ------------------------------------------------------------------
    # __init__ = "constructor". Se ejecuta automáticamente al crear un
    # objeto de esta clase. Sirve para guardar la configuración inicial.
    #
    # Los parámetros tienen valores por defecto (por eso aparecen con `=`).
    # Si quien usa la clase no los pasa, se usan esos valores.
    # ------------------------------------------------------------------
    def __init__(
        self,
        min_crop_size: int = 64,  # Tamaño mínimo (px) antes de agrandar
        upscale_factor: float = 2.0,  # Cuánto agrandar si es pequeño
        quiet_zone_ratio: float = 0.30,  # Cuánto margen blanco añadir (30%)
        min_quality: float = 15.0,  # Calidad mínima para aceptar lectura
    ):

        self.min_crop_size = min_crop_size
        self.upscale_factor = upscale_factor
        self.quiet_zone_ratio = quiet_zone_ratio

        # Comprensión de conjuntos: convertir cada formato a mayúsculas y
        # meterlo en un `set` (conjunto) para búsquedas rápidas.

        self.min_quality = min_quality

    # ==================================================================
    # UTILIDADES DE IMAGEN
    # Funciones pequeñas que se usarán en varios sitios.
    # ==================================================================


    def detectar_color_codigo(self, img: np.ndarray) -> str:
        """
        Determina si el ROI del código de barras es predominantemente azul, verde u otro.
        """
        # Convertir la imagen de BGR (OpenCV) a HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Definir los rangos de color en HSV.
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])

        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        total_pixeles = img.shape[0] * img.shape[1]
        pixeles_azules = cv2.countNonZero(mask_blue)
        pixeles_verdes = cv2.countNonZero(mask_green)
        porcentaje_azul = (pixeles_azules / total_pixeles) * 100
        porcentaje_verde = (pixeles_verdes / total_pixeles) * 100

        umbral = 15.0 
        if max(porcentaje_azul, porcentaje_verde) <= umbral:
            return "blanco"
        return "Azul" if porcentaje_azul > porcentaje_verde else "Verde"

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        # `img.shape` nos dice el tamaño de la imagen.
        # Si tiene 2 dimensiones → ya es gris, la devolvemos tal cual.
        if len(img.shape) == 2:
            return img

        # Si tiene 4 canales (BGRA: azul, verde, rojo, transparencia),
        # primero quitamos el canal de transparencia.
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Convertimos de BGR (color, típico de OpenCV) a gris.
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _add_quiet_zone(self, img: np.ndarray) -> np.ndarray:
        """
        Añade padding blanco alrededor. ZBar exige quiet zones limpias.

        Analogía: un lector humano necesita "espacio en blanco" alrededor
        de un texto para leerlo cómodamente. Un lector de códigos también
        necesita ese margen (llamado "zona de silencio") o se confunde.
        """
        # `h` = alto (height), `w` = ancho (width) de la imagen en píxeles.
        h, w = img.shape

        # Cuántos píxeles de margen añadir arriba/abajo y a los lados.
        # Es un porcentaje del tamaño (por defecto 30%). `max(1, ...)` evita
        # que el resultado sea 0 (lo que significaría "sin margen").
        pad_y = max(1, int(h * self.quiet_zone_ratio))
        pad_x = max(1, int(w * self.quiet_zone_ratio))

        # `copyMakeBorder` = "haz una copia con un borde".
        # BORDER_CONSTANT con value=255 → rellena con blanco puro.
        return cv2.copyMakeBorder(
            img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    # ==================================================================
    # PREPROCESAMIENTOS
    # Aquí generamos varias "versiones" de la misma imagen, cada una con
    # un filtro distinto. La idea: si una versión no funciona, quizá otra
    # sí. Es como probar varias recetas hasta que una sale bien.
    # ==================================================================

    def procesar_codigo_barras_azul(
        self, imagen_roi: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Recibe el recorte de la imagen y devuelve una versión con fondo blanco 
        y barras negras perfectamente definidas.
        """
        #  Convertir a espacio de color HSV
        hsv = cv2.cvtColor(imagen_roi, cv2.COLOR_BGR2HSV)

        #  Definir el rango del color AZUL 
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        #  Crear la máscara: El fondo azul será blanco, lo demás negro
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Encontrar el contorno más grande (la etiqueta azul)
        contornos, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            return None
        
        # Asumimos que el contorno más grande es la etiqueta
        c = max(contornos, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # (Esto elimina el fondo de cartón ruidoso)
        roi = imagen_roi[y:y+h, x:x+w]
        
        # Convertimos el recorte a HSV de nuevo
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Extraemos SOLO el canal V (Value / Brillo)
        # En este canal, el azul se ve claro y el texto negro se ve oscuro.
        canal_v = hsv_roi[:, :, 2]

        # Aplicar Umbralización de Otsu
        # Invertimos el resultado para que las letras/barras queden NEGRAS y el fondo BLANCO
        _, binaria_inv = cv2.threshold(canal_v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Invertimos el resultado final para que las barras/texto sean NEGROS y el fondo BLANCO
        resultado_final = cv2.bitwise_not(binaria_inv)
        
        # Limpieza morfológica para hacer las barras más nítidas
        # Si las barras se ven rotas, puedes usar una operación de cierre.
        kernel = np.ones((2,2), np.uint8)
        binaria_limpia = cv2.morphologyEx(resultado_final, cv2.MORPH_CLOSE, kernel)

        return binaria_limpia

    def procesar_codigo_barras_verde(
        self, imagen_roi: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Aísla la etiqueta verde y convierte sus barras y texto a negro sobre blanco.
        """
        hsv = cv2.cvtColor(imagen_roi, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        contornos, _ = cv2.findContours(
            mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contornos:
            return None

        x, y, w, h = cv2.boundingRect(max(contornos, key=cv2.contourArea))
        roi = imagen_roi[y:y + h, x:x + w]
        canal_v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2]

        _, binaria_inv = cv2.threshold(
            canal_v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        resultado = cv2.bitwise_not(binaria_inv)
        kernel = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(resultado, cv2.MORPH_CLOSE, kernel)

    def _preprocess_variants(self, img: np.ndarray, i: int) -> Dict[str, np.ndarray]:
        """
        Devuelve variantes de mejora.
        """
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        color = self.detectar_color_codigo(img)
        variants = {"original": img}
        if color == "Azul":
            processed = self.procesar_codigo_barras_azul(img)
            if processed is not None:
                variants["azul"] = processed
        elif color == "Verde":
            processed = self.procesar_codigo_barras_verde(img)
            if processed is not None:
                variants["verde"] = processed

        return variants

    # ==================================================================
    # DECODIFICACIÓN PRINCIPAL
    # ==================================================================
    def decode(
        self,
        image: np.ndarray,
        i: int = 0,
    ) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
        """
        Decodifica un código de barras en la imagen o en un crop específico.

        Args:
            image: imagen HWC (BGR, típicamente de cv2) o HW (gray).

        Returns:
            Tupla con las lecturas decodificadas, ordenadas por calidad descendente,
            y las variantes de imagen generadas. La lista de lecturas queda vacía
            si no se pudo decodificar.
        """
        # ---- Generar variantes preprocesadas ----
        variants = self._preprocess_variants(image, i)

        # ----  Probar primero cada variante sin modificar ----
        all_hits: List[Dict] = []  # lecturas válidas acumuladas

        for name, img in variants.items():
            try:
                results = zxingcpp.read_barcodes(img)
            except Exception:
                LOGGER.exception(
                    "ZXing falló con la variante %s (forma %s, tipo %s)",
                    name,
                    img.shape,
                    img.dtype,
                )
                continue

            for result in results:
                if not result.text:
                    continue
                all_hits.append(
                    {
                        "data": result.text,
                        "type": result.format.name,
                        "quality": 100.0,
                        "rect": {
                            "left": 0,
                            "top": 0,
                            "width": 0,
                            "height": 0,
                        },
                        "polygon": [],
                    }
                )
                if all_hits:                
                    break

            if all_hits:
                break

        # ---- 5. Deduplicar y ordenar ----
        seen = set()
        unique: List[Dict] = []
        for h in all_hits:
            key = (h["data"], h["type"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(h)

        # Ordenamos por calidad, de mayor a menor.
        # `reverse=True` = descendente.
        unique.sort(key=lambda x: x["quality"], reverse=True)

        return unique , variants
