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

    # `@staticmethod` significa "esta función no necesita acceder a `self`".
    # Es una utilidad genérica que podría existir fuera de la clase.
    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        """
        Convierte una imagen a escala de grises (blanco y negro).
        ¿Por qué? Porque leer un código de barras solo depende de las
        diferencias de intensidad (claro/oscuro), no del color.
        Además, trabajar en gris es 3 veces más rápido que en color.
        """
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


    def alinear_y_rotar_barras(self, image_to_align: np.ndarray) -> np.ndarray:
        """
        Detecta de manera robusta el ángulo de inclinación de un código de barras
        directamente desde la imagen original (color o escala de grises),
        aplica la transformada de Hough adaptativa y retorna la imagen rotada.
        """
        # 1. Convertir la imagen a escala de grises
        if len(image_to_align.shape) == 3:
            gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_to_align.copy()

        # 2. Aplicar un CLAHE de alta intensidad para resaltar bordes verticales
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        # 3. Detectar bordes de manera limpia
        edges = cv2.Canny(enhanced_gray, 50, 150, apertureSize=3)

        # 4. Transformada de Hough adaptativa
        hough_threshold = 150
        lines = None
        while hough_threshold >= 30:
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=hough_threshold)
            if lines is not None and len(lines) >= 8:
                break
            hough_threshold -= 15

        angles = []
        if lines is not None and len(lines) > 0:
            for line in lines:
                rho, theta = line[0]
                angle_deg = np.rad2deg(theta)

                # Orientar el ángulo respecto a la vertical (0 o 180 grados)
                if angle_deg > 90:
                    angle_deg -= 180

                # Filtrar: solo nos interesan componentes predominantemente verticales
                if abs(angle_deg) < 45:
                    angles.append(angle_deg)

            if len(angles) > 0:
                angulo_desviacion = np.median(angles)
            else:
                angulo_desviacion = 0.0
        else:
            angulo_desviacion = 0.0

        print(
            f"[Alineación] Desviación detectada: {angulo_desviacion:.2f} grados (usando umbral Hough: {max(hough_threshold, 30)})"
        )

        # 5. Rotar la imagen usando el centro geométrico
        h, w = gray.shape[:2]
        centro = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(centro, angulo_desviacion, 1.0)

        # Aplicar la matriz afín con fondo blanco (255) para preservar el área externa limpia
        img_rotada = cv2.warpAffine(
            image_to_align,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(image_to_align.shape) == 3 else 255,
        )

        return img_rotada

   
    @staticmethod
    def _rotate(img: np.ndarray, angle: int) -> np.ndarray:
        """
        Rota sin recortar (expande el canvas para no perder esquinas).

        ¿Por qué? Un código puede estar fotografiado torcido. El lector
        a veces solo lo reconoce si está perfectamente horizontal o
        vertical. Rotando probamos varias orientaciones.
        """
        # Si el ángulo es 0, no hay nada que hacer.
        if angle == 0:
            return img

        # Centro de la imagen (alrededor del cual giraremos).
        h, w = img.shape[:2]
        center = (w / 2, h / 2)

        # `getRotationMatrix2D` construye la "receta matemática" para rotar.
        # El 1.0 es la escala (no ampliar ni reducir).
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Cuando rotas un rectángulo, las esquinas "se salen".
        # Calculamos el nuevo tamaño para que quepan todas.
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Ajustamos la matriz para centrar la imagen en el nuevo lienzo.
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # `warpAffine` aplica la rotación. BORDER_CONSTANT + blanco = los
        # huecos que aparecen por la rotación se rellenan de blanco (no
        # de negro, que confundiría al lector).
        return cv2.warpAffine(
            img,
            M,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    # ==================================================================
    # PREPROCESAMIENTOS
    # Aquí generamos varias "versiones" de la misma imagen, cada una con
    # un filtro distinto. La idea: si una versión no funciona, quizá otra
    # sí. Es como probar varias recetas hasta que una sale bien.
    # ==================================================================
    def _preprocess_variants(self, gray: np.ndarray, i: int) -> Dict[str, np.ndarray]:
        """
        Devuelve primero la imagen original y luego variantes de mejora.
        """
        output_dir = Path(OUTPUT_DIR)
        aligned = self.alinear_y_rotar_barras(gray)

        variants = {"original": aligned}
        #log
        cv2.imwrite(str(output_dir / f"variant_original_{i}.png"), aligned)
     
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        variants["clahe"] = clahe.apply(aligned)
        #log
        cv2.imwrite(str(output_dir / f"variant_aligned_{i}.png"), aligned)

        
        clahe_img = variants["clahe"]
        _, otsu_on_clahe = cv2.threshold(
            clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        variants["clahe+otsu"] = otsu_on_clahe
        #log
        cv2.imwrite(str(output_dir / f"variant_clahe+otsu_{i}.png"), otsu_on_clahe)

        return variants

    # ==================================================================
    # DECODIFICACIÓN PRINCIPAL
    # ==================================================================
    def _try_decode_single(self, img: np.ndarray) -> List[Dict]:
        """
        Corre zxing-cpp sobre una imagen (gray o binarizada).
        Devuelve una lista de lecturas encontradas en esa imagen.
        """
        # try/except: "intenta esto, y si falla, no rompas el programa".
        # zxingcpp puede lanzar excepciones si la imagen es rara.
        try:
            # Aquí está la "magia": entregamos la imagen al lector y él
            # devuelve una lista de códigos encontrados.
            results = zxingcpp.read_barcodes(img)
        except Exception:
            LOGGER.exception(
                "ZXing falló al procesar una imagen de forma %s y tipo %s",
                img.shape,
                img.dtype,
            )
            return []

        out = []
        for r in results:
            # `r.format.name` es el tipo de código (CODE128, QR_CODE, ...)
            symbology = r.format.name

            # Si el usuario solo quiere ciertos formatos y este no está,
            # lo ignoramos.
            if not self._format_allowed(symbology.upper()):
                continue

            # Construimos un diccionario con la info que devolveremos.
            # `quality` fijo a 100.0 porque zxing no nos da un score
            # directo; el filtro real se hace con min_quality aguas abajo.
            out.append(
                {
                    "data": r.text,  # el texto del código
                    "type": symbology,  # el tipo (CODE128, EAN13...)
                    "quality": 100.0,  # calidad reportada (fija)
                    "rect": {"left": 0, "top": 0, "width": 0, "height": 0},
                    "polygon": [],  # coordenadas del código (no usadas)
                }
            )
        return out

    def decode(
        self,
        image: np.ndarray,
        i: int = 0,
    ) -> List[Dict]:
        """
        Decodifica un código de barras en la imagen o en un crop específico.

        Args:
            image: imagen HWC (BGR, típicamente de cv2) o HW (gray).
            return_attempts: si True, adjunta la lista de (variante, ángulo, hit).

        Returns:
            Lista de dicts con la decodificación, ordenados por calidad descendente.
            Vacía si no se pudo decodificar.
        """

        # ----  Convertir a gris ----
        gray = self._to_gray(image)

        # ---- Generar variantes preprocesadas ----
        variants = self._preprocess_variants(gray, i)

        # ----  Probar primero cada variante sin modificar ----
        all_hits: List[Dict] = []  # lecturas válidas acumuladas
        attempts: List[Dict] = []  # registro de intentos (para depurar)

        for name, img in variants.items():
            candidates = [
                (name, img),
                (f"{name}+quiet_zone", self._add_quiet_zone(img)),
            ]
            
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

        return unique
