import cv2
import numpy as np


def colorProm(ruta):
    salida = ''
    # Cargar la imagen
    imagen = cv2.imread(ruta)
    """
    cv2.namedWindow('Imagen con contornos', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Imagen con contornos', 800, 600)
    cv2.namedWindow('Contornos interpolados', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Contornos interpolados', 800, 600)
    """
    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar desenfoque para reducir ruido
    desenfoque = cv2.GaussianBlur(gris, (11, 11), 0)

    # Detectar bordes con Canny
    umbral_bajo = 45
    umbral_alto = 50
    bordes = cv2.Canny(desenfoque, umbral_bajo, umbral_alto)

    # Encontrar contornos y su jerarquía
    contornos, jerarquia = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar para quedarnos solo con los contornos externos y significativos
    contornos_externos = [contorno for contorno in contornos if
                          cv2.contourArea(contorno) > 100]  # Ajusta el área mínima

    # Cerrar contornos abiertos y suavizarlos
    contornos_cerrados = []
    for contorno in contornos_externos:
        # Aproximar el contorno
        epsilon = 0.01 * cv2.arcLength(contorno, True)  # Ajusta este valor si es necesario
        aproximado = cv2.approxPolyDP(contorno, epsilon, True)

        # Generar puntos interpolados
        puntos_interpolados = []
        for j in range(len(aproximado)):
            p1 = aproximado[j][0]
            p2 = aproximado[(j + 1) % len(aproximado)][0]  # Conectar el último punto con el primero
            # Interpolar entre p1 y p2
            for t in np.linspace(0, 1, num=20):  # Cambia `num` para más o menos puntos
                interpolacion = (1 - t) * p1 + t * p2
                puntos_interpolados.append(interpolacion)

        contornos_cerrados.append(np.array(puntos_interpolados, dtype=np.int32))

    # Visualizar contornos interpolados
    contorno_imagen = np.zeros_like(imagen)
    for contorno in contornos_cerrados:
        cv2.drawContours(contorno_imagen, [contorno], -1, (255, 255, 255), thickness=5)

    # Crear una máscara vacía del mismo tamaño que la imagen original
    mascara = np.zeros(imagen.shape[:2], dtype=np.uint8)

    # Dibujar los contornos externos interpolados en la máscara (rellenando las áreas internas)
    for contorno in contornos_cerrados:
        cv2.drawContours(mascara, [contorno], -1, 255, thickness=cv2.FILLED)

    # Extraer los píxeles dentro del contorno externo
    pixeles_en_contorno = imagen[mascara == 255]

    # Calcular el promedio de color (RGB) de los píxeles dentro del contorno
    if len(pixeles_en_contorno) > 0:
        color_promedio = np.mean(pixeles_en_contorno, axis=0)
        salida = f"{color_promedio[2]},{color_promedio[1]},{color_promedio[0]}\n"
    else:
        print("No se encontraron píxeles dentro del contorno.")
        print(ruta)

    return salida
    """
    # Dibujar los contornos sobre la imagen original
    cv2.drawContours(imagen, contornos_cerrados, -1, (0, 255, 0), 2)

    # Mostrar la imagen con contornos
    cv2.imshow('Contornos interpolados', contorno_imagen)
    cv2.imshow('Imagen con contornos', imagen)

    # Esperar a que se cierre la ventana con la imagen
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """