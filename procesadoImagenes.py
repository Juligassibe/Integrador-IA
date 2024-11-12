import cv2
import numpy as np
import os
from manejoArchivos import guardarCSV
from graficar import mostrarDatos

def procesadoImagen(imagenGris):
    # Ajustar el contraste y el brillo de la imagen en escala de grises
    alpha = 1.5  # Factor de contraste
    beta = 20    # Valor de brillo
    imagenAjustada = cv2.convertScaleAbs(imagenGris, alpha=alpha, beta=beta)

    # Aplicar desenfoque gaussiano para reducir el ruido
    imagenBorroso = cv2.GaussianBlur(imagenAjustada, (5, 5), 0)

    return imagenBorroso

def detectarContornos(imagen):
    # Aplicar Canny para detectar bordes
    bordes = cv2.Canny(imagen, 50, 150)

    # Crear un kernel para dilatación y erosión
    kernel = np.ones((3, 3), np.uint8)

    # Aplicar dilatación seguida de erosión para cerrar pequeños huecos en el contorno
    imagenDilatada = cv2.dilate(bordes, kernel, iterations=1)
    imagenProcesada = cv2.erode(imagenDilatada, kernel, iterations=1)

    return imagenProcesada

def contornoExterior(imagen):
    # Encontrar contornos, seleccionando solo el contorno exterior
    contornos, _ = cv2.findContours(imagen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Seleccionar el contorno con el área más grande (suponiendo que sea el contorno exterior)
    contornoExterior = max(contornos, key=cv2.contourArea)

    return contornoExterior

def mostrarImagenes(original, gris, sinRuido, procesada, mascara):
    # Mostrar como se fue modificando la imagen con el procesado
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original", 400, 400)
    cv2.imshow("Original", original)

    cv2.namedWindow("Escala de grises", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Escala de grises", 400, 400)
    cv2.imshow("Escala de grises", gris)

    cv2.namedWindow("Difuminada", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Difuminada", 400, 400)
    cv2.imshow("Difuminada", sinRuido)

    cv2.namedWindow("Imagen procesada", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Imagen procesada", 400, 400)
    cv2.imshow("Imagen procesada", procesada)

    cv2.namedWindow("Mascara rellena", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mascara rellena", 400, 400)
    cv2.imshow("Mascara rellena", mascara)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def colorMedio(ruta):
    # Cargar la imagen en color y convertirla a escala de grises
    imagenColor = cv2.imread(ruta)
    imagenGris = cv2.cvtColor(imagenColor, cv2.COLOR_BGR2GRAY)
    # Desenfocado de imagen para eliminar ruido
    imagenSinRuido = procesadoImagen(imagenGris)
    # Detección de contornos y procesado extra para que queden curvas cerradas
    imagenProcesada = detectarContornos(imagenSinRuido)
    # Se extrae el contorno exterior, utilizando como parámetro que sea el de área interior mas grande
    contorno = contornoExterior(imagenProcesada)

    # Crear una máscara y rellenar el contorno exterior
    mascara = np.zeros_like(imagenGris)
    cv2.drawContours(mascara, [contorno], -1, color=255, thickness=cv2.FILLED)

    # Calcular el color promedio dentro del contorno exterior utilizando la máscara
    mean_color = cv2.mean(imagenColor, mask=mascara)

    # Opcional: Mostrar la imagen con el contorno exterior resaltado
    # Mostrara 4 imágenes por cada imagen que se procesa, dejar desactivado a menos que se tengan pocas imagenes para procesar
    #mostrarImagenes(imagenColor, imagenGris, imagenSinRuido, imagenProcesada, mascara)

    return np.array(mean_color[:3]).round(2)

if __name__ == "__main__":
    vegetales = ['berenjena', 'camote', 'papa', 'zanahoria']
    medio = np.empty((0, 5))

    for i in range(len(vegetales)):
        ruta = f"Imagenes/{vegetales[i]}/"

        lista = [f for f in os.listdir(ruta) if not f.startswith('.')]
        lista = sorted(lista, key=lambda x: int(x.split(vegetales[i])[-1].split('.')[0]))

        for j in range(len(lista)):
            temp = colorMedio(f"{ruta}{lista[j]}")
            temp = np.append(temp, i)
            temp = np.append(temp, lista[j])
            medio = np.vstack((medio, temp))

    guardarCSV(medio, 'Resultados/Imagenes/puntos imagenes.csv')
    mostrarDatos('Resultados/Imagenes/puntos imagenes.csv')