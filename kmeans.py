import numpy as np
import os
from manejoArchivos import leerCSVImagenes, guardarCSV
from graficar import mostrarDatosImagenes
from procesadoImagenes import colorMedio

def calculoCentroides(nClusters = 4, tolerancia = 3):
    # Obtengo puntos sin clasificar
    x, y, z = leerCSVImagenes('Resultados/Imagenes/puntos imagenes.csv', False)
    # Valores iniciales de los centroides
    centroides = np.array([[30, 40, 20],
                           [90, 80, 160],
                           [80, 120, 150],
                           [50, 100, 180]])

    data = np.array([x, y, z]).T

    N = data.shape[0]

    while True:
        etiquetas = []
        clusterB = np.empty((0, 3))
        clusterC = np.empty((0, 3))
        clusterP = np.empty((0, 3))
        clusterZ = np.empty((0, 3))
        for i in range(N):
            distancias = []
            for j in range(nClusters):
                # Calculo distancia del punto a cada centroide
                distancias.append(np.linalg.norm(data[i, :] - centroides[j, :]))

            # Extraigo la minima, ese será el grupo al que pertenezca el punto
            etiquetas.append(np.argmin(distancias))

            # Guardo el punto en el grupo que corresponda
            if etiquetas[i] == 0:
                clusterB = np.vstack((clusterB, data[i, :]))
            elif etiquetas[i] == 1:
                clusterC = np.vstack((clusterC, data[i, :]))
            elif etiquetas[i] == 2:
                clusterP = np.vstack((clusterP, data[i, :]))
            else:
                clusterZ = np.vstack((clusterZ, data[i, :]))

        # Calculo centroides nuevos luego de clasificar todos los puntos, almaceno en temporal para comparar con iteración anterior
        temp = np.empty((4, 3))
        temp[0, :] = np.sum(clusterB, axis=0) / clusterB.shape[0]
        temp[1, :] = np.sum(clusterC, axis=0) / clusterC.shape[0]
        temp[2, :] = np.sum(clusterP, axis=0) / clusterP.shape[0]
        temp[3, :] = np.sum(clusterZ, axis=0) / clusterZ.shape[0]

        # Verifico si la distancia entre iteraciones sucesivas de centroides es menor que la tolerancia
        if np.max(np.linalg.norm(temp - centroides, axis=1)) < tolerancia:
            guardarCSV(temp, 'Resultados/Imagenes/centroides.csv')
            etiquetas = np.array(etiquetas)
            data = np.hstack((data, etiquetas[:, np.newaxis]))
            guardarCSV(data, 'Resultados/Imagenes/puntos clasificados.csv')
            return centroides
        centroides = temp

def kmeans():
    # Obtengo los centroides calculados previamente
    centroides = np.empty((4, 3))
    xc, yc, zc = leerCSVImagenes('Resultados/Imagenes/centroides.csv', False)   # Extraigo centroides de csv
    centroides[:, 0] = np.array(xc)
    centroides[:, 1] = np.array(yc)
    centroides[:, 2] = np.array(zc)

    # Imagenes a clasificar
    lista = [f for f in os.listdir('Temp/Imagenes/') if not f.startswith('.')]

    # Calculo los parámetros que las describen
    coordenadas = np.empty((0, 3))
    for imagen in lista:
        coordenadas = np.vstack((coordenadas, colorMedio(f"Temp/Imagenes/{imagen}")))

    # Verifico a que grupo pertenecen y las etiqueto
    etiquetas = []
    for coordenada in coordenadas:
        distancias = []
        for centroide in centroides:
            distancias.append(np.linalg.norm(centroide - coordenada))
        etiquetas.append(np.argmin(distancias))

    # Las renombro para identificarlas
    for i in range(len(lista)):
        if etiquetas[i] == 0:
            os.rename(f"Temp/Imagenes/{lista[i]}", f"Temp/Imagenes/berenjena.jpg")
        elif etiquetas[i] == 1:
            os.rename(f"Temp/Imagenes/{lista[i]}", f"Temp/Imagenes/camote.jpg")
        elif etiquetas[i] == 2:
            os.rename(f"Temp/Imagenes/{lista[i]}", f"Temp/Imagenes/papa.jpg")
        else:
            os.rename(f"Temp/Imagenes/{lista[i]}", f"Temp/Imagenes/zanahoria.jpg")
    # Mostrar posiciones de imagenes nuevas con las de la base de datos
    mostrarDatosImagenes('Resultados/Imagenes/puntos clasificados.csv', coordenadas)

def main():
    kmeans()

if __name__ == '__main__':
    main()