import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
from manejoArchivos import leerCSV
from procesadoAudio import procesarNuevo, mostrarDatos

np.set_printoptions(suppress=True)
matplotlib.use('TkAgg')

nroVecinos = 5

def calculoDistancia(xn, yn, zn, x, y, z):
    return math.sqrt((xn-x)**2 + (yn-y)**2 + (zn-z)**2)

def knn(audioNuevo):
    x, y, z, etiqueta, nombre = leerCSV('Resultados/Audios/Puntos.csv')
    print("Procesando audio...")
    posicionNuevo = procesarNuevo(audioNuevo)

    print("Calculando distancias...")
    distancias = np.empty((0, 1))
    for i in range(len(x)):
        distancias = np.vstack((distancias, calculoDistancia(posicionNuevo[0], posicionNuevo[1], posicionNuevo[2], x[i], y[i], z[i])))

    distancias = np.column_stack((distancias, np.array(etiqueta)))
    ordenados = distancias[np.argsort(distancias[:, 0])]

    contador = [0, 0, 0, 0]

    for i in range(nroVecinos):
        if ordenados[i, 1] == 0:
            contador[0]+= 1
        elif ordenados[i, 1] == 1:
            contador[1]+= 1
        elif ordenados[i, 1] == 2:
            contador[2]+= 1
        else:
            contador[3]+= 1

    indice = contador.index(max(contador))

    if indice == 0:
        return 'berenjena', posicionNuevo
    elif indice == 1:
        return 'camote', posicionNuevo
    elif indice == 2:
        return 'papa', posicionNuevo
    else:
        return 'zanahoria', posicionNuevo

def main():
    y, sr = librosa.load('Temp/Audios/clasificar.wav', sr=None)
    prediccion, coordenadas = knn(y)
    print(f"El audio dice: {prediccion}")
    mostrarDatos(coordenadas[0], coordenadas[1], coordenadas[2])


if __name__ == '__main__':
    main()









