import librosa                                                      # Para abrir audios
import numpy as np                                                  # Para manejo de vectores
import math                                                         # Para sqrt()
import os
from scipy.io.wavfile import write
from manejoArchivos import leerCSVAudios, guardarUltimaFila, moverAudio   # Realizar lectura de datos en archivos csv
from procesadoAudio import procesarNuevo, grabarAudio               # Para procesar audios y mostrar nubes de puntos
from graficar import mostrarDatosAudios

np.set_printoptions(suppress=True)  # Para que numpy no use notación científica

nroVecinos = 5

def knn(audioNuevo):
    # Extraigo puntos de la base de datos
    x, y, z, etiqueta, nombre = leerCSVAudios('Resultados/Audios/Puntos.csv')
    data = np.array([x, y, z]).T
    print("Procesando audio...")
    posicionNuevo = procesarNuevo(audioNuevo)

    print("Calculando distancias...")
    distancias = np.empty((0, 1))
    for i in range(len(x)):
        distancias = np.vstack((distancias, np.linalg.norm(posicionNuevo - data[i, :])))

    distancias = np.column_stack((distancias, np.array(etiqueta)))
    ordenados = distancias[np.argsort(distancias[:, 0])]

    # Cuento dentro de los 5 vecinos mas cercanos cuantos pertenecen a cada grupo
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

    # Obtengo el indice que tiene el contador mayor, es decir el grupo con mas vecinos al punto nuevo
    indice = contador.index(max(contador))

    if indice == 0:
        return 'berenjena', posicionNuevo
    elif indice == 1:
        return 'camote', posicionNuevo
    elif indice == 2:
        return 'papa', posicionNuevo
    else:
        return 'zanahoria', posicionNuevo

def agregarBaseDatosAudios():
    while True:
        print("1. Grabar audio")
        print("2. Utilizar audio de Temp/Audios/")
        opcion = input("Opción: ")

        if opcion == "1":
            audioNuevo = grabarAudio()
            break
        elif opcion == "2":
            lista = [f for f in os.listdir('Temp/Audios/') if not f.startswith('.')]
            audioNuevo, _ = librosa.load('Temp/Audios/' + lista[0], sr=None)
            break
        else:
            print("Opción no válida.")

    # Clasifico el audio nuevo y verifico si la predicción es correcta antes de agregarlo a la base de datos
    prediccion, coordenadas = knn(audioNuevo)

    print(f"Prediccion: {prediccion}")
    print(coordenadas)
    correcta = input("Es correcta? (0. No - 1. Si): ")

    if correcta == "1":
        print("Agregando audio a la base de datos...")
        n = len(os.listdir('Audios/' + prediccion + '/')) - 2  # Resto los archivos .directory y .gitkeep
        nombre = prediccion + str(n)

        if prediccion == 'berenjena':
            nuevo = [coordenadas[0], coordenadas[1], coordenadas[2], 0.0, nombre]
        elif prediccion == 'camote':
            nuevo = [coordenadas[0], coordenadas[1], coordenadas[2], 1.0, nombre]
        elif prediccion == 'papa':
            nuevo = [coordenadas[0], coordenadas[1], coordenadas[2], 2.0, nombre]
        else:
            nuevo = [coordenadas[0], coordenadas[1], coordenadas[2], 3.0, nombre]

        guardarUltimaFila(nuevo, 'Resultados/Audios/Puntos.csv')
        print(f"Agregado {nombre}.wav a la base de datos")

        if opcion == "1":
            write(f"Audios/{prediccion}/{nombre}.wav", 48000, audioNuevo)
        elif opcion == "2":
            moverAudio(prediccion, f"{nombre}.wav")
        mostrarDatosAudios('Resultados/Audios/Puntos.csv')
        return

    print("Predicción incorrecta, audio descartado...")

def analizarAudio():
    audio = grabarAudio()

    prediccion, coordenadas = knn(audio)

    print(f"La predicción es: {prediccion}")
    mostrarDatosAudios('Resultados/Audios/Puntos.csv', True, coordenadas[0], coordenadas[1], coordenadas[2], 4.0, 'nuevo')

    return prediccion

def main():
    y, sr = librosa.load('Temp/Audios/clasificar0.wav', sr=None)
    prediccion, coordenadas = knn(y)
    print(f"El audio dice: {prediccion}")
    mostrarDatosAudios('Resultados/Audios/Puntos.csv', True, coordenadas[0], coordenadas[1], coordenadas[2])


if __name__ == '__main__':
    main()