import os
from knn import knn
from manejoArchivos import guardarUltimaFila, moverAudio
from procesadoAudio import ejecutarAudios

def agregarBaseDatosAudios():
    prediccion, coordenadas = knn()

    print(f"Prediccion: {prediccion}")
    correcta = input("Es correcta (0. No - 1. Si): ")
    if correcta != "0":
        print("Agregando audio a la base de datos...")
        n = len(os.listdir('Audios/' + prediccion + '/'))
        nombre = prediccion + str(n) + '.wav'

        if prediccion == 'berenjena':
            nuevo = [coordenadas[0], coordenadas[1], coordenadas[2], 0.0, nombre]
        elif prediccion == 'camote':
            nuevo = [coordenadas[0], coordenadas[1], coordenadas[2], 1.0, nombre]
        elif prediccion == 'papa':
            nuevo = [coordenadas[0], coordenadas[1], coordenadas[2], 2.0, nombre]
        else:
            nuevo = [coordenadas[0], coordenadas[1], coordenadas[2], 3.0, nombre]

        guardarUltimaFila(nuevo)
        moverAudio(prediccion, nombre)

        return
    print("Predicción incorrecta, descartando audio...")

def main():
    while True:
        print("----0PCIONES----")
        print("1. Procesar imagenes de base de datos")
        print("2. Agregar audio a la base de datos")
        print("3. Ejecutar K-means y Knn")
        print("4. Salir")

        opcion = input("Opcion: ")

        if opcion == "1":
            print("Imagenes")

        elif opcion == "2":
            agregarBaseDatosAudios()
                    
        elif opcion == "3":
            print("kmeans")

        elif opcion == "4":
            break

        else:
            print("Opcion invalida")

if __name__ == '__main__':
    main()