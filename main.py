from knn import agregarBaseDatosAudios, analizarAudio
from kmeans import calculoCentroides
from procesadoImagenes import procesarBaseDatosImagenes

def main():
    while True:
        print("\n----0PCIONES----")
        print("1. Reprocesar base de datos de imágenes")
        print("2. Agregar audio a la base de datos")
        print("3. Ejecutar K-means y Knn")
        print("4. Salir")

        opcion = input("Opcion: ")

        if opcion == "1":
            procesarBaseDatosImagenes()
            print("Imagenes procesadas exitosamente...")
            calculoCentroides()
            print("Centroides calculados exitosamente...")

        elif opcion == "2":
            agregarBaseDatosAudios()

        elif opcion == "3":
            print("kmeans")

            analizarAudio()

        elif opcion == "4":
            break

        else:
            print("Opcion invalida")


if __name__ == '__main__':
    main()