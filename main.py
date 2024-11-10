import os                                                   # Para ver archivos en directorios
import librosa                                              # Para abrir archivos de audio
import sounddevice as sd                                    # Para grabar archivos de audio
from scipy.io.wavfile import write                          # Para escribir archivos de audio
from knn import knn, mostrarDatos                           # Para algoritmo Knn y mostrar de datos
from manejoArchivos import guardarUltimaFila, moverAudio    # Para mover archivos y escribir en csv

def grabarAudio():
    seguir = input("Presione ENTER para grabar...")
    buffer = sd.rec(int(1.5 * 48000), samplerate=48000, channels=1)  # Para eliminar curva rara al encender microfono para grabar
    sd.wait()

    print("Grabando...")

    audio = sd.rec(int(3 * 48000), samplerate=48000, channels=1)
    sd.wait()
    print("Grabacion finalizada...")
    return audio

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

    prediccion, coordenadas = knn(audioNuevo)

    print(f"Prediccion: {prediccion}")
    correcta = input("Es correcta? (0. No - 1. Si): ")

    if correcta == "1":
        print("Agregando audio a la base de datos...")
        n = len(os.listdir('Audios/' + prediccion + '/')) - 2 # Resto los archivos .directory y .gitkeep
        nombre = prediccion + str(n)

        if prediccion == 'berenjena':
            nuevo = [coordenadas[0], coordenadas[1], coordenadas[2], 0.0, nombre]
        elif prediccion == 'camote':
            nuevo = [coordenadas[0], coordenadas[1], coordenadas[2], 1.0, nombre]
        elif prediccion == 'papa':
            nuevo = [coordenadas[0], coordenadas[1], coordenadas[2], 2.0, nombre]
        else:
            nuevo = [coordenadas[0], coordenadas[1], coordenadas[2], 3.0, nombre]

        guardarUltimaFila(nuevo)
        print(f"Agregado {nombre}.wav a la base de datos")

        if opcion == "1":
            write(f"Audios/{prediccion}/{nombre}.wav", 48000, audioNuevo)
        elif opcion == "2":
            moverAudio(prediccion, f"{nombre}.wav")
        mostrarDatos()
        return

    print("Predicción incorrecta, descartando audio...")

def main():
    while True:
        print("\n----0PCIONES----")
        print("1. Agregar imagenes a la base de datos")
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

            audio = grabarAudio()

            prediccion, coordenadas = knn(audio)

            print(f"La predicción es: {prediccion}")
            mostrarDatos(True, coordenadas[0], coordenadas[1], coordenadas[2], 4.0, 'nuevo')

        elif opcion == "4":
            break

        else:
            print("Opcion invalida")

if __name__ == '__main__':
    main()