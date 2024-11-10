import librosa
import numpy as np
import csv
import os
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
from manejoArchivos import guardar, guardarCSV, leerCSV

np.set_printoptions(suppress=True)

matplotlib.use('TkAgg')

nroMfccs = 13
k = 1

def normalizar(audio):
    normalizado = librosa.util.normalize(audio)
    return normalizado

def eliminarSilencios(audio):
    sinSilencio, _ = librosa.effects.trim(audio, top_db=20)
    return sinSilencio

def filtrarAudio(audio, coef_preenfasis):
    filtrado = librosa.effects.preemphasis(audio, coef=coef_preenfasis)
    return filtrado

def conservarMayorAmplitud(audio, nSegmentos):
    duracionSegmento = len(audio) // nSegmentos
    yProcesado = []

    for i in range(nSegmentos):
        inicio = i * duracionSegmento
        fin = inicio + duracionSegmento
        segmento = audio[inicio:fin]

        if np.mean(np.abs(segmento)) > 0.5 * np.mean(np.abs(audio)):
            yProcesado.extend(segmento)

    return yProcesado

def procesarAudio(rutaAudio, nSegmentos=100, coef_preenfasis=0.99):
    # Cargar y normalizar el audio
    y, sr = librosa.load(rutaAudio, sr=None)
    yNormalizado = normalizar(y)

    # Eliminar silencios en los bordes de la pista
    ySinSilencio = eliminarSilencios(yNormalizado)

    # Filtrar ruido de baja frecuencia
    yFiltrado = filtrarAudio(ySinSilencio, coef_preenfasis)

    # Dividir en segmentos y conservar los de mayor amplitud
    yFinal = eliminarSilencios(yFiltrado)

    return np.array(yFinal)

def extraerMfcc(audio, sr = 48000, n_mfcc=nroMfccs, n_fft=1024, hop_length=512):
    # Subdivido en segmentos
    duracionSegmento = len(audio) // k
    mfccs = []

    for i in range(k):
        inicio = i * duracionSegmento
        fin = inicio + duracionSegmento
        segmentos = audio[inicio:fin]

        mfcc = librosa.feature.mfcc(y=segmentos, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        # Calcular la media de los MFCC a lo largo del tiempo (opcional)
        mfccsMedia = np.mean(mfcc, axis=1).round(1)
        mfccs.extend(mfccsMedia)

    mfccs = np.array(mfccs)

    return mfccs

def extraerZCR(audio):
    zcrs = librosa.feature.zero_crossing_rate(audio)
    return zcrs

#----------------------------------------------------------------------------------------------------------------------#

def ejecutarAudios(vegetal, id):
    ruta = 'Audios/' + vegetal + '/'+ vegetal + str(id) + '.wav'
    actual = procesarAudio(ruta)
    mfccs = extraerMfcc(actual)
    guardarCSV(vegetal + str(id), mfccs)

def procesarBaseDatosAudios():
    vegetales = ['berenjena', 'camote', 'papa', 'zanahoria']

    etiqueta = np.empty((0, 1))
    nombre = np.empty((0, 1))
    mfccs = np.empty((0, nroMfccs * k))
    zcr = np.empty((0, 1))

    for i in range(len(vegetales)):
        elementos = os.listdir('Audios/' + vegetales[i] + '/')
        elementos = sorted(elementos, key=lambda x: int(x.split(vegetales[i])[-1].split('.')[0]))

        for j in range(len(elementos)):
            actual = procesarAudio('Audios/' + vegetales[i] + '/' + elementos[j])
            mfccs = np.vstack((mfccs, extraerMfcc(actual)))
            zcr = np.vstack((zcr, np.mean(extraerZCR(actual), axis=1).round(5) * 10))
            etiqueta = np.vstack((etiqueta, np.array(i)))
            nombre = np.vstack((nombre, np.array(vegetales[i]+str(j))))

    caracteristicas = np.array([mfccs[:, 4], mfccs[:, 5]]).T
    caracteristicas = np.append(caracteristicas, zcr, axis=1)
    caracteristicas = np.append(caracteristicas, etiqueta, axis=1)
    caracteristicas = np.append(caracteristicas, nombre, axis=1)
    guardarCSV(caracteristicas)

def procesarNuevo(ruta):
    aProcesar = procesarAudio(ruta)
    mfcc = extraerMfcc(aProcesar)
    zcr = np.mean(extraerZCR(aProcesar), axis=1).round(5) * 10
    caracteristicas = np.array([mfcc[4], mfcc[5], zcr[0]])

    return caracteristicas

def mostrarDatos():
    x, y, z, etiqueta, nombre = leerCSV('Resultados/Audios/Puntos.csv')

    colores = {0: 'violet', 1: 'red', 2: 'brown', 3: 'orange'}
    coloresPuntos = [colores[e] for e in etiqueta]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=coloresPuntos, cmap='inferno')

    for i, name in enumerate(nombre):
        ax.text(x[i], y[i], z[i], name, fontsize=9, color='black')

    ax.set_xlabel('5to mfcc')
    ax.set_ylabel('6to mfcc')
    ax.set_zlabel('zcr promedio')

    plt.show()

if __name__ == '__main__':
    procesarBaseDatosAudios()
    mostrarDatos()