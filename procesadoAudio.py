import librosa                                              # Para procesado de audios
import numpy as np                                          # Para manejo de arrays
import os                                                   # Para ver archivos en directorios
from manejoArchivos import guardarCSV, leerCSV              # Para escribir en archivos csv
from graficar import mostrarDatos

np.set_printoptions(suppress=True)  # Para que numpy no use notación científica

# Número de caracteristicas MFCC
nroMfccs = 13

def normalizar(audio):
    normalizado = librosa.util.normalize(audio)
    return normalizado

def eliminarSilencios(audio):
    sinSilencio, _ = librosa.effects.trim(audio, top_db=20)
    return sinSilencio

def filtrarAudio(audio, coef_preenfasis):
    audio = np.squeeze(audio) # Para transformar matriz columna (N, 1) en vector (N,)
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

def procesarAudio(y, nSegmentos=100, coef_preenfasis=0.99):
    # Cargar y normalizar el audio
    yNormalizado = normalizar(y)

    # Eliminar silencios en los bordes de la pista
    ySinSilencio = eliminarSilencios(yNormalizado)

    # Filtrar ruido de baja frecuencia
    yFiltrado = filtrarAudio(ySinSilencio, coef_preenfasis)

    # Dividir en segmentos y conservar los de mayor amplitud
    yFinal = eliminarSilencios(yFiltrado)

    return np.array(yFinal)

def extraerMfcc(audio, sr = 48000, n_mfcc=nroMfccs, n_fft=1024, hop_length=512):
    # Cálculo de los MFCCs a lo largo del tiempo
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Calcular la media de los MFCC a lo largo del tiempo
    mfccs = np.mean(mfcc, axis=1).round(1)

    return mfccs

def extraerZCR(audio):
    # Cálculo de ZCRs en segmentos del audio
    zcrs = librosa.feature.zero_crossing_rate(audio)
    # Cálculo de ZCR promedio
    zcr = np.mean(zcrs, axis=1).round(5)
    return zcr

#----------------------------------------------------------------------------------------------------------------------#

def procesarBaseDatosAudios():
    vegetales = ['berenjena', 'camote', 'papa', 'zanahoria']

    etiqueta = np.empty((0, 1))
    nombre = np.empty((0, 1))
    mfccs = np.empty((0, nroMfccs))
    zcr = np.empty((0, 1))

    for i in range(len(vegetales)):
        elementos = [f for f in os.listdir('Audios/' + vegetales[i] + '/') if not f.startswith('.')]
        elementos = sorted(elementos, key=lambda x: int(x.split(vegetales[i])[-1].split('.')[0]))

        for j in range(len(elementos)):
            actual = procesarAudio('Audios/' + vegetales[i] + '/' + elementos[j])
            mfccs = np.vstack((mfccs, extraerMfcc(actual)))
            zcr = np.vstack((zcr, extraerZCR(actual) * 10))
            etiqueta = np.vstack((etiqueta, np.array(i)))
            nombre = np.vstack((nombre, np.array(vegetales[i]+str(j))))

    caracteristicas = np.array([mfccs[:, 4], mfccs[:, 5]]).T
    caracteristicas = np.append(caracteristicas, zcr, axis=1)
    caracteristicas = np.append(caracteristicas, etiqueta, axis=1)
    caracteristicas = np.append(caracteristicas, nombre, axis=1)
    guardarCSV(caracteristicas, 'Resultados/Audios/Puntos.csv')

def procesarNuevo(audio):
    procesado = procesarAudio(audio)

    mfcc = extraerMfcc(procesado)
    zcr = extraerZCR(procesado) * 10

    caracteristicas = np.array([mfcc[4], mfcc[5], zcr[0]])

    return caracteristicas

if __name__ == '__main__':
    procesarBaseDatosAudios()
    mostrarDatos()