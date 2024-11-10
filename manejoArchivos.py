import h5py
import csv
import numpy as np
import shutil

def guardar(verdura, matriz):
    if 'berenjena' in verdura:
        grupo = 'berenjena.h5'
    elif 'camote' in verdura:
        grupo = 'camote.h5'
    elif 'papa' in verdura:
        grupo = 'papa.h5'
    else:
        grupo = 'zanahoria.h5'
    with h5py.File('Resultados/Audios/' + grupo, 'a') as f:
        f.create_dataset(verdura, data=matriz)

def guardarSeparados(verdura, matriz):
    with open('Resultados/Audios/' + verdura + '.csv', 'w') as f:
        for fila in matriz:
            csv.writer(f).writerow(fila)

def guardarCSV(matriz):
    with open('Resultados/Audios/Puntos.csv', 'w') as f:
        for fila in matriz:
            csv.writer(f).writerow(fila)

def guardarUltimaFila(nuevo):
    with open('Resultados/Audios/Puntos.csv', 'a') as f:
        csv.writer(f).writerows([nuevo])

def leerCSV(ruta):
    datos = np.genfromtxt('Resultados/Audios/Puntos.csv', delimiter=',', dtype=None, names=('x', 'y', 'z', 'label', 'name'))
    x = datos['x'].astype(float).tolist()
    y = datos['y'].astype(float).tolist()
    z = datos['z'].astype(float).tolist()
    etiqueta = datos['label'].astype(int).tolist()
    nombre = datos['name'].astype(str).tolist()
    return x, y, z, etiqueta, nombre

def moverAudio(prediccion, nombre):
    shutil.move('Temp/Audios/clasificar.wav', 'Audios/' + prediccion + '/' + nombre)