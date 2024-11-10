import csv
import numpy as np
import shutil
import os

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
    lista = [f for f in os.listdir('Temp/Audios/') if not f.startswith('.')]
    shutil.move(f'Temp/Audios/{lista[0]}', 'Audios/' + prediccion + '/' + nombre)