import csv
import numpy as np
import shutil
import os

#-------------------------------------------------------GENERAL--------------------------------------------------------#

def guardarCSV(matriz, ruta):
    with open(ruta, 'w') as f:
        for fila in matriz:
            csv.writer(f).writerow(fila)

def guardarUltimaFila(nuevo, ruta):
    with open(ruta, 'a') as f:
        csv.writer(f).writerows([nuevo])

def leerCSV(ruta):
    datos = np.genfromtxt(ruta, delimiter=',', dtype=None, names=('x', 'y', 'z', 'label', 'name'))
    x = datos['x'].astype(float).tolist()
    y = datos['y'].astype(float).tolist()
    z = datos['z'].astype(float).tolist()
    etiqueta = datos['label'].astype(int).tolist()
    nombre = datos['name'].astype(str).tolist()
    return x, y, z, etiqueta, nombre

#--------------------------------------------------------AUDIOS--------------------------------------------------------#

def moverAudio(prediccion, nombre):
    lista = [f for f in os.listdir('Temp/Audios/') if not f.startswith('.')]
    shutil.move(f'Temp/Audios/{lista[0]}', 'Audios/' + prediccion + '/' + nombre)