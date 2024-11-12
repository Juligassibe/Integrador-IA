import matplotlib.pyplot as plt
import matplotlib
from manejoArchivos import leerCSVAudios, leerCSVImagenes

matplotlib.use('TkAgg')             # Para que matplotlib haga gráficos en ventana aparte

def mostrarDatosAudios(ruta, nuevo = False, xn = 0, yn = 0, zn = 0, etiquetaN = 4.0, nombreN = 'nombre'):
    x, y, z, etiqueta, nombre = leerCSVAudios(ruta)
    if nuevo:
        x.append(xn)
        y.append(yn)
        z.append(zn)
        etiqueta.append(etiquetaN)
        nombre.append(nombreN)

    colores = {0: 'violet', 1: 'red', 2: 'brown', 3: 'orange', 4: 'blue'}
    coloresPuntos = [colores[e] for e in etiqueta]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=coloresPuntos)

    """# DESCOMENTAR PARA MOSTRAR LOS NOMBRES DEL ARCHIVO QUE CORRESPONDE A CADA PUNTO
    for i, name in enumerate(nombre):
        ax.text(x[i], y[i], z[i], name, fontsize=9, color='black')
    """

    ax.set_xlabel('5to mfcc')
    ax.set_ylabel('6to mfcc')
    ax.set_zlabel('zcr promedio')

    plt.show()

def mostrarDatosImagenes(ruta):
    if 'clasificados' in ruta:
        x, y, z, etiqueta = leerCSVImagenes(ruta)
    else:
        x, y, z = leerCSVImagenes(ruta, False)

    xc, yc, zc = leerCSVImagenes('Resultados/Imagenes/centroides.csv', False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colores = ['violet', 'red', 'brown', 'orange']
    if 'clasificados' in ruta:
        coloresPuntos = [colores[e] for e in etiqueta]
        scatter = ax.scatter(x, y, z, c=coloresPuntos)
    else:
        scatter = ax.scatter(x, y, z)

    for i in range(len(xc)):
        ax.scatter(xc[i], yc[i], zc[i], color=colores[i], marker='s', s=100, label=f'Centroide {i + 1}')

    ax.set_xlabel('B')
    ax.set_ylabel('G')
    ax.set_zlabel('R')

    plt.show()
