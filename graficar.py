import matplotlib.pyplot as plt
import matplotlib
from manejoArchivos import leerCSV

matplotlib.use('TkAgg')             # Para que matplotlib haga gráficos en ventana aparte

def mostrarDatos(ruta, nuevo = False, xn = 0, yn = 0, zn = 0, etiquetaN = 4.0, nombreN = 'nombre'):
    x, y, z, etiqueta, nombre = leerCSV(ruta)
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
    if 'Audios' in ruta:
        ax.set_xlabel('5to mfcc')
        ax.set_ylabel('6to mfcc')
        ax.set_zlabel('zcr promedio')
    else:
        ax.set_xlabel('B')
        ax.set_ylabel('G')
        ax.set_zlabel('R')

    plt.show()