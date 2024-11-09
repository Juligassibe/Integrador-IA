import os

def renombrar_archivos(directorio, prefijo="clasificar"):
    # Obtiene todos los archivos del directorio
    archivos = [f for f in os.listdir(directorio) if os.path.isfile(os.path.join(directorio, f))]

    # Ordena los archivos para un renombrado consistente
    archivos.sort()

    # Recorre cada archivo y le asigna un nuevo nombre
    for i, archivo in enumerate(archivos):
        # Separa el nombre y la extensión del archivo original
        nombre, extension = os.path.splitext(archivo)

        # Define el nuevo nombre del archivo con el prefijo y el número
        nuevo_nombre = f"{prefijo}{extension}"

        # Ruta completa del archivo original y del nuevo nombre
        ruta_original = os.path.join(directorio, archivo)
        ruta_nueva = os.path.join(directorio, nuevo_nombre)

        # Renombra el archivo
        os.rename(ruta_original, ruta_nueva)
        print(f"Renombrado: {archivo} a {nuevo_nombre}")


# Llamada a la función (cambiar 'tu_directorio' por la ruta real)
directorio = "Temp/Audios"
renombrar_archivos(directorio)
