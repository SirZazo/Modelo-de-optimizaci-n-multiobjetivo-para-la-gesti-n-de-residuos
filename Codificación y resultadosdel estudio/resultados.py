import os
import re
from pathlib import Path
import pandas as pd
from collections import defaultdict

def obtener_objetivos_equilibrados(ruta, idx):
    try:
        df = pd.read_csv(ruta, sep=';')
        if all(col in df.columns for col in ['F0', 'F1', 'F2']):
            fila = df[df['ID'] == idx]
            if not fila.empty:
                return fila[['F0', 'F1', 'F2']].iloc[0].tolist()
            else:
                print(f"⚠️ Índice {idx} no encontrado en: {ruta}")
        else:
            print(f"⚠️ Columnas F0, F1, F2 no encontradas en: {ruta}")
    except Exception as e:
        print(f"❌ Error leyendo {ruta}: {e}")
    return None

def obtener_solucion_x(ruta_fichero, indice):
    """
    Dado un fichero CSV de soluciones y un índice (ID), devuelve la solución X correspondiente como lista.

    Args:
        ruta_fichero (str): Ruta al archivo .csv
        indice (int): Índice (ID) de la solución deseada

    Returns:
        list: Lista con los valores de la solución X correspondiente, o None si no se encuentra
    """
    try:
        df = pd.read_csv(ruta_fichero, sep=';', index_col='ID')
        if indice in df.index:
            return df.loc[indice].tolist()
        else:
            print(f"⚠️ Índice {indice} no encontrado en {ruta_fichero}")
            return None
    except Exception as e:
        print(f"❌ Error al leer {ruta_fichero}: {e}")
        return None

def leerMaxMinGlobales():
    # Ruta base donde se encuentran los ficheros
    RUTA_BASE = "WMP_v04.00/resultados"

    # Listas para acumular todos los objetivos
    valores_totales = []

    # Recorremos recursivamente
    for root, dirs, files in os.walk(RUTA_BASE):
        for file in files:
            if file.endswith("_objetivos.csv"):
                ruta = os.path.join(root, file)
                try:
                    df = pd.read_csv(ruta, sep=';')
                    if all(col in df.columns for col in ['F0', 'F1', 'F2']):
                        valores_totales.append(df[['F0', 'F1', 'F2']])
                        #print(f"✅ Leído: {ruta}")
                    else:
                        print(f"⚠️ Columnas faltantes en: {ruta}")
                except Exception as e:
                    print(f"❌ Error leyendo {ruta}: {e}")

    # Unimos todos los DataFrames y calculamos los extremos
    if valores_totales:
        df_global = pd.concat(valores_totales, ignore_index=True)

        minimos = df_global.min()
        maximos = df_global.max()

        resumen = pd.DataFrame({
            'minimos': minimos,
            'maximos': maximos
        })
        return resumen

    else:
        print("❌ No se encontraron ficheros válidos.")


def extraer_numero(nombre):
    """Extrae el número final antes de '_datos.csv' para ordenar correctamente"""
    coincidencia = re.search(r'_(\d+)_datos\.csv$', nombre)
    return int(coincidencia.group(1)) if coincidencia else float('inf')




# Ruta raíz que contiene las carpetas por modo de objetivos
RUTA_BASE = Path("WMP_v04.00/resultados")

# Diccionario estructurado: modo -> algoritmo -> lista de índices de puntos equilibrados
indices_equilibrados = defaultdict(lambda: defaultdict(list))

# Recorremos todos los modos de objetivos (incluye triobjetivo también)
for modo_dir in RUTA_BASE.iterdir():
    if modo_dir.is_dir() and not modo_dir.name.startswith("."):
        modo = modo_dir.name

        # Dentro de cada modo, recorremos los algoritmos
        for alg_dir in modo_dir.iterdir():
            if alg_dir.is_dir():
                algoritmo = alg_dir.name

                # Buscar archivos *_datos.csv ordenados por el número de prueba
                archivos = sorted(alg_dir.glob("*_datos.csv"), key=lambda x: extraer_numero(x.name))

                for archivo in archivos:
                    try:
                        with open(archivo, 'r') as f:
                            lineas = [l.strip() for l in f.readlines() if l.strip()]

                            if not lineas:
                                print(f"⚠️ Archivo vacío: {archivo.name}")
                                continue

                            ultima_linea = lineas[-1]  # última línea con datos
                            valores = ultima_linea.split(";")

                            # El último valor debe ser el índice equilibrado
                            idx = int(float(valores[-1]))  # convertimos a float y luego int por seguridad
                            indices_equilibrados[modo][algoritmo].append(idx)

                    except Exception as e:
                        print(f"❌ Error al leer {archivo.name}: {e}")

# Mostrar resultados agrupados
for modo, algoritmos in indices_equilibrados.items():
    print(f"\n🧭 Modo: {modo}")
    for algoritmo, indices in algoritmos.items():
        print(f"  🔹 {algoritmo}: {indices}")

# Diccionario para almacenar soluciones encontradas de los puntos equilibrados
soluciones_equilibradas = defaultdict(lambda: defaultdict(list))
for modo, algoritmos in indices_equilibrados.items():
    for algoritmo, indices in algoritmos.items():
        for i, idx in enumerate(indices, start=1):

            # ⚠️ Reemplazamos los guiones por guiones bajos solo para el nombre del archivo
            modo_filename = modo.replace("-", "_")
            # Construir la ruta
            ruta = f"WMP_v04.00/resultados/{modo}/{algoritmo}/{algoritmo}_{modo_filename}_{i}_soluciones.csv"

            # Obtener la solución
            solucion = obtener_solucion_x(ruta, idx)

            if solucion is not None:
                soluciones_equilibradas[modo][algoritmo].append(solucion)
                #print(f"✅ [{modo} | {algoritmo} | ejecución {i}] -> Solución extraída")
            else:
                print(f"⚠️  No se pudo leer solución en: {ruta} (índice {idx})")


# Diccionario para almacenar los objetivos de los puntos equilibrados
objetivos_equilibrados = defaultdict(lambda: defaultdict(list))

for modo, algoritmos in indices_equilibrados.items():
    for algoritmo, indices in algoritmos.items():
        for i, idx in enumerate(indices, start=1):
            # Ajustar nombre del archivo con guiones bajos
            modo_filename = modo.replace("-", "_")
            ruta = f"WMP_v04.00/resultados/{modo}/{algoritmo}/{algoritmo}_{modo_filename}_{i}_objetivos.csv"
            
            objetivos = obtener_objetivos_equilibrados(ruta, idx)
            if objetivos:
                objetivos_equilibrados[modo][algoritmo].append(objetivos)
                #print(f"✅ [{modo} | {algoritmo} | ejecución {i}] -> {objetivos}")
            else:
                print(f"⚠️  No se pudo extraer objetivos para [{modo} | {algoritmo} | ejecución {i}]")
"""
# Mostrar resultados finales
for modo, algoritmos in objetivos_equilibrados.items():
    print(f"\n🧭 Modo: {modo}")
    for algoritmo, lista_obj in algoritmos.items():
        print(f"  🔹 {algoritmo}: {lista_obj}")
"""

maxmin = leerMaxMinGlobales()

#print(maxmin)



# Niveles de satisfacción de los objetivos seleccionados

# Crear el diccionario de satisfacción
satisfaccion_equilibrados = defaultdict(lambda: defaultdict(list))

# Nombres de los objetivos
OBJETIVOS_NOMBRES = {0: "Económico", 1: "Uso del Terreno", 2: "Salud Pública"}

# Extraer mínimos y máximos globales
minimos = maxmin['minimos']
maximos = maxmin['maximos']



# Calcular satisfacción para cada solución equilibrada
for modo, algoritmos in objetivos_equilibrados.items():
    for algoritmo, lista_obj in algoritmos.items():
        for i, objetivos in enumerate(lista_obj):
            f0, f1, f2 = objetivos

            sat_f0 = (1 - (f0 - minimos['F0']) / (maximos['F0'] - minimos['F0'])) * 100
            sat_f1 = (1 - (f1 - minimos['F1']) / (maximos['F1'] - minimos['F1'])) * 100
            sat_f2 = (1 - (f2 - minimos['F2']) / (maximos['F2'] - minimos['F2'])) * 100

            # Guardar resultado
            satisfaccion_equilibrados[modo][algoritmo].append([sat_f0, sat_f1, sat_f2])

           # print(f"✅ [{modo} | {algoritmo} | ejecución {i+1}] → "
           #       f"Satisfacción: Económico={sat_f0:.2f}%, Uso={sat_f1:.2f}%, Salud={sat_f2:.2f}%")


mejor_satisfaccion = defaultdict(dict)

for modo, algoritmos in satisfaccion_equilibrados.items():
    for algoritmo, lista_satisfacciones in algoritmos.items():
        mejor_idx = -1
        mejor_media = -1

        for i, satisf in enumerate(lista_satisfacciones):
            media = sum(satisf) / len(satisf)
            if media > mejor_media:
                mejor_media = media
                mejor_idx = i

        # Acceder a los objetivos reales
        objetivos_reales = objetivos_equilibrados[modo][algoritmo][mejor_idx]

        mejor_satisfaccion[modo][algoritmo] = {
            'indice': mejor_idx,
            'media': mejor_media,
            'satisfaccion': lista_satisfacciones[mejor_idx],
            'objetivos': objetivos_reales
        }

# Mostrar resultados
print("\n-------- RESULTADOS DE LAS MEJORES SOLUCIONES ---------")
for modo, algoritmos in mejor_satisfaccion.items():
    print(f"\n🧭 Modo: {modo}")
    for algoritmo, datos in algoritmos.items():
        idx = datos['indice']
        media = datos['media']
        satisf = datos['satisfaccion']
        obj = datos['objetivos']

        niveles_fmt = [f"{valor:.2f}%" for valor in satisf]
        objetivos_fmt = [f"{o:.2f}" for o in obj]

        print(f"  🔹 {algoritmo}: fichero={idx+1}, media={media:.2f}%, niveles={niveles_fmt}, objetivos={objetivos_fmt}")


