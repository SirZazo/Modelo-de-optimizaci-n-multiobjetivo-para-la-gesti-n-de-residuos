import os
import pandas as pd
import numpy as np
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt

def globalminmax(RUTA_BASE):

    # Recorrer directorios recursivamente
    for root, dirs, files in os.walk(RUTA_BASE):
        print("---")
        for file in files:
            if file.endswith(".csv") and "gen_" in file:
                ruta_fichero = os.path.join(root, file)
                try:
                    df = pd.read_csv(ruta_fichero, sep=';')
                    if all(col in df.columns for col in ['F0', 'F1', 'F2']):
                        valores_totales.append(df[['F0', 'F1', 'F2']])
                except Exception as e:
                    print(f"Error al leer {ruta_fichero}: {e}")

    # Concatenar todos los DataFrames en uno solo
    if valores_totales:
        df_global = pd.concat(valores_totales, ignore_index=True)

        # Calcular min y max por columna
        minimos = df_global.min()
        maximos = df_global.max()

        print("\n📉 Mínimos globales:")
        print(minimos)
        print("\n📈 Máximos globales:")
        print(maximos)

        # Opcional: guardar en archivo CSV
        resumen = pd.DataFrame({
            'minimos': minimos,
            'maximos': maximos
        })
        resumen.to_csv("resumen_global_objetivos.csv", sep=';')
        print("\n✅ Resumen guardado en 'resumen_global_objetivos.csv'")
    else:
        print("❌ No se encontraron datos válidos.")


def normalizacion():

    # Cargar los min y max globales desde el resumen
    resumen = pd.read_csv("resumen_global_objetivos.csv", sep=';', index_col=0)
    minimos = resumen['minimos']
    maximos = resumen['maximos']
    RUTA_ENTRADA = "."
    RUTA_SALIDA = "Normalizados"
    # Creamos la estructura de carpetas
    for root, dirs, files in os.walk(RUTA_ENTRADA):
        for file in files:
            if file.endswith(".csv") and "gen_" in file:
                ruta_fichero = os.path.join(root, file)

                try:
                    df = pd.read_csv(ruta_fichero, sep=';')

                    if all(col in df.columns for col in ['F0', 'F1', 'F2']):
                        # Normalizar
                        for col in ['F0', 'F1', 'F2']:
                            df[col] = (df[col] - minimos[col]) / (maximos[col] - minimos[col] + 1e-8)

                        # Crear ruta destino
                        relative_path = os.path.relpath(root, RUTA_ENTRADA)
                        ruta_destino_dir = os.path.join(RUTA_SALIDA, relative_path)
                        os.makedirs(ruta_destino_dir, exist_ok=True)

                        # Guardar archivo normalizado
                        ruta_destino_file = os.path.join(ruta_destino_dir, file)
                        df.to_csv(ruta_destino_file, sep=';', index=False)
                        print(f"✅ Guardado: {ruta_destino_file}")

                except Exception as e:
                    print(f"⚠️ Error en {ruta_fichero}: {e}")



def hipervolumentriobjetivo():
        # Carpeta base donde están los datos normalizados
    BASE_DIR = "Normalizados/triobjetivo"
    RESULTADOS_DIR = "Resultados_HV"
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    # Iteramos por todos los ficheros CSV
    for root, dirs, files in os.walk(BASE_DIR):
        for file in sorted(files):
            if file.endswith(".csv") and "gen_" in file:
                ruta_csv = os.path.join(root, file)

                try:
                    df = pd.read_csv(ruta_csv, sep=';')

                    # Solo si tiene las columnas necesarias
                    columnas_objetivo = [col for col in df.columns if col.startswith("F")]
                    if not columnas_objetivo:
                        continue

                    F = df[columnas_objetivo].values
                    if F.shape[0] == 0:
                        continue

                    # Referencia de hipervolumen según número de objetivos
                    ref_point = np.ones(F.shape[1])  # [1.1,1.1]  o el [1.]

                    # Calcular hipervolumen
                    hv = HV(ref_point=ref_point)
                    valor_hv = hv(F)

                    # Construimos identificador
                    gen = file.split("_gen_")[-1].replace(".csv", "")
                    modo_objetivos = root.split(os.sep)[-3] if "trio" in root else root.split(os.sep)[-2]
                    algoritmo = root.split(os.sep)[-1]
                    nombre_resultado = f"hv_{modo_objetivos}_{algoritmo}.csv"
                    ruta_resultado = os.path.join(RESULTADOS_DIR, nombre_resultado)

                    # Añadir a CSV de resultados
                    with open(ruta_resultado, "a") as f:
                        f.write(f"{gen};{valor_hv:.6f}\n")

                    print(f"✅ HV {modo_objetivos} - {algoritmo} (gen {gen}): {valor_hv:.6f}")

                except Exception as e:
                    print(f"⚠️ Error procesando {ruta_csv}: {e}")

def hipervolumen_biobjetivo():
    BASE_DIR = "Normalizados"
    RESULTADOS_DIR = "Resultados_HV"
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    # Diccionario que mapea el modo al par de columnas necesarias
    pares_objetivos = {
        "economico-salud": [0, 2],  # F0 y F2
        "economico-uso": [0, 1],    # F0 y F1
        "salud-uso": [1, 2],        # F1 y F2
    }

    for modo_objetivos, columnas_indices in pares_objetivos.items():
        path_modo = os.path.join(BASE_DIR, modo_objetivos)

        for algoritmo in os.listdir(path_modo):
            path_algoritmo = os.path.join(path_modo, algoritmo)

            if not os.path.isdir(path_algoritmo):
                continue

            for file in sorted(os.listdir(path_algoritmo)):
                if file.endswith(".csv") and "gen_" in file:
                    ruta_csv = os.path.join(path_algoritmo, file)

                    try:
                        df = pd.read_csv(ruta_csv, sep=';')

                        # Obtener columnas F0, F1, F2
                        columnas_objetivo = [col for col in df.columns if col.startswith("F")]
                        if not columnas_objetivo:
                            continue

                        # Extraer solo las columnas necesarias
                        F = df.iloc[:, columnas_indices].values

                        if F.shape[0] == 0:
                            continue

                        # Referencia para biobjetivo: [1.0, 1.0]
                        ref_point = np.array([1.0, 1.0])

                        # Calcular hipervolumen
                        hv = HV(ref_point=ref_point)
                        valor_hv = hv(F)

                        # Extraer número de generación
                        gen = file.split("_gen_")[-1].replace(".csv", "")
                        nombre_resultado = f"hv_{modo_objetivos}_{algoritmo}.csv"
                        ruta_resultado = os.path.join(RESULTADOS_DIR, nombre_resultado)

                        # Añadir resultado
                        with open(ruta_resultado, "a") as f:
                            f.write(f"{gen};{valor_hv:.6f}\n")

                        print(f"✅ HV {modo_objetivos} - {algoritmo} (gen {gen}): {valor_hv:.6f}")

                    except Exception as e:
                        print(f"⚠️ Error procesando {ruta_csv}: {e}")

def showhipervolumen():

    RESULTADOS_DIR = "Resultados_HV"
        # Leer los archivos de resultados y graficarlos
    for file in os.listdir(RESULTADOS_DIR):
        if file.endswith(".csv") and file.startswith("hv_"):
            ruta = os.path.join(RESULTADOS_DIR, file)
            try:
                df = pd.read_csv(ruta, sep=";", header=None, names=["generacion", "hipervolumen"])
                df = df.sort_values("generacion")

                plt.figure(figsize=(8, 5))
                plt.plot(df["generacion"], df["hipervolumen"], marker='o', linestyle='-')
                plt.xlabel("Generación")
                plt.ylabel("Hipervolumen")
                plt.title(file.replace("hv_", "").replace(".csv", "").replace("_", " ").capitalize())
                plt.grid(True)

                nombre_img = file.replace(".csv", ".png")
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTADOS_DIR, nombre_img))
                plt.close()
                print(f"📊 Gráfico guardado: {nombre_img}")

            except Exception as e:
                print(f"❌ Error graficando {file}: {e}")

# Ruta raíz donde están las carpetas como "economico-uso", "salud-uso", etc.
RUTA_BASE = "."  # O el path absoluto si no estás en el mismo directorio


# Inicializar listas para acumular todos los valores
valores_totales = []
#globalminmax(RUTA_BASE)
#normalizacion()
#hipervolumentriobjetivo()
hipervolumen_biobjetivo()
showhipervolumen()



