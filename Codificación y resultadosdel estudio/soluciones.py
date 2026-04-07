import os
import re
import pandas as pd
from pathlib import Path
import numpy as np
from collections import defaultdict
from pymoo.indicators.hv import HV



BASE_RESULTADOS = Path("WMP_v04.00/resultados")
BASE_NORMALIZADO = Path("WMP_v04.00/normalizado")
BASE_SALIDA = Path("WMP_v04.00/soluciones")

def extraer_numero_ejecucion(nombre_archivo):
    match = re.search(r'_(\d+)_objetivos\.csv$', nombre_archivo)
    return match.group(1) if match else None

def unficarObjetivos():

    for modo_dir in BASE_RESULTADOS.iterdir():
        if modo_dir.is_dir():
            modo = modo_dir.name

            for alg_dir in modo_dir.iterdir():
                if alg_dir.is_dir():
                    algoritmo = alg_dir.name

                    # Crear carpeta de salida si no existe
                    salida_dir = BASE_SALIDA / modo
                    salida_dir.mkdir(parents=True, exist_ok=True)

                    objetivos_unificados = []

                    for fichero in alg_dir.glob("*_objetivos.csv"):
                        try:
                            df = pd.read_csv(fichero, sep=';')
                            if 'ID' not in df.columns:
                                print(f"⚠️ 'ID' no encontrado en {fichero.name}")
                                continue

                            num_ejecucion = extraer_numero_ejecucion(fichero.name)
                            if not num_ejecucion:
                                print(f"⚠️ No se pudo extraer número de ejecución en {fichero.name}")
                                continue

                            # Modificar los ID para incluir el número de ejecución
                            df['ID'] = df['ID'].apply(lambda x: f"{num_ejecucion}_{int(x)}")
                            objetivos_unificados.append(df)

                            print(f"✅ Procesado {fichero.name} (ejecución {num_ejecucion})")

                        except Exception as e:
                            print(f"❌ Error leyendo {fichero.name}: {e}")

                    # Guardar el CSV unificado
                    if objetivos_unificados:
                        df_final = pd.concat(objetivos_unificados, ignore_index=True)
                        ruta_salida = salida_dir / f"{algoritmo}_objetivos_unificado.csv"
                        df_final.to_csv(ruta_salida, sep=';', index=False)
                        print(f"📁 Guardado: {ruta_salida}")

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



def obtencionMinOrigen(maximo, minimo):                        

    BASE_SALIDA = Path("WMP_v04.00/soluciones")  # Ruta donde están los archivos *_unificado.csv

    for modo_dir in BASE_SALIDA.iterdir():
        if modo_dir.is_dir():
            modo = modo_dir.name

            for fichero in modo_dir.glob("*_objetivos_unificado.csv"):
                try:
                    df = pd.read_csv(fichero, sep=';')

                    if all(col in df.columns for col in ['F0', 'F1', 'F2']):
                        F = df[['F0', 'F1', 'F2']].values

                        distancias = np.linalg.norm(F, axis=1)
                        idx_equilibrado = np.argmin(distancias)

                        fila_equilibrada = df.iloc[[idx_equilibrado]]

                        # Obtener los valores de F0, F1, F2
                        f0, f1, f2 = fila_equilibrada.iloc[0][['F0', 'F1', 'F2']]

                        # Calcular satisfacciones
                        sat_f0 = (1 - (f0 - minimos['F0']) / (maximos['F0'] - minimos['F0'])) * 100
                        sat_f1 = (1 - (f1 - minimos['F1']) / (maximos['F1'] - minimos['F1'])) * 100
                        sat_f2 = (1 - (f2 - minimos['F2']) / (maximos['F2'] - minimos['F2'])) * 100

                        # Añadir al DataFrame
                        fila_equilibrada['SAT_F0'] = sat_f0
                        fila_equilibrada['SAT_F1'] = sat_f1
                        fila_equilibrada['SAT_F2'] = sat_f2
                        fila_equilibrada['SAT_MEDIA'] = (sat_f0 + sat_f1 + sat_f2) / 3

                        # Guardar
                        nombre_base = fichero.name.replace("_unificado", "_soluciones")
                        ruta_salida = modo_dir / nombre_base
                        fila_equilibrada.to_csv(ruta_salida, sep=';', index=False)

                        print(f"✅ {fichero.name} → solución + satisfacciones guardadas en {nombre_base}")

                    else:
                        print(f"⚠️ Columnas F0-F1-F2 no encontradas en: {fichero.name}")

                except Exception as e:
                    print(f"❌ Error procesando {fichero.name}: {e}")


# RECORREMOS EL SISTE MA DIRECTORIO DE RESULTADOS PARA NORMALIZAR LOS FICHEROS DE OBJETIVO 


def normalizacion(maximos,minimos):  

    for root, dirs, files in os.walk(BASE_RESULTADOS):
        for file in files:
            if file.endswith("_objetivos.csv"):
                ruta_entrada = os.path.join(root, file)
                
                # Crear ruta equivalente de salida
                ruta_salida = ruta_entrada.replace("resultados", "normalizacion")
                ruta_salida = ruta_entrada.replace("resultados", "soluciones/satisfacción")
                
                os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

                try:
                    df = pd.read_csv(ruta_entrada, sep=';')

                    if all(col in df.columns for col in ['F0', 'F1', 'F2']):
                        # Normalización
                        df['F0'] = (df['F0'] - minimos['F0']) / (maximos['F0'] - minimos['F0'])
                        df['F1'] = (df['F1'] - minimos['F1']) / (maximos['F1'] - minimos['F1'])
                        df['F2'] = (df['F2'] - minimos['F2']) / (maximos['F2'] - minimos['F2'])

                        # Media de satisfacción
                        df['SAT_MEDIA'] = df[['SAT_F0', 'SAT_F1', 'SAT_F2']].mean(axis=1)

                        df.to_csv(ruta_salida, sep=';', index=False)
                        print(f"✅ Normalizado: {ruta_entrada} → {ruta_salida}")
                    else:
                        print(f"⚠️ Columnas F0-F2 no encontradas en {ruta_entrada}")
                except Exception as e:
                                print(f"❌ Error procesando normalizacion {ruta_entrada}: {e}")

def calculoSatisfacion():
    for root, dirs, files in os.walk(BASE_RESULTADOS):
        for file in files:
            if file.endswith("_objetivos.csv"):
                ruta_entrada = os.path.join(root, file)
                ruta_salida = ruta_entrada.replace("resultados", "satisfaccion")

                os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

                try:
                    df = pd.read_csv(ruta_entrada, sep=';')

                    if all(col in df.columns for col in ['F0', 'F1', 'F2']):
                        # Calcular satisfacción
                        sat_f0 = (1 - (df['F0'] - minimos['F0']) / (maximos['F0'] - minimos['F0'])) * 100
                        sat_f1 = (1 - (df['F1'] - minimos['F1']) / (maximos['F1'] - minimos['F1'])) * 100
                        sat_f2 = (1 - (df['F2'] - minimos['F2']) / (maximos['F2'] - minimos['F2'])) * 100

                        df_salida = pd.DataFrame({
                            'SAT_F0': sat_f0,
                            'SAT_F1': sat_f1,
                            'SAT_F2': sat_f2
                        })

                        df_salida['SAT_MEDIA'] = df_salida.mean(axis=1)

                        # Si hay columna ID, conservarla
                        if 'ID' in df.columns:
                            df_salida.insert(0, 'ID', df['ID'])

                        df_salida.to_csv(ruta_salida, sep=';', index=False)
                        print(f"✅ Satisfacción generada: {ruta_entrada} → {ruta_salida}")
                    else:
                        print(f"⚠️ Columnas F0-F2 no encontradas en {ruta_entrada}")

                except Exception as e:
                    print(f"❌ Error procesando {ruta_entrada}: {e}")


def ficherosatisfa():

    RUTA_SATISFACCION = Path("WMP_v04.00/satisfaccion")
    RUTA_RESULTADOS = Path("WMP_v04.00/resultados")
    RUTA_MEJORES = Path("WMP_v04.00/mejores")
    RUTA_MEJORES.mkdir(parents=True, exist_ok=True)
    
    soluciones_por_modo = defaultdict(lambda: defaultdict(list))

    # Cargar todos los ficheros de satisfacción
    for root, dirs, files in os.walk(RUTA_SATISFACCION):
        for file in files:
            if file.endswith("_objetivos.csv"):
                ruta = Path(root) / file
                partes = ruta.parts

                # Obtener modo desde la carpeta
                modo = partes[partes.index("satisfaccion") + 1] if "satisfaccion" in partes else "desconocido"
                algoritmo = file.split("_")[0]

                try:
                    df = pd.read_csv(ruta, sep=';')
                    df['archivo'] = file.replace(".csv", "")
                    soluciones_por_modo[modo][algoritmo].append(df)
                except Exception as e:
                    print(f"❌ Error leyendo {ruta}: {e}")

    # Buscar la mejor solución por modo y algoritmo
    for modo, algoritmos in soluciones_por_modo.items():
        for algoritmo, lista_dfs in algoritmos.items():
            combinado = pd.concat(lista_dfs, ignore_index=True)

            if 'SAT_MEDIA' not in combinado.columns or 'ID' not in combinado.columns:
                print(f"⚠️ SAT_MEDIA o ID no encontradas en las soluciones de {modo}/{algoritmo}")
                continue

            mejor_idx = combinado['SAT_MEDIA'].idxmax()
            mejor_sol = combinado.loc[mejor_idx]

            archivo = mejor_sol['archivo']
            identificador = int(mejor_sol['ID'])

            # Construir la ruta al fichero original de resultados
            ruta_resultado = RUTA_RESULTADOS / modo / algoritmo / f"{archivo}.csv"

            try:
                df_original = pd.read_csv(ruta_resultado, sep=';')
                fila_obj = df_original[df_original['ID'] == identificador]

                if not fila_obj.empty:
                    f0, f1, f2 = fila_obj[['F0', 'F1', 'F2']].iloc[0]
                else:
                    print(f"⚠️ No se encontró ID={identificador} en {ruta_resultado}")
                    f0 = f1 = f2 = None

            except Exception as e:
                print(f"❌ Error leyendo original: {ruta_resultado}: {e}")
                f0 = f1 = f2 = None

            # Guardar todo
            id_compuesto = f"{archivo}_{identificador}"
            fila = pd.DataFrame([{
                'ID': id_compuesto,
                'F0': f0,
                'F1': f1,
                'F2': f2,
                'SAT_F0': mejor_sol['SAT_F0'],
                'SAT_F1': mejor_sol['SAT_F1'],
                'SAT_F2': mejor_sol['SAT_F2'],
                'SAT_MEDIA': mejor_sol['SAT_MEDIA']
            }])

            nombre_archivo = f"{algoritmo}_{modo.replace('-', '_')}_mejor.csv"
            ruta_salida = RUTA_MEJORES / nombre_archivo
            fila.to_csv(ruta_salida, sep=';', index=False)
            print(f"✅ Guardado: {ruta_salida}")



def calculoHipervolumen():

    # Ruta base donde están los ficheros normalizados
    RUTA_NORMALIZACION = Path("WMP_v04.00/normalizacion")
    RUTA_SALIDA = "hipervolumenes.csv"

    # Punto de referencia (un poco mayor que 1.0 para asegurar volumen válido)
    punto_referencia = np.array([1.0, 1.0, 1.0])

    # Inicializar lista para resultados
    resultados = []

    # Recorrer la estructura
    for root, dirs, files in os.walk(RUTA_NORMALIZACION):
        for file in files:
            if file.endswith(".csv") and "objetivos" in file:
                ruta = Path(root) / file
                try:
                    df = pd.read_csv(ruta, sep=';')
                    if all(col in df.columns for col in ['F0', 'F1', 'F2']):
                        F = df[['F0', 'F1', 'F2']].to_numpy()
                        hv = HV(ref_point=punto_referencia)
                        valor_hv = hv(F)

                        # Extraer modo, algoritmo, ejecución
                        partes = ruta.parts
                        modo = partes[-3]
                        algoritmo = partes[-2]
                        nombre = file.replace(".csv", "")
                        match = re.search(r'_(\d+)_objetivos', nombre)
                        ejecucion = int(match.group(1)) if match else -1  # -1 si no se encuentra

                        resultados.append({
                            "modo": modo,
                            "algoritmo": algoritmo,
                            "ejecucion": int(ejecucion),
                            "hipervolumen": valor_hv
                        })
                        print(f"✅ Calculado HV para {modo}/{algoritmo} → {file} = {valor_hv:.5f}")
                    else:
                        print(f"⚠️ Faltan columnas F0-F2 en {ruta}")
                except Exception as e:
                    print(f"❌ Error procesando {ruta}: {e}")

    # Guardar en CSV
    df_resultados = pd.DataFrame(resultados)
    df_resultados.sort_values(by=['modo', 'algoritmo', 'ejecucion'], inplace=True)
    df_resultados.to_csv(RUTA_SALIDA, index=False, sep=';')
    print(f"\n✅ Hipervolúmenes guardados en {RUTA_SALIDA}")



maxmin = leerMaxMinGlobales()

# Extraer mínimos y máximos globales
minimos = maxmin['minimos']
maximos = maxmin['maximos']

#ficherosatisfa()

calculoHipervolumen()

