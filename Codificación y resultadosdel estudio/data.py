import pandas as pd
import os
import pandas as pd
import numpy as np





def leer_matriz_localizacion(ruta_csv):
    """
    Lee una matriz binaria desde un CSV ignorando la primera fila y la primera columna.
    Devuelve un numpy.array de enteros (0 o 1).
    """
    df = pd.read_csv(ruta_csv, index_col=0)
    matriz = df.to_numpy(dtype=int)
    return matriz

def leer_matriz_distancias(ruta_archivo):
    # Leer el CSV con punto y coma como separador y usando la primera columna como índice
    df = pd.read_csv(ruta_archivo, sep=";", index_col=0)

    # Limpiar posibles espacios extra
    df.columns = df.columns.str.strip()
    df.index = df.index.str.strip()

    # Verificación básica
    if df.shape[0] != df.shape[1]:
        print("⚠️ La matriz no es cuadrada. Forma:", df.shape)

    matriz = df.to_numpy()
    return matriz  # 👈 solo devuelve la matriz

def read_waste_data(file_path):
    """Lee un archivo CSV y devuelve un vector con la cantidad de toneladas de residuos como float."""
    df = pd.read_csv(file_path, delimiter=';', usecols=['Waste(tonnes)'])
    
    # Convertir valores eliminando comas y transformándolos en flotantes
    waste_vector = df['Waste(tonnes)'].astype(float).values
    
    return waste_vector

def read_facility_data(file_path):
    """Lee un archivo CSV con datos de instalaciones y devuelve vectores de capacidad, costos fijos y costos operacionales."""
    df = pd.read_csv(file_path)
    
    # Filtrar por tipo de instalación
    sorting = df[df['Facility Type'] == 'Sorting Facility']
    incinerator = df[df['Facility Type'] == 'Incinerator']
    landfill = df[df['Facility Type'] == 'Landfill']
    
    # Vectores de capacidad
    Cjl = sorting['Storage Capacity'].values
    Ckl = incinerator['Storage Capacity'].values
    Ck_primal = landfill['Storage Capacity'].values
    
    # Vectores de costos fijos
    cjl = sorting['Fixed Cost'].values
    ckl = incinerator['Fixed Cost'].values
    c_primal = landfill['Fixed Cost'].values
    
    # Vectores de costos operacionales
    ojl = sorting['Operation Cost'].values
    okl = incinerator['Operation Cost'].values
    ok_primal = landfill['Operation Cost'].values
    
    return Cjl, Ckl, Ck_primal, cjl, ckl, c_primal, ojl, okl, ok_primal


def read_scaling_factors(sjl_path, skl_path, skl_prima_path):
    """Lee los archivos de factores de escala y los devuelve como matrices numpy de tipo float."""

    def read_and_convert(file_path):
        """Función auxiliar para leer y convertir un archivo CSV en una matriz de flotantes."""
        df = pd.read_csv(file_path, delimiter=';', header=None, decimal=",")  # Leer CSV con delimitador ';'
        df = df.applymap(lambda x: str(x).replace(',', '.'))  # Reemplazar ',' por '.'
        return df.astype(float).values  # Convertir a flotantes y devolver como matriz NumPy

    # Leer cada archivo y convertirlo en matriz de flotantes
    Sjl = read_and_convert(sjl_path)  # Matriz 37x3
    Skl = read_and_convert(skl_path)  # Matriz 30x3
    Skl_prima = read_and_convert(skl_prima_path)  # Matriz 30x3

    return Sjl, Skl, Skl_prima

def read_people_file(pij_path, pjk_path, pjk_prima_path):
    """Lee los archivos CSV y devuelve matrices numpy de flotantes."""
    def read_and_convert(file_path):
        """Función auxiliar para leer y convertir un archivo CSV en una matriz de flotantes."""
        df = pd.read_csv(file_path, header=None, delimiter=';', dtype=str)  # Leer como strings para manipulación
        df = df.applymap(lambda x: float(str(x)))  # Convertir valores a float
        return df.to_numpy()  # Convertir en matriz NumPy

    pij = read_and_convert(pij_path)
    pjk = read_and_convert(pjk_path)
    pjk_prima = read_and_convert(pjk_prima_path)
    
    return pij, pjk, pjk_prima

def read_people_facility_file(pjl_path, pkl_path, pkl_prima_path):
    """Lee los archivos CSV y devuelve matrices numpy de flotantes."""
    def read_and_convert(file_path):
        """Función auxiliar para leer y convertir un archivo CSV en una matriz de flotantes."""
        df = pd.read_csv(file_path, header=None, delimiter=';', dtype=str)  # Leer como strings para manipulación
        df = df.applymap(lambda x: float(str(x).replace(',', '.')))  # Convertir valores a float
        return df.to_numpy()  # Convertir en matriz NumPy

    pjl = read_and_convert(pjl_path)
    pkl = read_and_convert(pkl_path)
    pkl_prima = read_and_convert(pkl_prima_path)
    
    return pjl, pkl, pkl_prima


def read_facility_dalys(filepath):
    """Lee el archivo CSV usando pandas y almacena los valores en los vectores y variables correspondientes."""
    
    # Leer el archivo CSV con pandas
    df = pd.read_csv(filepath, delimiter=';')

    # Inicializar las variables
    dij = np.zeros(3)        # Sorting
    djk = np.zeros(3)        # Incinerator
    djk_prima = np.zeros(3)  # Landfill
    djl = 0.0  # Light Truck
    dkl = 0.0  # Heavy Truck
    dkl_prima = 0.0  # Heavy Truck (mismo valor que dkl)

    # Filtrar y asignar valores
    sorting_values = df[df["Facility"] == "Sorting"]["DAYLs"].values
    incinerator_values = df[df["Facility"] == "Incinerator"]["DAYLs"].values
    landfill_values = df[df["Facility"] == "Landfill"]["DAYLs"].values
    light_truck_value = df[df["Facility"] == "LigthTruck"]["DAYLs"].values
    heavy_truck_value = df[df["Facility"] == "HeavyTruck"]["DAYLs"].values

    # Asignar los valores asegurando que las longitudes sean correctas
    if len(sorting_values) == 3:
        dij = sorting_values
    if len(incinerator_values) == 3:
        djk = incinerator_values
    if len(landfill_values) == 3:
        djk_prima = landfill_values
    if len(light_truck_value) == 1:
        djl = light_truck_value[0]  # Convertir a escalar
    if len(heavy_truck_value) == 1:
        dkl = heavy_truck_value[0]  # Convertir a escalar
        dkl_prima = heavy_truck_value[0]  # Mismo valor para dkl_prima

    return dij, djk, djk_prima, djl, dkl, dkl_prima

# Ejemplo de uso
if __name__ == "__main__":

    np.set_printoptions(suppress=True, precision=6, linewidth=200, threshold=np.inf)
    
    facility_file = "WMP_v03.00/Data/costos_facilities.csv"
       
    Cjl, Ckl, Ck_primal, cjl, ckl, c_primal, ojl, okl, ok_primal = read_facility_data(facility_file)

    scaling_files = "WMP_v03.00/Data/Sjl.csv", "WMP_v03.00/Data/Skl.csv", "WMP_v03.00/Data/Skl_prima.csv"
    Sjl, Skl, Skl_prima = read_scaling_factors(scaling_files[0],scaling_files[1],scaling_files[2])
    
    people_file = "WMP_v03.00/Data/pij.csv","WMP_v03.00/Data/pjk.csv","WMP_v03.00/Data/pjk_prima.csv"  # Ruta al archivo CSV
    pij,pjk,pjk_prima = read_people_file(people_file[0],people_file[1],people_file[2])

    people_facility_file = "WMP_v03.00/Data/pjl.csv","WMP_v03.00/Data/pkl.csv","WMP_v03.00/Data/pkl_prima.csv"  # Ruta al archivo CSV
    pjl,pkl,pkl_prima = read_people_facility_file(people_facility_file[0],people_facility_file[1],people_facility_file[2])
        
    print("\nScaling data:")
    print("Sjl:", Sjl)
    print(Sjl.shape)

    print("\nScaling data:")
    print("Skl:", Sjl)
    print(Skl.shape)


    print("\nScaling data:")
    print("Skl_prima:", Sjl)
    print(Skl_prima.shape)



    print("\nStorage Capacity:")
    print("Cjl:", Cjl)
    print("Ckl:", Ckl)
    print("Ck_primal:", Ck_primal)
    
    print("\nFixed Costs:")
    print("cjl:", cjl)
    print("ckl:", ckl)
    print("c_primal:", c_primal)
    
    print("\nOperation Costs:")
    print("ojl:", ojl)
    print("okl:", okl)
    print("ok_primal:", ok_primal)

    print("\nMatriz Pij leída correctamente:")
    print(pij)
    print(pij.shape)

    print("\nMatriz Pij leída correctamente:")
    print(pjk)
    print(pjk.shape)

    print("\nMatriz Pij leída correctamente:")
    print(pjk_prima)
    print(pjk_prima.shape)

    print("\nMatriz Pij leída correctamente:")
    print(pjl)
    print(pjl.shape)

    print("\nMatriz Pij leída correctamente:")
    print(pkl)
    print(pkl.shape)

    print("\nMatriz Pij leída correctamente:")
    print(pkl_prima)
    print(pkl_prima.shape)

    filepath = "WMP_v03.00 copy/Data/DALYs_facility.csv"  # Cambia esto con la ruta real del archivo
    dij, djk, djk_prima, djl, dkl, dkl_prima = read_facility_dalys(filepath)

    print("📌 Datos extraídos:")
    print(f"dij: {dij}")
    print(f"djk: {djk}")
    print(f"djk_prima: {djk_prima}")
    print(f"djl: {djl}")
    print(f"dkl: {dkl}")
    print(f"dkl_prima: {dkl_prima}")

def exportar_datos (X,F,tracker,nombre = None, problem = None, modo_objetivos = None, algortimo = None, idx_equilibrado = None):



# Crear la estructura de directorios
    ruta_salida = os.path.join("resultados", modo_objetivos, algortimo)
    os.makedirs(ruta_salida, exist_ok=True)



    F = np.array([problem._evaluate3(x) for x in X])

    # Exportación de soluciones (X)
    df_x = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
    ruta_x = os.path.join(ruta_salida, f"{nombre}_soluciones.csv")
    df_x = df_x.drop_duplicates()
    df_x.to_csv(ruta_x, sep=';', index_label="ID")
    print(f"✅ Soluciones exportadas a: {ruta_x}")
    


    df_f = pd.DataFrame(F, columns=[f"F{i}" for i in range(F.shape[1])])
    ruta_f = os.path.join(ruta_salida, f"{nombre}_objetivos.csv")
    df_f = df_f.drop_duplicates()
    df_f.to_csv(ruta_f, sep=';', index_label="ID")
    print(f"✅ Objetivos exportados a: {ruta_f}")

    # Exportaciones de los valores minimos y maximos
    
    resumen = {

        "F0_min": tracker.best_vals[0],
        "F0_max": tracker.worst_vals[0],
        "F1_min": tracker.best_vals[1],
        "F1_max": tracker.worst_vals[1],
        "F2_min": tracker.best_vals[2],
        "F2_max": tracker.worst_vals[2],
        "Indice solucion equilibrada: ": idx_equilibrado,
    }

    df_resumen = pd.DataFrame([resumen])
    ruta_resumen = os.path.join(ruta_salida, f"{nombre}_datos.csv")
    df_resumen.to_csv(ruta_resumen, sep=';', index=False)
    print(f"✅ Resumen exportado a: {ruta_resumen}")


    