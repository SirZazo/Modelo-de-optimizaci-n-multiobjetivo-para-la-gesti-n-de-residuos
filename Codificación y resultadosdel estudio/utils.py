import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BASE = Path("WMP_v04.00/resultados")

# Dimensiones de instalaciones
N_CLASF = 37
N_INCIN = 30
N_VERT = 30
COLS = 3

LEN_CLASF = N_CLASF * COLS
LEN_INCIN = N_INCIN * COLS
LEN_VERT = N_VERT * COLS
LEN_TOTAL = LEN_CLASF + LEN_INCIN + LEN_VERT


def cargar_solucion(modo, algoritmo, ejecucion, id_sol):
    carpeta = BASE / modo / algoritmo.lower()
    modo_archivo = modo.replace("-", "_")

    archivo = carpeta / f"{algoritmo.lower()}_{modo_archivo}_{ejecucion}_soluciones.csv"

    if not archivo.exists():
        raise FileNotFoundError(f"No existe el archivo: {archivo}")

    df = pd.read_csv(archivo, sep=';')

    # Buscar ID exacto
    id_sol = int(id_sol)
    fila = df[df["ID"] == id_sol]

    if fila.empty:
        raise ValueError(f"No existe el individuo con ID {id_sol} en {archivo}")

    fila = fila.iloc[0]
    vector = fila.iloc[1:].to_numpy(dtype=float)
    return vector


def reconstruir_instalaciones(X):

    Xc = X[:LEN_TOTAL]

    pos = 0
    yjl = Xc[pos : pos + LEN_CLASF].reshape((N_CLASF, COLS))
    pos += LEN_CLASF

    ykl = Xc[pos : pos + LEN_INCIN].reshape((N_INCIN, COLS))
    pos += LEN_INCIN

    yk_primal = Xc[pos : pos + LEN_VERT].reshape((N_VERT, COLS))

    def contar(mat):
        return int(mat[:, 0].sum()), int(mat[:, 1].sum()), int(mat[:, 2].sum())

    c_pq, c_md, c_gr = contar(yjl)
    i_pq, i_md, i_gr = contar(ykl)
    v_pq, v_md, v_gr = contar(yk_primal)

    return {
        "clasif_pq": c_pq,
        "clasif_md": c_md,
        "clasif_gr": c_gr,
        "incin_pq": i_pq,
        "incin_md": i_md,
        "incin_gr": i_gr,
        "vert_pq": v_pq,
        "vert_md": v_md,
        "vert_gr": v_gr,
    }


def procesar_tabla(input_csv, salida_csv="reconstruccion_final.csv"):
    tabla = pd.read_csv(input_csv, sep=',')

    resultados = []

    for idx, row in tabla.iterrows():
        modo = row["MODO"]
        algoritmo = row["ALGORITMO"]
        ejecucion, id_sol = row["EJECUCION_SOL"].split("_")

        try:
            X = cargar_solucion(modo, algoritmo, ejecucion, id_sol)
            datos = reconstruir_instalaciones(X)

            resultados.append({
                "modo": modo,
                "algoritmo": algoritmo,
                "ejecucion": ejecucion,
                "id_sol": id_sol,
                **datos
            })

        except Exception as e:
            print(f"ERROR en {modo} - {algoritmo} - {ejecucion}_{id_sol}: {e}")

    df_out = pd.DataFrame(resultados)
    df_out.to_csv(salida_csv, sep=';', index=False)
    print(f"\nArchivo generado: {salida_csv}")


def plot_projections_with_highlight(csv_path, highlight_id=None, color=None, title = None):
    # Cargar CSV
    df = pd.read_csv(csv_path, sep=";")

    # Comprobar si se debe destacar un punto
    if highlight_id is not None:
        if highlight_id not in df["ID"].values:
            print(f"⚠ El ID {highlight_id} no existe en el archivo.")
            highlight_id = None
        else:
            df_high = df[df["ID"] == highlight_id]

    # Crear figura con 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    fig.suptitle(title, fontsize=16)
    # ---- PROYECCIÓN F0–F1 ----
    axs[0].scatter(df["F0"], df["F1"], s=25, alpha=0.5, color="blue")

    if highlight_id is not None:
        axs[0].scatter(df_high["F0"], df_high["F1"], 
                       s=70, c=color, edgecolors="blue", label=f"ID={highlight_id}")
        axs[0].legend()

    axs[0].set_xlabel("F0 (Coste)")
    axs[0].set_ylabel("F1 (Uso del suelo)")
    axs[0].set_title("Proyección F0–F1")
    axs[0].grid(True)


    # ---- PROYECCIÓN F0–F2 ----
    axs[1].scatter(df["F0"], df["F2"], s=25, alpha=0.5, color="blue")

    if highlight_id is not None:
        axs[1].scatter(df_high["F0"], df_high["F2"], 
                       s=70, c=color, edgecolors="black", label=f"ID={highlight_id}")
        axs[1].legend()

    axs[1].set_xlabel("F0 (Coste)")
    axs[1].set_ylabel("F2 (Impacto sanitario)")
    axs[1].set_title("Proyección F0–F2")
    axs[1].grid(True)


    # ---- PROYECCIÓN F1–F2 ----
    axs[2].scatter(df["F1"], df["F2"], s=25, alpha=0.5, color="blue")

    if highlight_id is not None:
        axs[2].scatter(df_high["F1"], df_high["F2"], 
                       s=70, c=color, edgecolors="black", label=f"ID={highlight_id}")
        axs[2].legend()

    axs[2].set_xlabel("F1 (Uso del suelo)")
    axs[2].set_ylabel("F2 (Impacto sanitario)")
    axs[2].set_title("Proyección F1–F2")
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()

from mpl_toolkits.mplot3d import Axes3D

def plot_3d_with_highlight(csv_path, highlight_id,color, title = None):
    df = pd.read_csv(csv_path, sep=";")

    if highlight_id not in df['ID'].values:
        print(f"⚠ El ID {highlight_id} no existe en el archivo.")
        return

    df_high = df[df['ID'] == highlight_id]

    fig = plt.figure(figsize=(8,6))
    fig.suptitle(title, fontsize=16)
    ax = fig.add_subplot(111, projection='3d')

    # Resto de puntos
    ax.scatter(df["F0"], df["F1"], df["F2"], c="blue", alpha=0.5)

    # Punto destacado
    ax.scatter(df_high["F0"], df_high["F1"], df_high["F2"],
               c=color, s=70, edgecolors="black")

    ax.set_xlabel("Costo Económico")
    ax.set_ylabel("Estrés uso de suelo")
    ax.set_zlabel("Impacto en la salud pública")
    ax.set_title(f"Frente 3D con ID={highlight_id} destacado")

    plt.tight_layout()
    plt.show()
# EJECUTAR
#procesar_tabla("WMP_v04.00/sol_recontruir.csv")

# Cargar un CSV trióbjetivo (cambia la ruta por el que quieras usar)

def plot_frente_pareto_con_proyecciones(csv_path,highlight_id, color="black", label="Algoritmo"):
    
    df = pd.read_csv(csv_path, sep=";")
    F0 = df["F0"].values
    F1 = df["F1"].values
    F2 = df["F2"].values

    # Límites reales del frente
    x_min, x_max = F0.min(), F0.max()
    y_min, y_max = F1.min(), F1.max()
    z_min, z_max = F2.min(), F2.max()


    # Figura
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

        # Punto destacado
    if highlight_id not in df['ID'].values:
        print(f"⚠ El ID {highlight_id} no existe en el archivo.")
        return

    df_high = df[df['ID'] == highlight_id]

    # Punto destacado
    ax.scatter(df_high["F0"], df_high["F1"], df_high["F2"],
               c="red", s=60, edgecolors="black",label=f"Solución destacada ({highlight_id})")

    # ------------------------------------------------
    # 1. NUBE 3D del frente (puntos reales)
    # ------------------------------------------------
    ax.scatter(F0, F1, F2,
               c=color,
               marker="^",      # TRIÁNGULOS
               s=40,
               alpha=0.9,
               label=label)

    # ------------------------------------------------
    # 2. PROYECCIONES ORTOGONALES (triángulos también)
    # ------------------------------------------------

    # Plano F2 = z_min → verde
    ax.scatter(F0, F1, np.full_like(F2, z_min),
               c="green",
               marker="^",
               s=25,
               alpha=0.5)

    # Plano F1 = y_min → azul
    ax.scatter(F0, np.full_like(F1, y_min), F2,
               c="blue",
               marker="^",
               s=25,
               alpha=0.5)

    # Plano F0 = x_min → rojo
    ax.scatter(np.full_like(F0, x_min), F1, F2,
               c="red",
               marker="^",
               s=25,
               alpha=0.5)

    # ------------------------------------------------
    # 3. Ajustes
    # ------------------------------------------------
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel("F0 (Coste económico)")
    ax.set_ylabel("F1 (Uso del suelo)")
    ax.set_zlabel("F2 (Impacto sanitario)")

    ax.view_init(elev=25, azim=35)
 

    ax.legend()
    plt.tight_layout()
    plt.show()

plot_frente_pareto_con_proyecciones(
    "WMP_v04.00/resultados/economico-salud/nsga3/nsga3_economico_salud_2_objetivos.csv", 
    highlight_id = 3,
    color = "black",
    label="NSGA-III"
)