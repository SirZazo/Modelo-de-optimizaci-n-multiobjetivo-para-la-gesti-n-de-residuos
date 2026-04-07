import numpy as np
import matplotlib.pyplot as plt
from problem import WasteManagementProblem
from numpy.linalg import norm
import pandas as pd


class ShowResults:


    @staticmethod
 
    def show_pareto_2d(F, X, tracker, problem, nombres_objetivos=None, objetivo=None):

        OBJETIVOS_NOMBRES = {
        0: "Económico",
        1: "Uso del Terreno",
        2: "Salud Pública"
        }


        if F.shape[1] != 2:
            raise ValueError("Se requieren exactamente dos objetivos para graficar en 2D")

        # Determinar qué objetivos están presentes
        match objetivo:
            case "economico-uso":
                objetivo1, objetivo2 = 0, 1
                objetivo3 = 2
            case "economico-salud":
                objetivo1, objetivo2 = 0, 2
                objetivo3 = 1
            case "salud-uso":
                objetivo1, objetivo2 = 2, 1
                objetivo3 = 0
            case _:
                raise ValueError("Modo no válido")

        # Calcular distancia al origen (0,0)
        distancias = np.linalg.norm(F, axis=1)
        idx_equilibrado = np.argmin(distancias)
        punto_equilibrado = F[idx_equilibrado]
        x_equilibrado = X[idx_equilibrado]

        # Mostrar el punto más equilibrado
        print("✅ Punto más equilibrado respecto a (0,0):")
        print(f"Index: {idx_equilibrado} - Obj1={punto_equilibrado[0]:.4f}, Obj2={punto_equilibrado[1]:.4f}")

        # Gráfico
        plt.figure(figsize=(8, 6))
        plt.scatter(F[:, 0], F[:, 1], c='green', label="Puntos Pareto")
        plt.scatter(*punto_equilibrado, c='red', s=60, label="Más Equilibrado (mín. distancia)")

        # Niveles de satisfacción de los objetivos seleccionados
        sigma_1 = (punto_equilibrado[0] - tracker.best_vals[objetivo1]) / (tracker.worst_vals[objetivo1] - tracker.best_vals[objetivo1])
        sigma_2 = (punto_equilibrado[1] - tracker.best_vals[objetivo2]) / (tracker.worst_vals[objetivo2] - tracker.best_vals[objetivo2])

        print(f"Nivel de satisfacción objetivo {OBJETIVOS_NOMBRES[objetivo1]}: {(1 - sigma_1) * 100:.2f}%")
        print(f"Nivel de satisfacción objetivo {OBJETIVOS_NOMBRES[objetivo2]}: {(1 - sigma_2) * 100:.2f}%")

        # ✅ Calcular objetivo faltante usando evaluate3
        f_all = problem._evaluate3(x_equilibrado)
        f_faltante = f_all[objetivo3]
        sigma_f = (f_faltante - tracker.best_vals[objetivo3]) / (tracker.worst_vals[objetivo3] - tracker.best_vals[objetivo3])
        print(f"Nivel de satisfacción objetivo ({OBJETIVOS_NOMBRES[objetivo3]}): {(1 - sigma_f) * 100:.2f}%")

        # Etiquetas
        if nombres_objetivos:
            plt.xlabel(nombres_objetivos[0])
            plt.ylabel(nombres_objetivos[1])
        else:
            plt.xlabel("Objetivo 1")
            plt.ylabel("Objetivo 2")

        plt.title("Frontera de Pareto en 2D")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return idx_equilibrado



    @staticmethod
    def show_pareto_3d(F, X ,tracker):

        # Cargar el resumen fuera de la función (o dentro si prefieres encapsularlo)
        #resumen = pd.read_csv("resumen_global_objetivos.csv", sep=';', index_col=0)
        # Convertir a arrays usando el índice correcto
        #min_vals = resumen.loc[:, 'minimos'].values
        #max_vals = resumen.loc[:, 'maximos'].values
        """
        Muestra una visualización 3D de la frontera de Pareto (si hay tres objetivos).
        Además, marca el punto más equilibrado (más cercano al origen).
        """
        if F.shape[1] != 3:
            raise ValueError("Se requieren exactamente tres objetivos para graficar en 3D")

        # Calcular distancia euclídea al origen (0,0,0) para cada punto
        distancias = np.linalg.norm(F, axis=1)
        idx_equilibrado = np.argmin(distancias)
        punto_equilibrado = F[idx_equilibrado]

        # Mostrar el punto más equilibrado
        print("✅ Punto más equilibrado respecto a (0,0,0):")
        print(f"Index: {idx_equilibrado} - Costo={punto_equilibrado[0]:.4f}, Uso={punto_equilibrado[1]:.4f}, Salud={punto_equilibrado[2]:.4f}")
        print(X[idx_equilibrado])
        
        # Gráfico
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], c='green', marker='o', label="Puntos Pareto")

        # Marcar el punto más equilibrado en rojo
        ax.scatter(*punto_equilibrado, c='red', s=50, label="Más Equilibrado (mín. distancia)")

        # Muestra los nivels de satisfación

        sigma_c = (punto_equilibrado[0] - tracker.best_vals[0]) / (tracker.worst_vals[0] - tracker.best_vals[0])
        sigma_u = (punto_equilibrado[1] - tracker.best_vals[1]) / (tracker.worst_vals[1] - tracker.best_vals[1])
        sigma_h = (punto_equilibrado[2] - tracker.best_vals[2]) / (tracker.worst_vals[2] - tracker.best_vals[2])

        print("Nivel de satisfación economico: ",(1 - sigma_c) * 100,"%" )
        print("Nivel de satisfación uso-tierra: ",(1 - sigma_u) * 100,"%")
        print("Nivel de satisfación salud: ",(1 - sigma_h) * 100,"%")


        ax.set_xlabel("Coste Económico")
        ax.set_ylabel("Uso del Terreno")
        ax.set_zlabel("Impacto Salud")
        plt.title("Frontera de Pareto en 3D")
        ax.legend()
        plt.tight_layout()
        plt.show()

        return idx_equilibrado

    # Opcional: retornar también el valor si quieres usarlo en otro lado
        return punto_equilibrado
  

    def plot_convergence(tracker):
        generaciones = range(len(tracker.min_costo))

        plt.figure(figsize=(15, 5))

        # Gráfico 1: Objetivo económico
        plt.subplot(1, 3, 1)
        plt.plot(generaciones, tracker.min_costo, label="Mínimo")
        plt.plot(generaciones, tracker.max_costo, label="Máximo")
        plt.title("Coste Económico")
        plt.xlabel("Generación")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)

        # Gráfico 2: Uso del terreno
        plt.subplot(1, 3, 2)
        plt.plot(generaciones, tracker.min_uso, label="Mínimo")
        plt.plot(generaciones, tracker.max_uso, label="Máximo")
        plt.title("Uso del Terreno")
        plt.xlabel("Generación")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)

        # Gráfico 3: Impacto en la salud
        plt.subplot(1, 3, 3)
        plt.plot(generaciones, tracker.min_salud, label="Mínimo")
        plt.plot(generaciones, tracker.max_salud, label="Máximo")
        plt.title("Impacto en la Salud")
        plt.xlabel("Generación")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def showHipervolumen(tracker):
        gens = tracker.generaciones
        hvs = tracker.hipervolumenes

        # Verificación para evitar error si está vacío
        if not gens or not hvs:
            print("⚠️ No hay datos de hipervolumen registrados.")
            return

        # Si `generaciones` guarda tuplas (gen, valor), desempaquetamos
        if isinstance(gens[0], tuple):
            gens = [g for g, _ in gens]
            hvs = [hv for _, hv in tracker.generaciones]

        plt.figure(figsize=(10, 6))
        plt.plot(gens, hvs, marker='o', linestyle='-', color='blue')
        plt.title("Evolución del Hipervolumen por Generación")
        plt.xlabel("Generación")
        plt.ylabel("Hipervolumen (normalizado)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def calcular_nivel_satisfaccion(valor, minimo, maximo):
        if maximo == minimo:
            return 1.0  # Evita división por cero: si no hay rango, se asume 100%
        sigma = (valor - minimo) / (maximo - minimo)
        return max(0.0, min(1.0, 1 - sigma))  # Clamp entre 0 y 1