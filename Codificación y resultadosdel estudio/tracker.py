
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
#from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import pandas as pd
import os
import numpy as np

class ConvergenceTracker(Callback):
    def __init__(self, problem, modo_objetivos, algortimo, nombre):
        super().__init__()
        self.problem = problem  # 👈 almacenas el objeto problemç
        self.modo_objetivos = modo_objetivos
        self.algoritmo = algortimo
        self.nombre = nombre
        self.best_vals = [np.inf, np.inf, np.inf]
        self.worst_vals = [-np.inf, -np.inf, -np.inf]
        self.hipervolumenes = []
        self.generaciones = []
        self.ref_point = np.array([-np.inf, -np.inf, -np.inf])  # valores grandes que dominen
        self.idxbest = 0
        self.idxworst = 0 

    def notify(self, algorithm):
        gen = algorithm.n_gen

        # Cálculo de las funciones objetivo
        listavalores =[]
        for ind in algorithm.pop:
            valores = self.problem._evaluate3(ind.X)  # 👈 ahora funciona
            listavalores.append(valores)

        # Actualizacion de maximos y minimos de las generaciones
        for i in range(3):  # 0: costo, 1: uso, 2: salud
            if valores[i] < self.best_vals[i]:
                self.best_vals[i] = valores[i]
                self.idxbest = i
            if valores[i] > self.worst_vals[i]:
                self.worst_vals[i] = valores[i]
                self.idxworst = i

        # Estudio de convergencia 
        # if algorithm.problem.n_obj in [2, 3] and gen % 100 == 0:
 
            #guardar_frente_pareto(listavalores, gen, self.modo_objetivos, self.algoritmo, self.nombre)  
        


        #print(f"🧠 Mejores hasta ahora: Costo={self.best_vals[0]:.4f}, Uso={self.best_vals[1]:.4f}, Salud={self.best_vals[2]:.4f}")
        #print(f"⚠️ Peores hasta ahora: Costo={self.worst_vals[0]:.4f}, Uso={self.worst_vals[1]:.4f}, Salud={self.worst_vals[2]:.4f}")


        """
        # Cálculo del hipervolumen cada 100 generaciones
        gen = algorithm.n_gen
        # Solo para problemas biobjetivo
        if algorithm.problem.n_obj == 3 and gen % 100 == 0:
            F = algorithm.pop.get("F")

            if np.all(np.isfinite(F)):
                # Calcular min y max reales de cada objetivo
                min_vals = np.min(F, axis=0)
                max_vals = np.max(F, axis=0)

                # Normalizar en rango [0, 1]
                F_scaled = min_max_scale(F, min_vals, max_vals)

                # Referencia un poco peor que el peor caso (todo normalizado)
                ref_point = np.array([1.0, 1.0, 1.0])

                hv = HV(ref_point=ref_point)
                valor_hv = hv(F_scaled)

                self.hipervolumenes.append((valor_hv))                
                self.generaciones.append((gen, valor_hv))
                print(f"📈 Gen {gen}: Hypervolumen 3D (escalado) = {valor_hv:.4f}")

        # Solo para problemas biobjetivo
        if algorithm.problem.n_obj == 2 and gen % 100 == 0:
            print("Calulando hipervolumen")
            F = algorithm.pop.get("F")
            # Calcular min y max reales
            min_vals = np.min(F, axis=0)
            max_vals = np.max(F, axis=0)
            # Normalizar manualmente
            F_scaled = min_max_scale(F, min_vals, max_vals)
            # Usamos un punto de referencia superior a [1, 1] en espacio escalado
            ref_point = np.array([1.0, 1.0])
            hv = HV(ref_point=ref_point)
            valor_hv = hv(F_scaled)

            self.hipervolumenes.append((valor_hv))
            self.generaciones.append(gen)
            print(f"📈 Gen {gen}: Hypervolumen (escalado) = {valor_hv:.4f}")
        """    

def min_max_scale(F, min_vals, max_vals):
    """Normaliza F en base a valores mínimos y máximos (por columnas)"""
    return (F - min_vals) / (max_vals - min_vals + 1e-8)  # Evita divisiones por cero

def construir_directorio(algoritmo, modo_objetivos):
    ruta = os.path.join(".", algoritmo, modo_objetivos)
    os.makedirs(ruta, exist_ok=True)
    return ruta  # Devuelve la ruta para usarla luego

def guardar_frente_pareto(valores, gen, algoritmo, modo_objetivos, nombre):
    """
    Guarda el frente de Pareto en un archivo dentro de ./algoritmo/modo_objetivos/
    """
    # Construir y crear ruta si no existe
    output_dir = construir_directorio(algoritmo, modo_objetivos)

    # Nombre del archivo nombre_objetivos_generaciones
    nombre_archivo =  nombre + "_" + modo_objetivos + "_" + f"gen_{gen}.csv"
    ruta_archivo = os.path.join(output_dir, nombre_archivo)


    # Guardar F en CSV
    valores = np.array(valores)
    df = pd.DataFrame(valores, columns=[f"F{i}" for i in range(valores.shape[1])])
    #df = df.drop_duplicates()
    df.to_csv(ruta_archivo, sep=';', index= False)

    print(f"✅ Frente de Pareto guardado en: {ruta_archivo}")