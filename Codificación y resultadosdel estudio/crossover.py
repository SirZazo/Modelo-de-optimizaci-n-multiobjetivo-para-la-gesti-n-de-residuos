from pymoo.core.crossover import Crossover
import numpy as np

class NoCrossover(Crossover):
    def __init__(self):
        # Definimos un crossover sin hijos (0)
        super().__init__(n_parents=2, n_offsprings=2)

    def _do(self, problem, X, **kwargs):
        #print("🚫 NoCrossover llamado, pero no hace nada.")
        return X  # Devuelve los mismos padres sin cambios