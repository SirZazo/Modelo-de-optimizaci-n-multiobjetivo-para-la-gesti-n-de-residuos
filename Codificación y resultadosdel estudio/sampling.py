import numpy as np
import random
from pymoo.core.sampling import Sampling


def generar_yjl_valido(num_sorting, num_sizes, Cjl, Di, p_open):
    """
    Genera una matriz yjl válida (clasificadoras abiertas y con tamaño asignado)
    que asegure capacidad >= demanda total, favoreciendo tamaños grandes.
    """
    total_demand = np.sum(Di)

    # Más peso para tamaños grandes
    size_weights = ([0.1, 0.2, 0.7])
 

    while True:
        yjl = np.zeros((num_sorting, num_sizes), dtype=int)
        capacidad_acumulada = 0

        for loc in range(num_sorting):
            if np.random.rand() < p_open:
                size = np.random.choice(num_sizes, p=size_weights)
                yjl[loc, size] = 1
                capacidad_acumulada += Cjl[size]
            
        if capacidad_acumulada >= total_demand:
            #print("",total_demand)
            #print("CAP SORTING",capacidad_acumulada)
            return yjl

def generar_ykl_valido(num_treatment, num_sizes, Ctreatment, flujo_total, p_open):
    """
    Genera una matriz ykl válida (instalaciones de tratamiento abiertas y con tamaño asignado)
    que asegure capacidad >= flujo total a tratar, favoreciendo tamaños grandes.
    """
    # Más peso para tamaños grandes
    size_weights = ([0.1, 0.2, 0.7])

    while True:
        ykl = np.zeros((num_treatment, num_sizes), dtype=int)
        capacidad_acumulada = 0

        for loc in range(num_treatment):
            if np.random.rand() < p_open:
                size = np.random.choice(num_sizes, p=size_weights)
                ykl[loc, size] = 1
                capacidad_acumulada += Ctreatment[size]
        #print(num_treatment)
        #print("flujo_total",flujo_total)
        #print("CAP SORTING",capacidad_acumulada)
        if capacidad_acumulada >= flujo_total:
            return ykl    
        
def generar_yk_primal_valido(num_landfills, num_sizes, Ckl_primal, flujo_total, p_open=1.0):
    """
    Genera una matriz yk_primal válida (vertederos abiertos y con tamaño asignado)
    que asegure capacidad >= flujo total a tratar, favoreciendo tamaños grandes.
    """
    # Pesos manuales con énfasis en tamaños grandes
    size_weights = np.array([0.1, 0.2, 0.7])
    size_weights = size_weights / size_weights.sum()

    while True:
        yk_primal = np.zeros((num_landfills, num_sizes), dtype=int)
        capacidad_acumulada = 0

        for loc in range(num_landfills):
            if np.random.rand() < p_open:
                size = np.random.choice(num_sizes, p=size_weights)
                yk_primal[loc, size] = 1
                capacidad_acumulada += Ckl_primal[size]

        if capacidad_acumulada >= flujo_total:
            return yk_primal
# Sampling para la generación de múltiples individuos
class RestrictedBinarySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        """
        Método que genera una población inicial de individuos respetando restricciones lógicas.
        """
        np.random.seed(None)  # Esto hace que cada llamada use una semilla distinta
        print(f"🚀 Generando {n_samples} muestras en Sampling")
        
        # Matriz para almacenar todos los samples
        X_samples = []
        
        while len(X_samples) < n_samples:
            # 📌 Acceder a los datos desde el problema definido
            Di = problem.Di
            Cjl, Ckl, Ckl_primal = problem.Cjl, problem.Ckl, problem.Ckl_primal
            
            # 📌 Parámetros del problema
            num_sorting, num_incinerators, num_landfills = problem.num_sorting, problem.num_incinerators, problem.num_landfills
            num_sizes, num_collection_centers = problem.num_sizes, problem.num_collection_centers
            
            small_truck_capacity, large_truck_capacity = problem.small_truck_capacity, problem.large_truck_capacity
            
            # 1. Inicialización de variables binarias y continuas
            yjl = np.zeros((num_sorting, num_sizes), dtype=int)
            ykl = np.zeros((num_incinerators, num_sizes), dtype=int)
            yk_primal = np.zeros((num_landfills, num_sizes), dtype=int)
            
            fij = np.zeros((num_collection_centers, num_sorting))
            fjk = np.zeros((num_sorting, num_incinerators))
            fjk_primal = np.zeros((num_sorting, num_landfills))
            
            xij = np.zeros((num_collection_centers, num_sorting))
            xjk = np.zeros((num_sorting, num_incinerators))
            xjk_primal = np.zeros((num_sorting, num_landfills))

            #Flujo de residuos para vertederos e incineradoras flujo para shorting es el 100% de los residuos.
            total_waste = np.sum(Di)
            flujo_incineracion = 0.5 * total_waste
            flujo_vertederos   = 0.5 * total_waste
            
            p_open = 1  # Probabilidad de apertura de instalaciones
            
            # Predeterminacion de matriz de localizaciones
            yjl = generar_yjl_valido(num_sorting, num_sizes, Cjl, Di, p_open)
            ykl = generar_ykl_valido(num_incinerators, num_sizes, Ckl, flujo_incineracion, p_open)
            yk_primal = generar_ykl_valido(num_landfills, num_sizes, Ckl_primal, flujo_vertederos, p_open)
                    
            
            # Distribución de residuos entre los centros
          

            # Capacidad total por clasificadora (según tamaño asignado)
            sorting_capacities = np.sum(yjl * Cjl, axis=1)  # shape: (num_sorting,)
            remaining_capacity = sorting_capacities.copy()  # para ir descontando lo ocupado

            # Inicialización de fij por si no está aún
            fij = np.zeros((num_collection_centers, num_sorting))

            # Distribución de residuos por centro de recolección
            for i in range(num_collection_centers):
                remaining_waste = Di[i]

                # Mientras quede residuo por repartir desde este centro
                while remaining_waste > 0:
                    # Filtramos clasificadoras abiertas y con capacidad disponible
                    open_centers = np.where((yjl.sum(axis=1) > 0) & (remaining_capacity > 0))[0]

                    if len(open_centers) == 0:
                        print(f"❌ No hay clasificadoras disponibles con capacidad para repartir la basura del centro {i}")
                        break  # Evita bucle infinito

                    # Selecciona una clasificadora aleatoria
                    selected = np.random.choice(open_centers)

                    # Determina cuánto puede aceptar
                    capacity = remaining_capacity[selected]
                    allocated = min(remaining_waste, capacity)

                    # Asignamos el residuo
                    fij[i, selected] += allocated
                    remaining_capacity[selected] -= allocated
                    remaining_waste -= allocated
            """
            print("Basura en centros de recolección",yjl)
            print("Basura total",np.sum(Di))
            print("Basura total calculada en los centros de recoleccióm", np.sum(fij,axis=0))
            print("Basura total calculada en los centros de recoleccióm", np.sum(np.sum(fij,axis=0)))
            """
             #---------------------------
            # Distribución incineradoras
            capacidades_incineradoras = np.sum(ykl * Ckl, axis=1)
            capacidad_restante_incineradoras = capacidades_incineradoras.copy()
            for j in range(num_sorting):
                if yjl[j, :].sum() > 0:
                    total_waste = fij[:, j].sum()
                    residuo_incinerar = 0.5 * total_waste

                    remaining = residuo_incinerar

                    while remaining > 0:
                        # Filtrar incineradoras abiertas y con capacidad
                        abiertas = (ykl.sum(axis=1) > 0) & (capacidad_restante_incineradoras > 0)
                        incineradoras_disponibles = np.where(abiertas)[0]

                        if len(incineradoras_disponibles) == 0:
                            print(f"⚠️ No hay incineradoras disponibles para j={j}")
                            break

                        seleccionada = np.random.choice(incineradoras_disponibles)
                        capacidad = capacidad_restante_incineradoras[seleccionada]
                        asignado = min(remaining, capacidad)

                        fjk[j, seleccionada] += asignado
                        capacidad_restante_incineradoras[seleccionada] -= asignado
                        remaining -= asignado
            """            
            print("shape", fjk.shape)
            print("Basura en incineradoras",ykl)
            print("Basura total",np.sum(Di/2))
            print("Basura total calculada en los centros de incineradora", np.sum(fjk,axis=0))
            print("Basura total calculada en los centros de incineradora", np.sum(np.sum(fjk,axis=0)))
            """
            # Distribución vertederos
                       
            capacidades_vertederos = np.sum(yk_primal * Ckl_primal, axis=1)
            capacidad_restante_vertederos = capacidades_vertederos.copy()

            # Para cada clasificadora abierta, repartir el 50% del residuo restante a vertederos
            for j in range(num_sorting):
                if yjl[j, :].sum() > 0:
                    total_waste = fij[:, j].sum()
                    residuo_vertedero = 0.5 * total_waste

                    remaining = residuo_vertedero

                    while remaining > 0:
                        abiertas = (yk_primal.sum(axis=1) > 0) & (capacidad_restante_vertederos > 0)
                        vertederos_disponibles = np.where(abiertas)[0]

                        if len(vertederos_disponibles) == 0:
                            print(f"⚠️ No hay vertederos disponibles para j={j}")
                            break

                        seleccionada = np.random.choice(vertederos_disponibles)
                        capacidad = capacidad_restante_vertederos[seleccionada]
                        asignado = min(remaining, capacidad)

                        fjk_primal[j, seleccionada] += asignado
                        capacidad_restante_vertederos[seleccionada] -= asignado
                        remaining -= asignado
            """
            print("shape", fjk_primal.shape)
            print("Basura en incineradoras",yk_primal)
            print("Basura total",np.sum(Di/2))
            print("Basura total calculada en los centros de incineradora", np.sum(fjk_primal,axis=0))
            print("Basura total calculada en los centros de incineradora", np.sum(np.sum(fjk_primal,axis=0)))
            """

            for center in range(num_collection_centers):
                for sorting in range(num_sorting):
                    xij[center, sorting] = np.ceil(fij[center, sorting] / small_truck_capacity)
            
            for sorting in range(num_sorting):
                for incinerator in range(num_incinerators):
                    xjk[sorting, incinerator] = np.ceil(fjk[sorting, incinerator] / large_truck_capacity)
                
                for landfill in range(num_landfills):
                    xjk_primal[sorting, landfill] = np.ceil(fjk_primal[sorting, landfill] / large_truck_capacity)

            # 🔥 Asegurar que las variables binarias sigan siendo binarias
            yjl = (yjl > 0).astype(int)
            ykl = (ykl > 0).astype(int)
            yk_primal = (yk_primal > 0).astype(int)


            # Convertir a vector de decisión
            X = np.hstack([
                yjl.flatten(), ykl.flatten(), yk_primal.flatten(),
                fij.flatten(), fjk.flatten(), fjk_primal.flatten(),
                xij.flatten(), xjk.flatten(), xjk_primal.flatten()
            ])
            
            X_samples.append(X)
        
        return np.array(X_samples)
