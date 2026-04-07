import numpy as np
from pymoo.core.mutation import Mutation

class CustomMutation(Mutation):
    def __init__(self, prob):
        """
        Mutación personalizada con tres tipos de mutación:
        - Mutación en las variables binarias y.
        - Mutación en las variables continuas f.
        - Mutación en las matrices fij, fjk y fjk_prima.
        
        :param prob: Probabilidad de que ocurra una mutación en general.
        :param prob_y: Probabilidad de que la mutación ocurra en y en lugar de f.
        :param prob_fij: Probabilidad de mutación en fij,fjk y fjk_prima.
    
        """
        super().__init__()
        self.prob = prob

        
 
    def _do(self, problem, x, **kwargs):
        
       # Aplica la mutación específica a cada individuo.
       # :param problem: Instancia del problema de optimización.
       # :param x: Matriz de soluciones (población actual).
       
        X_mut = np.copy(x)
        
        for i in range(x.shape[0]):
            index = 0  # Inicializar index antes de la reconstrucción
            
            # Variables binarias para la apertura de instalaciones
            yjl = X_mut[i, index:index + problem.num_sorting * problem.num_sizes].reshape(problem.num_sorting, problem.num_sizes)
            index += problem.num_sorting * problem.num_sizes

            ykl = X_mut[i, index:index + problem.num_incinerators * problem.num_sizes].reshape(problem.num_incinerators, problem.num_sizes)
            index += problem.num_incinerators * problem.num_sizes

            ykl_primal = X_mut[i, index:index + problem.num_landfills * problem.num_sizes].reshape(problem.num_landfills, problem.num_sizes)
            index += problem.num_landfills * problem.num_sizes
            
            # Variables continuas de flujo de residuos
            fij = X_mut[i, index:index + problem.num_collection_centers * problem.num_sorting].reshape(problem.num_collection_centers, problem.num_sorting)
            index += problem.num_collection_centers * problem.num_sorting

            fjk = X_mut[i, index:index + problem.num_sorting * problem.num_incinerators].reshape(problem.num_sorting, problem.num_incinerators)
            index += problem.num_sorting * problem.num_incinerators

            fjk_primal = X_mut[i, index:index + problem.num_sorting * problem.num_landfills].reshape(problem.num_sorting, problem.num_landfills)
            index += problem.num_sorting * problem.num_landfills

            # Variables enteras para el número de camiones
            xij = X_mut[i, index:index + problem.num_collection_centers * problem.num_sorting].reshape(problem.num_collection_centers, problem.num_sorting)
            index += problem.num_collection_centers * problem.num_sorting

            xjk = X_mut[i, index:index + problem.num_sorting * problem.num_incinerators].reshape(problem.num_sorting, problem.num_incinerators)
            index += problem.num_sorting * problem.num_incinerators

            xjk_primal = X_mut[i, index:index + problem.num_sorting * problem.num_landfills].reshape(problem.num_sorting, problem.num_landfills)

            print()

            # Aplicación de mutaciones
            if np.random.rand() < self.prob:
                tipo = np.random.randint(1,3)
            
                if tipo == 1:
                    #print("Mutacion sobre la apertura de instalaciones de clasificación")
                    mutacion_yjl_apertura_tamano(self,fij,fjk,fjk_primal,xij,xjk,xjk_primal,yjl,ykl,ykl_primal,problem.Cjl, problem.Di, problem, n_aperturas=2, n_tamano=2)


                if tipo == 2:

                    #MUTACIÓN SOBRE LAS INSTALACIONES DE CLASIFICACIÓN  (TRANSPASO DE RESIDUOS ENTRE CLASIFICADORAS DE MISMO CENTTRO DE RECOLECCIÓN)
                    #print("Mutación sobre el flujo de clasificadoras")
                    "PASO 1: Recorremos las localizaciones y buscamos un par de clasificadoras para realizar el traspaso de residuos"
                    """
                    * Las clasificadoras no pueden estar vacias
                    * La clasificadora que recibe residuos no puede estar llena
                    """
                    for x in range(np.random.randint(0, fij.shape[0]/10)):
                        for i in range(fij.shape[0]):  # Iteramos sobre centros de recolección
                            clasificadoras = np.where(fij[i, :] > 0)[0]  # Clasificadoras a las que ya envía residuos
                            
                        # Paso 1: Seleccion de pares de clasificadoras a las cuales se hará el traspaso de residuos
                            if len(clasificadoras) >= 2:
                                # Elegimos dos clasificadoras diferentes al azar
                                j1, j2 = np.random.choice(clasificadoras, size=2, replace=False)
                                
                                # Capacidad disponible de j2 (clasificadora receptora)
                                capacidad_total_j2 = np.sum(yjl[j2, :] * problem.Cjl)
                                carga_actual_j2 = np.sum(fij[:, j2])
                                capacidad_restante_j2 = capacidad_total_j2 - carga_actual_j2
                                
                                # Nos aseguramos que j2 pueda recibir y j1 tenga al menos 1 tonelada
                                if capacidad_restante_j2 >= 1 and fij[i, j1] >= 1:
                                    # Toneladas de residuo a traspasar
                                    max_traspaso = min(fij[i, j1], capacidad_restante_j2)
                                    cantidad = np.random.randint(1, int(max_traspaso) + 1)
                                    # Se realiza el traspaso de residuos
                                    fij[i, j1] -= cantidad
                                    fij[i, j2] += cantidad

                        # Paso 2: correción de matrices de incineradoras y vertederos 
                                    """
                                    Se plantea la correción de tal forma que no se varie la cantidad de residuos que recibe las incineradoras, sino que simplemente cambie 
                                    el origen de los residuos de esta, asi pues se recorrerá el vector correspondiente a la que traspasa y el vector de la traspasada
                                    y se realizarán los cabios oportunos sobre las incineradoras y vertederos teniendo en cuenta que la contribución es del 50% en cada una 
                                    """
                                    #print("Cantidad a traspasar", cantidad)
                                    T_parcial = cantidad/2  # Cantidad de residuos que le corresponden a las incineradoras y vertederos por separado
                                    corregir_fjk_traspaso(fjk,j1,j2,T_parcial)  # Correción sobre las incineradoras
                                    corregir_fjk_prima_traspaso(fjk_primal,j1,j2,T_parcial)

                        # Paso 3: correción de matrices de viajes para contemplar el nuevo escenario
                                    actualizar_camiones(j1, j2, fij, fjk, fjk_primal, xij, xjk, xjk_primal, problem.small_truck_capacity , problem.large_truck_capacity)
            
                if tipo == 3:
                    #print("Mutaciones sobre el flujo de vertederos e incineradoras")
                    "PASO 1: Recorremos las clasificadoras y buscamos un par de incineradoras para realizar el traspaso de residuos"
                    """
                    * Las incineradoras no pueden estar vacias
                    * La incineradora que recibe residuos no puede estar llena
                    """
                    for x in range(np.random.randint(0, fjk.shape[0]/10)):

                    # MUTACION EN INCINERADORAS  
                        for i in range(fjk.shape[0]):  # Iteramos sobre centros de clasificación
                         incineradoras = np.where(fjk[i, :] > 0)[0]  # Clasificadoras a las que ya envía residuos

                    # Paso 1: Seleccion de pares de incineradoras a las cuales se hará el traspaso de residuos
                        if len(incineradoras) >= 2:
                            # Elegimos dos incineradoras diferentes al azar
                            k1, k2 = np.random.choice(incineradoras, size=2, replace=False)
                            #print("Incineradoras seleccionadas", incineradoras)
                            #print("Par de incineradoras",k1,k2) 
                            # Capacidad disponible de k2 (incineradoras receptora)
                            capacidad_total_k2 = np.sum(ykl[k2, :] * problem.Ckl)
                            carga_actual_k2 = np.sum(fjk[:, k2])
                            capacidad_restante_k2 = capacidad_total_k2 - carga_actual_k2

                            # Nos aseguramos que k2 pueda recibir y k1 tenga al menos 1 tonelada
                            if capacidad_restante_k2 >= 1 and fjk[i, k1] >= 1:
                                # Toneladas de residuo a traspasar
                                max_traspaso = min(fjk[i, k1], capacidad_restante_k2)
                                cantidad = np.random.randint(1, int(max_traspaso) + 1)
                                #print("Estado de la las incineradoras antes...", fjk[i, k1],fjk[i, k2])
                                # Se realiza el traspaso de residuos
                                fjk[i, k1] -= cantidad
                                fjk[i, k2] += cantidad
                                #print("Cantidad traspasada", cantidad, "Estado de la las incineradoras despues...", fjk[i, k1],fjk[i, k2]) 
                        # Paso 2: Actualización del número de viajes
                                for j in [k1, k2]:
                                    for k in range(fjk.shape[1]):
                                        xjk[j, k] = np.ceil(fjk[j, k] / problem.large_truck_capacity)  
            
                    # MUTACION EN VERTEDEROS  
                        for i in range(fjk_primal.shape[0]):  # Iteramos sobre centros de clasificación
                            vertederos = np.where(fjk_primal[i, :] > 0)[0]  # Clasificadoras a las que ya envía residuos


                    # Paso 1: Seleccion de pares de vertederos a las cuales se hará el traspaso de residuos
                        if len(vertederos) >= 2:
                            # Elegimos dos vertederos diferentes al azar
                            v1, v2 = np.random.choice(vertederos, size=2, replace=False)
                            #print("Vertederos seleccionadas", vertederos) df = df.applymap(lambda x: float(str(x).replace(',', '.')))  # Convertir valores a float
                            #print("Par de Vertederos",v1,v2) 
                            # Capacidad disponible de k2 (vertedero receptora)
                            capacidad_total_v2 = np.sum(ykl_primal[v2, :] * problem.Ckl_primal)
                            carga_actual_v2 = np.sum(fjk_primal[:, v2])
                            capacidad_restante_v2 = capacidad_total_v2 - carga_actual_v2

                            # Nos aseguramos que v2 pueda recibir y v1 tenga al menos 1 tonelada
                            if capacidad_restante_v2 >= 1 and fjk_primal[i, v1] >= 1:
                                # Toneladas de residuo a traspasar
                                max_traspaso = min(fjk_primal[i, v1], capacidad_restante_v2)
                                cantidad = np.random.randint(1, int(max_traspaso) + 1)
                                #print("Estado de la las incineradoras antes...", fjk_primal[i, v1],fjk_primal[i, v2])
                                # Se realiza el traspaso de residuos
                                fjk_primal[i, v1] -= cantidad
                                fjk_primal[i, v2] += cantidad
                                #print("Cantidad traspasada", cantidad, "Estado de la los vertederos despues...", fjk_primal[i, v1],fjk_primal[i, v2]) 
                        # Paso 2: Actualización del número de viajes
                                for j in [v1, v2]:
                                    for k in range(fjk_primal.shape[1]):
                                        xjk_primal[j, k] = np.ceil(fjk_primal[j, k] / problem.large_truck_capacity)  
            
            return X_mut
        

def mutacion_yjl_apertura_tamano(self,fij,fjk,fjk_primal,xij,xjk,xjk_primal, yjl,ykl,ykl_primal ,Cjl, Di, problem, n_aperturas, n_tamano):
    num_sorting, num_sizes = yjl.shape

    # Paso 1: Generar una mutación plausible al problema
    

    clasi_ini_ca = []     # Clasificadoras que se tratatan de abrir inicialmente.
    clasi_ini_modify = [] # Clasificadoras que se modifican su tamño incialmente.
    clasi_new_ca = []     # Clasificadoras que se  abren para adaptarse al problema.
    clasi_new_modify = [] # Clasificadoras que se modifican su tamaño para adaptarse al problema.
    residuos_guardados = np.zeros(problem.num_collection_centers)  # Vector que acumula los residuos provenientes de clasificadoras cerradas (saber la localizacion de origen)
    # --------- MUTACIÓN DE APERTURA / CIERRE ---------

    n_aperturas = np.random.randint(0,num_sorting/10)
    for _ in range(n_aperturas):
        loc = np.random.randint(0, num_sorting)

        if yjl[loc, :].sum() == 0:
            # Estaba cerrado → lo abrimos con tamaño aleatorio

            size = np.random.randint(0, num_sizes)
            yjl[loc, :] = 0
            yjl[loc, size] = 1
            clasi_ini_ca.append(int(loc))
        else:
            # Estaba abierto → lo cerramos
            #print("Clasificadora cerrada ->", loc)
            yjl[loc, :] = 0

            # Guardamos la el contenido del vector en cada caso para saber de donde provienen los residuos eliminados
            residuos_guardados += fij[:, loc].copy()
            # Poner a cero las matrices relacionadas
            fij[:, loc] = 0
            fjk[loc, :] = 0
            fjk_primal[loc, :] = 0
            xij[:, loc] = 0
            xjk[loc, :] = 0
            xjk_primal[loc, :] = 0
         

    # --------- MUTACIÓN DE TAMAÑO ---------
    centros_abiertos = [i for i in range(num_sorting) if yjl[i, :].sum() > 0]

    n_tamano = np.random.randint(0,num_sorting/10)

    for _ in range(n_tamano):
        if not centros_abiertos:
            break

        loc = np.random.choice(centros_abiertos)
        clasi_ini_modify.append(int(loc))  # Almacenamos en el eje 0 las clasificadoras modificadas
        size_actual = np.argmax(yjl[loc, :])
        nuevos = [s for s in range(num_sizes) if s != size_actual]

        if nuevos:
            nuevo_size = np.random.choice(nuevos)
            yjl[loc, :] = 0
            yjl[loc, nuevo_size] = 1

            # En el caso de que la variación de tamaño haya sido menor debemos hacer una comprobacion sobre la capacidad y anotar
            # los residuos sobrantes en vector residuos_guardados

            # Comprobamos si la nueva capacidad es menor a la anterior
            capacidad_nueva = Cjl[nuevo_size]
            residuos_actuales = np.sum(fij[:, loc])

            if residuos_actuales > capacidad_nueva:
                exceso = residuos_actuales - capacidad_nueva
                # Guardamos el exceso para redistribuir
                sobrante = fij[:, loc].copy()

                # Reducimos proporcionalmente el contenido de la columna hasta la nueva capacidad
                if np.sum(sobrante) > 0:
                    proporciones = sobrante / np.sum(sobrante)
                    nueva_asignacion = proporciones * capacidad_nueva
                    fij[:, loc] = nueva_asignacion

                    # Lo que se ha perdido en la reasignación es el residuo sobrante
                    nuevo_total = np.sum(fij[:, loc])
                    exceso_real = residuos_actuales - nuevo_total

                    # Acumulamos el exceso en residuos_a_repartir
                    residuos_guardados += sobrante - fij[:, loc]
            redistribuir_residuos_incineradoras_vertederos(loc, fjk, fjk_primal,ykl,ykl_primal, problem.Ckl, problem.Ckl_primal)



    # --------- COMPROBACIÓN DE CAPACIDAD ---------
    capacidad_actual = np.sum(yjl * Cjl)
    demanda_total = np.sum(problem.Di)

    # Coreción de la mutacion si resuelta que el individuo no tiene la capacidad suficiente
    if capacidad_actual < demanda_total:
        #print("Capacidad insuficiente (clasificadoras): ", capacidad_actual - demanda_total)
        # Correción de la mutación
        while capacidad_actual < demanda_total:
            accion = np.random.choice(["abrir", "ampliar"])
            
            if accion == "abrir":
                # Buscar clasificadoras cerradas
                cerradas = [i for i in range(yjl.shape[0]) if yjl[i, :].sum() == 0]
                if cerradas:
                    loc = np.random.choice(cerradas)
                    clasi_new_ca.append(int(loc))
                    nuevo_size = np.random.choice(num_sizes)
                    yjl[loc, nuevo_size] = 1
                    capacidad_actual += Cjl[nuevo_size]

            elif accion == "ampliar":
                abiertas = [i for i in range(yjl.shape[0]) if yjl[i, :].sum() == 1]
                if abiertas:
                    loc = np.random.choice(abiertas)
                    size_actual = np.argmax(yjl[loc, :])
                    if size_actual < num_sizes - 1:  # aún se puede ampliar
                        clasi_new_modify.append(int(loc))
                        yjl[loc, size_actual] = 0
                        yjl[loc, size_actual + 1] = 1
                        capacidad_actual += Cjl[size_actual + 1] - Cjl[size_actual]
        #print("Capacidad alcanzada con un exceso de: (clasificadoras): ", capacidad_actual - demanda_total)

    # Paso 2: Corregir el individuo en función de los valores modificados.    
    
    # Cálculo del total de basura restante a distribuir
    #print("Basura a procesar", np.sum(problem.Di))
    #print("Basura actualmente almacenada en clasificadoras", np.sum(fij))
    #print("Basura actualmente almacenada en las incineradoras", np.sum(fjk))
    #print("Basura actualmente almacenada en los vertederos", np.sum(fjk_primal))



    residuos_restantes = np.sum(problem.Di) - np.sum(fij)
    #print("Basura restante", residuos_restantes)
    #print("Basura guardad", residuos_guardados.shape , np.sum(residuos_guardados))

    # Repartición de los residuos restantes
    fij = distribuir_residuos_guardados(fij, residuos_guardados, yjl, Cjl, clasi_ini_ca, clasi_ini_modify,clasi_new_ca, clasi_new_modify)
    
    
    clasificadoras_afectadas = set(clasi_ini_ca + clasi_ini_modify + clasi_new_ca + clasi_new_modify)
    
    # Reajuste de fjk y fjk_primal
    for j in clasificadoras_afectadas:
        distribucion_residuos_clasificadora(j, fij, fjk, fjk_primal, ykl, ykl_primal, problem.Ckl, problem.Ckl_primal)

    actualizar_camiones_post_mutacion(fij, fjk, fjk_primal, xij, xjk, xjk_primal,
                                   problem.small_truck_capacity, problem.large_truck_capacity)

    
    #print("Basura actualmente almacenada en clasificadoras final", np.sum(fij))
    #print("Basura actualmente almacenada en las incineradoras final", np.sum(fjk))
    #print("Basura actualmente almacenada en los vertederos final", np.sum(fjk_primal))




def corregir_fjk_traspaso(fjk, j1, j2, T_inc):
    traspasado = 0
    intentos = 0
    max_intentos = 200  # seguridad para evitar bucles infinitos

    while traspasado < T_inc and intentos < max_intentos:
        intentos += 1

        # Incineradoras donde j1 tiene flujo positivo
        incineradoras_j1 = np.where(fjk[j1, :] > 0)[0]
      

        if len(incineradoras_j1) == 0:
            break

        k = np.random.choice(incineradoras_j1)
        disponible = fjk[j1, k]
        pendiente = T_inc - traspasado
        a_traspasar = min(disponible, pendiente)


        #print("Valores iniciale de inicineradora j1", fjk[j1, k])
        #print("Valores iniciale de inicineradora j2", fjk[j2, k])

        # Traspaso•••••••••••••
        fjk[j1, k] -= a_traspasar
        fjk[j2, k] += a_traspasar
        traspasado += a_traspasar

        #print("Valores finales de inicineradora j1", fjk[j1, k])
        #print("Valores finales de inicineradora j2", fjk[j2, k])
        #print("Transpaso incineradoras ",intentos, "nº de toneladas traspasadas ",a_traspasar)

def corregir_fjk_prima_traspaso(fjk_prima, j1, j2, T_land):
    traspasado = 0
    intentos = 0
    max_intentos = 1000

    while traspasado < T_land and intentos < max_intentos:
        intentos += 1

        vertederos_j1 = np.where(fjk_prima[j1, :] > 0)[0]
        if len(vertederos_j1) == 0:
            break

        k = np.random.choice(vertederos_j1)
        disponible = fjk_prima[j1, k]
        pendiente = T_land - traspasado
        a_traspasar = min(disponible, pendiente)

        #print("Valores iniciale de vertedero j1", fjk_prima[j1, k])
        #print("Valores iniciale de vertedero j2", fjk_prima[j2, k])

        fjk_prima[j1, k] -= a_traspasar
        fjk_prima[j2, k] += a_traspasar
        traspasado += a_traspasar

        #print("Valores finales de vertedero j1", fjk_prima[j1, k])
        #print("Valores finales de vertedero j2", fjk_prima[j2, k])
        #print("Transpaso vertederos ",intentos, "nº de toneladas traspasadas ",a_traspasar)



def actualizar_camiones(j1, j2, fij, fjk, fjk_prima, xij, xjk, xjk_prima, small_cap, large_cap):
    for i in range(fij.shape[0]):
        for j in [j1, j2]:
            xij[i, j] = np.ceil(fij[i, j] / small_cap)

    for j in [j1, j2]:
        for k in range(fjk.shape[1]):
            xjk[j, k] = np.ceil(fjk[j, k] / large_cap)
        for k in range(fjk_prima.shape[1]):
            xjk_prima[j, k] = np.ceil(fjk_prima[j, k] / large_cap)



def distribucion_residuos_clasificadora(j, fij, fjk, fjk_primal, ykl, yk_primal, Ckl, Ckl_primal):
    """
    Reasigna los residuos de la clasificadora j a incineradoras y vertederos.
    Distribuye el 50% a incineradoras y el 50% a vertederos, respetando capacidades.
    """
    total_waste = fij[:, j].sum()
    incineracion = 0.5 * total_waste
    vertedero = 0.5 * total_waste

    # Calcular capacidades disponibles
    capacidad_restante_inc = (ykl * Ckl).sum(axis=1) - fjk.sum(axis=0)
    capacidad_restante_prim = (yk_primal * Ckl_primal).sum(axis=1) - fjk_primal.sum(axis=0)

    # Corrección: evitar negativos si ya hay exceso
    capacidad_restante_inc = np.clip(capacidad_restante_inc, 0, None)
    capacidad_restante_prim = np.clip(capacidad_restante_prim, 0, None)

    # Reparto a incineradoras
    while incineracion > 1e-6:
        abiertas = np.where((ykl.sum(axis=1) > 0) & (capacidad_restante_inc > 0))[0]
        if len(abiertas) == 0:
            print(f"⚠️ No hay incineradoras disponibles para clasificadora {j}")
            break

        seleccionada = np.random.choice(abiertas)
        capacidad = capacidad_restante_inc[seleccionada]
        asignado = min(incineracion, capacidad)
        fjk[j, seleccionada] += asignado
        capacidad_restante_inc[seleccionada] -= asignado
        incineracion -= asignado

    # Reparto a vertederos
    while vertedero > 1e-6:
        abiertas = np.where((yk_primal.sum(axis=1) > 0) & (capacidad_restante_prim > 0))[0]
        if len(abiertas) == 0:
            print(f"⚠️ No hay vertederos disponibles para clasificadora {j}")
            break

        seleccionada = np.random.choice(abiertas)
        capacidad = capacidad_restante_prim[seleccionada]
        asignado = min(vertedero, capacidad)
        fjk_primal[j, seleccionada] += asignado
        capacidad_restante_prim[seleccionada] -= asignado
        vertedero -= asignado



def redistribuir_residuos_incineradoras_vertederos(j, fjk, fjk_primal, ykl, yk_primal, Ckl, Ckl_primal):
    # 1. Obtener residuos actuales
    residuos_incineradora = np.sum(fjk[j, :])
    residuos_vertedero = np.sum(fjk_primal[j, :])
    
    total_residuo = residuos_incineradora + residuos_vertedero
    
    # 2. Reiniciar los flujos
    fjk[j, :] = 0
    fjk_primal[j, :] = 0

    # 3. Calcular nueva distribución: 50% a cada uno
    nuevo_residuo_incineradora = 0.5 * total_residuo
    nuevo_residuo_vertedero = 0.5 * total_residuo

    # 4. Capacidad disponible por incineradora y vertedero
    capacidad_inc = np.sum(ykl * Ckl, axis=1)
    capacidad_disponible_inc = capacidad_inc - np.sum(fjk, axis=0)
    abiertas_inc = np.where(capacidad_disponible_inc > 0)[0]

    capacidad_ver = np.sum(yk_primal * Ckl_primal, axis=1)
    capacidad_disponible_ver = capacidad_ver - np.sum(fjk_primal, axis=0)
    abiertas_ver = np.where(capacidad_disponible_ver > 0)[0]

    # 5. Redistribuir aleatoriamente (simple)
    while nuevo_residuo_incineradora > 0 and len(abiertas_inc) > 0:
        k = np.random.choice(abiertas_inc)
        asignar = min(nuevo_residuo_incineradora, capacidad_disponible_inc[k])
        fjk[j, k] += asignar
        nuevo_residuo_incineradora -= asignar
        capacidad_disponible_inc[k] -= asignar
        if capacidad_disponible_inc[k] <= 0:
            abiertas_inc = abiertas_inc[abiertas_inc != k]

    while nuevo_residuo_vertedero > 0 and len(abiertas_ver) > 0:
        k = np.random.choice(abiertas_ver)
        asignar = min(nuevo_residuo_vertedero, capacidad_disponible_ver[k])
        fjk_primal[j, k] += asignar
        nuevo_residuo_vertedero -= asignar
        capacidad_disponible_ver[k] -= asignar
        if capacidad_disponible_ver[k] <= 0:
            abiertas_ver = abiertas_ver[abiertas_ver != k]


def distribuir_residuos_guardados(fij, residuos_guardados, yjl, Cjl,
                                   clasi_ini_ca, clasi_ini_modify,
                                   clasi_new_ca, clasi_new_modify):
    """
    Distribuye los residuos acumulados en 'residuos_guardados' hacia las clasificadoras activas,
    siguiendo una estrategia en dos fases:
    1. Prioriza las clasificadoras modificadas.
    2. Si aún queda residuo por asignar, lo reparte entre cualquier clasificadora abierta con capacidad.
    """

    num_centros, num_clasificadoras = fij.shape

    # Clasificadoras candidatas ordenadas por prioridad
    clasificadoras_prioritarias = clasi_ini_ca + clasi_ini_modify + clasi_new_ca + clasi_new_modify

    # Quitar duplicados y filtrar solo abiertas
    clasificadoras_prioritarias = list({j for j in clasificadoras_prioritarias if yjl[j].sum() > 0})

    # Capacidad total y capacidad restante por clasificadora
    capacidad_total = np.sum(yjl * Cjl, axis=1)  # shape (num_clasificadoras,)
    residuos_actuales = np.sum(fij, axis=0)
    capacidad_restante = capacidad_total - residuos_actuales

    # Paso por cada centro de recolección
    for i in range(num_centros):
        residuo = residuos_guardados[i]

        # --- Fase 1: intentar con clasificadoras prioritarias ---
        for j in clasificadoras_prioritarias:
            if residuo <= 0:
                break
            if capacidad_restante[j] <= 0:
                continue

            asignar = min(residuo, capacidad_restante[j])
            fij[i, j] += asignar
            capacidad_restante[j] -= asignar
            residuo -= asignar

        # --- Fase 2: repartir el sobrante a cualquier clasificadora abierta con capacidad ---
        if residuo > 0:
            abiertas = [j for j in range(num_clasificadoras) if yjl[j].sum() > 0 and capacidad_restante[j] > 0]
            while residuo > 0 and abiertas:
                j = np.random.choice(abiertas)
                asignar = min(residuo, capacidad_restante[j])
                fij[i, j] += asignar
                capacidad_restante[j] -= asignar
                residuo -= asignar
                abiertas = [j for j in abiertas if capacidad_restante[j] > 0]

        residuos_guardados[i] = residuo  # Si quedó sin asignar, se mantiene

    residuos_totales_guardados = np.sum(residuos_guardados)
    if residuos_totales_guardados > 1e-6:
        print(f"⚠️ Atención: {residuos_totales_guardados} toneladas de residuos no se han podido asignar")


    return fij


def actualizar_camiones_post_mutacion(fij, fjk, fjk_primal, xij, xjk, xjk_primal,
                                      small_truck_capacity, large_truck_capacity):
    """
    Recalcula las matrices xij, xjk, xjk_primal después de modificar fij, fjk y fjk_primal.
    """
    # Recalcular xij: de centros de recolección a clasificadoras
    xij[:, :] = np.ceil(fij / small_truck_capacity)

    # Recalcular xjk: de clasificadoras a incineradoras
    xjk[:, :] = np.ceil(fjk / large_truck_capacity)

    # Recalcular xjk_primal: de clasificadoras a vertederos
    xjk_primal[:, :] = np.ceil(fjk_primal / large_truck_capacity)
