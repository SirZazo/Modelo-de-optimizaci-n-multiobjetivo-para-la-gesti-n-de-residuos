import numpy as np
import pandas as pd

class CostFunctions:

    def calcular_matriz_distancias_links(x, Ub_sorting, D):
        """
        Calcula una matriz (32x37) de distancias reales para los links activos en xij.
        
        Parámetros:
            xij         : matriz de viajes desde centro i a clasificadora j
            Ub_sorting  : matriz binaria (32x37), 1 si clasificadora j está en localización i
            D           : matriz de distancias (32x32)

        Retorna:
            dist_ij     : matriz (32x37) con distancias reales para cada link (i,j)
        """
        num_centros, num_clasificadoras = x.shape
        dist_ij = np.zeros_like(x, dtype=float)

        for i in range(num_centros):
            for j in range(num_clasificadoras):
                if x[i, j] > 0:
                    # Localización del centro i es i mismo
                    ubicacion_i = i

                    # Localización de la clasificadora j
                    ubicacion_j = np.argmax(Ub_sorting[:, j])  # columna j

                    # Distancia entre ambas localizaciones
                    dist_ij[i, j] = D[ubicacion_i, ubicacion_j]
        return dist_ij
    
    def calcular_matriz_distancias_links_facilities(x, Ub_sorting,Ub_facility,D):
        """
        Calcula una matriz (32x37) de distancias reales para los links activos en xij.
        
        Parámetros:
            xij         : matriz de viajes desde centro i a clasificadora j
            Ub_sorting  : matriz binaria (32x37), 1 si clasificadora j está en localización i
            D           : matriz de distancias (32x32)

        Retorna:
            dist_ij     : matriz (32x37) con distancias reales para cada link (i,j)
        """
        num_centros, num_clasificadoras = x.shape
        dist = np.zeros_like(x, dtype=float)

        for i in range(num_centros):
            for j in range(num_clasificadoras):
                if x[i, j] > 0:
                    # Localización del centro i es i mismo
                    ubicacion_i = np.argmax(Ub_sorting[:, i])  # columna j

                    # Localización de la clasificadora j
                    ubicacion_j = np.argmax(Ub_facility[:, j])  # columna j

                    # Distancia entre ambas localizaciones
                    dist[i, j] = D[ubicacion_i, ubicacion_j]
        return dist
    
    


    def F_c(yjl, ykl, yk_primal, xij, xjk, xjk_primal, 
            cjl, ckl, ckl_primal, tij, tjk, tjk_primal, 
            ojl, okl, ok_primal,distancias, distancias_clasificadora,
            distancias_incineradoras, distancias_vertedero):
        """ Calcula el costo total de apertura, operación y transporte. 

        print("Matriz de distancias", distancias, "\n", "Dimension de la matriz:", distancias.shape)
        print("Matriz de distancias clasificadora", distancias_clasificadora, "\n", "Dimension de la matriz:", distancias_clasificadora.shape)
        print("Matriz de distancias vertedero", distancias_vertedero, "\n", "Dimension de la matriz:", distancias_vertedero.shape)
        print("Matriz de distancias incineradora", distancias_incineradoras, "\n", "Dimension de la matriz:", distancias_incineradoras.shape)
        """
        


        F_c_yjl = np.sum(cjl * yjl, axis=1)
        F_c_ykl = np.sum(ckl * ykl, axis=1)
        F_c_ykprimal = np.sum(ckl_primal * yk_primal, axis=1)

        
        # 📌 Determinar el costo correcto basado en el tamaño activo en yjl
        selected_ojl = np.sum(yjl * ojl.T, axis=1)  # (37,)
        transport_cost_xij = tij + selected_ojl  # (37,)
        transport_cost_xij_expanded = np.expand_dims(transport_cost_xij, axis=0)  # (1, 37)

        # *****************Calculo de la matriz de distancias de los centros de recoleccion-centros de clasificacion,
       
        dist_ij = CostFunctions.calcular_matriz_distancias_links(xij,distancias_clasificadora,distancias)

        F_c_xij = np.sum(transport_cost_xij_expanded * dist_ij *xij, axis=(0, 1), keepdims=False)  # Escalar
        
        # 📌 Cálculo del costo de transporte y operación para xjk (incineradores)
        selected_okl = np.sum(ykl * okl.T, axis=1)  # (30,)from mpl_toolkits.mplot3d import Axes3D

        transport_cost_xjk = tjk + selected_okl  # (30,)
        transport_cost_xjk = transport_cost_xjk.reshape(1, -1)  # (1, 30)  # Matriz de coste de transporte

        dist_jk = CostFunctions.calcular_matriz_distancias_links_facilities(xjk,distancias_clasificadora,distancias_incineradoras,distancias) # Cálculo matriz de distancias
        F_c_xjk = np.sum(transport_cost_xjk * dist_jk * xjk, axis=(0, 1), keepdims=False)  # Escalar



        # 📌 Cálculo del costo de transporte y operación para xjk_primal (vertederos)
        selected_okprimal = np.sum(yk_primal * ok_primal.T, axis=1)  # (30,)
        transport_cost_xjk_primal = tjk_primal + selected_okprimal  # (30,)
        transport_cost_xjk_primal = transport_cost_xjk_primal.reshape(1, -1)  # (1, 30)
        dist_jk_primal = CostFunctions.calcular_matriz_distancias_links_facilities(xjk_primal,distancias_clasificadora,distancias_vertedero,distancias) # Cálculo matriz de distancias
        F_c_xjk_primal = np.sum(transport_cost_xjk_primal * dist_jk_primal * xjk_primal, axis=(0, 1), keepdims=False)  # Escalar
        
        F_c_yjl = np.sum(F_c_yjl).item()
        F_c_ykl = np.sum(F_c_ykl).item()
        F_c_ykprimal = np.sum(F_c_ykprimal).item()




        # 📌 Cálculo del costo total
        F_c = np.sum([
            F_c_xjk_primal,  # Costo transporte vertederos
            F_c_xjk,         # Costo transporte incineradores
            F_c_xij,         # Costo transporte recolección → clasificadoras
            F_c_ykprimal,    # Costo apertura y operación vertederos
            F_c_ykl,         # Costo apertura y operación incineradoras
            F_c_yjl          # Costo apertura y operación clasificadoras
        ], axis=0, keepdims=True)  # Se mantiene la dimensión correcta

        return F_c
    
    def F_u(yjl, ykl, yk_primal, Sjl, Skl, Skl_prima, num_sorting, num_incinerators, num_landfills, num_sizes):
        """
        Calcula la función objetivo de uso de tierra.

        Parámetros:
        - yjl: Matriz de activación de clasificadoras (num_sorting, num_sizes)
        - ykl: Matriz de activación de incineradoras (num_incinerators, num_sizes)
        - yk_primal: Matriz de activación de vertederos (num_landfills, num_sizes)
        - Sjl: Matriz de uso de tierra para clasificadoras
        - Skl: Matriz de uso de tierra para incineradoras
        - Skl_prima: Matriz de uso de tierra para vertederos
        - num_sorting: Número de clasificadoras
        - num_incinerators: Número de incineradoras
        - num_landfills: Número de vertederos
        - num_sizes: Número de tamaños posibles de instalaciones

        Retorna:
        - F_u: Escalar con el costo total de uso de tierra
        """

        # Cálculo del uso de tierra para cada tipo de instalación
        F_u_yjl = np.sum(Sjl.reshape(1, num_sorting, num_sizes) * yjl, axis=2)
        F_u_ykl = np.sum(Skl.reshape(1, num_incinerators, num_sizes) * ykl, axis=2)
        F_u_ykprimal = np.sum(Skl_prima.reshape(1, num_landfills, num_sizes) * yk_primal, axis=2)

        # Suma de contribuciones
        F_u = np.sum(F_u_yjl, axis=1, keepdims=True) + np.sum(F_u_ykl, axis=1, keepdims=True) + np.sum(F_u_ykprimal, axis=1, keepdims=True)

        return np.sum(F_u).item()
    
    def F_h(yjl, ykl, yk_primal, xij, xjk, xjk_primal, 
            pjl, pkl, pkl_prima, pij, pjk, pjk_prima, 
            djl, dkl, dkl_prima, dij, djk, djk_prima):
        """ Calcula el impacto en la salud de la gestión de residuos. """

        # 📌 Impacto en salud de instalaciones (clasificadoras, incineradoras, vertederos)
        F_h_yjl = np.sum(pjl * djl * yjl)  # Multiplicación escalar-matriz
        F_h_ykl = np.sum(pkl * dkl * ykl)  
        F_h_ykprimal = np.sum(pkl_prima * dkl_prima * yk_primal)  

        
        #TODO REVISAR
        # 📌 Expandir dij, djk y djk_prima a matrices compatibles usando las matrices binarias
        # Matrices de tamaño igual al número de instalaciones * los DAYLS asociados a ese tipo de intalacion
        selected_dij = np.sum(yjl * dij.T, axis=1)  # (37,)
        selected_djk = np.sum(ykl * djk.T, axis=1)  # (30,)
        selected_djk_primal = np.sum(yk_primal * djk_prima.T, axis=1)  # (30,)

        

        # 📌 Impacto en salud del transporte (recolección → clasificadoras → incineradores/vertederos)
        F_h_xij = np.sum(pij * selected_dij * xij)  
        F_h_xjk = np.sum(pjk * selected_djk * xjk)  
        F_h_xjk_primal = np.sum(pjk_prima * selected_djk_primal * xjk_primal)  

        # 📌 Sumar todos los componentes
        F_h_total = F_h_yjl + F_h_ykl + F_h_ykprimal + F_h_xij + F_h_xjk + F_h_xjk_primal

        return F_h_total
    

    