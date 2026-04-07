import numpy as np
from pymoo.core.problem import ElementwiseProblem
from funciones import CostFunctions

class WasteManagementProblem(ElementwiseProblem):
    def __init__(self, num_sorting, num_incinerators, num_landfills, num_collection_centers, num_sizes,
                 Di, Cjl, Ckl, Ckl_primal, cjl, ckl, ckl_primal, ojl, okl, ok_primal,
                 Sjl, Skl, Skl_prima, pij, pjk, pjk_prima,pjl,pkl,pkl_prima,dij,djk,djk_prima,djl,dkl,dkl_prima,distancias,distancias_clasificadora,distancias_incineradoras,distancias_vertedero, 
                 small_truck_capacity=16, large_truck_capacity=32, tij=0.16, tjk=0.25, tjk_primal=0.25, modo_objetivos = "triobjetivo", 
                 ):

        # Definir parámetros del problema
        self.num_sorting = num_sorting
        self.num_incinerators = num_incinerators
        self.num_landfills = num_landfills
        self.num_collection_centers = num_collection_centers
        self.num_sizes = num_sizes

        self.Di = Di  # Demanda de residuos     distancias_clasificadora=distancias_clasificadora,
        distancias_incineradoras=distancias_incineradoras
        distancias_vertedero=distancias_vertedero

        # Capacidades de Instalaciones
        self.Cjl = Cjl
        self.Ckl = Ckl
        self.Ckl_primal = Ckl_primal
        
        # Costos de Apertura
        self.cjl = cjl
        self.ckl = ckl
        self.ckl_primal = ckl_primal
        
        # Costos Operacionales
        self.ojl = ojl
        self.okl = okl
        self.ok_primal = ok_primal
        
        # Costos de transporte
        self.tij = tij
        self.tjk = tjk
        self.tjk_primal = tjk_primal

        # Matriz de uso de terreno
        self.Sjl = Sjl
        self.Skl = Skl
        self.Skl_prima = Skl_prima

        # Matriz de número de personas por links
        self.pij = pij
        self.pjk = pjk
        self.pjk_prima = pjk_prima

        # Matriz de número de personas por ubicación
        self.pjl = pjl
        self.pkl = pkl
        self.pkl_prima = pkl_prima

        # DAYLS instalaciones
        self.dij=dij
        self.djk=djk
        self.djk_prima=djk_prima

        # DALYS transporte
        self.djl=djl
        self.dkl=dkl 
        self.dkl_prima=dkl_prima

        # Capacidades de los camiones 
        self.small_truck_capacity = small_truck_capacity
        self.large_truck_capacity = large_truck_capacity

        self.modo_objetivos = modo_objetivos

        # Matriz de distancias entre distritos
        self.distancias=distancias

        # Localizaciones instalaciones
        self.distancias_clasificadora=distancias_clasificadora
        self.distancias_incineradoras=distancias_incineradoras
        self.distancias_vertedero=distancias_vertedero

        # Modos de objetivos
        modo = self.modo_objetivos
        if modo == "triobjetivo":
            n_obj = 3
        elif modo in ["economico-uso", "economico-salud", "salud-uso"]:
            n_obj = 2
        
        elif modo in ["economico","uso","salud"]:
            n_obj = 1
        else:
            raise ValueError("Modo de objetivos no válido")

        print("🌟 Número de objetivos: ", n_obj)
        # Definir el número de variables en el vector de decisión
        n_var = (num_sorting * num_sizes + 
                 num_incinerators * num_sizes + 
                 num_landfills * num_sizes +
                 num_collection_centers * num_sorting +
                 num_sorting * num_incinerators +
                 num_sorting * num_landfills +
                 num_collection_centers * num_sorting + 
                 num_sorting * num_incinerators +
                 num_sorting * num_landfills)

        # Inicialización de la clase base con múltiples objetivos
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=7, xl=0, xu=1)


    def _evaluate(self, x, out, *args, **kwargs):


        index = 0
        
        # 📌 Variables binarias para la apertura de instalaciones
        yjl = x[index:index + self.num_sorting * self.num_sizes].reshape(self.num_sorting, self.num_sizes)
        index += self.num_sorting * self.num_sizes

        ykl = x[index:index + self.num_incinerators * self.num_sizes].reshape(self.num_incinerators, self.num_sizes)
        index += self.num_incinerators * self.num_sizes

        yk_primal = x[index:index + self.num_landfills * self.num_sizes].reshape(self.num_landfills, self.num_sizes)
        index += self.num_landfills * self.num_sizes

        # 📌 Variables continuas de flujo de residuos
        fij = x[index:index + self.num_collection_centers * self.num_sorting].reshape(self.num_collection_centers, self.num_sorting)
        index += self.num_collection_centers * self.num_sorting

        fjk = x[index:index + self.num_sorting * self.num_incinerators].reshape(self.num_sorting, self.num_incinerators)
        index += self.num_sorting * self.num_incinerators

        fjk_primal = x[index:index + self.num_sorting * self.num_landfills].reshape(self.num_sorting, self.num_landfills)
        index += self.num_sorting * self.num_landfills

        # 📌 Variables enteras para el número de camiones
        xij = x[index:index + self.num_collection_centers * self.num_sorting].reshape(self.num_collection_centers, self.num_sorting)
        index += self.num_collection_centers * self.num_sorting

        xjk = x[index:index + self.num_sorting * self.num_incinerators].reshape(self.num_sorting, self.num_incinerators)
        index += self.num_sorting * self.num_incinerators

        xjk_primal = x[index:index + self.num_sorting * self.num_landfills].reshape(self.num_sorting, self.num_landfills)
        

    # FUNCIÓN OBJETICO USO DE TIERRA   
        F_u = CostFunctions.F_u(yjl, ykl, yk_primal, self.Sjl, self.Skl, self.Skl_prima, 
                self.num_sorting, self.num_incinerators, self.num_landfills, self.num_sizes)
        
        
    #FUNCION OBJETIVO IMPACTO EN LA SALUD
        F_h = CostFunctions.F_h(
            yjl, ykl, yk_primal, xij, xjk, xjk_primal, 
            self.pjl, self.pkl, self.pkl_prima, 
            self.pij, self.pjk, self.pjk_prima, 
            self.djl, self.dkl, self.dkl_prima, 
            self.dij, self.djk, self.djk_prima
        )
        

    #FUNCION OBJETIVO SALUD POBLACIÓN

        F_c = CostFunctions.F_c(yjl, ykl, yk_primal, xij, xjk, xjk_primal, 
                        self.cjl, self.ckl, self.ckl_primal, 
                        self.tij, self.tjk, self.tjk_primal, 
                        self.ojl, self.okl, self.ok_primal, self.distancias,
                        self.distancias_clasificadora,self.distancias_incineradoras,self.distancias_vertedero)
        

        # 📌 RESTRICCIONES

        # Restricción 1: ∑𝑓𝑖𝑗 = 𝐷𝑖 (Toda la basura generada en i debe ser transportada completamente a j)
        g_fij = np.linalg.norm(np.sum(fij, axis=1) - self.Di).item()  # Convierte a escalar

        # Restricción 2: yjl, ykl, yk_primal deben ser binarios (0 o 1)
        g_bin_yjl = np.sum(np.abs(yjl - np.round(yjl))).item()  # Convierte a escalar
        g_bin_ykl = np.sum(np.abs(ykl - np.round(ykl))).item()
        g_bin_ykprimal = np.sum(np.abs(yk_primal - np.round(yk_primal))).item()

        # Restricción 3: xij, xjk, xjk_primal deben ser enteros positivos
        g_xij = np.sum(np.abs(xij - np.round(xij)) + (xij < 0)).item()
        g_xjk = np.sum(np.abs(xjk - np.round(xjk)) + (xjk < 0)).item()
        g_xjkprimal = np.sum(np.abs(xjk_primal - np.round(xjk_primal)) + (xjk_primal < 0)).item()


        # Outputo de los objetivos
        modo = self.modo_objetivos
        # TRIOBJETIVO
        if modo == "triobjetivo":
            out["F"] = [float(F_c), float(F_u), float(F_h)]
        # BIOBJETIVO
        elif modo == "economico-uso":
            out["F"] = [float(F_c), float(F_u)]
        elif modo == "economico-salud":
            out["F"] = [float(F_c), float(F_h)]
        elif modo == "salud-uso":
            out["F"] = [float(F_h), float(F_u)]
        #MONOJETIVO
        elif modo == "economico":
            out["F"] = [float(F_c)]
        elif modo == "uso":
            out["F"] = [float(F_u)]
        elif modo == "salud":
            out["F"] = [float(F_h)]        


        
        # 📌 Guardamos en `out`
        #out["F"] = [float(F_c), float(F_u), float(F_h)]  # Convierte ambos a escalares 
        out["G"] = [g_fij, g_bin_yjl, g_bin_ykl, g_bin_ykprimal, g_xij, g_xjk, g_xjkprimal]  # Restricciones en forma correcta


        # EVALUATE 3

    def _evaluate3(self, x):
        index = 0
        
        # 📌 Variables binarias para la apertura de instalaciones
        yjl = x[index:index + self.num_sorting * self.num_sizes].reshape(self.num_sorting, self.num_sizes)
        index += self.num_sorting * self.num_sizes
        ykl = x[index:index + self.num_incinerators * self.num_sizes].reshape(self.num_incinerators, self.num_sizes)
        index += self.num_incinerators * self.num_sizes
        yk_primal = x[index:index + self.num_landfills * self.num_sizes].reshape(self.num_landfills, self.num_sizes)
        index += self.num_landfills * self.num_sizes
        # 📌 Variables continuas de flujo de residuos
        fij = x[index:index + self.num_collection_centers * self.num_sorting].reshape(self.num_collection_centers, self.num_sorting)
        index += self.num_collection_centers * self.num_sorting
        fjk = x[index:index + self.num_sorting * self.num_incinerators].reshape(self.num_sorting, self.num_incinerators)
        index += self.num_sorting * self.num_incinerators
        fjk_primal = x[index:index + self.num_sorting * self.num_landfills].reshape(self.num_sorting, self.num_landfills)
        index += self.num_sorting * self.num_landfills
        # 📌 Variables enteras para el número de camiones
        xij = x[index:index + self.num_collection_centers * self.num_sorting].reshape(self.num_collection_centers, self.num_sorting)
        index += self.num_collection_centers * self.num_sorting
        xjk = x[index:index + self.num_sorting * self.num_incinerators].reshape(self.num_sorting, self.num_incinerators)
        index += self.num_sorting * self.num_incinerators
        xjk_primal = x[index:index + self.num_sorting * self.num_landfills].reshape(self.num_sorting, self.num_landfills)
        
    # FUNCIÓN OBJETICO USO DE TIERRA   
        F_u = CostFunctions.F_u(yjl, ykl, yk_primal, self.Sjl, self.Skl, self.Skl_prima, 
                self.num_sorting, self.num_incinerators, self.num_landfills, self.num_sizes)
        
        
    #FUNCION OBJETIVO IMPACTO EN LA SALUD
        F_h = CostFunctions.F_h(
            yjl, ykl, yk_primal, xij, xjk, xjk_primal, 
            self.pjl, self.pkl, self.pkl_prima, 
            self.pij, self.pjk, self.pjk_prima, 
            self.djl, self.dkl, self.dkl_prima, 
            self.dij, self.djk, self.djk_prima
        )
        
    #FUNCION OBJETIVO SALUD POBLACIÓN
        F_c = CostFunctions.F_c(yjl, ykl, yk_primal, xij, xjk, xjk_primal, 
                        self.cjl, self.ckl, self.ckl_primal, 
                        self.tij, self.tjk, self.tjk_primal, 
                        self.ojl, self.okl, self.ok_primal, self.distancias,
                        self.distancias_clasificadora,self.distancias_incineradoras,self.distancias_vertedero)
        

        return [float(F_c), float(F_u), float(F_h)]



    
