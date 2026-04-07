import numpy as np
import pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.soo.nonconvex.ga import GA

from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.individual import Individual
from data import read_waste_data, read_facility_data, read_scaling_factors,read_people_file,read_people_facility_file, read_facility_dalys, leer_matriz_distancias,leer_matriz_localizacion, exportar_datos

from sampling import RestrictedBinarySampling
from pymoo.operators.crossover.sbx import SBX
from mutation import CustomMutation
from crossover import NoCrossover
import argparse
from show import ShowResults
from tracker import ConvergenceTracker
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting



# Lectura de datos
Di_path = "WMP_v04.00/Data/Di.csv"
Di = read_waste_data(Di_path)

facility_file = "WMP_v04.00/Data/costos_facilities.csv"
Cjl, Ckl, Ckl_primal, cjl, ckl, ckl_primal, ojl, okl, ok_primal = read_facility_data(facility_file)


scaling_files = "WMP_v04.00/Data/Sjl.csv", "WMP_v04.00/Data/Skl.csv", "WMP_v04.00/Data/Skl_prima.csv"
Sjl, Skl, Skl_prima = read_scaling_factors(scaling_files[0],scaling_files[1],scaling_files[2])

people_file = "WMP_v04.00/Data/pij.csv","WMP_v04.00/Data/pjk.csv","WMP_v04.00/Data/pjk_prima.csv"  # Ruta al archivo CSV
pij,pjk,pjk_prima = read_people_file(people_file[0],people_file[1],people_file[2])

people_facility_file = "WMP_v04.00/Data/pjl.csv","WMP_v04.00/Data/pkl.csv","WMP_v04.00/Data/pkl_prima.csv"  # Ruta al archivo CSV
pjl,pkl,pkl_prima = read_people_facility_file(people_facility_file[0],people_facility_file[1],people_facility_file[2])

dayls_file = "WMP_v04.00/Data/DALYs_facility.csv"  # Cambia esto con la ruta real del archivo
dij, djk, djk_prima, djl, dkl, dkl_prima = read_facility_dalys(dayls_file)

distance_file  = "WMP_v04.00/Data/plantilla_matriz_distancias.csv"
distancias = leer_matriz_distancias(distance_file)

# Matrices binarias de localizacion de incineradoras clasificadoras y vertederos

distance_clasificadora_file  = "WMP_v04.00/Data/binaria_clasificadoras.csv"
distancias_clasificadora = leer_matriz_localizacion(distance_clasificadora_file)

distance_vertedero_file  = "WMP_v04.00/Data/binaria_vertederos.csv"
distancias_vertedero = leer_matriz_localizacion(distance_vertedero_file)

distance_incineradora_file  = "WMP_v04.00/Data/binaria_incineradoras.csv"
distancias_incineradoras = leer_matriz_localizacion(distance_incineradora_file)

num_sizes = 3

# Parseador de argumentos por consola
parser = argparse.ArgumentParser(description="Selecciona el modo de objetivos para la optimización")

parser.add_argument(
    "--modo_objetivos",
    type=str,
    default="triobjetivo",
    choices=["triobjetivo", "economico-uso", "economico-salud", "salud-uso","economico","uso","salud"],
    help="Modo de objetivos a optimizar: triobjetivo, economico-uso, economico-salud o salud-uso"
)

parser.add_argument(
    "--algoritmo",
    type=str,
    default="nsga2",
    choices=["nsga2", "nsga3", "moead","ga"],
    help="Algoritmo a utilizar: nsga2, nsga3, ga o moead"
)


parser.add_argument(
    "--generaciones",
    type=int,
    default= 100,
    help='Numero de generaciones a ejecutar'
)

parser.add_argument(
    "--nombre",
    type=str,
    default="prueba",
    help="Nombre de la ejecución a guardar los datos"
)

args = parser.parse_args()

# Aquí puedes imprimir para verificar
print(f"🔧 Modo de objetivos seleccionado: {args.modo_objetivos}")

# Importacion del tipo de problem en especifico para cada algoritmo
if args.algoritmo == "moead":
    from problemMOEAD import WasteManagementProblem
else:
    from problem import WasteManagementProblem

# Crear instancia del problema
problem = WasteManagementProblem(
    num_sorting=37,
    num_incinerators=30,
    num_landfills=30,
    num_collection_centers=32,
    num_sizes=num_sizes,
    Cjl=Cjl,
    Ckl=Ckl,
    Ckl_primal=Ckl_primal,
    ojl=ojl,
    okl=okl,
    ok_primal=ok_primal,
    Di=Di,
    cjl=cjl,
    ckl=ckl,
    ckl_primal=ckl_primal,
    Sjl=Sjl, 
    Skl=Skl,
    Skl_prima=Skl_prima,
    pij=pij,
    pjk=pjk,
    pjk_prima=pjk_prima,
    pjl=pjl,  
    pkl=pkl,  
    pkl_prima=pkl_prima,  
    dij=dij,
    djk=djk,
    djk_prima=djk_prima,
    djl=djl,                
    dkl=dkl,               
    dkl_prima=dkl_prima,
    modo_objetivos=args.modo_objetivos,
    distancias=distancias,
    distancias_clasificadora=distancias_clasificadora ,
    distancias_incineradoras=distancias_incineradoras ,
    distancias_vertedero=distancias_vertedero
)






# Crear el algoritmo dinámicamente
if args.algoritmo == "nsga2":
    algorithm = NSGA2(
        pop_size=100,
        sampling=RestrictedBinarySampling(),
        mutation=CustomMutation(prob=1),
        crossover=NoCrossover(),
        eliminate_duplicates=False,
    )

elif args.algoritmo == "nsga3":
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
    algorithm = NSGA3(
        pop_size=100,
        ref_dirs=ref_dirs,
        sampling=RestrictedBinarySampling(),
        mutation=CustomMutation(prob=1),
        crossover=NoCrossover(),
        eliminate_duplicates=False,
    )

elif args.algoritmo == "moead":
    ref_dirs = get_reference_directions("uniform", problem.n_obj, n_partitions=12)
    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=4,
        sampling=RestrictedBinarySampling(),
        mutation=CustomMutation(prob=1),
        crossover=NoCrossover(),
        
    )

elif args.algoritmo == "ga":
    algorithm = GA(
        pop_size=20,
        sampling=RestrictedBinarySampling(),
        mutation=CustomMutation(prob=1),
        crossover=NoCrossover(),
        eliminate_duplicates=True
    )

# Definir la terminación
termination = get_termination("n_gen", args.generaciones)
ruta = ""
tracker = ConvergenceTracker(problem, args.modo_objetivos, args.algoritmo, args.nombre)
# Ejecutar la optimización
res = minimize(
    problem = problem,
    algorithm=algorithm,
    termination = termination,
    callback = tracker,
    seed=None,
    verbose= True
)


#valores del tracker
print(f"🧠 Mejores hasta ahora: Costo={tracker.best_vals[0]:.4f}, Uso={tracker.best_vals[1]:.4f}, Salud={tracker.best_vals[2]:.4f}")
print(f"⚠️ Peores hasta ahora: Costo={tracker.worst_vals[0]:.4f}, Uso={tracker.worst_vals[1]:.4f}, Salud={tracker.worst_vals[2]:.4f}")

X = res.pop.get("X")
F = res.pop.get("F")





modo = args.modo_objetivos

if modo == "triobjetivo":
   idx_equilibrado = ShowResults.show_pareto_3d(F, X ,tracker = tracker)
   print()
elif modo in ["economico-uso", "economico-salud", "salud-uso"]:
    if modo == "economico-uso":
        idx_equilibrado = ShowResults.show_pareto_2d(res.pop.get("F"), X , tracker, problem, nombres_objetivos=["Coste Económico", "Uso del Terreno"], objetivo = modo)

    elif modo == "economico-salud":
        idx_equilibrado = ShowResults.show_pareto_2d(res.pop.get("F"), X , tracker, problem, nombres_objetivos=["Coste Económico", "Impacto sobre la Salud"], objetivo = modo)

    elif modo == "salud-uso":
        idx_equilibrado = ShowResults.show_pareto_2d(res.pop.get("F"), X , tracker, problem, nombres_objetivos=["Uso del Terreno", "Impacto sobre la Salud"], objetivo = modo)

else:
    raise ValueError("Modo de objetivos no válido")

ShowResults.showHipervolumen(tracker=tracker)

exportar_datos(X,F,tracker, args.nombre,problem,args.modo_objetivos, args.algoritmo, idx_equilibrado)


