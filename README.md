# Modelo-de-optimizaci-n-multiobjetivo-para-la-gesti-n-de-residuos
# Multiobjective Optimization Model for Waste Management

Modelo de optimización multiobjetivo para el diseño de redes de gestión de residuos basado en algoritmos evolutivos.

Este proyecto implementa un sistema de optimización que determina:

- Localización de instalaciones
- Tamaño de instalaciones
- Asignación de residuos
- Flujo entre centros
- Minimización de costes e impactos

El problema se resuelve mediante algoritmos evolutivos multiobjetivo.

---

# Objetivos del modelo

El modelo optimiza simultáneamente:

- Coste económico total
- Impacto ambiental / uso del terreno
- Impacto en la salud pública

El resultado es un **frente de Pareto** con soluciones óptimas no dominadas.

---

# Algoritmos implementados

El sistema permite trabajar con:

- NSGA-II
- NSGA-III
- MOEA/D
- Genetic Algorithm (monoobjetivo)

Todos configurables y comparables.

---

# Características principales

- Generación de individuos viables
- Restricciones de capacidad
- Distribución realista de residuos
- Optimización multiobjetivo
- Comparación entre algoritmos
- Cálculo de hipervolumen
- Seguimiento de convergencia
- Normalización de resultados
- Análisis de soluciones

---

# Estructura del proyecto

---
Modelo-de-optimizacion-multiobjetivo-para-la-gestion-de-residuos/

│
├── Codificación y resultadosdel estudio/
│   │
│   ├── model.py
│   ├── problem.py
│   ├── problemMOEAD.py
│   ├── mutation.py
│   ├── crossover.py
│   ├── data.py
│   │
│   └── __pycache__/
│
├── Datos estudio de convergencia/
│   │
│   ├── Datos Max-Min/
│   ├── Normalizados/
│   ├── Resultados_HV/
│   ├── economico-salud/
│   ├── economico-uso/
│   └── triobjetivo/
│
├── Documentación/
│   │
│   ├── Memoria.pdf
│   ├── Resultados.docm
│   └── Análisis.docm
│
└── README.md

# Tecnologías utilizadas

- Python
- Pymoo
- Pandas
- NumPy
- Matplotlib

---

# Cómo ejecutar

Instalar dependencias:

---

# Resultados

El modelo genera:

- Frente de Pareto
- Soluciones óptimas
- Datos de convergencia
- Cálculo de hipervolumen
- Comparación de algoritmos

---

# Problema abordado

El sistema optimiza una red completa de gestión de residuos incluyendo:

- Centros de recogida
- Clasificadoras
- Incineradoras
- Vertederos

El modelo asegura:

- Cobertura total de demanda
- Restricciones de capacidad
- Distribución viable
- Minimización de impactos

---

# Autor

Álvaro Álvarez Zazo  
Computer Engineer  
Python | Optimization | Data
