import openrouteservice
import pandas as pd
from geopy.geocoders import Nominatim
import time

# === CONFIGURACIÓN ===
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjAxZmIxZTFmN2VkMzRmYjlhNjg5YzUxYTQ5MTcxY2Q3IiwiaCI6Im11cm11cjY0In0="
TIEMPO_ESPERA = 1.5  # segundos entre llamadas

# === LISTA DE LOCALIDADES ===
locations = [
    "Bang Duea", "Bang Kadi", "Bang Khayaeng", "Bang Khu Wat", "Bang Prok", "Bang Phut", "Bang Phun", "Bang Luang",
    "Ban Mai", "Ban Krachaeng", "Ban Klang", "Ban Chang", "Suan Phrik Thai", "Lak Hok", "Khlong Phra Udom",
    "Khu Kwang", "Khu Bang Luang", "Bo Ngoen", "Rahaeng", "Lat Lum Kaeo", "Na Mai", "Chiang Rak Yai",
    "Chiang Rak Noi", "Krachaeng", "Khlong Khwai", "Thai Ko", "Bang Toei", "Bang Pho Nuea", "Bang Krabue",
    "Ban Ngio", "Ban Pathum", "Sam Khok"
]

# === INICIALIZACIÓN ===
client = openrouteservice.Client(key=ORS_API_KEY)
geolocator = Nominatim(user_agent="matriz_distancias")
coords = {}

# === OBTENER COORDENADAS ===
print("📍 Obteniendo coordenadas...")
for loc in locations:
    location = geolocator.geocode(loc + ", Thailand")
    if location:
        coords[loc] = [location.longitude, location.latitude]
    else:
        print(f"❌ No se encontró {loc}")
        coords[loc] = None
    time.sleep(1)  # evita bloqueo

# === MATRIZ VACÍA ===
n = len(locations)
matrix = [[0.0 for _ in range(n)] for _ in range(n)]

# === CALCULAR SOLO TRIANGULO SUPERIOR (SIMÉTRICO) ===
print("🧮 Calculando distancias...")
for i in range(n):
    for j in range(i + 1, n):
        origen = locations[i]
        destino = locations[j]

        if coords[origen] is None or coords[destino] is None:
            distancia = 0.0
        else:
            try:
                route = client.directions(
                    coordinates=[coords[origen], coords[destino]],
                    profile='driving-car',
                    format='geojson'
                )
                distancia = route['features'][0]['properties']['segments'][0]['distance'] / 1000
                distancia = round(distancia, 2)
                print(f"✔️ {origen} → {destino}: {distancia} km")
            except Exception as e:
                print(f"⚠️ Error entre {origen} y {destino}: {e}")
                distancia = 0.0

        matrix[i][j] = distancia
        matrix[j][i] = distancia
        time.sleep(TIEMPO_ESPERA)

# === GUARDAR COMO CSV ===
df = pd.DataFrame(matrix, columns=locations, index=locations)
df.to_csv("matriz_distancias_simetrica.csv", encoding="utf-8")
print("✅ Matriz guardada como 'matriz_distancias_simetrica.csv'")
