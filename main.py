import numpy as np

print("Estamos creando una neurona para que aprenda a decidir si una planta necesita riego")

X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

print("Entradas (humedad %, temperatura °C): \n", X)
print("Respuestas esperadas: \n", y.ravel())

escala = np.array([100, 50])

X_normalizado = X / escala

print("\nDatos normalizados:")
print(X_normalizado)

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

rng = np.random.default_rng(7)
pesos_iniciales = rng.normal(size=(2, 1))
sesgo_inicial = 0.0

pesos = pesos_iniciales.copy()
sesgo = sesgo_inicial
tasa_aprendizaje = 0.5
epocas = 10000

print(f"\nEntrenamiento base: tasa = {tasa_aprendizaje}, épocas = {epocas}")
print("Pesos iniciales:", pesos_iniciales.ravel(), "| sesgo inicial:", sesgo_inicial)

for epoca in range(epocas):
    z = X_normalizado @ pesos + sesgo
    predicciones = sigmoide(z)

    error = predicciones - y
    gradiente_z = 2 * error * predicciones * (1 - predicciones) / len(X_normalizado)
    gradiente_pesos = X_normalizado.T @ gradiente_z
    gradiente_sesgo = np.sum(gradiente_z)

    pesos -= tasa_aprendizaje * gradiente_pesos
    sesgo -= tasa_aprendizaje * gradiente_sesgo

    if epoca % 1000 == 0:
        perdida = np.mean(error ** 2)
        print(f"Epoca {epoca:5} | error: {perdida: .6f}")

probabilidades = sigmoide(X_normalizado @ pesos + sesgo)
perdida_final = np.mean((probabilidades - y) ** 2)

print("\nPesos aprendidos:", pesos.ravel())
print("Sesgo aprendido", sesgo)
print(f"Peso de la humedad:     {pesos[0, 0]: .4f}")
print(f"Peso de la temperatura: {pesos[1, 0]: .4f}")
print(f"Sesgo:                  {sesgo: .4f}")
print(f"Error final (MSE):      {perdida_final: .6f}")

print("\nSignificado del signo de los pesos:")
if pesos[0, 0] < 0:
    print("Humedad negativa: a mas humedad, menor probabilidad de regar")
else:
    print("Humedad positiva: a mas humedad, mayor probabilidad de regar")
if pesos[1, 0] > 0:
    print("Temperatura positiva: a mas temperatura, mayor probabilidad de regar")
else:
    print("Temperatura negativa: a mas temperatura, menor probabilidad de regar")

print("\nProbamos lo aprendido:")
print("Si la salida es >= 0.5 = 1 (regar)")
print("Si es < 0.5 = 0 (no regar)")

respuestas = (probabilidades >= 0.5).astype(int)

for entrada, probabilidad, respuesta in zip(X, probabilidades.ravel(), respuestas.ravel()):
    print(f"{entrada.astype(int)} -> probabilidad {probabilidad:.4f} -> respuesta {respuesta}")

print("Aciertos:", np.sum(respuestas == y), "de", len(y))

assert np.array_equal(respuestas, y.astype(int))

print("\nPruebas con condiciones nuevas:")

X_nuevos = np.array([
    [75, 30], [45, 34], [25, 22], [50, 25], [30, 40]
], dtype=float)

X_nuevos_normalizado = X_nuevos / escala

probabilidades_nuevas = sigmoide(X_nuevos_normalizado @ pesos + sesgo)
respuestas_nuevas = (probabilidades_nuevas >= 0.5).astype(int)

for entrada, probabilidad, respuesta in zip(X_nuevos, probabilidades_nuevas.ravel(), respuestas_nuevas.ravel()):
    decision = "regar" if respuesta == 1 else "no regar"
    print(f"{entrada.astype(int)} -> probabilidad {probabilidad:.4f} -> respuesta {respuesta} ({decision})")

print("\nExperimentos con los parametros:")

experimentos = [
    ("Prueba base", 10000, 0.5),
    ("Pocas epocas", 100, 0.5),
    ("Cantidad intermedia", 1000, 0.5),
    ("Mas epocas", 20000, 0.5),
    ("Tasa pequena", 10000, 0.01),
    ("Tasa moderada", 10000, 0.1),
    ("Tasa alta", 10000, 1.0),
    ("Tasa muy alta", 10000, 2.0),
]

caso_prueba = np.array([[45, 34]], dtype=float) / escala

for nombre, epocas_exp, tasa_exp in experimentos:
    pesos_exp = pesos_iniciales.copy()
    sesgo_exp = sesgo_inicial
    historial = []

    for epoca in range(epocas_exp):
        z = X_normalizado @ pesos_exp + sesgo_exp
        predicciones = sigmoide(z)

        error = predicciones - y
        historial.append(np.mean(error ** 2))
        gradiente_z = 2 * error * predicciones * (1 - predicciones) / len(X_normalizado)
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = np.sum(gradiente_z)

        pesos_exp -= tasa_exp * gradiente_pesos
        sesgo_exp -= tasa_exp * gradiente_sesgo

    probabilidades_exp = sigmoide(X_normalizado @ pesos_exp + sesgo_exp)
    perdida_exp = np.mean((probabilidades_exp - y) ** 2)
    aciertos = np.sum((probabilidades_exp >= 0.5).astype(int) == y)
    probabilidad_caso = sigmoide(caso_prueba @ pesos_exp + sesgo_exp)[0, 0]
    subidas = np.sum(np.diff(historial) > 1e-12)

    print(f"{nombre:20} | epocas {epocas_exp:5} | tasa {tasa_exp:4} | error {perdida_exp:.6f} | "
          f"aciertos {aciertos}/10 | P(45%, 34°C) {probabilidad_caso:.4f} | subidas del error {subidas}")

print("\nPrueba con diferentes umbrales (sin volver a entrenar):")

umbrales = [0.4, 0.5, 0.6]
entradas = np.vstack([X, X_nuevos])
probabilidades_todas = sigmoide((entradas / escala) @ pesos + sesgo).ravel()

for entrada, probabilidad in zip(entradas, probabilidades_todas):
    respuestas_umbral = [int(probabilidad >= umbral) for umbral in umbrales]
    cambia = "<- cambia" if len(set(respuestas_umbral)) > 1 else ""
    print(f"{entrada.astype(int)} -> probabilidad {probabilidad:.4f} -> "
          f"u=0.4: {respuestas_umbral[0]} | u=0.5: {respuestas_umbral[1]} | u=0.6: {respuestas_umbral[2]} {cambia}")

print("Los pesos y el sesgo no cambian: el umbral solo se aplica despues de la sigmoide")
