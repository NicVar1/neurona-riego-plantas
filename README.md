# Neurona artificial para el riego de plantas

Taller de IA Generativa. La idea es hacer una neurona artificial que diga si hay que regar una planta o no, usando dos datos: la **humedad del suelo** y la **temperatura**. Está hecha en Python con `uv` y NumPy.

Ojo: los datos son de ejemplo para el taller, no sirven como recomendación real para regar una planta.

## El problema

La neurona recibe dos entradas y devuelve 1 o 0:

| Variable | Qué es | Rango |
|---|---|---|
| `x1` | Humedad del suelo (%) | 0 a 100 |
| `x2` | Temperatura (°C) | más o menos 0 a 50 |
| `y` | 1 = regar, 0 = no regar | 0 o 1 |

Estos son los datos con los que se entrena:

| Caso | Humedad | Temperatura | Salida |
|:-:|:-:|:-:|:-:|
| 1 | 80 % | 18 °C | 0 |
| 2 | 70 % | 22 °C | 0 |
| 3 | 65 % | 28 °C | 0 |
| 4 | 55 % | 25 °C | 0 |
| 5 | 50 % | 32 °C | 0 |
| 6 | 40 % | 30 °C | 1 |
| 7 | 35 % | 25 °C | 1 |
| 8 | 30 % | 32 °C | 1 |
| 9 | 20 % | 35 °C | 1 |
| 10 | 10 % | 38 °C | 1 |

## Cómo ejecutarlo

Hay que tener `uv` instalado. Después:

```bash
git clone <url-del-repo>
cd neurona-riego-plantas
uv sync
uv run main.py
```

`uv sync` instala NumPy y `uv run main.py` corre todo: el entrenamiento, las pruebas, los experimentos y la prueba de umbrales.

## Cómo funciona

Es parecido al perceptrón que hicimos en clase para la compuerta AND, pero con otros datos:

1. Primero se normalizan los datos: la humedad se divide entre 100 y la temperatura entre 50 (`escala = [100, 50]`), para que las dos queden entre 0 y 1.
2. Se calcula la suma ponderada: `z = X_normalizado @ pesos + sesgo`. Son 2 pesos (uno por entrada) y un sesgo.
3. A `z` se le aplica la **sigmoide**, que da una probabilidad entre 0 y 1.
4. Se calcula el error con el **error cuadrático medio**, comparando las predicciones con `y`.
5. Se sacan los gradientes:
   - `gradiente_z = 2 * error * predicciones * (1 - predicciones) / len(X_normalizado)`
   - `gradiente_pesos = X_normalizado.T @ gradiente_z`
   - `gradiente_sesgo = np.sum(gradiente_z)`
6. Se actualizan los pesos y el sesgo restándoles `tasa_aprendizaje * gradiente`.
7. Al final, si la probabilidad es >= 0.5 la respuesta es 1 (regar) y si no es 0 (no regar).

Los pesos iniciales salen de `np.random.default_rng(7)` con `rng.normal`, igual que en clase, y quedan en `[0.0012, 0.2987]` con sesgo `0.0`. Usé siempre la misma semilla para que todos los experimentos empiecen igual y así la diferencia se deba solo al parámetro que cambio.

## Resultados del entrenamiento base (tasa 0.5, 10000 épocas)

Así fue bajando el error:

| Época | 0 | 1000 | 2000 | 4000 | 6000 | 8000 | Final |
|---|---|---|---|---|---|---|---|
| Error (MSE) | 0.2468 | 0.0675 | 0.0497 | 0.0350 | 0.0275 | 0.0229 | 0.0196 |

Lo que aprendió:

| | Valor |
|---|---|
| Peso de la humedad | **-19.5032** |
| Peso de la temperatura | **+2.3101** |
| Sesgo | 7.4022 |
| Error final | 0.019645 |

**¿Qué significan los signos?**
- El peso de la **humedad es negativo**: entre más húmedo esté el suelo, menos probable es que haya que regar. 
- El peso de la **temperatura es positivo**: entre más calor, más probable es regar.
- El peso de la humedad es como 8 veces más grande que el de la temperatura, o sea que para la neurona la humedad es lo que más pesa al decidir ,el cambio entre regar y no regar pasa entre 50 % y 40 % de humedad.

Probabilidad que dio para cada caso:

| Caso | Humedad | Temp. | Esperado | Probabilidad | Salida |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 80 % | 18 °C | 0 | 0.0006 | 0 |
| 2 | 70 % | 22 °C | 0 | 0.0053 | 0 |
| 3 | 65 % | 28 °C | 0 | 0.0183 | 0 |
| 4 | 55 % | 25 °C | 0 | 0.1025 | 0 |
| 5 | 50 % | 32 °C | 0 | 0.2951 | 0 |
| 6 | 40 % | 30 °C | 1 | 0.7285 | 1 |
| 7 | 35 % | 25 °C | 1 | 0.8496 | 1  |
| 8 | 30 % | 32 °C | 1 | 0.9539 | 1 |
| 9 | 20 % | 35 °C | 1 | 0.9941 | 1 |
| 10 | 10 % | 38 °C | 1 | 0.9993 | 1 |

Acertó **los 10 casos**. Los que tienen menos seguridad son el 5 y el 6 (0.30 y 0.73) porque están justo en el límite entre regar y no regar.

## Pruebas con datos nuevos

Estos datos también se dividieron por la misma `escala` antes de pasarlos a la neurona.

| Humedad | Temperatura | Probabilidad | Decisión |
|:-:|:-:|:-:|:-:|
| 75 % | 30 °C | 0.0029 | 0 (no regar) |
| 45 % | 34 °C | 0.5490 | 1 (regar) |
| 25 % | 22 °C | 0.9719 | 1 (regar) |
| 50 % | 25 °C | 0.2325 | 0 (no regar) |
| 30 % | 40 °C | 0.9677 | 1 (regar) |

- **75 % y 30 °C**: hace calor pero el suelo está muy húmedo, y como la humedad pesa más, no riega.
- **45 % y 34 °C**: es el caso más dudoso (0.549). Queda entre el caso 5 (50 %, no regar) y el 6 (40 %, regar), y el calor lo empuja un poquito hacia regar.
- **25 % y 22 °C**: el suelo está seco, entonces riega aunque no haga tanto calor.
- **50 % y 25 °C**: se parece al caso 5 pero con menos calor, así que da todavía menos.
- **30 % y 40 °C**: seco y con mucho calor, riega sin duda.

## Experimentos cambiando los parámetros

En cada prueba cambié solo un parámetro y dejé lo demás igual a la prueba base.

| Experimento | Épocas | Tasa | Error final | Aciertos | P(45 %, 34 °C) | Cómo fue el aprendizaje |
|---|:-:|:-:|:-:|:-:|:-:|---|
| Prueba base | 10000 | 0.5 | 0.019645 | 10/10 | 0.5490 | Bien |
| Pocas épocas | 100 | 0.5 | 0.165623 | 10/10 | 0.5217 | Insuficiente |
| Cantidad intermedia | 1000 | 0.5 | 0.067481 | 10/10 | 0.5793 | Aceptable pero le faltó |
| Más épocas | 20000 | 0.5 | 0.011705 | 10/10 | 0.5263 | Mejoró poco |
| Tasa pequeña | 10000 | 0.01 | 0.130048 | 10/10 | 0.5388 | Muy lento |
| Tasa moderada | 10000 | 0.1 | 0.049728 | 10/10 | 0.5853 | Lento |
| Tasa alta | 10000 | 1.0 | 0.011705 | 10/10 | 0.5263 | Rápido |
| Tasa muy alta | 10000 | 2.0 | 0.006566 | 10/10 | 0.5082 | El más rápido, y no se volvió inestable |

Para saber si alguno era inestable, el programa cuenta cuántas veces el error subió de una época a otra. En los 8 experimentos dio **0**, o sea que ninguno fue inestable con estos datos.

Cosas que noté:
- Hasta con 100 épocas acierta los 10 casos, porque los datos son fáciles de separar. La diferencia está en qué tan segura queda la neurona: con 100 épocas las probabilidades quedan muy cerca de 0.5, y con 10000 quedan cerca de 0 o 1.
- **20000 épocas con tasa 0.5** y **10000 épocas con tasa 1.0** dieron exactamente el mismo error (0.011705). Me pareció curioso. Tiene sentido porque en los dos casos se avanza lo mismo en total: el doble de pasos de tamaño normal o la mitad de pasos del doble de tamaño.
- La probabilidad del caso 45 % y 34 °C cambia bastante entre experimentos (de 0.51 a 0.59), porque está muy cerca del límite y cualquier cambio en los pesos la mueve.

## Prueba con diferentes umbrales

Sin volver a entrenar (con los mismos pesos de la prueba base), probé los umbrales 0.4, 0.5 y 0.6:

| Dato | Probabilidad | u = 0.4 | u = 0.5 | u = 0.6 |
|---|:-:|:-:|:-:|:-:|
| Casos 1 a 5 | 0.0006 a 0.2951 | 0 | 0 | 0 |
| Casos 6 a 10 | 0.7285 a 0.9993 | 1 | 1 | 1 |
| 75 % y 30 °C | 0.0029 | 0 | 0 | 0 |
| **45 % y 34 °C** | **0.5490** | **1** | **1** | **0** |
| 25 % y 22 °C | 0.9719 | 1 | 1 | 1 |
| 50 % y 25 °C | 0.2325 | 0 | 0 | 0 |
| 30 % y 40 °C | 0.9677 | 1 | 1 | 1 |

El único que cambia es **45 % y 34 °C**, porque su probabilidad (0.549) queda entre 0.5 y 0.6. Con umbral 0.6 ya no alcanza y la respuesta pasa a "no regar". Los de entrenamiento no cambian porque ninguno tiene probabilidad entre 0.4 y 0.6.

Los pesos no cambian porque el umbral se aplica al final, después de la sigmoide. En el entrenamiento se usa la probabilidad tal cual para calcular el error, y el umbral no entra ahí. Cambiar el umbral solo cambia dónde se corta para decir 0 o 1.

## Análisis

**1. ¿Por qué fue necesario normalizar la humedad y la temperatura?**
Porque están en escalas diferentes: la humedad llega hasta 100 y la temperatura hasta 50 más o menos. Si no se normaliza, la humedad mete números más grandes en `z` y en los gradientes, entonces su peso se actualiza mucho más que el de la temperatura. Además, con números grandes la sigmoide se satura (da casi 0 o casi 1), su derivada queda casi en 0 y la neurona casi no aprende. Al dejar todo entre 0 y 1 las dos entradas quedan en igualdad de condiciones.

**2. ¿En qué operaciones se utilizó `X_normalizado` y para qué se conservó `X`?**
`X_normalizado` se usó en la suma ponderada (`z = X_normalizado @ pesos + sesgo`) y en el gradiente de los pesos (`X_normalizado.T @ gradiente_z`). `X` se dejó para mostrar los datos en % y °C cuando se imprimen los resultados, porque así se entienden mejor. Los datos nuevos también se dividen por la misma `escala` (`X_nuevos / escala`) y no se calcula una escala diferente.

**3. ¿Qué ocurrió al utilizar solamente 100 épocas?**
Le faltó entrenamiento. El error solo bajó de 0.247 a 0.166. Acertó los 10 casos porque el problema es fácil, pero las probabilidades quedaron muy cerca de 0.5 (para 45 % y 34 °C dio 0.522), o sea que la neurona no estaba segura de nada. Con un dato un poco diferente se podría equivocar.

**4. ¿Más épocas siempre produjeron una mejora importante?**
No. De 100 a 1000 épocas el error bajó bastante (0.166 a 0.067), y de 1000 a 10000 también (a 0.0196). Pero de 10000 a 20000, aunque se entrena el doble, solo bajó a 0.0117 y no ganó ningún acierto. Llega un punto en que los gradientes son muy pequeños y cada época mejora muy poquito.

**5. ¿Qué efecto tuvo una tasa de aprendizaje demasiado pequeña?**
Con 0.01 aprendió muy lento. Después de 10000 épocas el error quedó en 0.130, peor que la prueba base con solo 1000 épocas (0.067). Los pasos son tan chiquitos que necesitaría como 50 veces más épocas para llegar a lo mismo.

**6. ¿Qué efecto tuvo una tasa de aprendizaje alta o muy alta?**
Con 1.0 y 2.0 aprendió más rápido y no se volvió inestable: el error siempre bajó y con 2.0 quedó el error más bajo (0.0066). Creo que es porque los datos están normalizados y la derivada de la sigmoide (máximo 0.25) hace los gradientes pequeños. Aun así, una tasa muy alta puede hacer que la neurona se pase del mínimo y el error empiece a saltar. Con datos sin normalizar o con una tasa más grande seguramente pasaría.

**7. ¿Qué representa el signo del peso de la humedad?**
Es negativo (-19.50), o sea que la humedad va en contra de regar: entre más humedad, menos probabilidad de regar. Además es el peso más grande, entonces es lo que más influye.

**8. ¿Qué representa el signo del peso de la temperatura?**
Es positivo (+2.31), o sea que la temperatura va a favor de regar: entre más calor, más probabilidad de regar. Como es más pequeño, influye menos, pero sirve para desempatar casos dudosos como el de 45 % y 34 °C.

**9. ¿Por qué una probabilidad debe convertirse en 0 o 1 mediante un umbral?**
Porque la sigmoide da un número como 0.73, pero al final hay que decidir algo: regar o no regar, no se puede "regar un 73 %". El umbral convierte la probabilidad en esa decisión. También se puede mover según lo que convenga: con 0.4 riega cuando hay duda (mejor para la planta) y con 0.6 solo riega si está más seguro (ahorra agua).

**10. ¿Qué limitaciones tiene esta neurona para representar el riego de una planta real?**
- Solo se entrenó con 10 datos de ejemplo, no con datos reales.
- Solo mira dos cosas. No tiene en cuenta el tipo de planta, el tipo de suelo, si va a llover, la luz, la humedad del aire, etc.
- Una sola neurona solo puede separar con una línea recta, así que no aprende reglas más complicadas.
- Solo dice si regar o no, no cuánta agua ni a qué hora.
- No sabe cuándo fue la última vez que se regó.
- La escala supone que la temperatura no pasa de 50 °C, y con valores por fuera de eso no se sabe qué tan bien funcione.

## Archivos del repo

neurona-riego-plantas/
├── main.py          # la neurona, el entrenamiento, las pruebas y los experimentos
├── pyproject.toml   # configuración del proyecto con uv
├── uv.lock          # versiones de las dependencias
└── README.md

