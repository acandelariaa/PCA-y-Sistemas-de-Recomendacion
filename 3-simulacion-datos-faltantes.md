# 3. Simulación de Datos Faltantes (Entrenamiento / Prueba)

Para saber qué tan bien reconstruye el modelo los ratings, necesitamos **esconder algunos que ya conocemos** y luego ver si los predice correctamente. Es la forma de evaluar el sistema sin esperar a tener usuarios reales.

## ¿Por qué hacer esto?

Los datos faltantes de la matriz original son desconocidos — no podemos saber si la predicción es buena porque no hay un valor real con qué comparar. En cambio, si tomamos ratings que sí existen y los ocultamos artificialmente, entonces sí podemos medir el error entre el valor real y el predicho.

## Ocultando el 10% de los ratings

```python
# Crear una copia de la matriz para no modificar la original
user_item_matrix_nan = user_item_matrix.copy()

# Obtener los índices de los ratings existentes (no NaN)
known_ratings_indices = user_item_matrix_nan.stack().index

# Calcular el número de ratings a ocultar (10%)
num_ratings_to_hide = int(len(known_ratings_indices) * 0.10)

# Seleccionar aleatoriamente los índices a ocultar
indices_to_hide = np.random.choice(len(known_ratings_indices), num_ratings_to_hide, replace=False)

# Obtener las posiciones (user_id, item_id) de los ratings a ocultar
positions_to_hide = [known_ratings_indices[i] for i in indices_to_hide]

# Introducir NaNs en las posiciones seleccionadas
for user_id, item_id in positions_to_hide:
    user_item_matrix_nan.loc[user_id, item_id] = np.nan

print(f"Número total de ratings originales: {user_item_matrix.count().sum()}")
print(f"Número de NaNs introducidos: {num_ratings_to_hide}")
print(f"Número de ratings después de ocultar: {user_item_matrix_nan.count().sum()}")

print("\nMatriz Usuario-Ítem con NaNs (primeras 5 filas y columnas después de ocultar ratings):")
display(user_item_matrix_nan.iloc[:5, :5])
```

```
Número total de ratings originales: 100000
Número de NaNs introducidos: 10000
Número de ratings después de ocultar: 90000

Matriz Usuario-Ítem con NaNs (primeras 5 filas y columnas después de ocultar ratings):

item_id    1    2    3    4    5
user_id
1        5.0  3.0  4.0  3.0  3.0
2        4.0  NaN  NaN  NaN  NaN
3        NaN  NaN  NaN  NaN  NaN
4        NaN  NaN  NaN  NaN  NaN
5        4.0  3.0  NaN  NaN  NaN
```

El modelo trabajará sobre la matriz con 90,000 ratings visibles. Los 10,000 ocultos son nuestro set de prueba: los usaremos después para calcular el RMSE y saber qué tan bien se reconstruyeron.

---

*Siguiente paso → [4. Centrado de la Matriz por Usuario](4-centrado-matriz.md)*
