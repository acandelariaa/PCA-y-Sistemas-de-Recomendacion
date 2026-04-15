# 4. Centrado de la Matriz por Usuario

Antes de aplicar el algoritmo de reconstrucción, hacemos un ajuste importante: **restar a cada rating la media de ese usuario**.

## ¿Por qué centrar?

No todos los usuarios califican igual. Hay usuarios generosos que rara vez bajan de 4 estrellas, y usuarios exigentes que casi nunca superan el 3. Si no corregimos esto, el modelo confunde el estilo de calificación de cada usuario con la calidad real de la película.

Al centrar por usuario, lo que queda en la matriz ya no son los ratings absolutos, sino **qué tanto le gustó más o menos que el promedio**. Un valor positivo significa que le gustó más de lo habitual; uno negativo, menos.

## Cálculo del centrado

```python
# Calcular la media de los ratings para cada usuario (ignorando NaNs)
user_mean_ratings = user_item_matrix_nan.mean(axis=1)

# Centrar los ratings por usuario: rating - media_usuario
# .apply() recorre fila por fila (axis=1), restando la media correspondiente
user_item_matrix_centered = user_item_matrix_nan.apply(
    lambda row: row - user_mean_ratings[row.name], axis=1
)

print("\nMatriz Usuario-Ítem Centrada (primeras 5 filas y columnas):")
display(user_item_matrix_centered.iloc[:5, :5])
```

```
item_id         1         2         3         4         5
user_id
1        1.416309 -0.583691  0.416309 -0.583691 -0.583691
2        0.298246       NaN       NaN       NaN       NaN
3             NaN       NaN       NaN       NaN       NaN
4             NaN       NaN       NaN       NaN       NaN
5        1.080745  0.080745       NaN       NaN       NaN
```

Por ejemplo, el usuario 1 le dio un 5 a la película 1, pero su media es ~3.58, así que el valor centrado es +1.42: le gustó bastante más de lo que le suele gustar algo. Los `NaN` se mantienen igual — ahí es donde el algoritmo tendrá que inferir.

---

*Siguiente paso → [5. Imputación con SoftImpute](5-softimpute.md)*
