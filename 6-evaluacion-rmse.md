# 6. Evaluación de la Reconstrucción (RMSE)

Tenemos predicciones — ahora hay que medir qué tan buenas son. Usamos el **RMSE** (Root Mean Squared Error): el error cuadrático medio entre el valor real y el valor predicho. Cuanto más bajo, mejor reconstruye el modelo.

## RMSE de imputación en el dataset Wine

Primero comparamos los tres métodos sobre los datos de Wine, midiendo qué tan cerca estuvieron los valores imputados de los reales en las posiciones donde se introdujeron NaNs.

```python
from sklearn.metrics import mean_squared_error

# Función para calcular el RMSE solo en las posiciones que fueron imputadas
def calculate_imputation_rmse(original_data, imputed_data, nan_mask):
    original_imputed_values = original_data[nan_mask]
    imputed_values = imputed_data[nan_mask]
    if len(original_imputed_values) == 0:
        return 0.0
    return np.sqrt(mean_squared_error(original_imputed_values, imputed_values))

# Máscaras de posiciones NaN en train y test
nan_mask_train = np.isnan(X_train_nan)
nan_mask_test = np.isnan(X_test_nan)

print("\n--- Error de Reconstrucción (RMSE) en el conjunto de entrenamiento ---")
rmse_mean_train = calculate_imputation_rmse(X_train_scaled, X_train_mean_imputed, nan_mask_train)
rmse_knn_train = calculate_imputation_rmse(X_train_scaled, X_train_knn_imputed, nan_mask_train)
rmse_soft_train = calculate_imputation_rmse(X_train_scaled, X_train_soft_imputed, nan_mask_train)

print(f"RMSE (Imputación por Media) - Train: {rmse_mean_train:.4f}")
print(f"RMSE (Imputación por KNN) - Train: {rmse_knn_train:.4f}")
print(f"RMSE (Imputación por SoftImpute) - Train: {rmse_soft_train:.4f}")

print("\n--- Error de Reconstrucción (RMSE) en el conjunto de prueba ---")
rmse_mean_test = calculate_imputation_rmse(X_test_scaled, X_test_mean_imputed, nan_mask_test)
rmse_knn_test = calculate_imputation_rmse(X_test_scaled, X_test_knn_imputed, nan_mask_test)
rmse_soft_test = calculate_imputation_rmse(X_test_scaled, X_test_soft_imputed, nan_mask_test)

print(f"RMSE (Imputación por Media) - Test: {rmse_mean_test:.4f}")
print(f"RMSE (Imputación por KNN) - Test: {rmse_knn_test:.4f}")
print(f"RMSE (Imputación por SoftImpute) - Test: {rmse_soft_test:.4f}")
```

```
--- Error de Reconstrucción (RMSE) en el conjunto de entrenamiento ---
RMSE (Imputación por Media) - Train: 0.9896
RMSE (Imputación por KNN) - Train: 0.7718
RMSE (Imputación por SoftImpute) - Train: 0.7508

--- Error de Reconstrucción (RMSE) en el conjunto de prueba ---
RMSE (Imputación por Media) - Test: 0.9967
RMSE (Imputación por KNN) - Test: 0.7877
RMSE (Imputación por SoftImpute) - Test: 0.7729
```

| Método | RMSE Train | RMSE Test |
|--------|-----------|-----------|
| Media | 0.9896 | 0.9967 |
| KNN | 0.7718 | 0.7877 |
| SoftImpute | **0.7508** | **0.7729** |

SoftImpute gana en ambos conjuntos. La media es la peor opción: al ignorar la estructura de los datos, introduce bastante error. KNN mejora considerablemente. SoftImpute, al aprovechar la estructura global de la matriz, logra la reconstrucción más precisa.

## RMSE sobre los ratings escondidos (MovieLens)

Ahora medimos el error de reconstrucción directamente sobre los 10,000 ratings que ocultamos al inicio. Para hacerlo comparable, des-centramos los valores imputados sumando de nuevo la media de cada usuario antes de calcular el error.

```python
from sklearn.metrics import mean_squared_error

original_hidden_ratings = []
imputed_hidden_ratings = []

for user_id, item_id in positions_to_hide:
    # Rating original (no centrado)
    original_rating = user_item_matrix.loc[user_id, item_id]
    original_hidden_ratings.append(original_rating)

    # Rating imputado (centrado) por SoftImpute
    imputed_centered_rating = user_item_matrix_imputed_soft.loc[user_id, item_id]
    imputed_hidden_ratings.append(imputed_centered_rating)

original_hidden_ratings = np.array(original_hidden_ratings)
imputed_hidden_ratings = np.array(imputed_hidden_ratings)

# Calculamos el RMSE comparando el valor original centrado con el imputado centrado
rmse_soft_impute_hidden = np.sqrt(mean_squared_error(
    original_hidden_ratings - user_mean_ratings.loc[[p[0] for p in positions_to_hide]].values,
    imputed_hidden_ratings
))

print(f"RMSE de SoftImpute en los ratings escondidos: {rmse_soft_impute_hidden:.4f}")
```

```
RMSE de SoftImpute en los ratings escondidos: 0.9757
```

Un RMSE de ~0.98 en una escala de 1 a 5 significa que el modelo se equivoca, en promedio, en menos de 1 estrella. Para un sistema de recomendación es un resultado razonable — suficiente para ordenar bien las preferencias de un usuario aunque no acierte el número exacto.

---

*Siguiente paso → [7. Generación de Recomendaciones](7-recomendaciones.md)*
