# 5. Imputación de Valores Faltantes con SoftImpute

Con la matriz centrada, el siguiente paso es **rellenar los huecos** — predecir los ratings que faltan. Para esto probamos tres métodos y comparamos qué tan bien reconstruyen los valores reales.

## Contexto: tres métodos de imputación

Antes de llegar a SoftImpute, exploramos dos alternativas más simples sobre el dataset Wine con datos faltantes artificiales (30% de NaNs), para tener un punto de comparación.

### Imputación por Media

El método más directo: reemplaza cada `NaN` con la media de esa columna.

```python
from sklearn.impute import SimpleImputer, KNNImputer

# --- Imputación por Media ---
imputer_mean = SimpleImputer(strategy='mean')
X_train_mean_imputed = imputer_mean.fit_transform(X_train_nan)
X_test_mean_imputed = imputer_mean.transform(X_test_nan)

log_reg_model_mean = LogisticRegression(random_state=42, multi_class='ovr', solver='liblinear')
log_reg_model_mean.fit(X_train_mean_imputed, y_train)
y_pred_mean = log_reg_model_mean.predict(X_test_mean_imputed)
y_pred_proba_mean = log_reg_model_mean.predict_proba(X_test_mean_imputed)

accuracy_mean = accuracy_score(y_test, y_pred_mean)
f1_mean = f1_score(y_test, y_pred_mean, average='weighted')
auc_mean = roc_auc_score(y_test, y_pred_proba_mean, multi_class='ovr', average='weighted')

print(f"Accuracy (Media): {accuracy_mean:.4f}")
print(f"F1-Score (Media): {f1_mean:.4f}")
print(f"AUC (Media): {auc_mean:.4f}")
```

```
Accuracy (Media): 0.9444
F1-Score (Media): 0.9436
AUC (Media): 0.9978
```

### Imputación por KNN

Más sofisticado: busca los `k` vecinos más parecidos a cada muestra y promedia sus valores para rellenar el hueco.

```python
# --- Imputación por KNN ---
imputer_knn = KNNImputer(n_neighbors=5)
X_train_knn_imputed = imputer_knn.fit_transform(X_train_nan)
X_test_knn_imputed = imputer_knn.transform(X_test_nan)

log_reg_model_knn = LogisticRegression(random_state=42, multi_class='ovr', solver='liblinear')
log_reg_model_knn.fit(X_train_knn_imputed, y_train)
y_pred_knn = log_reg_model_knn.predict(X_test_knn_imputed)
y_pred_proba_knn = log_reg_model_knn.predict_proba(X_test_knn_imputed)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
auc_knn = roc_auc_score(y_test, y_pred_proba_knn, multi_class='ovr', average='weighted')

print(f"Accuracy (KNN): {accuracy_knn:.4f}")
print(f"F1-Score (KNN): {f1_knn:.4f}")
print(f"AUC (KNN): {auc_knn:.4f}")
```

```
Accuracy (KNN): 0.9630
F1-Score (KNN): 0.9626
AUC (KNN): 0.9815
```

## SoftImpute: imputación matricial

SoftImpute es el método elegido para la matriz usuario-ítem. En lugar de trabajar variable por variable, opera sobre **toda la matriz a la vez** usando factorización matricial — descompone la matriz en factores de baja dimensión y los usa para reconstruir los valores faltantes. Es el mismo principio que hace funcionar los sistemas de recomendación modernos.

### Aplicación sobre el dataset Wine

```python
from fancyimpute import SoftImpute

softimpute_imputer = SoftImpute(max_iters=100, verbose=False)
X_train_soft_imputed = softimpute_imputer.fit_transform(X_train_nan)
X_test_soft_imputed = softimpute_imputer.fit_transform(X_test_nan)

log_reg_model_soft = LogisticRegression(random_state=42, multi_class='ovr', solver='liblinear')
log_reg_model_soft.fit(X_train_soft_imputed, y_train)
y_pred_soft = log_reg_model_soft.predict(X_test_soft_imputed)
y_pred_proba_soft = log_reg_model_soft.predict_proba(X_test_soft_imputed)

accuracy_soft = accuracy_score(y_test, y_pred_soft)
f1_soft = f1_score(y_test, y_pred_soft, average='weighted')
auc_soft = roc_auc_score(y_test, y_pred_proba_soft, multi_class='ovr', average='weighted')

print(f"Accuracy (SoftImpute): {accuracy_soft:.4f}")
print(f"F1-Score (SoftImpute): {f1_soft:.4f}")
print(f"AUC (SoftImpute): {auc_soft:.4f}")
```

```
Accuracy (SoftImpute): 0.9630
F1-Score (SoftImpute): 0.9630
AUC (SoftImpute): 0.9956
```

### Aplicación sobre la matriz usuario-ítem (MovieLens)

```python
# Convertir la matriz centrada a array de NumPy para SoftImpute
user_item_matrix_centered_array = user_item_matrix_centered.values

# Inicializar y aplicar SoftImpute
softimpute_imputer_matrix = SoftImpute(max_iters=100, verbose=False)
user_item_matrix_imputed_soft_array = softimpute_imputer_matrix.fit_transform(
    user_item_matrix_centered_array
)

# Recuperar formato DataFrame con los índices originales
user_item_matrix_imputed_soft = pd.DataFrame(
    user_item_matrix_imputed_soft_array,
    index=user_item_matrix_centered.index,
    columns=user_item_matrix_centered.columns
)

print("Matriz Usuario-Ítem Imputada con SoftImpute (primeras 5 filas y columnas):")
display(user_item_matrix_imputed_soft.iloc[:5, :5])
```

```
item_id         1         2         3         4         5
user_id
1        1.416309 -0.583691  0.416309 -0.583691 -0.583691
2        0.298246 -0.234578 -0.172398 -0.223999  0.142844
3        0.305279  0.053282 -0.156032 -0.157800 -0.223375
4        0.123485  0.022087 -0.035070 -0.293353  0.008782
5        1.080745  0.080745 -0.378352  0.253740  0.088351
```

Los valores que antes eran `NaN` ahora tienen un número — la predicción de SoftImpute sobre cuánto le gustaría esa película a ese usuario, expresado como desviación respecto a su media habitual.

---

*Siguiente paso → [6. Evaluación de la Reconstrucción (RMSE)](6-evaluacion-rmse.md)*
