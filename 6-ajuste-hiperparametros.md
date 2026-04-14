# 6. Ajuste de Hiperparámetros

Con el modelo base funcionando, el siguiente paso es afinar su configuración para sacarle el mayor provecho posible.

## ¿Qué es el parámetro C?

En Regresión Logística, `C` controla la **regularización**: qué tan estrictamente el modelo penaliza los errores en entrenamiento.

- `C` pequeño → modelo más simple, menos riesgo de sobreajuste  
- `C` grande → modelo más flexible, puede memorizar el entrenamiento

No existe un valor correcto a priori, hay que buscarlo.

## Búsqueda con GridSearchCV

Probamos 7 valores de `C` distintos. Para cada uno, aplicamos **validación cruzada de 5 pliegues** — el dataset de entrenamiento se divide en 5 partes, el modelo se entrena 5 veces y se promedia su desempeño. Esto reduce el riesgo de que los resultados dependan de una partición afortunada.

La métrica que guía la búsqueda es el **F1-score**, elegida porque balancea precisión y recall — importante en un problema médico donde tanto los falsos positivos como los falsos negativos tienen consecuencias.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Definimos los valores de C a probar
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# Inicializamos el modelo base
log_reg = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')

# Configuramos la búsqueda con validación cruzada de 5 pliegues
grid_search = GridSearchCV(estimator=log_reg,
                           param_grid=param_grid,
                           cv=5,
                           scoring='f1',
                           n_jobs=-1,
                           verbose=1)

# Ajustamos sobre los datos de entrenamiento escalados
grid_search.fit(X_train_scaled, y_train)

# Extraemos los mejores parámetros
best_c = grid_search.best_params_['C']
best_f1_score = grid_search.best_score_

print(f"Mejor parámetro C encontrado: {best_c}")
print(f"Mejor F1 Score durante la validación cruzada: {best_f1_score:.4f}")
```

```
Fitting 5 folds for each of 7 candidates, totalling 35 fits
Mejor parámetro C encontrado: 0.1
Mejor F1 Score durante la validación cruzada: 0.9799
```

## Evaluación del mejor modelo

Con el mejor `C` encontrado, entrenamos un modelo nuevo y lo evaluamos sobre el set de prueba.

```python
# Entrenamos un nuevo modelo con el mejor C
best_model = LogisticRegression(C=best_c, random_state=42, solver='liblinear', class_weight='balanced')
best_model.fit(X_train_scaled, y_train)

# Predicciones sobre el set de prueba
y_pred_best = best_model.predict(X_test_scaled)
y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]

# Métricas
accuracy_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)
roc_auc_best = roc_auc_score(y_test, y_pred_proba_best)

print(f"\nResultados del modelo con el mejor C en el conjunto de prueba:")
print(f"Accuracy: {accuracy_best:.4f}")
print(f"F1 Score: {f1_best:.4f}")
print(f"AUC Score: {roc_auc_best:.4f}")

print("\nReporte de Clasificación con el mejor C:")
print(classification_report(y_test, y_pred_best))
```

```
Resultados del modelo con el mejor C en el conjunto de prueba:
Accuracy: 0.9825
F1 Score: 0.9860
AUC Score: 0.9987

Reporte de Clasificación con el mejor C:
              precision    recall  f1-score   support

           0       0.97      0.98      0.98        63
           1       0.99      0.98      0.99       108

    accuracy                           0.98       171
   macro avg       0.98      0.98      0.98       171
weighted avg       0.98      0.98      0.98       171
```

Los resultados son prácticamente idénticos al modelo base — lo que confirma que el modelo original ya era bastante robusto, y que `C = 0.1` es la configuración más confiable.

---

*Siguiente paso → [7. PCA: Reducción de Dimensionalidad](7-pca-varianza.md)*
