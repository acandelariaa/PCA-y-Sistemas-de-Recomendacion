# Procesamiento y exploración de datos

Primero, carguemos el dataset junto con las librerías necesarias:

```python
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

breast_cancer_data = load_breast_cancer()
```

Con el dataset cargado, lo convertimos a un DataFrame de pandas para facilitar su manipulación. También agregamos la columna `target` con su nombre legible, e imprimimos las variables (features) disponibles junto con las dimensiones del dataset:

```python
df = pd.DataFrame(data=breast_cancer_data.data, columns=breast_cancer_data.feature_names)
df['target'] = breast_cancer_data.target
df['target_names'] = df['target'].map({i: name for i, name in enumerate(breast_cancer_data.target_names)})

print("Feature names:", breast_cancer_data.feature_names)
print("Filas x columnas:", df.shape)
```

```text
Feature names: ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']

Filas x columnas: (569, 32)
```

Con esto confirmamos que el dataset cuenta con **569 observaciones** y **32 variables** para trabajar.

---

*Siguiente paso → [2-Distribucion de variables](2-distribucion_variables.md)*
