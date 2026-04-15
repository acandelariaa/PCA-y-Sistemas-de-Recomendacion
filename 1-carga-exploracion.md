# 1. Carga y Exploración del Dataset

Trabajamos con dos fuentes de datos: el dataset **Wine** de scikit-learn (para la parte de imputación) y el dataset **MovieLens** (para el sistema de recomendación). Empezamos conociendo su estructura antes de hacer cualquier cosa.

## El del vino lo puedo quitar

## Dataset Wine

El dataset Wine contiene resultados de un análisis químico de vinos de tres variedades distintas. Son 178 muestras con 13 características cada una.

```python
from sklearn.datasets import load_wine
import numpy as np

wine = load_wine()
```

```python
# Dimensiones del dataset
X = wine.data
y = wine.target

print(f"Dimensiones de las características (X): {X.shape}")
print(f"Dimensiones de las etiquetas (y): {y.shape}")

# Variables (nombres de las características)
print("\nNombres de las características:")
for i, feature_name in enumerate(wine.feature_names):
    print(f"  {i}: {feature_name}")

print("\nNombres de las clases (target):")
for i, target_name in enumerate(wine.target_names):
    print(f"  {i}: {target_name}")
```

```
Dimensiones de las características (X): (178, 13)
Dimensiones de las etiquetas (y): (178,)

Nombres de las características:
  0: alcohol
  1: malic_acid
  2: ash
  3: alcalinity_of_ash
  4: magnesium
  5: total_phenols
  6: flavanoids
  7: nonflavanoid_phenols
  8: proanthocyanins
  9: color_intensity
  10: hue
  11: od280/od315_of_diluted_wines
  12: proline

Nombres de las clases (target):
  0: class_0
  1: class_1
  2: class_2
```

## Dataset MovieLens

El dataset MovieLens contiene ratings de usuarios a películas. Se cargan dos archivos: `u.data` con las calificaciones y `u.item` con la información de cada película.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar u.data (ratings)
data_cols = ['user_id', 'item_id', 'rating', 'timestamp']
df_ratings = pd.read_csv('/content/drive/MyDrive/Inteligencia_Artificial_1/u.data',
                          sep='\t', names=data_cols)

# Cargar u.item (información de películas)
item_cols = ['item_id', 'movie_title'] + [str(i) for i in range(22)]
df_movies = pd.read_csv('/content/drive/MyDrive/Inteligencia_Artificial_1/u.item',
                         sep='|', names=item_cols, encoding='latin-1')
```

`u.data` tiene cuatro columnas: `user_id`, `item_id`, `rating` y `timestamp`. De `u.item` nos interesa principalmente `item_id` y `movie_title` para identificar las películas por nombre al momento de recomendar.

---

*Siguiente paso → [2. Construcción de la Matriz Usuario-Ítem](2-matriz-usuario-item.md)*
