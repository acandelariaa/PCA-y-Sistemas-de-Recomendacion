# 1. Carga y Exploración del Dataset

Trabajamos con la fuente de datos **MovieLens** (para el sistema de recomendación). Empezamos conociendo su estructura antes de hacer cualquier cosa.

## Dataset MovieLens

El dataset MovieLens contiene ratings de usuarios a películas. Se cargan dos archivos: `u.data` con las calificaciones y `u.item` con la información de cada película.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar u.data (ratings)
data_cols = ['user_id', 'item_id', 'rating', 'timestamp']
df_ratings = pd.read_csv('/content/drive/MyDrive/Inteligencia_Artificial_1/u.data', sep='\t', names=data_cols)

# Cargar u.item (información de películas)
item_cols = ['item_id', 'movie_title'] + [str(i) for i in range(22)] # item_id, movie_title, y 22 columnas más (géneros)
df_movies = pd.read_csv('/content/drive/MyDrive/Inteligencia_Artificial_1/u.item', sep='|', names=item_cols, encoding='latin-1')

# Crear la matriz usuario-ítem
# Usamos item_id para las columnas según la descripción del usuario
user_item_matrix = df_ratings.pivot_table(index='user_id', columns='item_id', values='rating')

print("Matriz Usuario-Ítem (primeras 5 filas y columnas):")
display(user_item_matrix.iloc[:5, :5])
```

`u.data` tiene cuatro columnas: `user_id`, `item_id`, `rating` y `timestamp`. De `u.item` nos interesa principalmente `item_id` y `movie_title` para identificar las películas por nombre al momento de recomendar.

---


---

*Siguiente paso → [2. Construcción de la Matriz Usuario-Ítem](2-matriz-usuario-item.md)*
