# 7. Generación de Recomendaciones para un Usuario

Con la matriz reconstruida, el último paso es convertir los ratings predichos en recomendaciones concretas. La lógica es directa: **las películas con mayor rating predicho que el usuario aún no ha visto son las que se recomiendan**.

## Des-centrado de los ratings

Recuerda que la matriz con la que trabajó SoftImpute estaba centrada — los valores son desviaciones respecto a la media de cada usuario, no ratings reales. Para recomendar, necesitamos volver a la escala original sumando la media del usuario.

```python
# Seleccionar un usuario aleatorio
random_user_id = np.random.choice(user_item_matrix.index)
print(f"Generando recomendaciones para el usuario con ID: {random_user_id}")

# Ratings imputados centrados para este usuario
user_imputed_ratings_centered = user_item_matrix_imputed_soft.loc[random_user_id]

# Des-centrar: sumar la media del usuario para volver a la escala 1-5
user_imputed_ratings_uncentered = user_imputed_ratings_centered + user_mean_ratings.loc[random_user_id]
```

## Filtrado de películas ya vistas

No tiene sentido recomendar algo que el usuario ya calificó. Identificamos las películas que sí ha visto y las eliminamos de los candidatos.

```python
# Películas que el usuario ya ha calificado (en la matriz original, antes de ocultar el 10%)
already_rated_movies = user_item_matrix.loc[random_user_id].dropna().index

# Eliminar esas películas de las candidatas a recomendar
recommendations = user_imputed_ratings_uncentered.drop(already_rated_movies, errors='ignore')
```

## Top 10 recomendaciones

```python
# Obtener las 10 películas con mayor rating predicho
top_n_recommendations = recommendations.nlargest(10)

# Convertir a DataFrame y unir con los títulos de las películas
top_n_df = top_n_recommendations.reset_index()
top_n_df.columns = ['item_id', 'predicted_rating']

recommended_movies_info = pd.merge(
    top_n_df,
    df_movies[['item_id', 'movie_title']],
    on='item_id',
    how='left'
)

print("\nRecomendaciones Detalladas:")
display(recommended_movies_info)
```

```
Generando recomendaciones para el usuario con ID: 318

Recomendaciones Detalladas:

   item_id  predicted_rating                          movie_title
0      272          4.795371             Good Will Hunting (1997)
1      153          4.587693          Fish Called Wanda, A (1988)
2      483          4.581363                    Casablanca (1942)
3      528          4.523439           Killing Fields, The (1984)
4      479          4.514134                       Vertigo (1958)
5       22          4.513852                    Braveheart (1995)
6      276          4.511060             Leaving Las Vegas (1995)
7      136          4.480939  Mr. Smith Goes to Washington (1939)
8      496          4.430311         It's a Wonderful Life (1946)
9       42          4.423244                        Clerks (1994)
```

El sistema sugiere 10 películas que este usuario probablemente valoraría alto, con ratings predichos entre 4.4 y 4.8 sobre 5. El resultado tiene sentido: son títulos clásicos y bien valorados en general, y el sistema los filtra y ordena según el perfil específico de este usuario.

## ¿Qué acabamos de hacer?

Resumiendo el flujo completo:

1. Tomamos 100,000 ratings reales y los organizamos en una matriz usuarios × películas
2. Ocultamos el 10% para poder evaluar
3. Centramos la matriz por usuario para normalizar los estilos de calificación
4. SoftImpute reconstruyó los ~1.6 millones de celdas vacías usando la estructura latente de la matriz
5. Des-centramos y filtramos las ya vistas
6. Las mejores predicciones se convierten en recomendaciones

Lo que hace el sistema de recomendación no es magia — es una **reconstrucción matricial**. El algoritmo encuentra patrones ocultos entre usuarios y películas, y los usa para estimar qué le gustaría a cada persona lo que aún no ha visto.

---

*← [6. Evaluación de la Reconstrucción (RMSE)](6-evaluacion-rmse.md)*
