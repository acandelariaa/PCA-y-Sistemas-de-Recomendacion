# Separacion de datos y estandarización

Ahora, vamos a partir nuestros datos en train y test, esto lo haremos con una proporción de 70/30 de los datos originales, siempre tratando de que los datos de train tengan las mismas observaciones que los de test.

>Python Code

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features (X) and target (y)
X = df.drop(['target', 'target_names'], axis=1)
y = df['target']

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
```
>Output

```text
Shape of X_train: (398, 30)
Shape of X_test: (171, 30)
Shape of y_train: (398,)
Shape of y_test: (171,)
```

Con esto nos aseguramos que tenemos la misma proporcion en ambos splits de datos.

Ahora vamos a estandarizar estos datos con la función de `scaler.fit`, donde basicamente le estamos diciendo al modelo que escale o estandarice los datos. Para esto utilizaremos una media de 0 y una desviacion estandar de 1.

La razon de estandarizar los datos es que como las variables tienen unidades distintas, el modelo podria darles mas peso en el poder de clasificacion a variables con una escala mas grande. Esto ademas de que PCA es extremadamente sensible a las escalas de las varuables, por lo que sin una estandarización, los resultados podrian estar incorrectos o sesgados.


>Python Code

```python
# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("First 5 rows of scaled X_train:\n", X_train_scaled[:5])
print("Mean of X_train_scaled (should be close to 0):\n", X_train_scaled.mean(axis=0))
print("Standard deviation of X_train_scaled (should be close to 1):\n", X_train_scaled.std(axis=0))
```

>Output
```text
First 5 rows of scaled X_train:
 [[-0.12348985 -0.29680142 -0.17050713 -0.20861569 -1.2016799  -0.7731696
  -0.76231194 -0.93324109 -1.22994935 -0.94816603 -0.53359339 -0.86028757
  -0.61678096 -0.39177533 -1.35556152 -0.52503193 -0.4817033  -0.97940018
  -0.88459317 -0.68548672 -0.19761978 -0.5067476  -0.30791001 -0.27357592
  -1.50742388 -0.44926047 -0.57223884 -0.84082156 -0.8563616  -0.76574773]
 [-0.22826757 -0.65795149 -0.25377521 -0.2965028  -1.80463697 -0.58761605
  -0.09198533 -0.54268359 -1.41998468 -0.61249143 -0.83040055 -0.12266723
  -0.78254381 -0.53126109 -0.36490698  0.40861926  0.57668457 -0.2482875
  -1.03572382  0.10768859 -0.42291745 -0.45849468 -0.4652873  -0.43812681
  -1.27301714  0.02704209  0.31804488 -0.37706655 -1.3415819  -0.41480748]
 [ 0.14553402 -1.23056444  0.24583328 -0.01024193  0.5191843   1.57000613
   0.73231958  0.38658307  1.05420084  1.57422827  0.48747836  0.59258929
   0.90918448  0.18132474  0.93956737  1.50696696  0.68362272  0.62223771
   0.76910084  0.62438565  0.03602226 -1.1922718   0.20386884 -0.12744491
  -0.02487735  0.77080169  0.27261182 -0.04762652 -0.08997059  0.4882635 ]
 [-0.35853176 -0.67220742 -0.40093712 -0.40001429 -1.20386189 -0.9706502
  -0.63470419 -0.6549921   0.0965718  -0.82798624 -0.72594925 -0.5687868
  -0.65466961 -0.50893518 -0.56458126 -0.52781397 -0.30147421 -0.56308997
  -0.23011004 -0.61901108 -0.50218886 -0.58328671 -0.50099984 -0.49338644
  -0.95989496 -0.66349557 -0.47014208 -0.49351467  0.22654729 -0.80289938]
 [-0.15747182  0.96722386 -0.20884342 -0.24153848 -0.25469546 -0.7006297
  -0.75034872 -0.63746879 -0.51824839 -0.64288172 -0.20265859  0.25520414
  -0.38329214 -0.1916614  -0.38130535 -0.24293351 -0.4014164  -0.45965487
   0.06215914 -0.45622129 -0.19553369  0.59641391 -0.29610671 -0.26673426
  -0.44237358 -0.65608493 -0.83513799 -0.65980195 -0.38720819 -0.80061313]]
Mean of X_train_scaled (should be close to 0):
 [-1.20869255e-15 -3.93292322e-15 -3.50361839e-15  2.23160407e-16
 -2.97863353e-15  9.79674187e-16  5.20521649e-16  2.25949912e-16
 -1.35005072e-15  3.02103401e-16  5.08247827e-16 -3.65425166e-16
 -1.17996065e-16 -6.12575317e-16 -1.19056077e-15 -2.49270175e-15
 -3.00429698e-16  1.11663889e-15 -3.51756591e-16 -1.23017174e-16
  6.05880505e-16  3.82162197e-16 -1.60396543e-15 -1.79644128e-15
 -2.81461063e-16 -1.61233394e-16 -4.34604893e-16  7.29455580e-16
  1.52753299e-15 -3.03609734e-15]
Standard deviation of X_train_scaled (should be close to 1):
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1.]
```

Listo, ya que nuestros datos estan estandarizados, ahora podemos realizar algun tipo de modelo para clasificar los tipos de tumores segun este conjunto de datos.

---
*Siguiente paso → [4 - Regresion Logistica](4-RL_metricas.md)*
