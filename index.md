# PCA y Sistemas de Recomendación
**Tutorial dedicado a explicar los algoritmos que conforman el Principal Component Analysis y los Sistemas de Recomendación**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

---

## Descripción

Este proyecto implementa **Principal Component Analysis (PCA)** y **Sistemas de Recomendación** para reducir la dimensionalidad de datos de usuarios y productos, y generar recomendaciones personalizadas basadas en similitud latente.

**Dataset**: MovieLens / Dataset de interacciones usuario-producto  
**Objetivo**: Aplicar PCA para compresión de características y construir un sistema de recomendación basado en filtrado colaborativo y descomposición matricial.

### Recursos

|Archivos||
|---|---|
| Dataset 1 | [wine.csv](ratings.csv) |
| Dataset 2 | [breast_cancer.csv](ratings.csv) |
| **Notebook** | [PCA.ipynb](PCA.ipynb) |
| **Notebook** | [sistemas_recomendacion.ipynb](PCA_Recomendacion.ipynb) |

---

## Metodología

- Exploración y preprocesamiento de la matriz de interacciones usuario-ítem.
- Análisis de varianza explicada y selección del número óptimo de componentes principales.
- Aplicación de PCA para reducción de dimensionalidad y visualización del espacio latente.
- Interpretación geométrica de los componentes principales y su relación con patrones de preferencia.
- Construcción del sistema de recomendación mediante filtrado colaborativo (usuario-usuario e ítem-ítem).
- Implementación de SVD (Descomposición en Valores Singulares) como extensión de PCA para recomendación.
- Evaluación del sistema con métricas de desempeño (RMSE, precisión, recall).
- Comparación e interpretación crítica entre enfoques basados en PCA y SVD.

---

## Pasos a seguir para un PCA

- [Implementacion PCA](PCA.md)

## Pasos a seguir para un sistema de recomendación

- [Sistemas de recomendacion](sistemas_recomendacion.md)

---

## Referencias

- Anthropic. (2025). Claude (claude-sonnet-4-6) [Large language model]. https://www.anthropic.com

---

**By**  
*Juan Angel Candelaria Rodriguez*  
Universidad de Monterrey | Inteligencia Artificial
