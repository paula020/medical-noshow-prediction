# Sistema de Predicción de No-Show Médico

**Proyecto I - Especialización en Ciencia de Datos e Inteligencia Artificial**

Sistema de Machine Learning para predecir la probabilidad de que un paciente no asista a su cita médica, permitiendo a clínicas y hospitales optimizar sus recursos y reducir costos operativos.

---

## Descripción del Proyecto

Este proyecto implementa un análisis completo de datos médicos utilizando técnicas de ciencia de datos y machine learning para predecir la inasistencia de pacientes a citas programadas.

### Dataset
- **Fuente**: Kaggle - Medical Appointment No Shows
- **Registros**: 110,527 citas médicas
- **Variables**: 14 características del paciente y la cita
- **Objetivo**: Predecir No-Show (0: Asistió, 1: No Asistió)

### Objetivos

Ayudar a clínicas y hospitales a:
- Reducir costos por citas perdidas
- Optimizar agendas médicas
- Mejorar la atención al paciente mediante gestión predictiva

---

## Estructura del Proyecto

```
medical-noshow-prediction/
│
├── data/                          # Datos del proyecto
│   ├── KaggleV2-May-2016.csv     # Dataset original
│   ├── processed/                 # Datos procesados
│   └── synthetic/                 # Datos sintéticos generados
│
├── notebooks/                     # Jupyter notebooks
│   └── 01_analisis_completo.ipynb # Análisis EDA y modelamiento
│
├── src/                           # Código fuente modular
│   ├── __init__.py
│   ├── data_loader.py            # Carga y limpieza de datos
│   ├── preprocessing.py          # Pipelines de preprocesamiento
│   ├── synthetic_data.py         # Generación de datos sintéticos (SMOTE)
│   ├── models.py                 # Entrenamiento y evaluación de modelos
│   └── visualization.py          # Visualizaciones con Plotly
│
├── app/                           # Aplicación Streamlit
│   └── streamlit_app.py          # Interfaz web interactiva
│
├── models/                        # Modelos entrenados (generados al ejecutar)
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   └── preprocessor.pkl
│
├── requirements.txt               # Dependencias del proyecto
├── README.md                      # Este archivo
└── CLAUDE.md                      # Guía para Claude Code
```

---

## Instalación y Configuración

### Requisitos Previos
- Python 3.11 o superior
- pip o conda instalado

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/medical-noshow-prediction.git
cd medical-noshow-prediction
```

### Paso 2: Crear Entorno Virtual

**Opción A: Con conda (recomendado)**
```bash
conda create -n medicalPrediction python=3.11
conda activate medicalPrediction
```

**Opción B: Con venv**
```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/Mac:
source .venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

---

## Uso del Proyecto

### 1. Ejecutar el Análisis Completo

Abrir y ejecutar el notebook principal:

```bash
jupyter notebook notebooks/01_analisis_completo.ipynb
```

Este notebook incluye:
- Carga y exploración de datos
- Análisis exploratorio (EDA) con visualizaciones Plotly
- Implementación de pipelines con `ColumnTransformer`
- Generación de datos sintéticos con SMOTE
- Entrenamiento de modelos (Random Forest y Regresión Logística)
- Comparación de modelos con y sin datos sintéticos
- Análisis de importancia de variables
- Guardado de modelos entrenados

### 2. Ejecutar la Aplicación Streamlit

Una vez entrenados los modelos, ejecutar la interfaz web:

```bash
streamlit run app/streamlit_app.py
```

La aplicación se abrirá en `http://localhost:8501` y permite:
- Explorar los datos de forma interactiva
- Visualizar análisis y métricas
- Realizar predicciones individuales
- Comparar resultados de modelos

---

## Metodología

### 1. Procesamiento de Datos
- Limpieza de valores inconsistentes (edades negativas, outliers)
- Conversión de tipos de datos apropiados
- Creación de features: días de anticipación, día de la semana, conteo de condiciones crónicas

### 2. Pipeline de Preprocesamiento
```python
- Variables Numéricas: Imputación (mediana) + StandardScaler
- Variables Categóricas: Imputación (moda) + OneHotEncoder
- ColumnTransformer: Combina ambos pipelines
```

### 3. Generación de Datos Sintéticos
- Técnica: SMOTE (Synthetic Minority Over-sampling Technique)
- Objetivo: Balancear clases (20% no-show vs 80% asistencia)
- Implementado con `imbalanced-learn`

### 4. Modelos Implementados
1. **Random Forest Classifier**
   - 100 árboles
   - Profundidad máxima: 10
   - Mejor desempeño general

2. **Regresión Logística**
   - Solver: lbfgs
   - Más interpretable
   - Baseline para comparación

### 5. Métricas de Evaluación
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## Resultados Principales

### Hallazgos Clave
1. **Desbalance de Clases**: ~20% de tasa de no-show en datos originales
2. **Impacto de SMOTE**: Mejora significativa en recall y F1-score
3. **Mejor Modelo**: Random Forest con datos sintéticos
4. **Variables Importantes**:
   - Días de anticipación (DaysAdvance)
   - Edad del paciente
   - Recepción de SMS recordatorio
   - Condiciones crónicas

### Aplicaciones Prácticas
- Sistema de alertas tempranas para citas de alto riesgo
- Optimización de envío de recordatorios SMS
- Mejor gestión y planificación de agendas médicas
- Sobreagendar estratégicamente basado en probabilidades

---

## Librerías Principales Utilizadas

- **Análisis de Datos**: pandas, numpy
- **Visualización**: plotly, seaborn, matplotlib
- **Machine Learning**: scikit-learn, imbalanced-learn
- **Análisis Estadístico**: scipy, statsmodels
- **Aplicación Web**: streamlit
- **Utilidades**: joblib, pathlib

Ver `requirements.txt` para lista completa de versiones.

---

## Equipo de Desarrollo

- Diego Fernando Nuñez Dia
- Diego Fernando Castaneda Loaiza
- Paula Carolina Barrera Camargo
- Mateo Atehortua Arango

---

## Notas Técnicas

### Reproducibilidad
- Todos los procesos aleatorios usan `random_state=42`
- Los modelos entrenados se guardan en `models/`
- Los datos procesados mantienen trazabilidad

### Consideraciones Éticas
- Este sistema es una herramienta de apoyo a la decisión
- Las predicciones deben interpretarse por profesionales de salud
- No debe usarse como único criterio para cancelar o denegar citas

### Mejoras Futuras
- Incorporar más variables contextuales (clima, transporte)
- Implementar modelos de deep learning
- Sistema de feedback continuo con datos reales
- API REST para integración con sistemas hospitalarios

---

## Licencia

Este proyecto es de uso académico para la Especialización en Ciencia de Datos e IA.

---

## Contacto y Soporte

Para preguntas o sugerencias sobre este proyecto, contactar al equipo de desarrollo.

**Última actualización**: Octubre 2025
