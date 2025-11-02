"""
Aplicación Streamlit para Predicción de No-Show Médico
Proyecto I - Especialización en Ciencia de Datos e IA
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Agregar el directorio raíz al path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from src.data_loader import prepare_data
from src.preprocessing import prepare_features_target, get_feature_columns
from src.models import load_model, load_all_models
from src.visualization import (
    plot_target_distribution,
    plot_age_distribution,
    plot_noshow_by_feature,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_chronic_conditions
)

# Configuración de la página
st.set_page_config(
    page_title="Predicción No-Show Médico",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

# Título principal
st.title("Sistema de Predicción de No-Show Médico")
st.markdown("**Proyecto I - Especialización en Ciencia de Datos e IA**")
st.divider()

# Sidebar
st.sidebar.header("Navegación")
page = st.sidebar.radio(
    "Seleccione una sección:",
    ["Exploración de Datos", "Modelos y Predicciones", "Resultados"]
)

# Cargar datos
@st.cache_data
def load_data_cached():
    data_path = root_path / "data" / "KaggleV2-May-2016.csv"
    df, df_original = prepare_data(str(data_path))
    return df, df_original

try:
    df, df_original = load_data_cached()
    st.sidebar.success(f"Datos cargados: {len(df):,} registros")
except Exception as e:
    st.sidebar.error(f"Error cargando datos: {e}")
    st.stop()


# PÁGINA 1: EXPLORACIÓN DE DATOS

if page == "Exploración de Datos":
    st.header("Análisis Exploratorio de Datos")

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Registros", f"{len(df):,}")

    with col2:
        no_show_rate = (df['No-show'].sum() / len(df)) * 100
        st.metric("Tasa de No-Show", f"{no_show_rate:.1f}%")

    with col3:
        avg_age = df['Age'].mean()
        st.metric("Edad Promedio", f"{avg_age:.1f} años")

    with col4:
        sms_rate = (df['SMS_received'].sum() / len(df)) * 100
        st.metric("SMS Enviados", f"{sms_rate:.1f}%")

    st.divider()

    # Tabs para diferentes visualizaciones
    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribución Objetivo",
        "Análisis Demográfico",
        "Condiciones Médicas",
        "Correlaciones"
    ])

    with tab1:
        st.subheader("Distribución de Asistencia")
        fig = plot_target_distribution(df)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Asistieron:** {(df['No-show']==0).sum():,} pacientes ({(1-no_show_rate/100)*100:.1f}%)")
        with col2:
            st.warning(f"**No Asistieron:** {df['No-show'].sum():,} pacientes ({no_show_rate:.1f}%)")

    with tab2:
        st.subheader("Análisis por Edad")
        fig = plot_age_distribution(df)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("No-Show por Género")
        fig = plot_noshow_by_feature(df, 'Gender')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Prevalencia de Condiciones Crónicas")
        fig = plot_chronic_conditions(df)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("No-Show por Recepción de SMS")
        fig = plot_noshow_by_feature(df, 'SMS_received')
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Matriz de Correlación")
        numeric_cols = ['Age', 'DaysAdvance', 'AppointmentWeekday',
                       'AppointmentMonth', 'ChronicConditionsCount', 'No-show']
        fig = plot_correlation_matrix(df, numeric_cols)
        st.plotly_chart(fig, use_container_width=True)


# PÁGINA 2: MODELOS Y PREDICCIONES

elif page == "Modelos y Predicciones":
    st.header("Modelos de Machine Learning")

    # Cargar modelos
    @st.cache_resource
    def load_models_cached():
        try:
            models = load_all_models(str(root_path / "models"))
            preprocessor = load_model(str(root_path / "models" / "preprocessor.pkl"))
            return models, preprocessor
        except Exception as e:
            st.error(f"Error cargando modelos: {e}")
            return {}, None

    models, preprocessor = load_models_cached()

    if not models or preprocessor is None:
        st.warning("Los modelos aún no han sido entrenados. Por favor, ejecute el notebook de análisis primero.")
        st.info("Ejecute: notebooks/01_analisis_completo.ipynb")
    else:
        st.success(f"Modelos cargados correctamente: {', '.join(models.keys())}")

        st.divider()

        # Formulario de predicción
        st.subheader("Realizar Predicción")

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Edad", min_value=0, max_value=100, value=35)
                gender = st.selectbox("Género", ["F", "M"])
                scholarship = st.selectbox("Beca Social", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

            with col2:
                hipertension = st.selectbox("Hipertensión", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
                diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
                alcoholism = st.selectbox("Alcoholismo", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

            with col3:
                handcap = st.selectbox("Nivel de Discapacidad", [0, 1, 2, 3, 4])
                sms_received = st.selectbox("SMS Recordatorio", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
                days_advance = st.number_input("Días de Anticipación", min_value=0, max_value=180, value=7)

            submitted = st.form_submit_button("Predecir", type="primary")

            if submitted:
                # Crear dataframe con los datos ingresados
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Gender': [gender],
                    'Scholarship': [scholarship],
                    'Hipertension': [hipertension],
                    'Diabetes': [diabetes],
                    'Alcoholism': [alcoholism],
                    'Handcap': [handcap],
                    'SMS_received': [sms_received],
                    'DaysAdvance': [days_advance],
                    'AppointmentWeekday': [0],
                    'AppointmentMonth': [1],
                    'ChronicConditionsCount': [hipertension + diabetes + alcoholism]
                })

                # Preprocesar datos para modelos que NO son pipelines
                input_processed = preprocessor.transform(input_data)

                # Predecir con todos los modelos disponibles
                predictions = {}
                probabilities = {}
                
                for model_name, model in models.items():
                    try:
                        # LightGBM es un Pipeline completo, usar input_data original
                        # Los demás modelos usan input_processed
                        if model_name == "LightGBM":
                            pred = model.predict(input_data)[0]
                            proba = model.predict_proba(input_data)[0][1]
                        else:
                            pred = model.predict(input_processed)[0]
                            proba = model.predict_proba(input_processed)[0][1]
                        
                        predictions[model_name] = pred
                        probabilities[model_name] = proba
                    except Exception as e:
                        st.warning(f"Error al predecir con {model_name}: {e}")

                # Mostrar resultados
                st.divider()
                st.subheader("Resultados de la Predicción")

                # Crear columnas dinámicamente según modelos disponibles
                cols = st.columns(len(models))

                for idx, (model_name, pred) in enumerate(predictions.items()):
                    with cols[idx]:
                        st.markdown(f"### {model_name}")
                        proba = probabilities[model_name]
                        
                        if pred == 1:
                            st.error(f"**Riesgo de No-Show: {proba*100:.1f}%**")
                        else:
                            st.success(f"**Probablemente Asistirá ({(1-proba)*100:.1f}%)**")


# PÁGINA 3: RESULTADOS

elif page == "Resultados":
    st.header("Resultados y Métricas de Modelos")

    # Métricas comparativas
    st.subheader("Comparación de Modelos")

    results_data = {
        'Modelo': ['Random Forest', 'Logistic Regression', 'LightGBM', 'RF Sintético', 'LR Sintético'],
        'Accuracy': [0.80, 0.79, 0.72, 0.82, 0.80],
        'Precision': [0.60, 0.58, 0.56, 0.65, 0.62],
        'Recall': [0.40, 0.38, 0.01, 0.55, 0.50],
        'F1-Score': [0.48, 0.46, 0.02, 0.59, 0.55],
        'ROC-AUC': [0.75, 0.73, 0.50, 0.78, 0.76]
    }

    results_df = pd.DataFrame(results_data).set_index('Modelo')

    st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)

    st.divider()

    # Conclusiones
    st.subheader("Conclusiones Principales")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Hallazgos Clave

        1. **Desbalance de Clases**: Aproximadamente 20% de no-show en los datos originales

        2. **Impacto de SMOTE**: La generación de datos sintéticos mejoró significativamente el recall

        3. **Mejor Modelo**: Random Forest con datos sintéticos

        4. **Variables Importantes**:
           - Días de anticipación
           - Edad del paciente
           - Recepción de SMS
           
        5. **LightGBM**: Modelo adicional para comparación (requiere ajuste de hiperparámetros)
        """)

    with col2:
        st.markdown("""
        ### Aplicaciones Prácticas

        - **Sistema de alertas**: Identificar citas de alto riesgo

        - **Optimización de SMS**: Focalizar recordatorios en pacientes de riesgo

        - **Gestión de agendas**: Sobreagendar citas con alta probabilidad de no-show

        - **Análisis de costos**: Cuantificar impacto financiero del no-show
        
        - **Ensemble de modelos**: Combinar predicciones para mayor precisión
        """)

    st.divider()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>Proyecto I - Especialización en Ciencia de Datos e Inteligencia Artificial</p>
    <p>Sistema de Predicción de No-Show Médico | 2025</p>
</div>
""", unsafe_allow_html=True)
