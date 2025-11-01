"""
Módulo para visualizaciones con Plotly
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List


def plot_target_distribution(df: pd.DataFrame, target_col: str = 'No-show') -> go.Figure:
    """
    Gráfico de distribución de la variable objetivo.

    Args:
        df: DataFrame
        target_col: Nombre de la columna objetivo

    Returns:
        Figura de Plotly
    """
    counts = df[target_col].value_counts()
    labels = ['Asistió', 'No Asistió']

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=counts.values,
            text=counts.values,
            textposition='auto',
            marker_color=['#2ecc71', '#e74c3c']
        )
    ])

    fig.update_layout(
        title='Distribución de Asistencia a Citas Médicas',
        xaxis_title='Estado de Asistencia',
        yaxis_title='Cantidad de Pacientes',
        template='plotly_white'
    )

    return fig


def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histograma de distribución de edades.

    Args:
        df: DataFrame

    Returns:
        Figura de Plotly
    """
    fig = px.histogram(
        df,
        x='Age',
        nbins=30,
        title='Distribución de Edades de Pacientes',
        labels={'Age': 'Edad', 'count': 'Frecuencia'},
        color_discrete_sequence=['#3498db']
    )

    fig.update_layout(template='plotly_white')

    return fig


def plot_noshow_by_feature(
    df: pd.DataFrame,
    feature: str,
    target_col: str = 'No-show'
) -> go.Figure:
    """
    Gráfico de barras de no-show por característica.

    Args:
        df: DataFrame
        feature: Nombre de la feature a analizar
        target_col: Nombre de la columna objetivo

    Returns:
        Figura de Plotly
    """
    grouped = df.groupby([feature, target_col]).size().reset_index(name='count')

    fig = px.bar(
        grouped,
        x=feature,
        y='count',
        color=target_col,
        barmode='group',
        title=f'Asistencia vs {feature}',
        labels={target_col: 'Estado', 'count': 'Cantidad'},
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
    )

    fig.update_layout(template='plotly_white')

    return fig


def plot_correlation_matrix(df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
    """
    Mapa de correlación de variables numéricas.

    Args:
        df: DataFrame
        numeric_cols: Lista de columnas numéricas

    Returns:
        Figura de Plotly
    """
    corr_matrix = df[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlación")
    ))

    fig.update_layout(
        title='Matriz de Correlación',
        template='plotly_white',
        width=800,
        height=800
    )

    return fig


def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """
    Gráfico de importancia de features.

    Args:
        importance_df: DataFrame con columnas 'feature' e 'importance'

    Returns:
        Figura de Plotly
    """
    fig = go.Figure(data=[
        go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color='#3498db'
        )
    ])

    fig.update_layout(
        title='Importancia de Variables en el Modelo',
        xaxis_title='Importancia',
        yaxis_title='Variable',
        template='plotly_white',
        height=500
    )

    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Gráfico de comparación de modelos.

    Args:
        comparison_df: DataFrame con métricas de modelos

    Returns:
        Figura de Plotly
    """
    metrics = comparison_df.columns.tolist()
    models = comparison_df.index.tolist()

    fig = go.Figure()

    for model in models:
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=comparison_df.loc[model].values,
            text=comparison_df.loc[model].values.round(3),
            textposition='auto'
        ))

    fig.update_layout(
        title='Comparación de Modelos',
        xaxis_title='Métrica',
        yaxis_title='Valor',
        barmode='group',
        template='plotly_white'
    )

    return fig


def plot_chronic_conditions(df: pd.DataFrame) -> go.Figure:
    """
    Gráfico de condiciones crónicas.

    Args:
        df: DataFrame

    Returns:
        Figura de Plotly
    """
    conditions = ['Hipertension', 'Diabetes', 'Alcoholism']
    counts = [df[col].sum() for col in conditions]

    fig = go.Figure(data=[
        go.Bar(
            x=conditions,
            y=counts,
            text=counts,
            textposition='auto',
            marker_color=['#e67e22', '#9b59b6', '#34495e']
        )
    ])

    fig.update_layout(
        title='Prevalencia de Condiciones Crónicas',
        xaxis_title='Condición',
        yaxis_title='Número de Pacientes',
        template='plotly_white'
    )

    return fig
