"""
Módulo para carga y limpieza inicial de datos
"""

import pandas as pd
from pathlib import Path
from typing import Tuple


def load_data(file_path: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.

    Args:
        file_path: Ruta al archivo CSV

    Returns:
        DataFrame con los datos cargados
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset: elimina duplicados, valores nulos y outliers críticos.

    Args:
        df: DataFrame original

    Returns:
        DataFrame limpio
    """
    df_clean = df.copy()

    # Eliminar duplicados
    df_clean = df_clean.drop_duplicates()

    # Eliminar valores nulos si existen
    df_clean = df_clean.dropna()

    # Corregir edad: eliminar edades negativas y mayores a 100
    df_clean = df_clean[(df_clean['Age'] >= 0) & (df_clean['Age'] <= 100)]

    return df_clean


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las columnas a los tipos de datos apropiados.

    Args:
        df: DataFrame original

    Returns:
        DataFrame con tipos de datos convertidos
    """
    df_converted = df.copy()

    # Convertir variables categóricas
    categorical_cols = [
        'Gender', 'Neighbourhood', 'Scholarship', 'Hipertension',
        'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received'
    ]
    df_converted[categorical_cols] = df_converted[categorical_cols].astype('category')

    # Convertir fechas
    df_converted['AppointmentDay'] = pd.to_datetime(df_converted['AppointmentDay'])
    df_converted['ScheduledDay'] = pd.to_datetime(df_converted['ScheduledDay'])

    # Convertir variable objetivo a binaria
    df_converted['No-show'] = df_converted['No-show'].map({'No': 0, 'Yes': 1})

    return df_converted


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea nuevas features a partir de los datos existentes.

    Args:
        df: DataFrame con fechas convertidas

    Returns:
        DataFrame con nuevas features
    """
    df_features = df.copy()

    # Días de anticipación entre el agendamiento y la cita
    df_features['DaysAdvance'] = (
        df_features['AppointmentDay'] - df_features['ScheduledDay']
    ).dt.days

    # Día de la semana de la cita (0=Lunes, 6=Domingo)
    df_features['AppointmentWeekday'] = df_features['AppointmentDay'].dt.dayofweek

    # Mes de la cita
    df_features['AppointmentMonth'] = df_features['AppointmentDay'].dt.month

    # Crear variable de múltiples condiciones crónicas
    chronic_conditions = ['Hipertension', 'Diabetes', 'Alcoholism']
    df_features['ChronicConditionsCount'] = df_features[chronic_conditions].sum(axis=1)

    return df_features


def prepare_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completo de preparación de datos.

    Args:
        file_path: Ruta al archivo CSV

    Returns:
        Tupla con (datos procesados, datos originales)
    """
    # Cargar datos
    df_original = load_data(file_path)

    # Limpiar datos
    df_clean = clean_data(df_original)

    # Convertir tipos de datos
    df_converted = convert_dtypes(df_clean)

    # Crear features
    df_final = create_features(df_converted)

    return df_final, df_original
