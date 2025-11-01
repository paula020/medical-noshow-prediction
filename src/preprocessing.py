"""
Módulo para pipelines de preprocesamiento con ColumnTransformer
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List


def get_feature_columns() -> Tuple[List[str], List[str], List[str]]:
    """
    Define las columnas numéricas y categóricas para el pipeline.

    Returns:
        Tupla con (columnas numéricas, columnas categóricas, columnas a eliminar)
    """
    # Variables numéricas
    numeric_features = [
        'Age',
        'DaysAdvance',
        'AppointmentWeekday',
        'AppointmentMonth',
        'ChronicConditionsCount'
    ]

    # Variables categóricas
    categorical_features = [
        'Gender',
        'Scholarship',
        'Hipertension',
        'Diabetes',
        'Alcoholism',
        'Handcap',
        'SMS_received'
    ]

    # Variables a eliminar (no relevantes para el modelo)
    drop_features = [
        'PatientId',
        'AppointmentID',
        'ScheduledDay',
        'AppointmentDay',
        'Neighbourhood'  # Demasiadas categorías (81), se elimina para simplificar
    ]

    return numeric_features, categorical_features, drop_features


def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Crea el pipeline de preprocesamiento con ColumnTransformer.

    Returns:
        ColumnTransformer configurado
    """
    numeric_features, categorical_features, _ = get_feature_columns()

    # Pipeline para variables numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para variables categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # ColumnTransformer que combina ambos pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor


def prepare_features_target(
    df: pd.DataFrame,
    target_col: str = 'No-show'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa las features y el target, eliminando columnas irrelevantes.

    Args:
        df: DataFrame completo
        target_col: Nombre de la columna objetivo

    Returns:
        Tupla (X, y) con features y target
    """
    _, _, drop_features = get_feature_columns()

    # Eliminar columnas irrelevantes
    X = df.drop(columns=drop_features + [target_col])

    # Variable objetivo
    y = df[target_col]

    return X, y


def get_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame) -> List[str]:
    """
    Obtiene los nombres de las features después del preprocesamiento.

    Args:
        preprocessor: ColumnTransformer fitted
        X: DataFrame de features originales

    Returns:
        Lista con nombres de features transformadas
    """
    numeric_features, categorical_features, _ = get_feature_columns()

    # Nombres de features numéricas
    numeric_names = numeric_features

    # Nombres de features categóricas después de OneHotEncoder
    categorical_names = []
    if hasattr(preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
        categorical_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
            categorical_features
        ).tolist()

    return numeric_names + categorical_names
