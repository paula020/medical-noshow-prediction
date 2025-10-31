"""
Módulo para entrenamiento y evaluación de modelos de Machine Learning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Tuple, Any
import joblib


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Entrena un modelo Random Forest.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        random_state: Semilla para reproducibilidad

    Returns:
        Modelo Random Forest entrenado
    """
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    return rf_model


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> LogisticRegression:
    """
    Entrena un modelo de Regresión Logística.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        random_state: Semilla para reproducibilidad

    Returns:
        Modelo de Regresión Logística entrenado
    """
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver='lbfgs',
        n_jobs=-1
    )

    lr_model.fit(X_train, y_train)

    return lr_model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evalúa un modelo y retorna métricas de desempeño.

    Args:
        model: Modelo entrenado
        X_test: Features de prueba
        y_test: Target de prueba

    Returns:
        Diccionario con métricas de evaluación
    """
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    return metrics


def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: list,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Obtiene la importancia de las features de un modelo Random Forest.

    Args:
        model: Modelo Random Forest entrenado
        feature_names: Lista con nombres de features
        top_n: Número de features más importantes a retornar

    Returns:
        DataFrame con importancia de features
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

        # Crear DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return feature_importance_df.head(top_n)
    else:
        return pd.DataFrame()


def compare_models(
    results: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Compara los resultados de múltiples modelos.

    Args:
        results: Diccionario con resultados de cada modelo

    Returns:
        DataFrame con comparación de modelos
    """
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.round(4)

    return comparison_df


def save_model(model: Any, filepath: str) -> None:
    """
    Guarda un modelo entrenado en disco.

    Args:
        model: Modelo a guardar
        filepath: Ruta donde guardar el modelo
    """
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """
    Carga un modelo desde disco.

    Args:
        filepath: Ruta del modelo guardado

    Returns:
        Modelo cargado
    """
    return joblib.load(filepath)
