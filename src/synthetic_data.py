"""
Módulo para generación de datos sintéticos usando SMOTE
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from typing import Tuple


def generate_synthetic_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera datos sintéticos usando SMOTE para balancear las clases.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        random_state: Semilla para reproducibilidad

    Returns:
        Tupla (X_resampled, y_resampled) con datos balanceados
    """
    # Verificar desbalance de clases
    unique, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique, counts))

    # Aplicar SMOTE solo si hay desbalance significativo
    minority_class_ratio = min(counts) / max(counts)

    if minority_class_ratio < 0.5:
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        return X_resampled, y_resampled
    else:
        return X_train, y_train


def get_synthetic_data_stats(
    X_original: np.ndarray,
    y_original: np.ndarray,
    X_synthetic: np.ndarray,
    y_synthetic: np.ndarray
) -> dict:
    """
    Calcula estadísticas sobre los datos sintéticos generados.

    Args:
        X_original: Features originales
        y_original: Target original
        X_synthetic: Features con datos sintéticos
        y_synthetic: Target con datos sintéticos

    Returns:
        Diccionario con estadísticas
    """
    original_counts = np.unique(y_original, return_counts=True)
    synthetic_counts = np.unique(y_synthetic, return_counts=True)

    stats = {
        'original_samples': len(y_original),
        'synthetic_samples': len(y_synthetic),
        'original_class_0': original_counts[1][0] if 0 in original_counts[0] else 0,
        'original_class_1': original_counts[1][1] if 1 in original_counts[0] else 0,
        'synthetic_class_0': synthetic_counts[1][0] if 0 in synthetic_counts[0] else 0,
        'synthetic_class_1': synthetic_counts[1][1] if 1 in synthetic_counts[0] else 0,
        'samples_added': len(y_synthetic) - len(y_original)
    }

    return stats
