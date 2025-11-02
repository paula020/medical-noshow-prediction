from pathlib import Path
import pandas as pd
import sys

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from src.models import load_all_models, load_model


def main() -> int:
    models_dir = root_path / "models"
    models = load_all_models(str(models_dir))
    preprocessor = load_model(str(models_dir / "preprocessor.pkl"))

    if not models:
        print("ERROR: No se encontraron modelos en models/")
        return 1
    if preprocessor is None:
        print("ERROR: No se pudo cargar el preprocessor.pkl")
        return 1

    # Entrada de prueba (coincide con el formulario de Streamlit)
    sample = pd.DataFrame({
        'Age': [35],
        'Gender': ['F'],
        'Scholarship': [1],
        'Hipertension': [0],
        'Diabetes': [0],
        'Alcoholism': [0],
        'Handcap': [0],
        'SMS_received': [1],
        'DaysAdvance': [7],
        'AppointmentWeekday': [0],
        'AppointmentMonth': [1],
        'ChronicConditionsCount': [0]
    })

    X = preprocessor.transform(sample)

    print(f"Modelos cargados: {', '.join(models.keys())}")
    successes = 0
    for name, model in models.items():
        # Algunos modelos (p.ej. LGBM en pipeline) requieren DataFrame original
        try:
            pred = int(model.predict(X)[0])
            proba = float(model.predict_proba(X)[0][1])
            print(f"{name}: pred={pred}, p_no_show={proba:.4f}")
            successes += 1
            continue
        except Exception as e1:
            try:
                pred = int(model.predict(sample)[0])
                proba = float(model.predict_proba(sample)[0][1])
                print(f"{name}: pred={pred}, p_no_show={proba:.4f}")
                successes += 1
                continue
            except Exception as e2:
                print(f"{name}: ERROR en prediccion ({e2})")

    if successes >= 1:
        print("SMOKE TEST OK")
        return 0
    else:
        print("SMOKE TEST FALLIDO: ninguna prediccion exitosa")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
