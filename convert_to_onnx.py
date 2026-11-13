
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


MODEL_PATH = Path("model.pickle")
ONNX_PATH = Path("model.onnx")
DATA_PATH = Path("src/uber.csv")

FEATURES = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "passenger_count",
]


def main():
    # 1. Загружаем обученную sklearn-модель
    model = joblib.load(MODEL_PATH)
    print("Sklearn-модель загружена")

    # 2. Проверяем порядок признаков
    actual_features = list(model.feature_names_in_)
    if actual_features != FEATURES:
        raise ValueError(
            f"Несовпадение признаков: {actual_features}"
        )

    # 3. Описываем входы ONNX
    initial_types = [
        (feature, FloatTensorType([None, 1]))
        for feature in FEATURES
    ]

    # 4. Конвертируем Pipeline
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=17,
    )

    # 5. Проверяем структуру ONNX
    onnx.checker.check_model(onnx_model)

    # 6. Сохраняем модель
    onnx.save_model(onnx_model, str(ONNX_PATH))
    print(f"ONNX-модель сохранена: {ONNX_PATH}")

    # 7. Загружаем тестовые данные
    df = pd.read_csv(DATA_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES).head(10)

    if df.empty:
        raise ValueError("Нет данных для проверки модели")

    X = df[FEATURES].astype(np.float32)

    # 8. Предсказания sklearn
    sklearn_predictions = model.predict(X)

    # 9. Предсказания ONNX Runtime
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    onnx_inputs = {
        feature: X[[feature]].to_numpy(dtype=np.float32)
        for feature in FEATURES
    }

    onnx_predictions = session.run(
        None, onnx_inputs
    )[0].ravel()

    # 10. Сравниваем результаты
    max_difference = np.max(
        np.abs(sklearn_predictions - onnx_predictions)
    )

    print("Sklearn:", sklearn_predictions[:5])
    print("ONNX:", onnx_predictions[:5])
    print("Максимальное расхождение:", max_difference)

    np.testing.assert_allclose(
        sklearn_predictions,
        onnx_predictions,
        rtol=1e-3,
        atol=1e-3,
    )

    print("Проверка пройдена: предсказания совпадают")


if __name__ == "__main__":
    main()
