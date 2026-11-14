# Module 8 Сервис рекомендации цен на поездку такси.

## Структура проекта
```bash
├── src - сервис для запуска API и обучения модели
```

## Быстрый старт

- **Требования**: Python 3.12 (см. `.python-version`), macOS/zsh
- **Путь проекта**: `<path_to_project>`

### 1) Установка окружения
```bash
cd <path_to_project>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Запуск API
Вариант A (рекомендуется, с авто‑перезапуском):
```bash
cd <path_to_project>
source .venv/bin/activate
uvicorn main:app --app-dir <path_to_project>/src/service --host 0.0.0.0 --port 32000 --reload
```

Вариант B (через модуль Python):
```bash
cd <path_to_project>
source .venv/bin/activate
python -m src.service.main  
```

Проверка:
```bash
curl -s http://127.0.0.1:32000/
```
Документация Swagger будет по адресу:
- `http://127.0.0.1:32000/docs`

Пример запроса на предсказание:
```bash
curl -s -X POST http://127.0.0.1:32000/api/predict/ \
  -H 'Content-Type: application/json' \
  -d '{
        "data": [
          {
            "pickup_latitude": 40.7614327,
            "pickup_longitude": -73.9798156,
            "dropoff_latitude": 40.6413111,
            "dropoff_longitude": -73.7781391,
            "passenger_count": 2
          }
        ]
      }'
```

UI-форма (простая HTML):
- `http://127.0.0.1:32000/predict/form/`

### 3) Обучение модели
Скрипт обучения читает датасет `src/uber.csv` и сохраняет модель в `src/service/model.pkl`.

Быстрый запуск по умолчанию:
```bash
cd <path_to_project>
source .venv/bin/activate
python <path_to_project>/src/service/train_and_save_model.py
```

С указанием параметров:
```bash
python <path_to_project>/src/service/train_and_save_model.py \
  --csv <path_to_project>/src/uber.csv \
  --output <path_to_project>/src/service/model.pkl \
  --test-size 0.2 \
  --random-state 42
```

- **Ожидаемые колонки** в CSV: `pickup_latitude`, `pickup_longitude`, `dropoff_latitude`, `dropoff_longitude`, `passenger_count`, таргет — `fare_amount`.
- После обучения сервис автоматически начнёт использовать обновлённый `model.pkl` при следующем запуске.

### 4) Полезное
- Установка зависимостей повторно: `pip install -r requirements.txt`
- Запуск тестов: `pytest`\


## 5) KServe
Проверка InferenceService:

```bash
kubectl get inferenceservice -n std
```

![InferenceService.png](docs%2FInferenceService.png)


### ServiceAccount и Secret

```bash
kubectl get serviceaccount taxi-s3-sa -n std
kubectl get secret taxi-s3-secret -n std
```

![ServiceAccount_Secret.png](docs%2FServiceAccount_Secret.png)

### Проверка API V2

```bash
curl -sS -i -X POST \
  http://127.0.0.1:8082/v2/models/taxi-price-inference/infer \
  -H 'Host: taxi-price-inference-predictor.std.svc.cluster.local' \
  -H 'Content-Type: application/json' \
  -d @k8s/inference_request.json
```
Ответ API v2
![inference-v2.png](docs%2Finference-v2.png)

## 6) ONNX

Конвертация и профилирование модели:

```bash
python convert_to_onnx.py
python benchmark_onnx.py
```

Результаты 1000 измерений сохранены в [profile_onnx_YuliyaKirichenko.json](profile_onnx_YuliyaKirichenko.json).

| Метрика | Время, мс |
|---|---:|
| Mean | 0.01560 |
| p50 | 0.01550 |
| p95 | 0.01629 |
| p99 | 0.02143 |

Результаты ONNX
![onnx-profile.png](docs%2Fonnx-profile.png)

## 7) S3

Модель model.pickle загружена в Yandex Object Storage:
![s3-model.png](docs%2Fs3-model.png)

## 8) Автомасштабирование

Параметры Knative:

| Параметр | Значение |
|---|---|
| Metric | rps |
| Target | 50 |
| Target utilization | 70% |
| Min replicas | 1 |
| Max replicas | 5 |
| Initial scale | 2 |

![Knative.png](docs%2FKnative.png)

Результаты нагрузочного тестирования (`load_test.py`):

| Показатель | Значение |
|---|---:|
| Длительность | 60 секунд |
| Параллельных потоков | 12 |
| Успешных запросов | 29544 |
| Ошибок | 0 |
| Средний RPS | 492.25 |

Результаты нагрузочного тестирования
![load-test.png](docs%2Fload-test.png)