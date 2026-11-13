
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd


MODEL_PATH = Path("model.onnx")
DATA_PATH = Path("src/uber.csv")
REPORT_PATH = Path("profile_onnx_YuliyaKirichenko.json")

FEATURES = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "passenger_count",
]

RUNS = 1000
WARMUP_RUNS = 100


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES)

    if df.empty:
        raise ValueError("Нет данных для тестирования")

    sample = df[FEATURES].iloc[[0]].astype(np.float32)

    inputs = {
        feature: sample[[feature]].to_numpy(dtype=np.float32)
        for feature in FEATURES
    }

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    for _ in range(WARMUP_RUNS):
        session.run(None, inputs)

    latencies = []

    for _ in range(RUNS):
        start = time.perf_counter()
        session.run(None, inputs)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    latencies = np.array(latencies)

    results = {
        "runs": RUNS,
        "min_s": float(np.min(latencies)),
        "mean_s": float(np.mean(latencies)),
        "p50_s": float(np.percentile(latencies, 50)),
        "p95_s": float(np.percentile(latencies, 95)),
        "p99_s": float(np.percentile(latencies, 99)),
        "max_s": float(np.max(latencies)),
    }

    REPORT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))

    p99_ms = results["p99_s"] * 1000

    print(f"\nP99 latency: {p99_ms:.4f} ms")

    if p99_ms <= 20:
        print("PASS: P99 <= 20 ms")
    else:
        print("FAIL: P99 > 20 ms")

    print(f"Отчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    main()
