
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "http://127.0.0.1:8082/v2/models/taxi-price-inference/infer"
HOST = "taxi-price-inference-predictor.std.svc.cluster.local"

DURATION = 60
WORKERS = 12

REQUEST_FILE = Path(__file__).parent / "k8s" / "inference_request.json"


def send_requests(deadline, payload):
    success = 0
    errors = 0

    while time.monotonic() < deadline:
        request = urllib.request.Request(
            URL,
            data=payload,
            headers={
                "Host": HOST,
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status == 200:
                    success += 1
                else:
                    errors += 1
        except Exception:
            errors += 1

    return success, errors


def main():
    payload = REQUEST_FILE.read_bytes()

    print(f"Нагрузочный тест: {WORKERS} потоков, {DURATION} секунд")

    start = time.monotonic()
    deadline = start + DURATION

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = list(
            executor.map(
                lambda _: send_requests(deadline, payload),
                range(WORKERS),
            )
        )

    elapsed = time.monotonic() - start
    success = sum(item[0] for item in results)
    errors = sum(item[1] for item in results)

    print(f"Успешных запросов: {success}")
    print(f"Ошибок: {errors}")
    print(f"Средний RPS: {(success + errors) / elapsed:.2f}")


if __name__ == "__main__":
    main()
