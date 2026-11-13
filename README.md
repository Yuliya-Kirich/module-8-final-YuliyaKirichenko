
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