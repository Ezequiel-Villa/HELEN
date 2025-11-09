# Helen – Arquitectura distribuida para inferencia en la nube

Este repositorio contiene todos los componentes necesarios para ejecutar Helen
separando la captura/UI en una Raspberry Pi y la inferencia pesada en un
servidor Flask alojado en AWS EC2.  A continuación se describe la arquitectura,
los archivos relevantes y los pasos para poner en marcha ambos extremos.

---

## 1. Componentes clave del repositorio

| Rol | Archivo/Directorio | Descripción |
| --- | --- | --- |
| API de inferencia en la nube | `cloud_inference_server.py` | Servidor Flask que carga el modelo PyTorch una sola vez y expone los endpoints `/health`, `/metadata` y `/predict`. |
| Utilidades del modelo | `model/cloud_inference.py` | Envuelve la carga del checkpoint y ofrece `run_inference` reutilizable. |
| Cliente HTTP para la Raspberry | `model/cloud_inference_client.py` | Gestiona peticiones a la API remota con reintentos y caché de metadata. |
| Pipeline de captura en Raspberry | `model/inference_classifier_video.py` | Usa MediaPipe para obtener landmarks, genera features y consulta el backend remoto. |
| Envío de gestos al backend local | `model/backendConexion.py` | Publica gestos detectados hacia la UI/WebSocket local (URL configurable por entorno). |
| Frontend web local | `backend/server.py` + `frontend/` | Mantiene la UI, eventos SSE y Socket.IO para mostrar el estado. |

---

## 2. Arquitectura distribuida

1. **Raspberry Pi 5 (frontend)**
   - Captura frames de cámara y obtiene landmarks con MediaPipe.
   - Construye secuencias de características y las envía por HTTP al backend en EC2 usando `model/cloud_inference_client.py`.
   - Recibe probabilidades/etiquetas y decide cuándo disparar una acción, que se publica al backend/UI local mediante `model/backendConexion.py`.

2. **AWS EC2 (backend de inferencia)**
   - Instancia Ubuntu 22.04 (t3.micro, Python 3.11 sin GPU).
   - Ejecuta `cloud_inference_server.py`, el cual carga el checkpoint GRU (`model/models/sequence_bigru.pt`) una sola vez.
   - Expone el endpoint `POST /predict` para recibir secuencias de landmarks (normalizados) y devolver la clase predicha.

3. **Formato de datos intercambiado**
   - Request JSON:
     ```json
     {
       "sequence": [[...], [...], ...],   // lista de T vectores (F_total)
       "mask": [[m_left, m_right], ...]   // opcional, presencia por mano
     }
     ```
   - Response JSON:
     ```json
     {
       "prediction": "Inicio",
       "probabilities": [0.01, 0.90, ...],
       "top_index": 1,
       "top_probability": 0.90,
       "elapsed_ms": 12.4
     }
     ```

---

## 3. Backend de inferencia en EC2

1. **Clonar y preparar entorno**
   ```bash
   git clone <URL_DE_MI_REPO>
   cd HELEN
   python3.11 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Variables de entorno útiles**
   - `HELEN_MODEL_PATHS`: lista separada por `:` con rutas al checkpoint (opcional, por defecto se busca en `model/models/sequence_bigru.pt`).
   - `HELEN_LOG_LEVEL`: nivel de logging (`INFO` por defecto).

3. **Ejecución en modo desarrollo**
   ```bash
   source venv/bin/activate
   python cloud_inference_server.py
   ```
   - Servirá en `http://0.0.0.0:5000/`.
   - Endpoints disponibles: `/health`, `/metadata`, `/predict`.

4. **Ejecución en producción con Gunicorn**
   ```bash
   source venv/bin/activate
   gunicorn cloud_inference_server:app --bind 0.0.0.0:5000 --workers 2
   ```
   Ajusta el número de workers según recursos.  Gunicorn reutiliza la carga del modelo sin recrearlo.

5. **Comprobación rápida desde la propia instancia**
   ```bash
   curl http://127.0.0.1:5000/health
   curl http://127.0.0.1:5000/metadata
   ```

---

## 4. Cliente de inferencia en Raspberry Pi

1. **Preparar entorno**
   ```bash
   git clone <URL_DE_MI_REPO>
   cd HELEN
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Configurar endpoints mediante variables**
   ```bash
   export HELEN_CLOUD_API_URL="http://<IP_EC2>:5000"
   export HELEN_BACKEND_GESTURE_URL="http://127.0.0.1:5000/gestures/gesture-key"  # UI local
   ```
   Opcionalmente ajusta `HELEN_CLOUD_TIMEOUT`, `HELEN_CLOUD_RETRIES` o `HELEN_CLOUD_METADATA_TTL` si necesitas timeouts más amplios.

3. **Ejecutar el pipeline de captura**
   ```bash
   source venv/bin/activate
   python model/inference_classifier_video.py
   ```
   El script obtendrá la metadata del backend remoto al arrancar, comenzará a capturar la cámara y enviará secuencias al endpoint `/predict`.

---

## 5. Pruebas manuales del endpoint `/predict`

Con el servidor corriendo, puedes probar la API desde cualquier máquina con Python:

```python
import requests

API_URL = "http://<IP_EC2>:5000/predict"
payload = {
    "sequence": [[0.1] * 84] * 10,  # 10 frames, 84 features (solo ejemplo)
    "mask": [[1.0, 0.0]] * 10
}
response = requests.post(API_URL, json=payload, timeout=5)
print(response.json())
```

Si los `feature_dim` o la longitud de la secuencia no coinciden con lo esperado, el servidor devolverá un `HTTP 400` con el detalle del error.

---

## 6. Variables de entorno relevantes

| Variable | Uso | Valor por defecto |
| --- | --- | --- |
| `HELEN_MODEL_PATHS` | Rutas alternativas al checkpoint del modelo en el servidor. | `model/models/sequence_bigru.pt:models/sequence_bigru.pt:/mnt/data/sequence_bigru.pt` |
| `HELEN_LOG_LEVEL` | Nivel de log del servidor Flask. | `INFO` |
| `HELEN_CLOUD_API_URL` | URL base del backend de inferencia vista desde la Raspberry. | `http://127.0.0.1:8000` |
| `HELEN_CLOUD_TIMEOUT` | Timeout (s) por petición del cliente. | `5.0` |
| `HELEN_CLOUD_RETRIES` | Reintentos automáticos ante fallos temporales. | `2` |
| `HELEN_CLOUD_METADATA_TTL` | Tiempo (s) que se cachea la metadata del modelo. | `300` |
| `HELEN_BACKEND_GESTURE_URL` | Endpoint del backend local para publicar gestos detectados. | `http://127.0.0.1:5000/gestures/gesture-key` |

---

## 7. Advertencias y recomendaciones

- `t3.micro` dispone de recursos limitados; espera latencias entre 10–40 ms por inferencia dependiendo de la longitud de la secuencia y la carga de red.
- El modelo opera sobre landmarks normalizados (84 valores si solo se usa XY, 126 si incluye Z). Asegúrate de que la Raspberry mantenga el mismo preprocesamiento que el entrenamiento.
- Maneja interrupciones de red: el cliente devuelve `None` si falla la comunicación y el pipeline omite ese frame.
- Si deseas reducir el ancho de banda, mantén el envío de landmarks en lugar de frames completos y considera comprimir a `float16` antes de enviarlos (se podría extender fácilmente en el cliente/servidor).
- Para despliegues públicos, coloca el servidor detrás de un `reverse proxy` (NGINX) con HTTPS y aplica autenticación si es necesario.

---

¡Listo! Con estos pasos tendrás la inferencia pesada en EC2 y la interacción con el usuario en la Raspberry Pi.
