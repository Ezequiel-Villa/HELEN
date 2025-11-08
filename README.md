# Proyecto Helen - Guía Rápida de Instalación y Ejecución

Bienvenido a Helen, un proyecto que combina un backend en Flask con un modelo de inferencia en tiempo real.  
Este documento explica cómo configurar el entorno, instalar dependencias y ejecutar todo el sistema paso a paso.

---

## 1. Requisitos previos

Antes de comenzar, asegúrate de tener instalado lo siguiente:
- Python 3.12 o superior  
- pip actualizado (`python -m pip install --upgrade pip`)  
- Archivo `requirements.txt` en la raíz del proyecto  

Verificar versión de Python:
```
python --version
```

---

## 2. Instalación del entorno virtual

### 🟢 Opción automática (recomendada)

Desde la carpeta raíz del proyecto, ejecuta:
```
powershell -ExecutionPolicy Bypass -File .\scripts\helen-setup.ps1
```

El script realiza automáticamente:
1. Crea la carpeta `venv` si no existe  
2. Instala las dependencias de `requirements.txt`  
3. Verifica la instalación de Python y pip  
4. Configura el entorno para su uso  

Al finalizar verás algo como:
```
[HELEN] Setup completado correctamente.
Entorno: venv
Python:  venv\Scripts\python.exe
```

---

### ⚙️ Opción manual (instalación paso a paso)

1. Crear entorno virtual:
```
python -m venv venv
```

2. Activar entorno virtual:
```
.\venv\Scripts\activate
```

3. Actualizar pip:
```
python -m pip install --upgrade pip
```

4. Instalar dependencias:
```
pip install -r requirements.txt
```

---

## 3. Ejecución del proyecto

El sistema Helen está compuesto por dos procesos:
1. Backend (Flask) → `backendHelen/server.py`
2. Modelo de inferencia → `Hellen_model_RN/inference_classifier_video.py`

Ambos deben ejecutarse al mismo tiempo.

---

### 🟢 Opción automática (recomendada)

Ejecuta el siguiente comando:
```
powershell -ExecutionPolicy Bypass -File .\scripts\helen-run.ps1
```

El script hará lo siguiente:
- Iniciará el backend (Flask) en el puerto **5000**  
- Iniciará el modelo de inferencia  
- Abrirá automáticamente el navegador en `http://localhost:5000`  
- Guardará los logs en la carpeta `reports/logs/`  

Para detener la ejecución: presiona **Ctrl + C** en la terminal.

Ejemplo de salida:
```
[HELEN] Lanzando backend...
[HELEN] Lanzando modelo...
[HELEN] Backend PID=4524 | Modelo PID=7712
[HELEN] Logs:
  - backend-20251107.out.log
  - model-20251107.out.log
=== Presiona Ctrl+C para detener ===
```

---

### ⚙️ Opción manual (dos terminales)

Abre dos terminales y activa el entorno virtual en ambas.

**Terminal 1:**
```
.\venv\Scripts\python.exe .\backendHelen\server.py
```

**Terminal 2:**
```
.\venv\Scripts\python.exe .\Hellen_model_RN\inference_classifier_video.py
```

---

## 4. Logs

Los registros se guardan automáticamente en:
```
reports/logs/
```

Ejemplo:
```
backend-YYYYMMDD-HHMMSS.out.log
backend-YYYYMMDD-HHMMSS.err.log
model-YYYYMMDD-HHMMSS.out.log
model-YYYYMMDD-HHMMSS.err.log
```

Ver logs en tiempo real:
```
Get-Content reports/logs/backend-*.out.log -Wait
```

---

## 5. Solución de problemas comunes

- **Python no se reconoce:**  
  Reinstala Python y marca “Add to PATH” durante la instalación.

- **Puerto 5000 en uso:**  
  Ejecuta el script con otro puerto:  
  ```
  .\scripts\helen-run.ps1 -Port 5050
  ```

- **El entorno virtual no existe:**  
  Ejecuta nuevamente:  
  ```
  .\scripts\helen-setup.ps1
  ```

- **PowerShell no permite ejecutar scripts:**  
  Ejecuta:  
  ```
  Set-ExecutionPolicy -Scope Process Bypass
  ```

---

## 6. Recomendación final

Primera ejecución (o después de clonar el proyecto):
```
.\scripts\helen-setup.ps1
```

Cada vez que trabajes o desarrolles:
```
.\scripts\helen-run.ps1
```

Esto automatiza todo el proceso y evita pasos manuales.

---

## 7. Instrucciones para Linux o macOS (opcional)

Instalación:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Ejecución:
```
# Terminal 1
python3 backendHelen/server.py

# Terminal 2
cd Hellen_model_RN && python3 inference_classifier_video.py
```

---

✅ **FIN**
