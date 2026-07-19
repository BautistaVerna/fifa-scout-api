# FIFA Scout API

Backend en **FastAPI** que sirve los dos modelos de Machine Learning del proyecto
FIFA Scout (Trabajo Práctico 4 — Inteligencia Artificial y Aprendizaje Automático I,
Licenciatura en Ciencias de Datos, UCA) a través de una API REST, para que un
frontend web pueda consumirlos en tiempo real sin necesidad de correr Python.

No reentrena ni redefine el análisis hecho en los TPs anteriores: carga los
artefactos ya entrenados y aplica exactamente la misma lógica de features,
filtros y umbrales que se usó durante el desarrollo del modelo.

- **API en producción:** https://fifa-scout-api.onrender.com
- **Documentación interactiva (Swagger):** https://fifa-scout-api.onrender.com/docs
- **Frontend que la consume:** https://fifa-scout.lovable.app ([repo](https://github.com/BautistaVerna/evo-scout-ai))

## Arquitectura del proyecto

Este TP está dividido en **dos repositorios** en lugar de una única app Streamlit:

| Repo | Contenido | Por qué |
|---|---|---|
| **`fifa-scout-api`** (este repo) | Modelos serializados, dataset, lógica de ML, API REST | Es la parte en Python |
| [`evo-scout-ai`](https://github.com/BautistaVerna/evo-scout-ai) | Interfaz web (React) | Consume esta API por HTTP/JSON |

El frontend nunca toca el dataset ni los modelos directamente — todo pasa por
peticiones HTTP a esta API. Este repo es el que contiene el **modelo
serializado** y el **`requirements.txt`** pedidos en el entregable.

## Modelos servidos

| Modelo | Tarea | Algoritmo | Archivo serializado | Métrica |
|---|---|---|---|---|
| Regresor de potencial | Predicción (TP2) | Gradient Boosting Optimizado | `modelo_gb_pipeline.joblib` | ver `/model-info` |
| Clasificador de posición | Clasificación (TP3) | SVC (kernel RBF), exportado a ONNX | `svc_fifa_scout_lovable.onnx` | 87,15 % accuracy en test — el mejor de 5 modelos comparados (Naive Bayes, KNN, Árbol de Decisión, Regresión Logística, SVC) |

Se desplegaron **ambos modelos** en la misma aplicación (predicción y
clasificación), permitiendo al usuario elegir qué análisis realizar desde el
frontend.

El escalado (`StandardScaler`) del clasificador queda embebido en el propio
grafo ONNX, así que la API solo arma el vector de features crudas en el orden
exacto de entrenamiento antes de llamar a `onnxruntime`.

## Ejecución local

Requisitos: Python 3.11.

```bash
git clone https://github.com/BautistaVerna/fifa-scout-api.git
cd fifa-scout-api

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

uvicorn main:app --reload
```

La API queda arriba en `http://127.0.0.1:8000`. Para confirmar que cargó bien
el dataset y los modelos:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","jugadores_en_dataset":17699}
```

Documentación interactiva de todos los endpoints en
`http://127.0.0.1:8000/docs` (Swagger UI, autogenerada por FastAPI).

## Endpoints principales

| Endpoint | Función |
|---|---|
| `GET /health` | Chequeo de vida. |
| `GET /filters/options`, `GET /features/ranges` | Metadata para armar los formularios del frontend. |
| `GET /players` | Lista filtrada de jugadores + KPIs + puntos para el scatter. |
| `GET /players/hidden-gems` | Jugadores jóvenes/baratos con alto potencial predicho. |
| `GET /players/position-stats` | Agregados por posición sobre la población filtrada. |
| `GET /players/{id}` | Ficha completa de un jugador. |
| `POST /predict`, `POST /predict-batch` | Predicción de potencial: individual o por CSV. |
| `GET /model-info`, `GET /template.csv` | Metadata del regresor y plantilla para carga en lote. |
| `GET /classifier/status`, `GET /classifier/ranges` | Disponibilidad y rangos válidos del clasificador. |
| `POST /classifier/predict` | Predicción de posición natural a partir de atributos de un jugador. |

## Despliegue

Desplegado en [Render](https://render.com) (plan free), con auto-deploy en cada
push a `main`. Ver [`DEPLOY_API.md`](DEPLOY_API.md) para el paso a paso y los
problemas puntuales que aparecieron al desplegar (versión de Python, wheels de
`scipy`, etc.).

## Dataset

`fifa_players_model_ready.csv` — 17.699 jugadores, el mismo dataset preprocesado
usado en TP1–TP3, incluido en este repo para que la app pueda alimentarse de él
sin depender de una fuente externa.
