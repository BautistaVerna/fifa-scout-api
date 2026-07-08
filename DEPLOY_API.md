# Desplegar la API de FIFA Scout (Render, gratis)

Esta carpeta (`fifa-scout-api`) contiene un backend FastAPI que expone el mismo
modelo (`modelo_gb_pipeline.joblib`) y el mismo dataset que usaba la app de
Streamlit, a través de una API REST. Lovable va a consumir esta API — no
necesita correr Python.

## Paso 1 — Subir esta carpeta a un repo de GitHub

Puede ser el mismo repo `FIFA-Scout` (en una carpeta `api/`) o uno nuevo,
por ejemplo `fifa-scout-api`. Contenido necesario:

```
main.py
requirements.txt
render.yaml
modelo_gb_pipeline.joblib
fifa_players_model_ready.csv
```

## Paso 2 — Crear el servicio en Render

1. Entrá a [render.com](https://render.com) y creá una cuenta (podés usar tu GitHub).
2. **New +** → **Web Service**.
3. Conectá el repo donde subiste la carpeta `fifa-scout-api`.
4. Si detecta `render.yaml`, va a completar todo solo. Si no, configurá a mano:
   - **Runtime**: Python 3
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
5. **Create Web Service**. La primera build tarda 3-5 minutos (instala scikit-learn, shap, etc).

## Paso 3 — Anotar la URL pública

Cuando termine el deploy, Render te da una URL fija, algo como:

```
https://fifa-scout-api.onrender.com
```

Probala abriendo `https://fifa-scout-api.onrender.com/docs` — ahí vas a ver
la documentación interactiva (Swagger) con todos los endpoints, y podés
probarlos desde el navegador sin escribir código.

Esa URL es la que le vas a pasar a Lovable (reemplazando el placeholder
`{{API_URL}}` en el prompt).

## Importante — plan gratuito

El plan free de Render "duerme" el servicio después de 15 minutos sin
tráfico. El primer request después de dormir tarda ~30-50 segundos en
responder (arranca el servidor). Los siguientes son instantáneos. Esto es
normal — avisale a Lovable que muestre un estado de carga claro
("conectando con el servidor…") para que no parezca que la app está rota.

## Endpoints disponibles

| Método | Ruta | Uso |
|---|---|---|
| GET | `/health` | chequeo de que el servicio está vivo |
| GET | `/filters/options` | posiciones y nacionalidades disponibles |
| GET | `/features/ranges` | min/max reales de cada atributo (para los sliders) |
| GET | `/players` | lista filtrada + KPIs + datos para el scatter (Explorador de Joyas) |
| GET | `/players/hidden-gems` | joyas ocultas (mismos criterios que la app original) |
| POST | `/predict` | predicción individual (Predictor Individual) |
| POST | `/predict-batch` | predicción en lote desde un CSV (Análisis en Lote) |
| GET | `/template.csv` | plantilla CSV descargable |
| GET | `/model-info` | comparativa de modelos, importancia de variables, residuos, dataset |
| GET | `/classifier/status` | estado del clasificador de posición (todavía no disponible) |
