FIFA Scout — reemplazá {{API_URL}} en todo el prompt por la URL real de tu API (ej: https://fifa-scout-api.onrender.com) antes de pegarlo en Lovable.

---

Quiero construir "FIFA Scout", una app web de scouting de jugadores de fútbol con Machine Learning. Es un producto de datos, no un juego: la audiencia son scouts/analistas que exploran un dataset de ~17.700 jugadores y usan un modelo entrenado (ya existente, servido por una API externa) para predecir el potencial de crecimiento de un jugador.

No hay backend propio que programar: toda la data y las predicciones vienen de una API REST ya desplegada en {{API_URL}}. La app es un frontend puro (React + Vite + TypeScript + Tailwind + shadcn/ui) que consume esa API con fetch.

## Sistema de diseño — "dark sports-tech"

Aplicar estos tokens como variables CSS / config de Tailwind, no hardcodear hex sueltos en los componentes:

- Fondo de página: #0B1220
- Superficie de tarjetas: #16213B
- Superficie elevada (hero, header): #1B2A47
- Borde sutil: #2C3B57
- Texto principal: #F3F4F6
- Texto secundario/muted: #94A3B8
- Acento primario (botones, badges — SIEMPRE con texto oscuro encima, no blanco): #16A34A
- Acento claro (íconos, gráficos, hover): #22C55E
- Texto sobre fondo verde: #08170D
- Dorado (tier "élite", usar con moderación): #D4AF37
- Ámbar (alertas suaves): #F59E0B
- Azul neutro (gráficos de conteo/ranking sin carga semántica): #5B8DEF

Tipografía: "Sora" (600/700/800) para títulos y números grandes, "Inter" (400-700) para cuerpo y datos, ambas de Google Fonts. Números en tablas y KPIs con `font-variant-numeric: tabular-nums`.

Íconos: **lucide-react**, trazo 1.5-2px, sin emojis en ningún lugar de la interfaz.

Estilo general: minimalista, tarjetas con `border-radius` 10-14px, borde 1px sutil (`#2C3B57`), sin gradientes decorativos salvo un detalle: el header de cada página ("hero") lleva un borde izquierdo de 4px en el verde claro (#22C55E) como acento de marca. Foco visible obligatorio en todo elemento interactivo (outline 2px verde claro) — nunca usar `outline: none` sin reemplazo. Botón primario: fondo verde `#16A34A`, texto oscuro `#08170D`, hover a `#22C55E`.

## Estructura de la app

Layout con sidebar fijo a la izquierda (colapsable) + contenido principal. En el sidebar: logo (ícono `circle` o similar de lucide con un patrón simple, no hace falta pelota literal) + "FIFA Scout" + "Scouting con Machine Learning", debajo la navegación con 5 ítems (ícono + texto), cada uno resaltado cuando está activo con el acento verde. Router: usar rutas reales (`/`, `/predictor`, `/lote`, `/clasificador`, `/modelo`) para que el estado de navegación quede en la URL.

Cada página empieza con un "hero header": tarjeta con ícono de lucide (26px, color verde claro) + título (Sora, 1.4rem) + subtítulo (Inter, color muted), fondo `#1B2A47`, borde izquierdo verde.

### Página 1 — Explorador de Joyas (`/`)

Ícono: `search`.

- Sidebar de filtros (o panel colapsable arriba en mobile): multiselect de posición, slider de rango de edad (15-45), slider de rango de valor de mercado en millones €, multiselect de nacionalidad, slider de overall mínimo (50-99). Las opciones de posición/nacionalidad salen de `GET {{API_URL}}/filters/options`.
- Al cambiar cualquier filtro, llamar `GET {{API_URL}}/players` con los filtros como query params (`posiciones`, `edad_min`, `edad_max`, `valor_min`, `valor_max`, `nacionalidades`, `overall_min`, `limit=500`).
- 3 tarjetas KPI arriba: "Jugadores filtrados", "Total en el dataset", "Potencial promedio" — vienen en `respuesta.kpis`.
- Scatter chart (usar `recharts`, `ScatterChart`): eje X = valor de mercado en millones, eje Y = potencial predicho, color por edad (escala de `#22C55E` → `#D4AF37` → `#F59E0B`), tamaño de punto por overall. Datos en `respuesta.scatter`. Fondo del chart `#16213B`, grilla `#22304A`, texto `#94A3B8`.
- Debajo, sección "Joyas Ocultas" (ícono `gem`): llamar `GET {{API_URL}}/players/hidden-gems` con los mismos filtros. Si `criterios_relajados` no es null, mostrar un aviso explicando que se relajaron los criterios. Tabla con nombre, nacionalidad, club, edad, overall, potencial predicho (con barra de progreso), valor formateado en € (M/K).

### Página 2 — Predictor Individual (`/predictor`)

Ícono: `zap`.

- Formulario en 3 columnas (tarjetas con borde), sliders con rango real traído de `GET {{API_URL}}/features/ranges`: edad, overall rating, visión, agilidad, entrada de pie, fuerza | reputación internacional (1-5), pie malo (1-5), gambeta (1-5), índice de ataque, índice de defensa | índice de juego asociado, índice físico, selects de cláusula de rescisión (sí/no), pie preferido, nacionalidad (de `/filters/options`), posición específica (de `/filters/options`).
- Botón primario "Predecir potencial" → `POST {{API_URL}}/predict` con el payload (ver forma exacta abajo). Mostrar loading state mientras espera (el primer request puede tardar hasta 50s si el servidor estaba dormido — mostrar mensaje "Conectando con el servidor, puede tardar unos segundos…").
- Resultado: número grande (Sora, 4rem) con el potencial predicho, badge con la clasificación (`clasificacion`), gauge/velocímetro (rango 50-99, colores por tramo: gris `#2C3B57` hasta 65, azul `#5B8DEF` hasta 82, verde `#16A34A` hasta 88, dorado `#D4AF37` hasta 99).
- Si `top_features` no es null, mostrar un bar chart horizontal con las variables de mayor impacto (`feature` + `impacto`), barras verdes para impacto positivo y ámbar para negativo.

Forma del payload de `/predict`:
```json
{
  "age": 22, "overall_rating": 70, "vision": 55, "agility": 65,
  "standing_tackle": 48, "strength": 65,
  "international_reputation": 1, "weak_foot": 3, "skill_moves": 2,
  "attack_score": 48.0, "defense_score": 48.0,
  "playmaking_score": 55.0, "physical_score": 65.0,
  "has_release_clause": true, "preferred_foot": "Derecho",
  "nacionalidad": "Argentina", "posicion": "Arquero"
}
```
Respuesta: `{ "potencial_predicho": 79.1, "clasificacion": "Titular sólido", "top_features": [{"feature": "overall_rating", "impacto": 4.2}, ...] | null }`

### Página 3 — Análisis en Lote (`/lote`)

Ícono: `folder-open`.

- Botón "Descargar plantilla CSV" → enlaza directo a `GET {{API_URL}}/template.csv`.
- Dropzone de carga de un CSV (drag & drop + botón "Elegir archivo"), texto en español, límite razonable (~10MB, avisar si supera).
- Al subir, `POST {{API_URL}}/predict-batch` como `multipart/form-data` con el campo `archivo`.
- Mostrar loading, luego: si hay error (400), mostrar el mensaje tal cual viene del backend (dice qué columnas faltan). Si OK: 4 KPIs (jugadores procesados, potencial promedio/máximo/mínimo), aviso si hubo `filas_invalidas_excluidas`, tabla con todos los `resultados` (paginada si son muchas filas), botón para descargar esos resultados como CSV (generarlo client-side con los datos ya recibidos).

### Página 4 — Clasificador de Posición (`/clasificador`)

Ícono: `crosshair` o `target`.

- Llamar `GET {{API_URL}}/classifier/status`. Como hoy `disponible` es `false`, mostrar un estado vacío claro: ícono, "Esta funcionalidad está en desarrollo y se habilitará próximamente", sin exponer nombres de archivos ni jerga técnica.
- Debajo, gráfico de barras horizontal con la distribución de posiciones del dataset (`GET {{API_URL}}/model-info` → `distribucion_posiciones`), color azul neutro `#5B8DEF`.
- Dejar el componente preparado (con un comentario) para cuando `disponible` sea `true`: en ese caso debería mostrar un formulario similar al Predictor Individual.

### Página 5 — Sobre el Modelo (`/modelo`)

Ícono: `bar-chart-3`.

Todo sale de `GET {{API_URL}}/model-info`:
- Tabla "Comparación de modelos evaluados" (`comparativa_modelos`): columnas Modelo, R² Test (con barra de progreso 0.8-1.0), MAE, RMSE, Gap train-test (%).
- Bar chart horizontal de R² por modelo, color continuo de gris `#94A3B8` a verde claro `#22C55E`.
- Texto explicando por qué se eligió Gradient Boosting Optimizado, usando `hiperparametros_gb` (learning_rate, max_depth, n_estimators, subsample) y las métricas del modelo ganador.
- Bar chart horizontal "Importancia de variables" (`importancia_variables`), azul neutro `#5B8DEF`, ordenado descendente.
- Histograma "Distribución de residuos" (`residuos_histograma`: `conteos` + `bordes`), verde claro `#22C55E`.
- 3 KPIs "Sobre los datos" (`dataset`: total_jugadores, rango de edad, nacionalidades_distintas).
- Bar chart horizontal de distribución de posiciones (mismo dato que en la página 4), azul neutro.

## Manejo de errores y estados

- Todas las páginas necesitan estado de loading (skeleton o spinner con texto descriptivo) y estado de error (mensaje claro + botón "Reintentar"), porque dependen de una API externa que puede tardar en el primer request (cold start del hosting gratuito).
- Nunca mostrar una pantalla en blanco silenciosa si un fetch falla.

## No tocar

Los umbrales de "joyas ocultas", las métricas de comparación de modelos, los hiperparámetros y la lógica de features vienen fijos desde la API — no inventar ni recalcular nada de eso en el frontend, solo mostrar lo que devuelve cada endpoint.
