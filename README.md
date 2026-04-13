🌿 AgroInsight — Análisis de Producción Agrícola Colombia

Proyecto de Inteligencia Artificial para predecir, clasificar y segmentar la producción agrícola municipal colombiana.
Bootcamp Talento Tech · IA Nivel Integrador · Universidad de Antioquia


📌 ¿Qué es AgroInsight?
AgroInsight es una aplicación de ciencia de datos que analiza 12 años de registros agrícolas colombianos (2006–2018) del Ministerio de Agricultura y Desarrollo Rural (MADR) para responder tres preguntas clave:
MóduloPreguntaTécnicaM1 — Regresión¿Cuánto va a producir este cultivo?Random Forest + Red Neuronal con EmbeddingsM2 — Clasificación¿Vale la pena sembrar?Random Forest + Red Neuronal con EmbeddingsM3 — Clustering¿Qué perfil agrícola tiene esta región?K-Means + Autoencoder

🗂️ Dataset

Fuente: datos.gov.co — Evaluaciones Agropecuarias Municipales EVA V1
ID: 2pnw-mmge
Período: 2006–2018
Registros limpios: ~163,000
Cobertura: 1,018 municipios · 32 departamentos · Colombia


🏗️ Arquitectura del proyecto
AgroInsight/
├── app.py                    # App Streamlit (demo interactiva)
├── requirements.txt          # Dependencias
├── SETUP.md                  # Instrucciones instalación local
├── notebook/
│   └── AgroInsight_Definitivo.ipynb   # Notebook completo Colab
├── modelos/                  # Modelos entrenados (.pkl, .keras, .h5)
├── graficas/                 # Gráficas exportadas (.png)
└── mapas/                    # Mapas interactivos Plotly (.html)

🤖 Modelos
M1 — Regresión de Producción

Target: log1p(produccion_t) → convertido con expm1 para interpretación
Features: área sembrada + ciclo + departamento + grupo de cultivo
Random Forest: R² = 0.83 (escala log)
Red Neuronal: Embeddings para variables categóricas + capas densas

M2 — Clasificación Alta/Baja Producción

Target: prod_alta = 1 si producción > mediana nacional (141 t)
Balance: 50/50 por definición de la mediana
Random Forest: AUC-ROC > 0.90
Red Neuronal: Embeddings + sigmoid

M3 — Clustering Municipal

Vector: 5 features por municipio (producción, rendimiento, área, diversidad, grupos)
K-Means: k=4 clusters con perfiles agrícolas interpretables
Autoencoder: Reducción a espacio latente 2D para visualización

Perfiles agrícolas identificados:
PerfilColorDescripción🔴 Alta producción agroindustrialRojoCaña, palma, arroz industrial🟠 Grande / Alta diversidadNaranjaMúltiples cultivos a gran escala🔵 Media escala extensivaAzulCereales, tubérculos, leguminosas🟢 Pequeña escala / SubsistenciaVerdeFrutales, hortalizas, pancoger

🗺️ Mapas interactivos
La app incluye 3 mapas choropleth de Colombia:

Perfil dominante — ¿Qué tipo de agricultura domina cada región?
Rendimiento t/ha — ¿Dónde es más eficiente el suelo?
Producción total — ¿Cuáles son los grandes polos agrícolas?


🚀 Instalación local
bash# Clonar repositorio
git clone https://github.com/luisepm92/AgroInsight.git
cd AgroInsight

# Crear entorno virtual
python3 -m venv venv_agroinsight
source venv_agroinsight/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Lanzar app
streamlit run app.py

⚠️ Los modelos pesados (m1_rf_opt.pkl, m2_rf_opt.pkl) no están en el repositorio por límites de GitHub. Están disponibles en Google Drive — ver SETUP.md para instrucciones.


🛠️ Stack tecnológico
CategoríaTecnologíasLenguajePython 3.11MLscikit-learn, TensorFlow/KerasDatospandas, numpyVisualizaciónPlotly, Matplotlib, SeabornAppStreamlitFuente datosSocrata API (datos.gov.co)EntrenamientoGoogle Colab (GPU T4)

👥 Equipo
Ing. Mecánico Luis Porras; Ing. Sistemas Cesar Beltran; Estadístico Natalia Mosquera
Institución: Universidad de Antioquia
Programa: Bootcamp Talento Tech · IA Nivel Integrador
Ciudad: Medellín, Colombia · 2026
