"""
AgroInsight — App de Presentación
Bootcamp Talento Tech · IA Nivel Integrador · Universidad de Antioquia

Correr con: streamlit run app.py
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px
import joblib
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay, confusion_matrix

# ─────────────────────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = 'AgroInsight',
    page_icon   = '🌿',
    layout      = 'wide',
    initial_sidebar_state = 'expanded',
)

# ─────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────
import os
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELOS  = os.path.join(BASE_DIR, 'modelos')
RUTA_GRAFICAS = os.path.join(BASE_DIR, 'graficas')
RUTA_MAPAS    = os.path.join(BASE_DIR, 'mapas')

# ─────────────────────────────────────────────────────────────
# Carga de datos (cacheada)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    tf_import_error = None
    try:
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            import keras
        m1_nn = keras.models.load_model(f'{RUTA_MODELOS}/m1_nn_model.keras')
        m2_nn = keras.models.load_model(f'{RUTA_MODELOS}/m2_nn_model.keras')
    except Exception as e:
        tf_import_error = str(e)
        m1_nn = None
        m2_nn = None

    modelos = {
        # NN M1 — predicción de producción
        'm1_scaler_nn': joblib.load(f'{RUTA_MODELOS}/m1_scaler_num_nn.pkl'),
        'm1_le_dep':    joblib.load(f'{RUTA_MODELOS}/m1_le_dep.pkl'),
        'm1_le_grp':    joblib.load(f'{RUTA_MODELOS}/m1_le_grp.pkl'),
        'm1_le_ciclo':  joblib.load(f'{RUTA_MODELOS}/m1_le_ciclo.pkl'),
        'm1_log_max':   joblib.load(f'{RUTA_MODELOS}/m1_log_max.pkl'),
        'm1_niveles':   joblib.load(f'{RUTA_MODELOS}/m1_niveles.pkl'),
        'm1_nn':        m1_nn,
        # NN M2 — clasificación alta/baja
        'm2_scaler_nn': joblib.load(f'{RUTA_MODELOS}/m2_scaler_num_nn.pkl'),
        'm2_le_dep':    joblib.load(f'{RUTA_MODELOS}/m2_le_dep.pkl'),
        'm2_le_grp':    joblib.load(f'{RUTA_MODELOS}/m2_le_grp.pkl'),
        'm2_le_ciclo':  joblib.load(f'{RUTA_MODELOS}/m2_le_ciclo.pkl'),
        'm2_mediana':   joblib.load(f'{RUTA_MODELOS}/m2_mediana.pkl'),
        'm2_nn':        m2_nn,
        'tf_error':     tf_import_error,
    }
    return modelos

@st.cache_data
def cargar_datos():
    data_limpia     = joblib.load(f'{RUTA_MODELOS}/data_limpia.pkl')
    listas          = joblib.load(f'{RUTA_MODELOS}/listas_categorias.pkl')
    with open(f'{RUTA_MODELOS}/metricas.json', encoding='utf-8') as f:
        metricas = json.load(f)
    return data_limpia, listas, metricas

def cargar_mapa(nombre):
    ruta = f'{RUTA_MAPAS}/{nombre}'
    if os.path.exists(ruta):
        with open(ruta, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def cargar_grafica(nombre):
    ruta = f'{RUTA_GRAFICAS}/{nombre}'
    if os.path.exists(ruta):
        return ruta
    return None

# ─────────────────────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #1D9E75;
        margin: 4px 0;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #1D9E75; }
    .metric-label { font-size: 13px; color: #666; margin-top: 4px; }
    .winner-badge {
        background: #1D9E75; color: white;
        padding: 3px 10px; border-radius: 12px;
        font-size: 11px; font-weight: 600;
    }
    .section-title {
        font-size: 22px; font-weight: 700;
        color: #1a1a1a; margin-bottom: 8px;
    }
    .tag {
        display: inline-block;
        background: #e8f5ee; color: #0F6E56;
        padding: 2px 10px; border-radius: 12px;
        font-size: 12px; margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Header superior
# ─────────────────────────────────────────────────────────────
col_logo, col_titulo, col_info = st.columns([1, 4, 2])
with col_logo:
    st.markdown('<div style="font-size:48px;padding-top:8px">🌿</div>', unsafe_allow_html=True)
with col_titulo:
    st.markdown('<h1 style="margin:0;padding-top:8px;font-size:28px">AgroInsight</h1>', unsafe_allow_html=True)
    st.markdown('<p style="margin:0;color:#888;font-size:13px">Análisis de Producción Agrícola Colombia · EVA V1 · 2006–2018</p>', unsafe_allow_html=True)
with col_info:
    st.markdown('<p style="text-align:right;color:#888;font-size:11px;padding-top:12px">Bootcamp Talento Tech · IA Nivel Integrador<br>Universidad de Antioquia</p>', unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# Navegación horizontal con botones
# ─────────────────────────────────────────────────────────────
SECCIONES = [
    ('🏠', 'Inicio'),
    ('📊', 'EDA y Limpieza'),
    ('📈', 'M1 — Regresión'),
    ('🎯', 'M2 — Clasificación'),
    ('🗺️', 'M3 — Clustering'),
    ('🤖', 'Demo Interactiva'),
]

if 'seccion' not in st.session_state:
    st.session_state['seccion'] = 'Inicio'

cols_nav = st.columns(len(SECCIONES))
for idx, (icono, nombre) in enumerate(SECCIONES):
    with cols_nav[idx]:
        activo = st.session_state['seccion'] == nombre
        estilo = (
            "background:#1D9E75;color:white;border:none;border-radius:8px;"
            "padding:8px 4px;width:100%;cursor:pointer;font-size:13px;font-weight:600;"
        ) if activo else (
            "background:#f0f2f6;color:#444;border:1px solid #ddd;border-radius:8px;"
            "padding:8px 4px;width:100%;cursor:pointer;font-size:13px;"
        )
        st.markdown(
            f'<button style="{estilo}" onclick="void(0)">{icono}<br>{nombre}</button>',
            unsafe_allow_html=True
        )
        if st.button(f'{icono} {nombre}', key=f'nav_{idx}', use_container_width=True,
                     type='primary' if activo else 'secondary'):
            st.session_state['seccion'] = nombre
            st.rerun()

seccion_raw = st.session_state['seccion']
# Mapear al formato que usa el resto del código
_mapa = {
    'Inicio':           '🏠 Inicio',
    'EDA y Limpieza':   '📊 EDA y Limpieza',
    'M1 — Regresión':   '📈 M1 — Regresión',
    'M2 — Clasificación':'🎯 M2 — Clasificación',
    'M3 — Clustering':  '🗺️ M3 — Clustering',
    'Demo Interactiva': '🤖 Demo Interactiva',
}
seccion = _mapa[seccion_raw]

st.divider()

# ─────────────────────────────────────────────────────────────
# SECCIÓN: INICIO
# ─────────────────────────────────────────────────────────────
if seccion == '🏠 Inicio':
    st.markdown('<p class="section-title">AgroInsight — Análisis de Producción Agrícola Colombia</p>', unsafe_allow_html=True)
    st.markdown('Proyecto de IA para predecir, clasificar y segmentar la producción agrícola municipal usando datos del MADR.')

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('### ¿Qué responde AgroInsight?')
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">M1</div>
                <div class="metric-label">¿Cuánto va a producir este cultivo? → Regresión</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class="metric-card" style="border-left-color:#7F77DD">
                <div class="metric-value" style="color:#7F77DD">M2</div>
                <div class="metric-label">¿Vale la pena sembrar? → Clasificación alta/baja</div>
            </div>""", unsafe_allow_html=True)
        with col_c:
            st.markdown("""
            <div class="metric-card" style="border-left-color:#EF9F27">
                <div class="metric-value" style="color:#EF9F27">M3</div>
                <div class="metric-label">¿Qué perfil tiene esta región? → Clustering</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('### Dataset')
        st.markdown("""
        **Fuente:** MADR — Evaluaciones Agropecuarias Municipales EVA V1  
        **Portal:** datos.gov.co · ID: `2pnw-mmge`  
        **Período:** 2006–2018  
        """)

    with col2:
        st.markdown('### Equipo')
        for rol in ['Ing. Mecánico', 'Ing. Sistemas', 'Estadístico']:
            st.markdown(f'<span class="tag">{rol}</span>', unsafe_allow_html=True)

        st.markdown('### Stack')
        for tech in ['scikit-learn', 'TensorFlow/Keras', 'Plotly', 'Streamlit', 'Pandas']:
            st.markdown(f'<span class="tag">{tech}</span>', unsafe_allow_html=True)

    st.divider()

    # Stats del dataset
    try:
        data_limpia, listas, metricas = cargar_datos()
        ds = metricas['dataset']
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('Registros limpios', f'{ds["registros_limpios"]:,}')
        c2.metric('Departamentos', ds['departamentos'])
        c3.metric('Municipios', ds['municipios'])
        c4.metric('Cultivos', ds['cultivos'])
        c5.metric('Años', '2006–2018')
    except:
        st.info('Correr el notebook primero para cargar las métricas.')

# ─────────────────────────────────────────────────────────────
# SECCIÓN: EDA
# ─────────────────────────────────────────────────────────────
elif seccion == '📊 EDA y Limpieza':
    st.markdown('<p class="section-title">Análisis Exploratorio y Limpieza de Datos</p>', unsafe_allow_html=True)

    try:
        data_limpia, listas, metricas = cargar_datos()

        # Pipeline de limpieza
        st.markdown('### Pipeline de limpieza')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Registros crudos', '205,242')
        col2.metric('Tras limpieza', f'{len(data_limpia):,}')
        col3.metric('Eliminados', f'{205242 - len(data_limpia):,}')
        col4.metric('Retención', f'{len(data_limpia)/205242*100:.1f}%')

        st.divider()

        # Gráficas exportadas
        st.markdown('### Distribución de producción')
        g1 = cargar_grafica('eda_distribucion_produccion.png')
        if g1:
            st.image(g1, use_container_width=True)
        else:
            # Generar dinámicamente si no existe
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            axes[0].hist(data_limpia['produccion_t'], bins=80, color='steelblue', alpha=0.7)
            axes[0].set_title('Escala original'); axes[0].set_xlabel('Producción (t)')
            axes[1].hist(np.log1p(data_limpia['produccion_t']), bins=80, color='darkorange', alpha=0.7)
            axes[1].set_title('Escala log1p'); axes[1].set_xlabel('log1p(Producción)')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('### Top 10 departamentos')
            g2 = cargar_grafica('eda_top_departamentos.png')
            if g2: st.image(g2, use_container_width=True)
            else:
                top = data_limpia.groupby('departamento')['produccion_t'].sum().sort_values(ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(8, 4))
                top.plot(kind='bar', ax=ax, color='teal', alpha=0.8)
                ax.set_xlabel(''); ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig); plt.close()

        with col2:
            st.markdown('### Ciclo de cultivo')
            g4 = cargar_grafica('eda_es_semestral.png')
            if g4: st.image(g4, use_container_width=True)
            else:
                counts = data_limpia['es_semestral'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(['Anual', 'Semestral'], counts.values, color=['steelblue','darkorange'], alpha=0.8)
                st.pyplot(fig); plt.close()

        st.markdown('### Producción por grupo de cultivo')
        g3 = cargar_grafica('eda_produccion_por_grupo.png')
        if g3: st.image(g3, use_container_width=True)

        st.divider()
        st.markdown('### Explorador de datos')
        with st.expander('Ver muestra del dataset limpio'):
            st.dataframe(data_limpia.sample(500).reset_index(drop=True), use_container_width=True)

    except FileNotFoundError:
        st.warning('Datos no encontrados. Correr la Sección 7 del notebook primero.')

# ─────────────────────────────────────────────────────────────
# SECCIÓN: M1
# ─────────────────────────────────────────────────────────────
elif seccion == '📈 M1 — Regresión':
    st.markdown('<p class="section-title">Módulo 1 — Regresión de Producción Agrícola</p>', unsafe_allow_html=True)
    st.markdown('**Objetivo:** Predecir la producción en toneladas dado un conjunto de condiciones de siembra.')

    try:
        _, _, metricas = cargar_datos()
        m1 = metricas['M1_Regresion']
        rf = m1['Random_Forest']
        nn = m1['Red_Neuronal']

        # Features
        st.markdown(f"""
        **Target:** `log1p(produccion_t)` — distribución log-normal, convertido con `expm1` para la app  
        **Features:** `area_sembrada_ha` + `es_semestral` + OHE(`departamento`, `grupo_cultivo`, `ciclo_cultivo`) = **{m1['n_features']} features**  
        **Split:** {m1['n_train']:,} entrenamiento · {m1['n_test']:,} prueba (80/20)
        """)

        st.divider()
        st.markdown('### Comparativa RF vs Red Neuronal')

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('#### 🌿 Random Forest')
            ganador_r2 = rf['r2_log_test'] >= nn['r2_log_test']
            m_col1, m_col2 = st.columns(2)
            m_col1.metric('R² Log (prueba)', rf['r2_log_test'],
                          delta=f"gap {rf['gap_train_test']:+.4f}")
            m_col2.metric('RMSE Log', rf['rmse_log_test'])
            st.metric('R² Toneladas', rf['r2_ton_test'])
            st.metric('MAE Log', rf['mae_log_test'])
            if ganador_r2:
                st.markdown('<span class="winner-badge">✓ Mejor R² log</span>', unsafe_allow_html=True)

        with col2:
            st.markdown('#### 🧠 Red Neuronal con Embeddings')
            m_col1, m_col2 = st.columns(2)
            m_col1.metric('R² Log (prueba)', nn['r2_log_test'],
                          delta=f"gap {nn['gap_train_test']:+.4f}")
            m_col2.metric('RMSE Log', nn['rmse_log_test'])
            st.metric('R² Toneladas', nn['r2_ton_test'])
            st.metric('MAE Log', nn['mae_log_test'])
            if not ganador_r2:
                st.markdown('<span class="winner-badge">✓ Mejor R² log</span>', unsafe_allow_html=True)

        st.divider()

        # Tabla comparativa completa
        st.markdown('### Tabla comparativa')
        tabla = pd.DataFrame({
            'Métrica': ['R² Log Entrenamiento', 'R² Log Prueba', 'RMSE Log Prueba',
                        'MAE Log Prueba', 'Gap train/test', 'R² Toneladas', 'RMSE Toneladas'],
            'Random Forest': [rf['r2_log_train'], rf['r2_log_test'], rf['rmse_log_test'],
                              rf['mae_log_test'], rf['gap_train_test'], rf['r2_ton_test'],
                              f"{rf['rmse_ton_test']:,.0f} t"],
            'Red Neuronal':  [nn['r2_log_train'], nn['r2_log_test'], nn['rmse_log_test'],
                              nn['mae_log_test'], nn['gap_train_test'], nn['r2_ton_test'],
                              f"{nn['rmse_ton_test']:,.0f} t"],
        })
        st.dataframe(tabla, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown('### Gráficas de desempeño')
        g1 = cargar_grafica('m1_actual_vs_predicho.png')
        if g1: st.image(g1, use_container_width=True)

        g2 = cargar_grafica('m1_curvas_aprendizaje.png')
        if g2:
            st.markdown('### Curvas de aprendizaje — Red Neuronal')
            st.image(g2, use_container_width=True)

        # Niveles interpretativos
        st.divider()
        st.markdown('### Niveles interpretativos para la demo')
        niveles_data = [
            {'Nivel': 1, 'Categoría': 'Subsistencia',      'Rango': '< 30 t'},
            {'Nivel': 2, 'Categoría': 'Pequeño productor', 'Rango': '30–105 t'},
            {'Nivel': 3, 'Categoría': 'Mediano',           'Rango': '105–329 t'},
            {'Nivel': 4, 'Categoría': 'Grande',            'Rango': '329–1,236 t'},
            {'Nivel': 5, 'Categoría': 'Agroindustrial',    'Rango': '> 1,236 t'},
        ]
        st.dataframe(pd.DataFrame(niveles_data), use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.warning('Métricas no encontradas. Correr la Sección 7 del notebook primero.')

# ─────────────────────────────────────────────────────────────
# SECCIÓN: M2
# ─────────────────────────────────────────────────────────────
elif seccion == '🎯 M2 — Clasificación':
    st.markdown('<p class="section-title">Módulo 2 — Clasificación Producción Alta vs Baja</p>', unsafe_allow_html=True)
    st.markdown('**Objetivo:** Clasificar si la producción superará la mediana nacional.')

    try:
        _, _, metricas = cargar_datos()
        m2 = metricas['M2_Clasificacion']
        rf = m2['Random_Forest']
        nn = m2['Red_Neuronal']

        st.markdown(f"""
        **Target:** `prod_alta` = 1 si `produccion_t` > **{m2['umbral_t']:.0f} t** (mediana nacional)  
        **Balance:** 50% clase alta · 50% clase baja — perfectamente balanceado por definición de la mediana  
        **Features:** {m2['n_features']} features · Split: {m2['n_train']:,} entrenamiento · {m2['n_test']:,} prueba
        """)

        st.divider()
        st.markdown('### Comparativa RF vs Red Neuronal')

        col1, col2 = st.columns(2)
        ganador_auc = rf['auc_roc'] >= nn['auc_roc']

        with col1:
            st.markdown('#### 🌿 Random Forest')
            c1, c2 = st.columns(2)
            c1.metric('Accuracy (prueba)', f"{rf['accuracy_test']:.4f}",
                      delta=f"gap {rf['gap_train_test']:+.4f}")
            c2.metric('AUC-ROC', f"{rf['auc_roc']:.4f}")
            c1, c2 = st.columns(2)
            c1.metric('Precisión', f"{rf['precision']:.4f}")
            c2.metric('Recall', f"{rf['recall']:.4f}")
            st.metric('F1-Score', f"{rf['f1_score']:.4f}")
            if ganador_auc:
                st.markdown('<span class="winner-badge">✓ Mejor AUC-ROC</span>', unsafe_allow_html=True)

        with col2:
            st.markdown('#### 🧠 Red Neuronal con Embeddings')
            c1, c2 = st.columns(2)
            c1.metric('Accuracy (prueba)', f"{nn['accuracy_test']:.4f}",
                      delta=f"gap {nn['gap_train_test']:+.4f}")
            c2.metric('AUC-ROC', f"{nn['auc_roc']:.4f}")
            c1, c2 = st.columns(2)
            c1.metric('Precisión', f"{nn['precision']:.4f}")
            c2.metric('Recall', f"{nn['recall']:.4f}")
            st.metric('F1-Score', f"{nn['f1_score']:.4f}")
            if not ganador_auc:
                st.markdown('<span class="winner-badge">✓ Mejor AUC-ROC</span>', unsafe_allow_html=True)

        st.divider()
        st.markdown('### Tabla comparativa')
        tabla = pd.DataFrame({
            'Métrica': ['Accuracy Entrenamiento', 'Accuracy Prueba', 'Precisión',
                        'Recall', 'F1-Score', 'AUC-ROC', 'Gap train/test'],
            'Random Forest': [rf['accuracy_train'], rf['accuracy_test'], rf['precision'],
                              rf['recall'], rf['f1_score'], rf['auc_roc'], rf['gap_train_test']],
            'Red Neuronal':  [nn['accuracy_train'], nn['accuracy_test'], nn['precision'],
                              nn['recall'], nn['f1_score'], nn['auc_roc'], nn['gap_train_test']],
        })
        st.dataframe(tabla, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown('### Gráficas de desempeño')
        g1 = cargar_grafica('m2_roc_comparativa.png')
        if g1: st.image(g1, use_container_width=True)

        g2 = cargar_grafica('m2_curvas_aprendizaje.png')
        if g2:
            st.markdown('### Curvas de aprendizaje — Red Neuronal')
            st.image(g2, use_container_width=True)

    except FileNotFoundError:
        st.warning('Métricas no encontradas. Correr la Sección 7 del notebook primero.')

# ─────────────────────────────────────────────────────────────
# SECCIÓN: M3
# ─────────────────────────────────────────────────────────────
elif seccion == '🗺️ M3 — Clustering':
    st.markdown('<p class="section-title">Módulo 3 — Mapa Agrícola de Colombia</p>', unsafe_allow_html=True)
    st.markdown('Perfil agrícola histórico por departamento 2006–2018. Hover sobre cada departamento para ver detalles.')

    tab1, tab2, tab3 = st.tabs([
        '🌿 Perfil dominante',
        '📊 Rendimiento t/ha',
        '🏭 Producción total'
    ])

    with tab1:
        html = cargar_mapa('mapa_perfiles_agricolas.html')
        if html:
            components.html(html, height=750, scrolling=False)
        else:
            st.info('Mapa no encontrado. Verificar carpeta de mapas.')

    with tab2:
        html = cargar_mapa('mapa_rendimiento_colombia.html')
        if html:
            components.html(html, height=750, scrolling=False)
        else:
            st.info('Mapa no encontrado.')

    with tab3:
        html = cargar_mapa('mapa_produccion_colombia.html')
        if html:
            components.html(html, height=750, scrolling=False)
        else:
            st.info('Mapa no encontrado.')


elif seccion == '🤖 Demo Interactiva':
    st.markdown('<p class="section-title">Demo Interactiva — Predicción en Tiempo Real</p>', unsafe_allow_html=True)
    st.markdown('Ingresa los parámetros de siembra y obtén predicciones de M1 y M2 en tiempo real.')

    try:
        _, listas, metricas = cargar_datos()
        modelos = cargar_modelos()

        DEPARTAMENTOS  = listas['departamentos']
        GRUPOS_CULTIVO = listas['grupos_cultivo']
        CICLOS         = listas['ciclos']
        mediana        = metricas['M2_Clasificacion']['umbral_t']

        # Inputs
        st.markdown('### Parámetros de siembra')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            departamento = st.selectbox('Departamento', DEPARTAMENTOS)
        with col2:
            grupo_cultivo = st.selectbox('Grupo de cultivo', GRUPOS_CULTIVO)
        with col3:
            ciclo_cultivo = st.selectbox('Ciclo de cultivo', CICLOS)
        with col4:
            area_ha = st.number_input('Área sembrada (ha)', min_value=1.0,
                                       max_value=50000.0, value=100.0, step=10.0)

        es_semestral = 1 if str(ciclo_cultivo).upper() == 'TRANSITORIO' else 0
        st.caption(f'Ciclo detectado: {"Semestral (1)" if es_semestral else "Anual (0)"}')

        if st.button('🔍 Predecir', type='primary', use_container_width=True):
            st.divider()

            def clasificar_nivel(ton):
                limites = [(30,1,'Subsistencia','< 30 t'),
                           (105,2,'Pequeño productor','30–105 t'),
                           (329,3,'Mediano','105–329 t'),
                           (1236,4,'Grande','329–1,236 t')]
                for lim, niv, nom, rng in limites:
                    if ton < lim:
                        return niv, f'Nivel {niv} — {nom} ({rng})'
                return 5, 'Nivel 5 — Agroindustrial (> 1,236 t)'

            col_m1, col_m2 = st.columns(2)

            # ── M1: Predicción de producción con NN ─────────────
            with col_m1:
                st.markdown('### 📈 Predicción de producción')
                try:
                    log_max = modelos['m1_log_max']

                    # Preparar inputs NN M1
                    sem = 1 if str(ciclo_cultivo).upper() == 'TRANSITORIO' else 0
                    X_num = modelos['m1_scaler_nn'].transform([[area_ha, sem]])
                    X_dep = modelos['m1_le_dep'].transform([departamento]).reshape(-1, 1)
                    X_grp = modelos['m1_le_grp'].transform([grupo_cultivo]).reshape(-1, 1)
                    X_ciclo = modelos['m1_le_ciclo'].transform([ciclo_cultivo]).reshape(-1, 1)

                    # Predecir en escala log y convertir a toneladas
                    pred_log = float(np.clip(
                        modelos['m1_nn'].predict(
                            {'numericas': X_num,
                             'departamento': X_dep,
                             'grupo_cultivo': X_grp,
                             'ciclo_cultivo': X_ciclo},
                            verbose=0
                        ).flatten()[0],
                        0, log_max
                    ))
                    pred_ton = np.expm1(pred_log)
                    niv, lbl = clasificar_nivel(pred_ton)

                    st.metric('Producción estimada', f'{pred_ton:,.0f} t')
                    st.info(lbl)

                    st.markdown('**Nivel de producción**')
                    niveles_labels = ['Subsistencia','Pequeño','Mediano','Grande','Agroindustrial']
                    cols_niv = st.columns(5)
                    for n, lab in enumerate(niveles_labels, 1):
                        with cols_niv[n-1]:
                            color = '#1D9E75' if n == niv else '#e0e0e0'
                            st.markdown(
                                f'<div style="background:{color};border-radius:6px;'
                                f'padding:6px 2px;text-align:center;font-size:10px;'
                                f'color:{"white" if n==niv else "#888"}">{lab}</div>',
                                unsafe_allow_html=True
                            )
                except Exception as e:
                    st.error(f'Error en predicción: {e}')

            # ── M2: ¿Vale la pena sembrar? con NN ───────────────
            with col_m2:
                st.markdown('### 🎯 ¿Vale la pena sembrar?')
                st.caption(f'Umbral: {mediana:.0f} t (mediana nacional 2006–2018)')
                try:
                    sem = 1 if str(ciclo_cultivo).upper() == 'TRANSITORIO' else 0
                    X_num = modelos['m2_scaler_nn'].transform([[area_ha, sem]])
                    X_dep = modelos['m2_le_dep'].transform([departamento]).reshape(-1, 1)
                    X_grp = modelos['m2_le_grp'].transform([grupo_cultivo]).reshape(-1, 1)
                    X_ciclo = modelos['m2_le_ciclo'].transform([ciclo_cultivo]).reshape(-1, 1)

                    prob = float(modelos['m2_nn'].predict(
                        {'numericas': X_num,
                         'departamento': X_dep,
                         'grupo_cultivo': X_grp,
                         'ciclo_cultivo': X_ciclo},
                        verbose=0
                    ).flatten()[0])
                    pred = int(prob >= 0.5)

                    if pred == 1:
                        st.success('✅ PRODUCCIÓN ALTA')
                    else:
                        st.warning('⚠️ PRODUCCIÓN BAJA')

                    st.markdown(f'**{prob*100:.1f}%** de probabilidad de superar la mediana nacional')
                    st.progress(float(prob))

                    recomendacion = (
                        'Condiciones favorables — se recomienda proceder con la siembra.'
                        if pred == 1 else
                        'Producción esperada por debajo de la mediana — evaluar alternativas.'
                    )
                    st.caption(recomendacion)
                except Exception as e:
                    st.error(f'Error en predicción: {e}')

    except FileNotFoundError:
        st.warning('Datos no encontrados. Correr la Sección 7 del notebook primero.')
