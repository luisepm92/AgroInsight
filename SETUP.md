# AgroInsight — Instalación y configuración local (Mac)

## Requisitos previos
- Python 3.10 o 3.11 (recomendado)
- VSCode con extensión **Jupyter** instalada
- Terminal (viene con Mac)

Verificar Python:
```bash
python3 --version
```

---

## Paso a paso

### 1. Crear la carpeta del proyecto
La carpeta ya existe en `/Users/blackrave/AgroInsight`.  
Copiar los archivos descargados aquí:
```
/Users/blackrave/AgroInsight/
├── notebook/
│   └── AgroInsight_Local.ipynb
├── app.py
└── requirements.txt
```

### 2. Abrir Terminal y navegar al proyecto
```bash
cd /Users/blackrave/AgroInsight
```

### 3. Crear el entorno virtual
```bash
python3 -m venv venv_agroinsight
```
Esto crea la carpeta `venv_agroinsight/` dentro de tu proyecto.

### 4. Activar el entorno virtual
```bash
source venv_agroinsight/bin/activate
```
Sabrás que está activo cuando veas `(venv_agroinsight)` al inicio del prompt.

### 5. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
⚠️ Esto puede tardar 5-10 minutos la primera vez.

### 6. Registrar el kernel para VSCode/Jupyter
```bash
pip install ipykernel
python -m ipykernel install --user --name=venv_agroinsight --display-name="AgroInsight"
```

### 7. Abrir VSCode
```bash
code .
```

---

## Correr el notebook

1. Abrir `notebook/AgroInsight_Local.ipynb` en VSCode
2. En la esquina superior derecha: **Select Kernel → AgroInsight**
3. `Run All` o correr sección por sección

**Tiempo estimado:** 30–50 min (dependiendo del CPU)

---

## Lanzar la app Streamlit

Una vez que el notebook terminó de correr y todos los archivos están en `modelos/`:

```bash
# Asegurarse de estar en la carpeta del proyecto con el env activo
cd /Users/blackrave/AgroInsight
source venv_agroinsight/bin/activate

# Lanzar la app
streamlit run app.py
```

La app se abre automáticamente en `http://localhost:8501`

---

## Desactivar el entorno virtual
```bash
deactivate
```

---

## Solución de problemas comunes

**Error: `tensorflow` no instala en Mac M1/M2/M3**
```bash
pip install tensorflow-macos
pip install tensorflow-metal  # para usar la GPU del chip Apple
```

**Error: `ModuleNotFoundError` en el notebook**
Verificar que el kernel seleccionado es "AgroInsight" y no el Python del sistema.

**La app no abre en el browser**
Ir manualmente a `http://localhost:8501`

**Error de certificado con la API Socrata**
```bash
pip install certifi
```
Y al inicio del notebook agregar:
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```
