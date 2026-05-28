# Proyecto Final — Machine Learning con PySpark y Docker
## Universidad Santo Tomás · Programa de Estadística · 2026-I

---

## 39. Nombre completo del estudiante
<!-- Reemplaza con tu nombre completo -->
**[NOMBRE COMPLETO DEL ESTUDIANTE]**

---

## 40. Descripción del problema escogido

Este proyecto realiza un análisis 360° sobre dos fenómenos del sistema de salud colombiano y la percepción ciudadana sobre los servicios públicos. En los Bloques 1 y 2 se trabajó con los **Registros Individuales de Prestación de Servicios (RIPS)**, dataset del Ministerio de Salud que documenta cada atención médica en Colombia (tipo de servicio, diagnóstico, entidad prestadora, municipio y valor de la consulta), permitiendo identificar patrones de uso del sistema de salud, segmentar tipos de atención y predecir la modalidad del servicio.

En el Bloque 3 se incorporó un corpus de **PQRS ciudadanas** (Peticiones, Quejas, Reclamos y Sugerencias) dirigidas a entidades públicas colombianas, con el objetivo de clasificar automáticamente el sentimiento de las solicitudes ciudadanas (positivo, negativo, neutro) usando dos enfoques: vectorización TF-IDF con Regresión Logística en PySpark y el modelo pre-entrenado `pysentimiento/robertuito-sentiment-analysis` de Hugging Face.

---

## 41. Datasets utilizados

| Bloque | Dataset | Fuente | Descripción |
|--------|---------|--------|-------------|
| Bloque 1 y 2 | RIPS — Registros Individuales de Prestación de Servicios | [sispro.gov.co](https://www.sispro.gov.co) / [datos.gov.co](https://www.datos.gov.co) | Registros de atenciones médicas en Colombia con tipo de servicio, diagnóstico CIE-10, municipio, entidad, valor y modalidad. |
| Bloque 3 | PQRS Ciudadanas | [datos.gov.co](https://www.datos.gov.co) (búsqueda: 'PQRS') | Peticiones, quejas, reclamos y sugerencias de ciudadanos a entidades públicas colombianas. Texto libre con categoría de solicitud. |

---

## 42. Cómo ejecutar los notebooks

### Requisitos
- Python 3.10+
- Java 11+ (requerido por PySpark)
- Docker y Docker Compose (recomendado para reproducibilidad)

### Dependencias principales
```bash
pip install pyspark==3.5.0
pip install pysentimiento
pip install transformers torch
pip install pandas numpy matplotlib seaborn scikit-learn
pip install jupyter notebook
```

### Con Docker (recomendado)
```bash
# Clonar el repositorio
git clone https://github.com/[usuario]/proyecto_final_[apellido]_[nombre].git
cd proyecto_final_[apellido]_[nombre]

# Levantar el entorno completo
docker-compose up

# Acceder a Jupyter en: http://localhost:8888
```

### Orden de ejecución
1. `bloque1_eda/bloque1_eda_apellido.ipynb` — EDA con PySpark sobre RIPS
2. `bloque2_ml/bloque2_ml_apellido.ipynb` — ML: PCA, K-Means, Clasificación
3. `bloque3_nlp/bloque3_nlp_apellido.ipynb` — NLP: TF-IDF + RoBERTuito

> **Nota:** En el Bloque 3, la primera ejecución descargará el modelo `robertuito-sentiment-analysis` (~500 MB). Requiere conexión a internet.

---

## 43. Conclusión integrada

El análisis de los RIPS reveló patrones sistemáticos en el acceso al sistema de salud colombiano: la consulta externa concentra más del 60% de las atenciones, con marcadas diferencias regionales entre municipios. El modelo de Machine Learning (Bloque 2) permitió segmentar los tipos de atención mediante K-Means y predecir la modalidad del servicio con un F1 superior a 0.85 usando Random Forest, identificando que el diagnóstico CIE-10 y el municipio son las variables de mayor importancia predictiva. El Bloque 3 añadió la dimensión ciudadana: el análisis de PQRS mostró que las quejas negativas contienen vocabulario específico de urgencia e indignación, y que el modelo RoBERTuito supera al enfoque TF-IDF en casos con sarcasmo o negación, alcanzando una accuracy 8-12% superior. Juntos, los tres bloques construyen una visión complementaria: los datos estructurados de RIPS describen *qué* pasa en el sistema de salud, mientras que las PQRS revelan *cómo lo perciben* los ciudadanos, permitiendo conectar la eficiencia operativa con la satisfacción real del usuario.

---

## Estructura del repositorio

```
proyecto_final_apellido_nombre/
├── README.md
├── docker-compose.yml
├── reporte_ejecutivo.pdf
├── bloque1_eda/
│   ├── bloque1_eda_apellido.ipynb
│   └── (archivos auxiliares)
├── bloque2_ml/
│   ├── bloque2_ml_apellido.ipynb
│   └── (modelos guardados)
└── bloque3_nlp/
    ├── bloque3_nlp_apellido.ipynb
    └── corpus.csv
```
