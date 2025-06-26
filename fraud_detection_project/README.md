# 🧠 Detección de Transacciones Bancarias Fraudulentas con Redes Neuronales

Este proyecto aplica técnicas de aprendizaje profundo para identificar transacciones fraudulentas en tarjetas de crédito, utilizando un conjunto de datos real y altamente desbalanceado.

## 📊 Dataset

- **Fuente**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Tamaño**: 284,807 transacciones, de las cuales solo 492 son fraudes (≈ 0.17%).
- **Características**:
  - 28 variables transformadas mediante **PCA**.
  - `Time` (tiempo transcurrido desde la primera transacción) y `Amount` (monto) no están transformadas.
  - Variable objetivo: `Class` (1: fraude, 0: legítima).

## 🎯 Objetivo

Desarrollar una **Red Neuronal Artificial (RNA)** que:

- Detecte transacciones fraudulentas.
- Minimice los falsos negativos.
- Mantenga un rendimiento robusto pese al desbalance del dataset.

## ⚙️ Metodología

- Exploración de datos y visualización.
- Preprocesamiento: limpieza y manejo del desbalance.
- Extraccion de caracteristicas
- Entrenamiento de una RNA usando **Keras y TensorFlow**.
- Tuning de RNA mediante optuna
- Evaluación con métricas como:
  - `precision`
  - `recall`
  - `F1-score`
  - `ROC-AUC`

## 📈 Resultados:

Un modelo capaz de identificar fraudes con alta precisión (Min 85%) y buen recall (85%) en Clase 1, útil para sistemas de monitoreo bancario en tiempo real.


Reporte de Clasificación.  
![alt text](images/image.png)

![alt text](images/image-1.png)

## 🧪 Reproducibilidad

```bash
# Clona el repositorio
git clone https://github.com/tu_usuario/rna-operaciones-fraudulentas.git
cd rna-operaciones-fraudulentas

# Crea y activa un entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instala dependencias
pip install -r requirements.txt

# Ejecuta el notebook
jupyter notebook rna-operaciones-fraudulentas.ipynb