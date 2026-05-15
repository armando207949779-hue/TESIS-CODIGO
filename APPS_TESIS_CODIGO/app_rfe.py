# app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report
)


st.set_page_config(
    page_title="Selección de variables con RFE",
    layout="wide"
)

st.title("Selección de variables con RFE")
st.write("Carga un archivo Parquet, elige tu variable objetivo y aplica Recursive Feature Elimination.")


# -----------------------------
# Cargar datos
# -----------------------------
uploaded_file = st.file_uploader(
    "Carga tu archivo .parquet",
    type=["parquet"]
)

if uploaded_file is None:
    st.info("Sube un archivo Parquet para comenzar.")
    st.stop()

try:
    df = pd.read_parquet(uploaded_file)
except Exception as e:
    st.error(f"No se pudo leer el archivo Parquet: {e}")
    st.stop()

st.subheader("Vista previa de los datos")
st.dataframe(df.head())

st.write(f"Filas: **{df.shape[0]}** | Columnas: **{df.shape[1]}**")


# -----------------------------
# Configuración
# -----------------------------
st.sidebar.header("Configuración del modelo")

target_col = st.sidebar.selectbox(
    "Variable objetivo",
    options=df.columns
)

problem_type = st.sidebar.selectbox(
    "Tipo de problema",
    options=["Regresión", "Clasificación"]
)

available_features = [col for col in df.columns if col != target_col]

selected_features = st.sidebar.multiselect(
    "Variables predictoras",
    options=available_features,
    default=available_features
)

if len(selected_features) == 0:
    st.warning("Selecciona al menos una variable predictora.")
    st.stop()

n_features_to_select = st.sidebar.slider(
    "Número de variables a seleccionar",
    min_value=1,
    max_value=len(selected_features),
    value=min(5, len(selected_features))
)

model_choice = st.sidebar.selectbox(
    "Modelo base para RFE",
    options=[
        "Regresión lineal / logística",
        "Random Forest"
    ]
)

test_size = st.sidebar.slider(
    "Porcentaje para prueba",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05
)

random_state = st.sidebar.number_input(
    "Random state",
    min_value=0,
    value=42,
    step=1
)


# -----------------------------
# Preparar datos
# -----------------------------
data = df[selected_features + [target_col]].copy()

# Eliminar filas con NA para simplificar
data = data.dropna()

X = data[selected_features]
y = data[target_col]

if data.shape[0] == 0:
    st.error("Después de eliminar valores faltantes, no quedan filas disponibles.")
    st.stop()

numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

st.subheader("Tipos de variables detectadas")

col1, col2 = st.columns(2)

with col1:
    st.write("**Variables numéricas**")
    st.write(numeric_features if numeric_features else "Ninguna")

with col2:
    st.write("**Variables categóricas**")
    st.write(categorical_features if categorical_features else "Ninguna")


# -----------------------------
# Preprocesamiento
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ],
    remainder="drop"
)


# -----------------------------
# Elegir estimador
# -----------------------------
if problem_type == "Regresión":
    if model_choice == "Regresión lineal / logística":
        estimator = LinearRegression()
    else:
        estimator = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        )
else:
    if model_choice == "Regresión lineal / logística":
        estimator = LogisticRegression(
            max_iter=1000,
            solver="liblinear"
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        )


# -----------------------------
# Botón para ejecutar
# -----------------------------
run_button = st.button("Ejecutar RFE")

if not run_button:
    st.stop()


# -----------------------------
# Train/test split
# -----------------------------
try:
    stratify = y if problem_type == "Clasificación" and y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
except Exception as e:
    st.error(f"Error al dividir los datos: {e}")
    st.stop()


# -----------------------------
# Transformar variables
# -----------------------------
try:
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
except Exception as e:
    st.error(f"Error en el preprocesamiento: {e}")
    st.stop()


# -----------------------------
# Ejecutar RFE
# -----------------------------
try:
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features_to_select
    )

    rfe.fit(X_train_transformed, y_train)

except Exception as e:
    st.error(f"Error ejecutando RFE: {e}")
    st.stop()


# -----------------------------
# Resultados de selección
# -----------------------------
results = pd.DataFrame({
    "variable_transformada": feature_names,
    "seleccionada": rfe.support_,
    "ranking": rfe.ranking_
}).sort_values(["ranking", "variable_transformada"])

selected_transformed_features = results.loc[
    results["seleccionada"], 
    "variable_transformada"
].tolist()

st.subheader("Ranking de variables según RFE")
st.dataframe(results)

st.subheader("Variables seleccionadas")
st.write(selected_transformed_features)


# -----------------------------
# Entrenar modelo final con variables seleccionadas
# -----------------------------
try:
    X_train_selected = rfe.transform(X_train_transformed)
    X_test_selected = rfe.transform(X_test_transformed)

    final_model = estimator
    final_model.fit(X_train_selected, y_train)

    y_pred = final_model.predict(X_test_selected)

except Exception as e:
    st.error(f"Error entrenando el modelo final: {e}")
    st.stop()


# -----------------------------
# Métricas
# -----------------------------
st.subheader("Evaluación del modelo con variables seleccionadas")

if problem_type == "Regresión":
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("RMSE", round(rmse, 4))

    with col2:
        st.metric("R²", round(r2, 4))

else:
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", round(accuracy, 4))

    with col2:
        st.metric("F1 weighted", round(f1, 4))

    st.text("Classification report")
    st.text(classification_report(y_test, y_pred))


# -----------------------------
# Descargar resultados
# -----------------------------
csv = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descargar ranking de variables",
    data=csv,
    file_name="rfe_ranking_variables.csv",
    mime="text/csv"
)
