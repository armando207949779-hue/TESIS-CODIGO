import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from catboost import CatBoostRegressor
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="CatBoost Regressor",
    layout="wide"
)

st.title("CatBoost Regressor")
st.write("Sube un CSV o Parquet, elige target y predictores, y entrena un modelo de regresión.")


@st.cache_data
def cargar_california_housing():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df["target"] = data.target
    return df


def cargar_archivo(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    else:
        raise ValueError("Formato no soportado. Usa CSV o Parquet.")


uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo CSV o Parquet",
    type=["csv", "parquet"]
)

if uploaded_file is not None:
    df = cargar_archivo(uploaded_file)
    st.sidebar.success("Archivo cargado correctamente.")
else:
    df = cargar_california_housing()
    st.sidebar.info("Usando dataset por defecto: California Housing.")


st.subheader("Vista previa del dataset")
st.dataframe(df.head())

st.write(f"Filas: **{df.shape[0]}** | Columnas: **{df.shape[1]}**")


columnas = df.columns.tolist()

target = st.sidebar.selectbox(
    "Elige la variable target",
    columnas,
    index=columnas.index("target") if "target" in columnas else 0
)

predictoras_default = [col for col in columnas if col != target]

predictoras = st.sidebar.multiselect(
    "Elige las variables predictoras",
    [col for col in columnas if col != target],
    default=predictoras_default
)

test_size = st.sidebar.slider(
    "Tamaño del conjunto de prueba",
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

iterations = st.sidebar.slider(
    "Iteraciones",
    min_value=100,
    max_value=3000,
    value=1000,
    step=100
)

learning_rate = st.sidebar.slider(
    "Learning rate",
    min_value=0.001,
    max_value=0.3,
    value=0.05,
    step=0.001,
    format="%.3f"
)

depth = st.sidebar.slider(
    "Profundidad del árbol",
    min_value=2,
    max_value=12,
    value=6,
    step=1
)


if len(predictoras) == 0:
    st.warning("Selecciona al menos una variable predictora.")
    st.stop()


df_model = df[[target] + predictoras].copy()

for col in df_model.columns:
    if df_model[col].dtype == "object":
        df_model[col] = df_model[col].astype("category")

df_model = df_model.dropna()

X = df_model[predictoras]
y = df_model[target]

cat_features = [
    col for col in X.columns
    if str(X[col].dtype) == "category"
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)


if st.sidebar.button("Entrenar CatBoost"):
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features if len(cat_features) > 0 else None
    )

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Métricas del modelo")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:,.4f}")
    col2.metric("RMSE", f"{rmse:,.4f}")
    col3.metric("R²", f"{r2:,.4f}")

    st.subheader("Gráfico: Real vs Predicho")

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_test, y_pred, alpha=0.6)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--"
    )

    ax.set_xlabel("Valor real")
    ax.set_ylabel("Valor predicho")
    ax.set_title("Real vs Predicho")

    st.pyplot(fig)

    st.subheader("Importancia de variables")

    importancias = pd.DataFrame({
        "variable": X.columns,
        "importancia": model.get_feature_importance()
    }).sort_values("importancia", ascending=False)

    st.dataframe(importancias)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(importancias["variable"], importancias["importancia"])
    ax2.set_xlabel("Variable")
    ax2.set_ylabel("Importancia")
    ax2.set_title("Importancia de variables")
    ax2.tick_params(axis="x", rotation=45)

    st.pyplot(fig2)

    st.subheader("Predicciones")

    resultados = pd.DataFrame({
        "real": y_test.values,
        "predicho": y_pred
    })

    st.dataframe(resultados.head(100))