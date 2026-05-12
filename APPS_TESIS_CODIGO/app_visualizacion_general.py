# app.py

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing


# =========================
# CONFIGURACIÓN GENERAL
# =========================

st.set_page_config(
    page_title="Visualizador de Datos",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Visualizador Interactivo de CSV / Parquet")
st.write(
    "Sube un archivo `.csv` o `.parquet`, o usa por defecto el dataset "
    "**California Housing**."
)


# =========================
# FUNCIONES
# =========================

@st.cache_data
def cargar_california_housing():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return df


def cargar_archivo(uploaded_file):
    if uploaded_file is None:
        return cargar_california_housing()

    nombre = uploaded_file.name.lower()

    if nombre.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if nombre.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)

    st.error("Formato no soportado. Usa CSV o Parquet.")
    return None


def obtener_columnas(df):
    columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()
    columnas_categoricas = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return columnas_numericas, columnas_categoricas


def mostrar_grafico_individual(df, tipo_grafico, x, y=None, hue=None):
    fig, ax = plt.subplots(figsize=(9, 5))

    try:
        if tipo_grafico == "Histograma":
            sns.histplot(data=df, x=x, hue=hue, kde=True, ax=ax)

        elif tipo_grafico == "Boxplot":
            sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax)

        elif tipo_grafico == "Violin plot":
            sns.violinplot(data=df, x=x, y=y, hue=hue, ax=ax)

        elif tipo_grafico == "Scatter plot":
            sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)

        elif tipo_grafico == "Line plot":
            sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax)

        elif tipo_grafico == "Bar plot":
            sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)

        elif tipo_grafico == "KDE plot":
            sns.kdeplot(data=df, x=x, hue=hue, fill=True, ax=ax)

        elif tipo_grafico == "Count plot":
            sns.countplot(data=df, x=x, hue=hue, ax=ax)

        elif tipo_grafico == "Regresión":
            sns.regplot(data=df, x=x, y=y, ax=ax)

        ax.set_title(tipo_grafico)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"No se pudo generar el gráfico: {e}")


def mostrar_matriz_graficos(df, columnas, tipo_matriz):
    try:
        if tipo_matriz == "Pairplot":
            fig = sns.pairplot(df[columnas].dropna())
            st.pyplot(fig)

        elif tipo_matriz == "Matriz de correlación":
            corr = df[columnas].corr()

            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(
                corr,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                ax=ax
            )
            ax.set_title("Matriz de correlación")
            st.pyplot(fig)

        elif tipo_matriz == "Heatmap de valores":
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(
                df[columnas].dropna().head(100),
                cmap="viridis",
                ax=ax
            )
            ax.set_title("Heatmap de valores")
            st.pyplot(fig)

        elif tipo_matriz == "Scatter matrix":
            from pandas.plotting import scatter_matrix

            fig = plt.figure(figsize=(12, 10))
            scatter_matrix(
                df[columnas].dropna(),
                figsize=(12, 10),
                diagonal="hist"
            )
            st.pyplot(fig)

    except Exception as e:
        st.warning(f"No se pudo generar la matriz: {e}")


# =========================
# CARGA DE DATOS
# =========================

st.sidebar.header("📁 Carga de datos")

uploaded_file = st.sidebar.file_uploader(
    "Sube un archivo CSV o Parquet",
    type=["csv", "parquet"]
)

df = cargar_archivo(uploaded_file)

if df is None:
    st.stop()

columnas_numericas, columnas_categoricas = obtener_columnas(df)
todas_columnas = df.columns.tolist()

st.sidebar.success(
    "Dataset cargado correctamente"
    if uploaded_file
    else "Usando California Housing por defecto"
)


# =========================
# INFORMACIÓN DEL DATASET
# =========================

st.subheader("📌 Vista general del dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Filas", df.shape[0])

with col2:
    st.metric("Columnas", df.shape[1])

with col3:
    st.metric("Valores nulos", int(df.isnull().sum().sum()))

with st.expander("Ver primeras filas"):
    st.dataframe(df.head(100), use_container_width=True)

with st.expander("Ver tipos de datos"):
    tipos = pd.DataFrame({
        "Columna": df.columns,
        "Tipo": df.dtypes.astype(str),
        "Nulos": df.isnull().sum().values,
        "Únicos": df.nunique().values
    })
    st.dataframe(tipos, use_container_width=True)

with st.expander("Resumen estadístico"):
    st.dataframe(df.describe(include="all").T, use_container_width=True)


# =========================
# VISUALIZACIONES
# =========================

st.sidebar.header("📊 Visualizaciones")

modo = st.sidebar.radio(
    "Modo de visualización",
    [
        "Individual",
        "Agrupada / Matriz"
    ]
)


# =========================
# MODO INDIVIDUAL
# =========================

if modo == "Individual":
    st.subheader("📈 Visualización individual")

    tipos_individuales = [
        "Histograma",
        "Boxplot",
        "Violin plot",
        "Scatter plot",
        "Line plot",
        "Bar plot",
        "KDE plot",
        "Count plot",
        "Regresión"
    ]

    tipo_grafico = st.sidebar.selectbox(
        "Tipo de gráfico",
        tipos_individuales
    )

    necesita_y = tipo_grafico in [
        "Boxplot",
        "Violin plot",
        "Scatter plot",
        "Line plot",
        "Bar plot",
        "Regresión"
    ]

    if tipo_grafico in ["Count plot"]:
        opciones_x = todas_columnas
    else:
        opciones_x = columnas_numericas + columnas_categoricas

    x = st.sidebar.selectbox(
        "Columna X",
        opciones_x
    )

    y = None
    if necesita_y:
        y = st.sidebar.selectbox(
            "Columna Y",
            columnas_numericas
        )

    usar_hue = st.sidebar.checkbox("Usar agrupación por color Hue")

    hue = None
    if usar_hue:
        hue = st.sidebar.selectbox(
            "Columna Hue",
            columnas_categoricas + columnas_numericas
        )

    mostrar_grafico_individual(
        df=df,
        tipo_grafico=tipo_grafico,
        x=x,
        y=y,
        hue=hue
    )


# =========================
# MODO AGRUPADO / MATRIZ
# =========================

else:
    st.subheader("🔢 Visualización agrupada / matriz")

    tipos_matriz = [
        "Pairplot",
        "Matriz de correlación",
        "Heatmap de valores",
        "Scatter matrix"
    ]

    tipo_matriz = st.sidebar.selectbox(
        "Tipo de matriz",
        tipos_matriz
    )

    columnas_matriz = st.sidebar.multiselect(
        "Selecciona columnas numéricas",
        columnas_numericas,
        default=columnas_numericas[:5]
    )

    if len(columnas_matriz) < 2:
        st.warning("Selecciona al menos 2 columnas numéricas.")
    else:
        mostrar_matriz_graficos(
            df=df,
            columnas=columnas_matriz,
            tipo_matriz=tipo_matriz
        )


# =========================
# ANÁLISIS EXTRA OPCIONAL
# =========================

st.sidebar.header("⚙️ Opciones extra")

mostrar_nulos = st.sidebar.checkbox("Mostrar análisis de valores nulos")
mostrar_distribuciones = st.sidebar.checkbox("Mostrar distribuciones automáticas")
mostrar_correlaciones = st.sidebar.checkbox("Mostrar correlaciones ordenadas")


if mostrar_nulos:
    st.subheader("🧩 Análisis de valores nulos")

    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)

    if nulos.empty:
        st.info("No hay valores nulos en el dataset.")
    else:
        st.dataframe(nulos.rename("Cantidad de nulos"))

        fig, ax = plt.subplots(figsize=(10, 5))
        nulos.plot(kind="bar", ax=ax)
        ax.set_title("Valores nulos por columna")
        ax.set_ylabel("Cantidad")
        plt.xticks(rotation=45)
        st.pyplot(fig)


if mostrar_distribuciones:
    st.subheader("📉 Distribuciones automáticas")

    if len(columnas_numericas) == 0:
        st.warning("No hay columnas numéricas para graficar.")
    else:
        columnas_dist = st.multiselect(
            "Columnas para mostrar distribución",
            columnas_numericas,
            default=columnas_numericas[:3]
        )

        for col in columnas_dist:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribución de {col}")
            st.pyplot(fig)


if mostrar_correlaciones:
    st.subheader("🔗 Correlaciones ordenadas")

    if len(columnas_numericas) < 2:
        st.warning("Se necesitan al menos 2 columnas numéricas.")
    else:
        variable_objetivo = st.selectbox(
            "Selecciona variable objetivo",
            columnas_numericas
        )

        corr = df[columnas_numericas].corr()[variable_objetivo]
        corr = corr.drop(variable_objetivo).sort_values(ascending=False)

        st.dataframe(
            corr.rename("Correlación").to_frame(),
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        corr.plot(kind="bar", ax=ax)
        ax.set_title(f"Correlaciones con {variable_objetivo}")
        ax.set_ylabel("Correlación")
        plt.xticks(rotation=45)
        st.pyplot(fig)


# =========================
# PIE
# =========================

st.caption("Aplicación creada con Streamlit, Pandas, Seaborn y Matplotlib.")