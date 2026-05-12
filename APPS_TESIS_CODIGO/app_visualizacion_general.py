# app_visualizacion_general.py

import math
import warnings

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import fetch_california_housing
from pandas.plotting import scatter_matrix

warnings.filterwarnings("ignore")


# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================

st.set_page_config(
    page_title="Visualizador General de Datos",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Visualizador General de Datos")
st.write(
    "Sube un archivo CSV o Parquet, o usa por defecto el dataset "
    "**California Housing**. Puedes generar gráficos individuales, "
    "gráficos múltiples en grilla, matrices y series temporales."
)


# ======================================================
# FUNCIONES DE CARGA
# ======================================================

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


def detectar_columnas(df):
    columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()

    columnas_categoricas = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    columnas_fecha = df.select_dtypes(
        include=["datetime64", "datetime64[ns]"]
    ).columns.tolist()

    return columnas_numericas, columnas_categoricas, columnas_fecha


def intentar_convertir_fechas(df):
    df_temp = df.copy()
    posibles_fechas = []

    for col in df_temp.columns:
        if df_temp[col].dtype == "object":
            try:
                muestra = df_temp[col].dropna().astype(str).head(50)

                if len(muestra) > 0:
                    convertida = pd.to_datetime(muestra, errors="coerce")
                    porcentaje_valido = convertida.notna().mean()

                    if porcentaje_valido >= 0.7:
                        df_temp[col] = pd.to_datetime(df_temp[col], errors="coerce")
                        posibles_fechas.append(col)

            except Exception:
                pass

    columnas_fecha = df_temp.select_dtypes(
        include=["datetime64", "datetime64[ns]"]
    ).columns.tolist()

    columnas_fecha = list(set(columnas_fecha + posibles_fechas))

    return df_temp, columnas_fecha


def limitar_dataframe(df, max_filas):
    if len(df) > max_filas:
        return df.sample(max_filas, random_state=42)
    return df


# ======================================================
# FUNCIONES DE GRÁFICOS INDIVIDUALES
# ======================================================

def grafico_individual(df, tipo, x, y=None, hue=None):
    fig, ax = plt.subplots(figsize=(10, 5))

    try:
        if tipo == "Histograma":
            sns.histplot(data=df, x=x, hue=hue, kde=True, ax=ax)

        elif tipo == "Boxplot":
            sns.boxplot(data=df, x=x if x else None, y=y, hue=hue, ax=ax)

        elif tipo == "Violin plot":
            sns.violinplot(data=df, x=x if x else None, y=y, hue=hue, ax=ax)

        elif tipo == "KDE plot":
            sns.kdeplot(data=df, x=x, hue=hue, fill=True, ax=ax)

        elif tipo == "Count plot":
            sns.countplot(data=df, x=x, hue=hue, ax=ax)

        elif tipo == "Scatter plot":
            sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)

        elif tipo == "Line plot":
            sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax)

        elif tipo == "Bar plot":
            sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)

        elif tipo == "Regresión":
            sns.regplot(data=df, x=x, y=y, ax=ax)

        ax.set_title(tipo)
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"No se pudo generar el gráfico individual: {e}")


# ======================================================
# FUNCIONES DE GRÁFICOS MÚLTIPLES EN GRILLA
# ======================================================

def crear_grilla(n_graficos, columnas_por_fila):
    filas = math.ceil(n_graficos / columnas_por_fila)
    return filas, columnas_por_fila


def graficos_multiples_univariados(
    df,
    columnas,
    tipo_grafico,
    columnas_por_fila=3,
    hue=None
):
    n = len(columnas)

    if n == 0:
        st.warning("Selecciona al menos una columna.")
        return

    filas, cols = crear_grilla(n, columnas_por_fila)

    fig, axes = plt.subplots(
        filas,
        cols,
        figsize=(cols * 5, filas * 4)
    )

    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(columnas):
        ax = axes[i]

        try:
            if tipo_grafico == "Histogramas múltiples":
                sns.histplot(data=df, x=col, kde=True, hue=hue, ax=ax)

            elif tipo_grafico == "Boxplots múltiples":
                sns.boxplot(data=df, y=col, ax=ax)

            elif tipo_grafico == "Violin plots múltiples":
                sns.violinplot(data=df, y=col, ax=ax)

            elif tipo_grafico == "KDE plots múltiples":
                sns.kdeplot(data=df, x=col, fill=True, hue=hue, ax=ax)

            elif tipo_grafico == "Distribución + Rug plot":
                sns.histplot(data=df, x=col, kde=True, ax=ax)
                sns.rugplot(data=df, x=col, ax=ax)

            ax.set_title(col)
            ax.tick_params(axis="x", rotation=45)

        except Exception as e:
            ax.set_title(f"{col}\nError")
            ax.text(
                0.5,
                0.5,
                str(e),
                ha="center",
                va="center",
                transform=ax.transAxes
            )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    st.pyplot(fig)


def graficos_multiples_bivariados(
    df,
    columnas_x,
    y_objetivo,
    tipo_grafico,
    columnas_por_fila=3,
    hue=None
):
    n = len(columnas_x)

    if n == 0:
        st.warning("Selecciona al menos una columna X.")
        return

    filas, cols = crear_grilla(n, columnas_por_fila)

    fig, axes = plt.subplots(
        filas,
        cols,
        figsize=(cols * 5, filas * 4)
    )

    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(columnas_x):
        ax = axes[i]

        try:
            if tipo_grafico == "Scatter múltiples contra objetivo":
                sns.scatterplot(data=df, x=col, y=y_objetivo, hue=hue, ax=ax)

            elif tipo_grafico == "Regresiones múltiples contra objetivo":
                sns.regplot(data=df, x=col, y=y_objetivo, ax=ax)

            elif tipo_grafico == "Line plots múltiples contra objetivo":
                sns.lineplot(data=df, x=col, y=y_objetivo, hue=hue, ax=ax)

            elif tipo_grafico == "Bar plots múltiples contra objetivo":
                sns.barplot(data=df, x=col, y=y_objetivo, hue=hue, ax=ax)

            ax.set_title(f"{col} vs {y_objetivo}")
            ax.tick_params(axis="x", rotation=45)

        except Exception as e:
            ax.set_title(f"{col}\nError")
            ax.text(
                0.5,
                0.5,
                str(e),
                ha="center",
                va="center",
                transform=ax.transAxes
            )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    st.pyplot(fig)


def graficos_multiples_box_violin_por_categoria(
    df,
    columnas_numericas,
    categoria,
    tipo_grafico,
    columnas_por_fila=3
):
    n = len(columnas_numericas)

    if n == 0:
        st.warning("Selecciona al menos una columna numérica.")
        return

    filas, cols = crear_grilla(n, columnas_por_fila)

    fig, axes = plt.subplots(
        filas,
        cols,
        figsize=(cols * 5, filas * 4)
    )

    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(columnas_numericas):
        ax = axes[i]

        try:
            if tipo_grafico == "Boxplots por categoría":
                sns.boxplot(data=df, x=categoria, y=col, ax=ax)

            elif tipo_grafico == "Violin plots por categoría":
                sns.violinplot(data=df, x=categoria, y=col, ax=ax)

            ax.set_title(f"{col} por {categoria}")
            ax.tick_params(axis="x", rotation=45)

        except Exception as e:
            ax.set_title(f"{col}\nError")
            ax.text(
                0.5,
                0.5,
                str(e),
                ha="center",
                va="center",
                transform=ax.transAxes
            )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    st.pyplot(fig)


# ======================================================
# FUNCIONES DE MATRICES
# ======================================================

def mostrar_matriz(df, columnas, tipo_matriz):
    if len(columnas) < 2:
        st.warning("Selecciona al menos 2 columnas numéricas.")
        return

    try:
        if tipo_matriz == "Matriz de correlación":
            corr = df[columnas].corr()

            fig, ax = plt.subplots(figsize=(11, 8))
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
            fig, ax = plt.subplots(figsize=(11, 8))
            sns.heatmap(
                df[columnas].dropna().head(100),
                cmap="viridis",
                ax=ax
            )
            ax.set_title("Heatmap de valores")
            st.pyplot(fig)

        elif tipo_matriz == "Pairplot":
            df_pair = df[columnas].dropna()
            fig = sns.pairplot(df_pair)
            st.pyplot(fig)

        elif tipo_matriz == "Scatter matrix":
            fig = plt.figure(figsize=(12, 10))
            scatter_matrix(
                df[columnas].dropna(),
                figsize=(12, 10),
                diagonal="hist"
            )
            st.pyplot(fig)

    except Exception as e:
        st.warning(f"No se pudo generar la matriz: {e}")


# ======================================================
# FUNCIONES DE SERIES TEMPORALES
# ======================================================

def preparar_serie_temporal(df, fecha_col, columnas_valor, frecuencia=None, agregacion="Media"):
    df_ts = df.copy()
    df_ts[fecha_col] = pd.to_datetime(df_ts[fecha_col], errors="coerce")
    df_ts = df_ts.dropna(subset=[fecha_col])
    df_ts = df_ts.sort_values(fecha_col)

    if frecuencia is None or frecuencia == "Sin remuestreo":
        return df_ts

    df_ts = df_ts.set_index(fecha_col)

    if agregacion == "Media":
        df_resampled = df_ts[columnas_valor].resample(frecuencia).mean()
    elif agregacion == "Suma":
        df_resampled = df_ts[columnas_valor].resample(frecuencia).sum()
    elif agregacion == "Mediana":
        df_resampled = df_ts[columnas_valor].resample(frecuencia).median()
    elif agregacion == "Máximo":
        df_resampled = df_ts[columnas_valor].resample(frecuencia).max()
    elif agregacion == "Mínimo":
        df_resampled = df_ts[columnas_valor].resample(frecuencia).min()
    else:
        df_resampled = df_ts[columnas_valor].resample(frecuencia).mean()

    df_resampled = df_resampled.reset_index()

    return df_resampled


def grafico_serie_temporal(
    df,
    fecha_col,
    columnas_valor,
    tipo_temporal,
    frecuencia=None,
    agregacion="Media"
):
    if len(columnas_valor) == 0:
        st.warning("Selecciona al menos una columna numérica para la serie temporal.")
        return

    df_ts = preparar_serie_temporal(
        df=df,
        fecha_col=fecha_col,
        columnas_valor=columnas_valor,
        frecuencia=frecuencia,
        agregacion=agregacion
    )

    try:
        if tipo_temporal == "Línea temporal":
            fig = px.line(
                df_ts,
                x=fecha_col,
                y=columnas_valor,
                title="Serie temporal - Línea"
            )

        elif tipo_temporal == "Área temporal":
            fig = px.area(
                df_ts,
                x=fecha_col,
                y=columnas_valor,
                title="Serie temporal - Área"
            )

        elif tipo_temporal == "Barras temporales":
            fig = px.bar(
                df_ts,
                x=fecha_col,
                y=columnas_valor,
                title="Serie temporal - Barras"
            )

        elif tipo_temporal == "Dispersión temporal":
            fig = px.scatter(
                df_ts,
                x=fecha_col,
                y=columnas_valor,
                title="Serie temporal - Dispersión"
            )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"No se pudo generar la serie temporal: {e}")


def multiples_series_temporales(
    df,
    fecha_col,
    columnas_valor,
    tipo_temporal,
    columnas_por_fila=3,
    frecuencia=None,
    agregacion="Media"
):
    if len(columnas_valor) == 0:
        st.warning("Selecciona al menos una columna numérica.")
        return

    df_ts = preparar_serie_temporal(
        df=df,
        fecha_col=fecha_col,
        columnas_valor=columnas_valor,
        frecuencia=frecuencia,
        agregacion=agregacion
    )

    n = len(columnas_valor)
    filas, cols = crear_grilla(n, columnas_por_fila)

    fig, axes = plt.subplots(
        filas,
        cols,
        figsize=(cols * 5, filas * 4)
    )

    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(columnas_valor):
        ax = axes[i]

        try:
            if tipo_temporal == "Líneas múltiples en grilla":
                ax.plot(df_ts[fecha_col], df_ts[col])

            elif tipo_temporal == "Barras múltiples en grilla":
                ax.bar(df_ts[fecha_col], df_ts[col])

            elif tipo_temporal == "Dispersión múltiple en grilla":
                ax.scatter(df_ts[fecha_col], df_ts[col])

            ax.set_title(col)
            ax.tick_params(axis="x", rotation=45)

        except Exception as e:
            ax.set_title(f"{col}\nError")
            ax.text(
                0.5,
                0.5,
                str(e),
                ha="center",
                va="center",
                transform=ax.transAxes
            )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    st.pyplot(fig)


# ======================================================
# SIDEBAR - CARGA DE DATOS
# ======================================================

st.sidebar.header("📁 Carga de datos")

uploaded_file = st.sidebar.file_uploader(
    "Sube un archivo CSV o Parquet",
    type=["csv", "parquet"]
)

df = cargar_archivo(uploaded_file)

if df is None:
    st.stop()

df, columnas_fecha_detectadas = intentar_convertir_fechas(df)

columnas_numericas, columnas_categoricas, columnas_fecha_tipo = detectar_columnas(df)

columnas_fecha = list(set(columnas_fecha_detectadas + columnas_fecha_tipo))
todas_columnas = df.columns.tolist()

st.sidebar.success(
    "Archivo cargado correctamente"
    if uploaded_file
    else "Usando California Housing por defecto"
)


# ======================================================
# OPCIONES GENERALES
# ======================================================

st.sidebar.header("⚙️ Opciones generales")

max_filas = st.sidebar.slider(
    "Máximo de filas para graficar",
    min_value=100,
    max_value=20000,
    value=5000,
    step=100
)

df_plot = limitar_dataframe(df, max_filas)

st.sidebar.write(f"Filas usadas para gráficos: **{len(df_plot)}**")


# ======================================================
# INFORMACIÓN DEL DATASET
# ======================================================

st.subheader("📌 Vista general del dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Filas", df.shape[0])

with col2:
    st.metric("Columnas", df.shape[1])

with col3:
    st.metric("Columnas numéricas", len(columnas_numericas))

with col4:
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


# ======================================================
# SELECCIÓN DE MODO
# ======================================================

st.sidebar.header("📊 Visualizaciones")

modo = st.sidebar.radio(
    "Modo de visualización",
    [
        "Individual",
        "Múltiple en grilla",
        "Agrupada / Matriz",
        "Series temporales"
    ]
)


# ======================================================
# MODO INDIVIDUAL
# ======================================================

if modo == "Individual":
    st.subheader("📈 Visualización individual")

    tipos_individuales = [
        "Histograma",
        "Boxplot",
        "Violin plot",
        "KDE plot",
        "Count plot",
        "Scatter plot",
        "Line plot",
        "Bar plot",
        "Regresión"
    ]

    tipo_grafico = st.sidebar.selectbox(
        "Tipo de gráfico",
        tipos_individuales
    )

    usar_hue = st.sidebar.checkbox("Usar Hue / color por grupo")

    hue = None
    if usar_hue:
        hue = st.sidebar.selectbox(
            "Columna Hue",
            columnas_categoricas + columnas_numericas
        )

    x = None
    y = None

    if tipo_grafico in ["Histograma", "KDE plot"]:
        x = st.sidebar.selectbox("Columna X", columnas_numericas)

    elif tipo_grafico == "Count plot":
        x = st.sidebar.selectbox("Columna X", todas_columnas)

    elif tipo_grafico in ["Boxplot", "Violin plot"]:
        usar_categoria = st.sidebar.checkbox("Usar columna categórica en X")

        if usar_categoria and len(columnas_categoricas) > 0:
            x = st.sidebar.selectbox("Columna categórica X", columnas_categoricas)
        else:
            x = None

        y = st.sidebar.selectbox("Columna numérica Y", columnas_numericas)

    elif tipo_grafico in ["Scatter plot", "Line plot", "Bar plot", "Regresión"]:
        x = st.sidebar.selectbox("Columna X", todas_columnas)
        y = st.sidebar.selectbox("Columna Y", columnas_numericas)

    grafico_individual(
        df=df_plot,
        tipo=tipo_grafico,
        x=x,
        y=y,
        hue=hue
    )


# ======================================================
# MODO MÚLTIPLE EN GRILLA
# ======================================================

elif modo == "Múltiple en grilla":
    st.subheader("🧩 Visualizaciones múltiples en grilla")

    st.write(
        "Este modo genera automáticamente varios gráficos en una sola figura, "
        "por ejemplo en formato 3x3, 3x4 o 4x4."
    )

    tipos_multiples = [
        "Histogramas múltiples",
        "Boxplots múltiples",
        "Violin plots múltiples",
        "KDE plots múltiples",
        "Distribución + Rug plot",
        "Scatter múltiples contra objetivo",
        "Regresiones múltiples contra objetivo",
        "Line plots múltiples contra objetivo",
        "Bar plots múltiples contra objetivo",
        "Boxplots por categoría",
        "Violin plots por categoría"
    ]

    tipo_multiple = st.sidebar.selectbox(
        "Tipo de gráfico múltiple",
        tipos_multiples
    )

    columnas_por_fila = st.sidebar.selectbox(
        "Columnas por fila",
        [2, 3, 4, 5],
        index=1
    )

    max_columnas_default = min(9, len(columnas_numericas))

    usar_hue_multiple = st.sidebar.checkbox("Usar Hue en gráficos múltiples")
    hue_multiple = None

    if usar_hue_multiple and len(columnas_categoricas + columnas_numericas) > 0:
        hue_multiple = st.sidebar.selectbox(
            "Columna Hue",
            columnas_categoricas + columnas_numericas
        )

    if tipo_multiple in [
        "Histogramas múltiples",
        "Boxplots múltiples",
        "Violin plots múltiples",
        "KDE plots múltiples",
        "Distribución + Rug plot"
    ]:
        columnas_seleccionadas = st.sidebar.multiselect(
            "Columnas numéricas a graficar",
            columnas_numericas,
            default=columnas_numericas[:max_columnas_default]
        )

        graficos_multiples_univariados(
            df=df_plot,
            columnas=columnas_seleccionadas,
            tipo_grafico=tipo_multiple,
            columnas_por_fila=columnas_por_fila,
            hue=hue_multiple
        )

    elif tipo_multiple in [
        "Scatter múltiples contra objetivo",
        "Regresiones múltiples contra objetivo",
        "Line plots múltiples contra objetivo",
        "Bar plots múltiples contra objetivo"
    ]:
        y_objetivo = st.sidebar.selectbox(
            "Variable objetivo Y",
            columnas_numericas
        )

        columnas_x = st.sidebar.multiselect(
            "Columnas X",
            [c for c in columnas_numericas if c != y_objetivo],
            default=[c for c in columnas_numericas if c != y_objetivo][:max_columnas_default]
        )

        graficos_multiples_bivariados(
            df=df_plot,
            columnas_x=columnas_x,
            y_objetivo=y_objetivo,
            tipo_grafico=tipo_multiple,
            columnas_por_fila=columnas_por_fila,
            hue=hue_multiple
        )

    elif tipo_multiple in [
        "Boxplots por categoría",
        "Violin plots por categoría"
    ]:
        if len(columnas_categoricas) == 0:
            st.warning("No hay columnas categóricas disponibles.")
        else:
            categoria = st.sidebar.selectbox(
                "Columna categórica",
                columnas_categoricas
            )

            columnas_box = st.sidebar.multiselect(
                "Columnas numéricas",
                columnas_numericas,
                default=columnas_numericas[:max_columnas_default]
            )

            graficos_multiples_box_violin_por_categoria(
                df=df_plot,
                columnas_numericas=columnas_box,
                categoria=categoria,
                tipo_grafico=tipo_multiple,
                columnas_por_fila=columnas_por_fila
            )


# ======================================================
# MODO AGRUPADO / MATRIZ
# ======================================================

elif modo == "Agrupada / Matriz":
    st.subheader("🔢 Visualización agrupada / matriz")

    tipos_matriz = [
        "Matriz de correlación",
        "Heatmap de valores",
        "Pairplot",
        "Scatter matrix"
    ]

    tipo_matriz = st.sidebar.selectbox(
        "Tipo de matriz",
        tipos_matriz
    )

    columnas_matriz = st.sidebar.multiselect(
        "Selecciona columnas numéricas",
        columnas_numericas,
        default=columnas_numericas[:min(6, len(columnas_numericas))]
    )

    mostrar_matriz(
        df=df_plot,
        columnas=columnas_matriz,
        tipo_matriz=tipo_matriz
    )


# ======================================================
# MODO SERIES TEMPORALES
# ======================================================

elif modo == "Series temporales":
    st.subheader("⏱️ Series temporales")

    if len(columnas_fecha) == 0:
        st.warning(
            "No se detectaron columnas de fecha automáticamente. "
            "Puedes intentar convertir una columna manualmente."
        )

        posible_fecha = st.sidebar.selectbox(
            "Selecciona una columna para intentar usar como fecha",
            todas_columnas
        )

        df_plot[posible_fecha] = pd.to_datetime(
            df_plot[posible_fecha],
            errors="coerce"
        )

        fecha_col = posible_fecha

    else:
        fecha_col = st.sidebar.selectbox(
            "Columna de fecha",
            columnas_fecha
        )

    columnas_valor = st.sidebar.multiselect(
        "Columnas numéricas para graficar",
        columnas_numericas,
        default=columnas_numericas[:min(3, len(columnas_numericas))]
    )

    tipo_temporal = st.sidebar.selectbox(
        "Tipo de gráfico temporal",
        [
            "Línea temporal",
            "Área temporal",
            "Barras temporales",
            "Dispersión temporal",
            "Líneas múltiples en grilla",
            "Barras múltiples en grilla",
            "Dispersión múltiple en grilla"
        ]
    )

    usar_remuestreo = st.sidebar.checkbox("Agrupar / remuestrear por frecuencia")

    frecuencia = "Sin remuestreo"
    agregacion = "Media"

    if usar_remuestreo:
        frecuencia = st.sidebar.selectbox(
            "Frecuencia",
            [
                "D",
                "W",
                "M",
                "Q",
                "Y"
            ],
            format_func=lambda x: {
                "D": "Diario",
                "W": "Semanal",
                "M": "Mensual",
                "Q": "Trimestral",
                "Y": "Anual"
            }.get(x, x)
        )

        agregacion = st.sidebar.selectbox(
            "Agregación",
            [
                "Media",
                "Suma",
                "Mediana",
                "Máximo",
                "Mínimo"
            ]
        )

    if tipo_temporal in [
        "Línea temporal",
        "Área temporal",
        "Barras temporales",
        "Dispersión temporal"
    ]:
        grafico_serie_temporal(
            df=df_plot,
            fecha_col=fecha_col,
            columnas_valor=columnas_valor,
            tipo_temporal=tipo_temporal,
            frecuencia=frecuencia,
            agregacion=agregacion
        )

    else:
        columnas_por_fila_ts = st.sidebar.selectbox(
            "Columnas por fila en grilla temporal",
            [2, 3, 4, 5],
            index=1
        )

        multiples_series_temporales(
            df=df_plot,
            fecha_col=fecha_col,
            columnas_valor=columnas_valor,
            tipo_temporal=tipo_temporal,
            columnas_por_fila=columnas_por_fila_ts,
            frecuencia=frecuencia,
            agregacion=agregacion
        )


# ======================================================
# OPCIONES EXTRA
# ======================================================

st.sidebar.header("🧪 Análisis extra")

mostrar_nulos = st.sidebar.checkbox("Mostrar análisis de valores nulos")
mostrar_correlaciones = st.sidebar.checkbox("Mostrar correlaciones ordenadas")
mostrar_categoricas = st.sidebar.checkbox("Mostrar conteos de categóricas")


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
        ax.tick_params(axis="x", rotation=45)
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

        fig, ax = plt.subplots(figsize=(10, 5))
        corr.plot(kind="bar", ax=ax)
        ax.set_title(f"Correlaciones con {variable_objetivo}")
        ax.set_ylabel("Correlación")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)


if mostrar_categoricas:
    st.subheader("🏷️ Conteo de variables categóricas")

    if len(columnas_categoricas) == 0:
        st.info("No hay columnas categóricas detectadas.")
    else:
        col_cat = st.selectbox(
            "Selecciona columna categórica",
            columnas_categoricas
        )

        conteo = df[col_cat].value_counts().head(30)

        st.dataframe(
            conteo.rename("Conteo").to_frame(),
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        conteo.plot(kind="bar", ax=ax)
        ax.set_title(f"Conteo de {col_cat}")
        ax.set_ylabel("Frecuencia")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)


# ======================================================
# PIE
# ======================================================

st.caption(
    "Aplicación creada con Streamlit, Pandas, NumPy, Matplotlib, Seaborn, Plotly y Scikit-learn."
)
