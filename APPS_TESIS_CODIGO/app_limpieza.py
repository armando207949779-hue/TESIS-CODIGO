# app.py

import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing


st.set_page_config(
    page_title="Limpieza de DataFrame",
    page_icon="🧹",
    layout="wide"
)


# ---------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------

@st.cache_data
def cargar_california_housing():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    return df


def leer_archivo(uploaded_file):
    nombre = uploaded_file.name.lower()

    if nombre.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    elif nombre.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)

    elif nombre.endswith(".xlsx") or nombre.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    else:
        raise ValueError("Formato no soportado")


def calcular_tabla_estadisticas(df, columnas_numericas):
    filas = []

    for col in columnas_numericas:
        serie = df[col].dropna()

        p1 = serie.quantile(0.01)
        p5 = serie.quantile(0.05)
        p95 = serie.quantile(0.95)
        p99 = serie.quantile(0.99)

        filas.append({
            "Variable": col,
            "Min": serie.min(),
            "P1": p1,
            "P5": p5,
            "Límite inferior": p1,
            "Límite superior": p99,
            "P95": p95,
            "P99": p99,
            "Max": serie.max()
        })

    return pd.DataFrame(filas)


def limpiar_dataframe(df, limites):
    df_limpio = df.copy()
    resumen = []

    total_inicial = len(df_limpio)

    for variable, lims in limites.items():
        limite_inferior = lims["limite_inferior"]
        limite_superior = lims["limite_superior"]

        antes = len(df_limpio)

        mascara_valida = (
            df_limpio[variable].isna()
            | (
                (df_limpio[variable] >= limite_inferior)
                & (df_limpio[variable] <= limite_superior)
            )
        )

        df_limpio = df_limpio[mascara_valida].copy()

        despues = len(df_limpio)
        eliminados = antes - despues

        resumen.append({
            "Variable": variable,
            "Límite inferior usado": limite_inferior,
            "Límite superior usado": limite_superior,
            "Registros antes": antes,
            "Registros después": despues,
            "Registros eliminados": eliminados,
            "% eliminado sobre paso": 100 * eliminados / antes if antes > 0 else 0,
            "% eliminado sobre total inicial": 100 * eliminados / total_inicial if total_inicial > 0 else 0
        })

    resumen_df = pd.DataFrame(resumen)

    return df_limpio, resumen_df


def convertir_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def convertir_parquet(df):
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    return buffer


# ---------------------------------------------------------
# Interfaz principal
# ---------------------------------------------------------

st.title("Limpieza de DataFrame por límites percentiles")

st.write(
    """
    Sube un archivo CSV, Parquet o Excel.  
    Si no subes ningún archivo, se cargará automáticamente el dataset California Housing.
    """
)

uploaded_file = st.file_uploader(
    "Sube tu archivo",
    type=["csv", "parquet", "xlsx", "xls"]
)


# ---------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------

try:
    if uploaded_file is not None:
        df = leer_archivo(uploaded_file)
        st.success(f"Archivo cargado correctamente: {uploaded_file.name}")
    else:
        df = cargar_california_housing()
        st.info("No se subió archivo. Se está usando California Housing por defecto.")

except Exception as e:
    st.error(f"Error al cargar el archivo: {e}")
    st.stop()


st.subheader("Vista previa del DataFrame")
st.dataframe(df.head(100), use_container_width=True)

st.write(f"Filas: **{df.shape[0]:,}**")
st.write(f"Columnas: **{df.shape[1]:,}**")


# ---------------------------------------------------------
# Selección de variables numéricas
# ---------------------------------------------------------

columnas_numericas = df.select_dtypes(include=["number"]).columns.tolist()

if len(columnas_numericas) == 0:
    st.warning("El DataFrame no tiene columnas numéricas para limpiar.")
    st.stop()

st.subheader("Variables numéricas detectadas")
st.write(columnas_numericas)


# ---------------------------------------------------------
# Tabla de estadísticas
# ---------------------------------------------------------

tabla_stats = calcular_tabla_estadisticas(df, columnas_numericas)

st.subheader("Tabla de límites por variable")

st.write(
    """
    Por defecto, el **límite inferior** es el percentil 1 y el 
    **límite superior** es el percentil 99.  
    Puedes modificar los límites antes de confirmar la limpieza.
    """
)

limites_usuario = {}

for i, fila in tabla_stats.iterrows():
    variable = fila["Variable"]

    with st.expander(f"Variable: {variable}", expanded=True):
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9)

        c1.metric("Variable", variable)
        c2.metric("Min", f"{fila['Min']:.4f}")
        c3.metric("P1", f"{fila['P1']:.4f}")
        c4.metric("P5", f"{fila['P5']:.4f}")

        limite_inferior = c5.number_input(
            "Límite inferior",
            value=float(fila["Límite inferior"]),
            key=f"lim_inf_{variable}",
            format="%.6f"
        )

        limite_superior = c6.number_input(
            "Límite superior",
            value=float(fila["Límite superior"]),
            key=f"lim_sup_{variable}",
            format="%.6f"
        )

        c7.metric("P95", f"{fila['P95']:.4f}")
        c8.metric("P99", f"{fila['P99']:.4f}")
        c9.metric("Max", f"{fila['Max']:.4f}")

        limites_usuario[variable] = {
            "limite_inferior": limite_inferior,
            "limite_superior": limite_superior
        }


# ---------------------------------------------------------
# Tabla resumen editable visual
# ---------------------------------------------------------

st.subheader("Resumen de límites definidos")

tabla_limites = tabla_stats.copy()
tabla_limites["Límite inferior"] = tabla_limites["Variable"].map(
    lambda x: limites_usuario[x]["limite_inferior"]
)
tabla_limites["Límite superior"] = tabla_limites["Variable"].map(
    lambda x: limites_usuario[x]["limite_superior"]
)

st.dataframe(
    tabla_limites[
        [
            "Variable",
            "Min",
            "P1",
            "P5",
            "Límite inferior",
            "Límite superior",
            "P95",
            "P99",
            "Max"
        ]
    ],
    use_container_width=True
)


# ---------------------------------------------------------
# Confirmar limpieza
# ---------------------------------------------------------

st.subheader("Confirmar limpieza")

confirmar = st.button("Confirmar y limpiar DataFrame")

if confirmar:
    errores = []

    for variable, lims in limites_usuario.items():
        if lims["limite_inferior"] > lims["limite_superior"]:
            errores.append(variable)

    if errores:
        st.error(
            "Hay variables donde el límite inferior es mayor que el límite superior: "
            + ", ".join(errores)
        )
        st.stop()

    df_limpio, resumen_limpieza = limpiar_dataframe(df, limites_usuario)

    st.session_state["df_limpio"] = df_limpio
    st.session_state["resumen_limpieza"] = resumen_limpieza

    st.success("Limpieza realizada correctamente.")


# ---------------------------------------------------------
# Resultados de limpieza
# ---------------------------------------------------------

if "df_limpio" in st.session_state:
    df_limpio = st.session_state["df_limpio"]
    resumen_limpieza = st.session_state["resumen_limpieza"]

    st.subheader("Resumen general")

    filas_originales = len(df)
    filas_limpias = len(df_limpio)
    filas_eliminadas = filas_originales - filas_limpias
    porcentaje_eliminado = (
        100 * filas_eliminadas / filas_originales
        if filas_originales > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Filas originales", f"{filas_originales:,}")
    col2.metric("Filas limpias", f"{filas_limpias:,}")
    col3.metric("Filas eliminadas", f"{filas_eliminadas:,}")
    col4.metric("% eliminado", f"{porcentaje_eliminado:.2f}%")

    st.subheader("Resumen de limpieza por variable")
    st.dataframe(resumen_limpieza, use_container_width=True)

    st.subheader("Gráfico porcentual de limpieza por variable")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        resumen_limpieza["Variable"],
        resumen_limpieza["% eliminado sobre total inicial"]
    )

    ax.set_xlabel("Variable")
    ax.set_ylabel("% eliminado sobre total inicial")
    ax.set_title("Porcentaje de registros eliminados por variable")
    ax.tick_params(axis="x", rotation=45)

    st.pyplot(fig)

    st.subheader("Vista previa del DataFrame limpio")
    st.dataframe(df_limpio.head(100), use_container_width=True)

    st.subheader("Descargar DataFrame limpio")

    csv_limpio = convertir_csv(df_limpio)
    parquet_limpio = convertir_parquet(df_limpio)

    col_csv, col_parquet = st.columns(2)

    with col_csv:
        st.download_button(
            label="Descargar CSV limpio",
            data=csv_limpio,
            file_name="dataframe_limpio.csv",
            mime="text/csv"
        )

    with col_parquet:
        st.download_button(
            label="Descargar Parquet limpio",
            data=parquet_limpio,
            file_name="dataframe_limpio.parquet",
            mime="application/octet-stream"
        )