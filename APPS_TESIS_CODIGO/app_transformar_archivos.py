# app.py

import streamlit as st
import pandas as pd
from io import BytesIO


st.set_page_config(
    page_title="Convertidor de archivos",
    page_icon="📄",
    layout="wide"
)

st.title("Convertidor de Excel, CSV y Parquet")

st.write(
    "Sube un archivo Excel, CSV o Parquet y conviértelo al formato que necesites."
)


def leer_archivo(uploaded_file):
    nombre = uploaded_file.name.lower()

    if nombre.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    elif nombre.endswith(".xlsx") or nombre.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    elif nombre.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)

    else:
        raise ValueError("Formato no soportado")


def convertir_a_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def convertir_a_excel(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos")

    output.seek(0)
    return output


def preparar_para_parquet(df):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(["nan", "None", "NaT"], "")

    return df


def convertir_a_parquet(df):
    output = BytesIO()

    df_parquet = preparar_para_parquet(df)

    df_parquet.to_parquet(
        output,
        index=False,
        engine="pyarrow"
    )

    output.seek(0)
    return output


uploaded_file = st.file_uploader(
    "Selecciona un archivo",
    type=["csv", "xlsx", "xls", "parquet"]
)

if uploaded_file is not None:
    try:
        df = leer_archivo(uploaded_file)

        st.success("Archivo cargado correctamente")

        st.subheader("Vista previa")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Información del archivo")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Filas", df.shape[0])

        with col2:
            st.metric("Columnas", df.shape[1])

        st.subheader("Tipos de datos")
        st.dataframe(
            df.dtypes.astype(str).reset_index().rename(
                columns={"index": "Columna", 0: "Tipo"}
            ),
            use_container_width=True
        )

        formato_salida = st.selectbox(
            "Selecciona el formato de salida",
            ["Excel", "CSV", "Parquet"]
        )

        nombre_base = uploaded_file.name.rsplit(".", 1)[0]

        if formato_salida == "CSV":
            archivo_convertido = convertir_a_csv(df)
            nombre_salida = f"{nombre_base}.csv"
            mime_type = "text/csv"

        elif formato_salida == "Excel":
            archivo_convertido = convertir_a_excel(df)
            nombre_salida = f"{nombre_base}.xlsx"
            mime_type = (
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )

        elif formato_salida == "Parquet":
            archivo_convertido = convertir_a_parquet(df)
            nombre_salida = f"{nombre_base}.parquet"
            mime_type = "application/octet-stream"

        st.download_button(
            label=f"Descargar como {formato_salida}",
            data=archivo_convertido,
            file_name=nombre_salida,
            mime=mime_type
        )

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
