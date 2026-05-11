# app.py
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Nivel de llenado SAG 2",
    layout="wide"
)

st.title("Nivel de llenado SAG 2")
st.caption("Grind Out y Crash Stop en gráficos separados, diferenciados por método y revisión")

data = [
    {
        "Molino": "SAG 2",
        "Campaña": "20-21",
        "Fecha": "18/08/2022",
        "Procedimiento": "Grind Out",
        "Metodo": "Faro (Elecmetal)",
        "Valor": 16.7,
        "Revision": "PDTE",
        "Comentario": "-"
    },
    {
        "Molino": "SAG 2",
        "Campaña": "21-22",
        "Fecha": "27/01/2023",
        "Procedimiento": "Grind Out",
        "Metodo": "Faro (Elecmetal)",
        "Valor": 15.6,
        "Revision": "PDTE",
        "Comentario": "-"
    },
    {
        "Molino": "SAG 2",
        "Campaña": "22",
        "Fecha": "06/06/2023",
        "Procedimiento": "Crash Stop",
        "Metodo": "Faro (Elecmetal)",
        "Valor": 35.5,
        "Revision": "PDTE",
        "Comentario": "-"
    },
    {
        "Molino": "SAG 2",
        "Campaña": "25",
        "Fecha": "05/12/2024",
        "Procedimiento": "Grind Out",
        "Metodo": "Faro (Elecmetal)",
        "Valor": 17.5,
        "Revision": "Valido",
        "Comentario": "~ Estabilización de celda de carga"
    },
    {
        "Molino": "SAG 2",
        "Campaña": "25",
        "Fecha": "05/01/2025",
        "Procedimiento": "Grind Out",
        "Metodo": "Conteo Lifter (SGS)",
        "Valor": 16.9,
        "Revision": "Valido",
        "Comentario": "~ Estabilización de celda de carga"
    },
    {
        "Molino": "SAG 2",
        "Campaña": "25",
        "Fecha": "13/01/2025",
        "Procedimiento": "Grind Out",
        "Metodo": "Conteo Lifter (SGS)",
        "Valor": 17.0,
        "Revision": "Valido",
        "Comentario": "~ Estabilización de celda de carga"
    },
    {
        "Molino": "SAG 2",
        "Campaña": "26",
        "Fecha": "16/03/2025",
        "Procedimiento": "Grind Out",
        "Metodo": "Conteo Lifter (SGS)",
        "Valor": 17.0,
        "Revision": "Valido",
        "Comentario": "~ Estabilización de celda de carga"
    },
    {
        "Molino": "SAG 2",
        "Campaña": "26",
        "Fecha": "31/03/2025",
        "Procedimiento": "Crash Stop",
        "Metodo": "Conteo Lifter (SGS)",
        "Valor": 31.4,
        "Revision": "Valido",
        "Comentario": "Relativa operación estable previo detención"
    },
    {
        "Molino": "SAG 2",
        "Campaña": "26",
        "Fecha": "03/06/2025",
        "Procedimiento": "Grind Out",
        "Metodo": "Faro (Elecmetal)",
        "Valor": 19.2,
        "Revision": "Valido",
        "Comentario": "~ Estabilización de celda de carga"
    },
    {
        "Molino": "SAG 2",
        "Campaña": "26",
        "Fecha": "03/06/2025",
        "Procedimiento": "Grind Out",
        "Metodo": "Conteo Lifter (SGS)",
        "Valor": 18.0,
        "Revision": "Valido",
        "Comentario": "~ Estabilización de celda de carga"
    },
    {
        "Molino": "SAG 2",
        "Campaña": "26",
        "Fecha": "05/06/2025",
        "Procedimiento": "Crash Stop",
        "Metodo": "Conteo Lifter (SGS)",
        "Valor": 27.9,
        "Revision": "No valido",
        "Comentario": "No hay operación estable previo a detención"
    },
    {
        "Molino": "SAG 2",
        "Campaña": "26",
        "Fecha": "11/06/2025",
        "Procedimiento": "Crash Stop",
        "Metodo": "Conteo Lifter (SGS)",
        "Valor": 31.1,
        "Revision": "No valido",
        "Comentario": "No hay operación estable previo a detención"
    },
    {
        "Molino": "SAG 2",
        "Campaña": "26",
        "Fecha": "19/07/2025",
        "Procedimiento": "Grind Out",
        "Metodo": "Conteo Lifter (SGS)",
        "Valor": 18.1,
        "Revision": "Valido",
        "Comentario": "~ Estabilización de celda de carga"
    },
    {
        "Molino": "SAG 2",
        "Campaña": "26",
        "Fecha": "20/07/2025",
        "Procedimiento": "Crash Stop",
        "Metodo": "Conteo Lifter (SGS)",
        "Valor": 31.0,
        "Revision": "Valido",
        "Comentario": "~ Operación estable previo detención"
    },
    {
        "Molino": "SAG 2",
        "Campaña": "27",
        "Fecha": "26/07/2025",
        "Procedimiento": "Grind Out",
        "Metodo": "Faro (Elecmetal)",
        "Valor": 18.6,
        "Revision": "Valido",
        "Comentario": "~ Estabilización de celda de carga"
    },
    {
        "Molino": "SAG 2",
        "Campaña": "27",
        "Fecha": "26/07/2025",
        "Procedimiento": "Crash Stop",
        "Metodo": "Faro (Elecmetal)",
        "Valor": 30.7,
        "Revision": "Valido",
        "Comentario": "~ Operación estable previo detención"
    },
]

df = pd.DataFrame(data)
df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y")
df = df.sort_values("Fecha")

st.sidebar.header("Filtros")

molinos = st.sidebar.multiselect(
    "Molino",
    options=sorted(df["Molino"].unique()),
    default=sorted(df["Molino"].unique())
)

campañas = st.sidebar.multiselect(
    "Campaña",
    options=sorted(df["Campaña"].unique()),
    default=sorted(df["Campaña"].unique())
)

procedimientos = st.sidebar.multiselect(
    "Procedimiento",
    options=sorted(df["Procedimiento"].unique()),
    default=sorted(df["Procedimiento"].unique())
)

metodos = st.sidebar.multiselect(
    "Método",
    options=sorted(df["Metodo"].unique()),
    default=sorted(df["Metodo"].unique())
)

revisiones = st.sidebar.multiselect(
    "Revisión",
    options=sorted(df["Revision"].unique()),
    default=sorted(df["Revision"].unique())
)

fecha_min = df["Fecha"].min().date()
fecha_max = df["Fecha"].max().date()

rango_fechas = st.sidebar.date_input(
    "Rango de fechas",
    value=(fecha_min, fecha_max),
    min_value=fecha_min,
    max_value=fecha_max,
    format="DD/MM/YYYY"
)

if len(rango_fechas) == 2:
    fecha_inicio = pd.to_datetime(rango_fechas[0])
    fecha_fin = pd.to_datetime(rango_fechas[1])
else:
    fecha_inicio = pd.to_datetime(fecha_min)
    fecha_fin = pd.to_datetime(fecha_max)

df_filtrado = df[
    (df["Molino"].isin(molinos)) &
    (df["Campaña"].isin(campañas)) &
    (df["Procedimiento"].isin(procedimientos)) &
    (df["Metodo"].isin(metodos)) &
    (df["Revision"].isin(revisiones)) &
    (df["Fecha"] >= fecha_inicio) &
    (df["Fecha"] <= fecha_fin)
].copy()


def crear_grafico(df_tipo, titulo):
    fig = px.scatter(
        df_tipo,
        x="Fecha",
        y="Valor",
        color="Metodo",
        symbol="Revision",
        text="Valor",
        hover_data={
            "Molino": True,
            "Campaña": True,
            "Fecha": "|%d/%m/%Y",
            "Procedimiento": True,
            "Metodo": True,
            "Valor": ":.1f",
            "Revision": True,
            "Comentario": True
        },
        labels={
            "Fecha": "Fecha",
            "Valor": "Nivel de llenado (%)",
            "Metodo": "Método",
            "Procedimiento": "Procedimiento",
            "Revision": "Revisión",
            "Comentario": "Comentario"
        },
        title=titulo
    )

    fig.update_traces(
        mode="markers+text",
        textposition="top center",
        marker=dict(size=11)
    )

    fig.update_layout(
        height=450,
        xaxis_title="Fecha",
        yaxis_title="Nivel de llenado (%)",
        legend_title="Método / Revisión",
        hovermode="closest"
    )

    return fig


df_grind_out = df_filtrado[df_filtrado["Procedimiento"] == "Grind Out"]
df_crash_stop = df_filtrado[df_filtrado["Procedimiento"] == "Crash Stop"]

if not df_grind_out.empty:
    fig_grind_out = crear_grafico(
        df_grind_out,
        "Nivel de llenado - Grind Out"
    )
    st.plotly_chart(fig_grind_out, use_container_width=True)
else:
    st.info("No hay datos disponibles para Grind Out con los filtros seleccionados.")

if not df_crash_stop.empty:
    fig_crash_stop = crear_grafico(
        df_crash_stop,
        "Nivel de llenado - Crash Stop"
    )
    st.plotly_chart(fig_crash_stop, use_container_width=True)
else:
    st.info("No hay datos disponibles para Crash Stop con los filtros seleccionados.")


st.subheader("Datos filtrados")

tabla = df_filtrado.copy()
tabla["Fecha"] = tabla["Fecha"].dt.strftime("%d/%m/%Y")

tabla = tabla.rename(columns={
    "Metodo": "Método",
    "Revision": "Revisión",
    "Valor": "Nivel de llenado, %"
})

tabla = tabla[
    [
        "Molino",
        "Campaña",
        "Fecha",
        "Procedimiento",
        "Método",
        "Nivel de llenado, %",
        "Revisión",
        "Comentario"
    ]
]

st.dataframe(
    tabla,
    use_container_width=True,
    hide_index=True
)
