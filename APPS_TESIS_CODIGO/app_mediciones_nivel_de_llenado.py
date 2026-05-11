import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Nivel de llenado SAG",
    layout="wide"
)

st.title("Nivel de llenado SAG")
st.caption("Grind Out y Crash Stop en gráficos separados, diferenciados por molino, método y revisión")

# Nota:
# - Los datos originales de SAG 2 se mantienen.
# - Los datos agregados para SAG 1, SAG 3 y SAG 4 se cargan con Metodo = "Conteo Lifter (SGS)"
#   cuando provienen de conteo/lifter o cuando no se especificó otro método en la tabla entregada.
# - Campaña se asigna siguiendo la lógica visible en SAG 2:
#   Campaña 25: hasta enero 2025, Campaña 26: marzo a julio 2025 antes del 26/07/2025,
#   Campaña 27: desde 26/07/2025.

data = [
    # =========================
    # SAG 2 - Datos existentes
    # =========================
    {
        "Molino": "SAG 2",
        "Campaña": "20-21",
        "Fecha": "18/08/2022",
        "Hora": "",
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
        "Hora": "",
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
        "Hora": "",
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
        "Hora": "",
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
        "Hora": "",
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
        "Hora": "",
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
        "Hora": "",
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
        "Hora": "",
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
        "Hora": "",
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
        "Hora": "",
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
        "Hora": "",
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
        "Hora": "",
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
        "Hora": "",
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
        "Hora": "",
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
        "Hora": "",
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
        "Hora": "",
        "Procedimiento": "Crash Stop",
        "Metodo": "Faro (Elecmetal)",
        "Valor": 30.7,
        "Revision": "Valido",
        "Comentario": "~ Operación estable previo detención"
    },

    # =========================
    # SAG 1 - Datos agregados
    # =========================
    {"Molino": "SAG 1", "Campaña": "25", "Fecha": "03/01/2025", "Hora": "12:05:00", "Procedimiento": "Grind Out", "Metodo": "Conteo Lifter (SGS)", "Valor": 16.3, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 1", "Campaña": "26", "Fecha": "17/02/2025", "Hora": "12:30:00", "Procedimiento": "Crash Stop", "Metodo": "Conteo Lifter (SGS)", "Valor": 23.7, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 1", "Campaña": "26", "Fecha": "22/03/2025", "Hora": "11:05:00", "Procedimiento": "Grind Out", "Metodo": "Conteo Lifter (SGS)", "Valor": 19.2, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 1", "Campaña": "26", "Fecha": "26/04/2025", "Hora": "23:45:00", "Procedimiento": "Grind Out", "Metodo": "Conteo Lifter (SGS)", "Valor": 17.3, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 1", "Campaña": "26", "Fecha": "12/06/2025", "Hora": "12:00:00", "Procedimiento": "Grind Out", "Metodo": "Conteo Lifter (SGS)", "Valor": 17.5, "Revision": "Valido", "Comentario": "-"},

    # =========================
    # SAG 3 - Datos agregados
    # =========================
    {"Molino": "SAG 3", "Campaña": "25", "Fecha": "06/01/2025", "Hora": "10:45:00", "Procedimiento": "Grind Out", "Metodo": "Conteo Lifter (SGS)", "Valor": 17.9, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 3", "Campaña": "26", "Fecha": "17/03/2025", "Hora": "00:45:00", "Procedimiento": "Grind Out", "Metodo": "Conteo Lifter (SGS)", "Valor": 18.6, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 3", "Campaña": "26", "Fecha": "30/03/2025", "Hora": "19:30:00", "Procedimiento": "Grind Out", "Metodo": "Conteo Lifter (SGS)", "Valor": 17.2, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 3", "Campaña": "26", "Fecha": "26/04/2025", "Hora": "17:30:00", "Procedimiento": "Crash Stop", "Metodo": "Conteo Lifter (SGS)", "Valor": 29.7, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 3", "Campaña": "26", "Fecha": "15/05/2025", "Hora": "17:30:00", "Procedimiento": "Crash Stop", "Metodo": "Conteo Lifter (SGS)", "Valor": 28.7, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 3", "Campaña": "26", "Fecha": "15/05/2025", "Hora": "18:00:00", "Procedimiento": "Grind Out", "Metodo": "Conteo Lifter (SGS)", "Valor": 17.9, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 3", "Campaña": "26", "Fecha": "15/05/2025", "Hora": "17:45:00", "Procedimiento": "Crash Stop", "Metodo": "Conteo Lifter (SGS)", "Valor": 29.2, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 3", "Campaña": "26", "Fecha": "30/05/2025", "Hora": "12:50:00", "Procedimiento": "Grind Out", "Metodo": "Conteo Lifter (SGS)", "Valor": 18.5, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 3", "Campaña": "26", "Fecha": "24/06/2025", "Hora": "14:30:00", "Procedimiento": "Grind Out", "Metodo": "Conteo Lifter (SGS)", "Valor": 18.1, "Revision": "Valido", "Comentario": "-"},

    # =========================
    # SAG 4 - Datos agregados
    # =========================
    {"Molino": "SAG 4", "Campaña": "25", "Fecha": "27/01/2025", "Hora": "07:15:00", "Procedimiento": "Grind Out", "Metodo": "Conteo Lifter (SGS)", "Valor": 14.6, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 4", "Campaña": "26", "Fecha": "02/04/2025", "Hora": "12:40:00", "Procedimiento": "Grind Out", "Metodo": "Conteo Lifter (SGS)", "Valor": 16.5, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 4", "Campaña": "26", "Fecha": "10/04/2025", "Hora": "17:10:00", "Procedimiento": "Crash Stop", "Metodo": "Conteo Lifter (SGS)", "Valor": 27.4, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 4", "Campaña": "26", "Fecha": "25/06/2025", "Hora": "18:00:00", "Procedimiento": "Grind Out", "Metodo": "Conteo Lifter (SGS)", "Valor": 16.5, "Revision": "Valido", "Comentario": "-"},
    {"Molino": "SAG 4", "Campaña": "26", "Fecha": "02/07/2025", "Hora": "23:00:00", "Procedimiento": "Crash Stop", "Metodo": "Conteo Lifter (SGS)", "Valor": 25.9, "Revision": "Valido", "Comentario": "-"},
]

df = pd.DataFrame(data)
df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y")
df = df.sort_values(["Fecha", "Hora", "Molino"])

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
        color="Molino",
        symbol="Revision",
        text="Valor",
        hover_data={
            "Molino": True,
            "Campaña": True,
            "Fecha": "|%d/%m/%Y",
            "Hora": True,
            "Procedimiento": True,
            "Metodo": True,
            "Valor": ":.1f",
            "Revision": True,
            "Comentario": True
        },
        labels={
            "Fecha": "Fecha",
            "Valor": "Nivel de llenado (%)",
            "Molino": "Molino",
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
        height=500,
        xaxis_title="Fecha",
        yaxis_title="Nivel de llenado (%)",
        legend_title="Molino / Revisión",
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
        "Hora",
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
