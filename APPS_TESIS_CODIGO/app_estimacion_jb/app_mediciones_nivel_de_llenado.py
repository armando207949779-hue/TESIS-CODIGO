# app.py
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Nivel de llenado SAG 2",
    layout="wide"
)

st.title("Nivel de llenado SAG 2")
st.caption("Grind Out y Crash Stop diferenciados por método")

data = [
    {"Fecha": "18/08/2022", "Metodo": "Faro (Elecmetal)", "Tipo": "Grind Out", "Valor": 16.7},
    {"Fecha": "27/01/2023", "Metodo": "Faro (Elecmetal)", "Tipo": "Grind Out", "Valor": 15.6},
    {"Fecha": "06/06/2023", "Metodo": "Faro (Elecmetal)", "Tipo": "Crash Stop", "Valor": 35.5},
    {"Fecha": "05/12/2024", "Metodo": "Faro (Elecmetal)", "Tipo": "Grind Out", "Valor": 17.5},
    {"Fecha": "05/01/2025", "Metodo": "Conteo Lifter (SGS)", "Tipo": "Grind Out", "Valor": 16.9},
    {"Fecha": "13/01/2025", "Metodo": "Conteo Lifter (SGS)", "Tipo": "Grind Out", "Valor": 17.0},
    {"Fecha": "16/03/2025", "Metodo": "Conteo Lifter (SGS)", "Tipo": "Grind Out", "Valor": 17.0},
    {"Fecha": "31/03/2025", "Metodo": "Conteo Lifter (SGS)", "Tipo": "Crash Stop", "Valor": 31.4},
    {"Fecha": "03/06/2025", "Metodo": "Faro (Elecmetal)", "Tipo": "Grind Out", "Valor": 19.2},
    {"Fecha": "03/06/2025", "Metodo": "Conteo Lifter (SGS)", "Tipo": "Grind Out", "Valor": 18.0},
    {"Fecha": "05/06/2025", "Metodo": "Conteo Lifter (SGS)", "Tipo": "Crash Stop", "Valor": 27.9},
    {"Fecha": "11/06/2025", "Metodo": "Conteo Lifter (SGS)", "Tipo": "Crash Stop", "Valor": 31.1},
    {"Fecha": "19/07/2025", "Metodo": "Conteo Lifter (SGS)", "Tipo": "Grind Out", "Valor": 18.1},
    {"Fecha": "20/07/2025", "Metodo": "Conteo Lifter (SGS)", "Tipo": "Crash Stop", "Valor": 31.0},
    {"Fecha": "26/07/2025", "Metodo": "Faro (Elecmetal)", "Tipo": "Grind Out", "Valor": 18.6},
    {"Fecha": "26/07/2025", "Metodo": "Faro (Elecmetal)", "Tipo": "Crash Stop", "Valor": 30.7},
]

df = pd.DataFrame(data)
df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y")
df = df.sort_values("Fecha")

st.sidebar.header("Filtros")

tipos = st.sidebar.multiselect(
    "Tipo de medición",
    options=sorted(df["Tipo"].unique()),
    default=sorted(df["Tipo"].unique())
)

metodos = st.sidebar.multiselect(
    "Método",
    options=sorted(df["Metodo"].unique()),
    default=sorted(df["Metodo"].unique())
)

df_filtrado = df[
    (df["Tipo"].isin(tipos)) &
    (df["Metodo"].isin(metodos))
].copy()

fig = px.scatter(
    df_filtrado,
    x="Fecha",
    y="Valor",
    color="Metodo",
    symbol="Tipo",
    facet_row="Tipo",
    text="Valor",
    hover_data={
        "Fecha": "|%d/%m/%Y",
        "Metodo": True,
        "Tipo": True,
        "Valor": ":.1f"
    },
    labels={
        "Fecha": "Fecha",
        "Valor": "Nivel de llenado (%)",
        "Metodo": "Método",
        "Tipo": "Tipo de medición"
    },
    title="Nivel de llenado por fecha, método y tipo de medición"
)

fig.update_traces(
    mode="markers+text",
    textposition="top center",
    marker=dict(size=11)
)

fig.update_layout(
    height=700,
    xaxis_title="Fecha",
    yaxis_title="Nivel de llenado (%)",
    legend_title="Método",
    hovermode="closest"
)

fig.update_yaxes(matches=None)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Datos")
tabla = df_filtrado.copy()
tabla["Fecha"] = tabla["Fecha"].dt.strftime("%d/%m/%Y")
tabla = tabla.rename(columns={
    "Metodo": "Método",
    "Tipo": "Tipo de medición",
    "Valor": "Valor (%)"
})

st.dataframe(tabla, use_container_width=True)