# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
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
    classification_report,
    confusion_matrix,
)


st.set_page_config(
    page_title="Selección de variables con RFE",
    layout="wide"
)

st.title("Selección visual de variables con RFE")
st.write(
    "Por defecto se carga California Housing. También puedes subir un archivo Parquet, "
    "elegir target y predictores con checkboxes, ejecutar RFE y ver gráficos de resultados."
)

st.markdown("""
<style>
.variable-chip {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    margin: 0.18rem 0.18rem 0.18rem 0;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
}
.chip-selected {
    background-color: #d1fae5;
    color: #065f46;
    border: 1px solid #10b981;
}
.chip-not-selected {
    background-color: #f3f4f6;
    color: #6b7280;
    border: 1px solid #d1d5db;
}
.legend-box {
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 4px;
    margin-right: 6px;
    vertical-align: middle;
}
.legend-selected { background-color: #10b981; }
.legend-not-selected { background-color: #d1d5db; }
.result-card {
    padding: 1rem;
    border-radius: 0.8rem;
    border: 1px solid #e5e7eb;
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Cargar datos
# -----------------------------
st.sidebar.header("Datos")

uploaded_file = st.sidebar.file_uploader(
    "Opcional: carga tu archivo .parquet",
    type=["parquet"]
)

@st.cache_data
def load_california_housing():
    california = fetch_california_housing(as_frame=True)
    df_california = california.frame.copy()
    return df_california

if uploaded_file is None:
    df = load_california_housing()
    st.info("Usando dataset por defecto: California Housing.")
else:
    try:
        df = pd.read_parquet(uploaded_file)
        st.success("Archivo Parquet cargado correctamente.")
    except Exception as e:
        st.error(f"No se pudo leer el archivo Parquet: {e}")
        st.stop()

st.subheader("Vista previa de los datos")

metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric("Filas", f"{df.shape[0]:,}")
with metric_col2:
    st.metric("Columnas", f"{df.shape[1]:,}")
with metric_col3:
    st.metric("Valores faltantes", f"{int(df.isna().sum().sum()):,}")

with st.expander("Ver primeras filas", expanded=True):
    st.dataframe(df.head(), use_container_width=True)

numeric_preview_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
if numeric_preview_cols:
    st.subheader("Exploración rápida")
    preview_col = st.selectbox(
        "Variable numérica para visualizar distribución",
        options=numeric_preview_cols,
        index=numeric_preview_cols.index("MedHouseVal") if "MedHouseVal" in numeric_preview_cols else 0,
    )
    st.bar_chart(df[preview_col].dropna().value_counts(bins=30).sort_index())


# -----------------------------
# Configuración
# -----------------------------
st.sidebar.header("Configuración del modelo")

columns = df.columns.tolist()

default_target = "MedHouseVal" if "MedHouseVal" in columns else columns[-1]

st.sidebar.markdown("### Variable objetivo")
st.sidebar.caption("Marca exactamente una columna como target.")

target_candidates = []
for col in columns:
    checked = st.sidebar.checkbox(
        label=col,
        value=(col == default_target),
        key=f"target_{col}"
    )
    if checked:
        target_candidates.append(col)

if len(target_candidates) != 1:
    st.warning("Selecciona exactamente una variable objetivo en la barra lateral.")
    st.stop()

target_col = target_candidates[0]

available_features = [col for col in columns if col != target_col]

st.sidebar.markdown("### Variables predictoras")
st.sidebar.caption("Marca las columnas que quieres usar como predictores.")

selected_features = []
for col in available_features:
    checked = st.sidebar.checkbox(
        label=col,
        value=True,
        key=f"feature_{col}"
    )
    if checked:
        selected_features.append(col)

if len(selected_features) == 0:
    st.warning("Selecciona al menos una variable predictora.")
    st.stop()

problem_type = st.sidebar.selectbox(
    "Tipo de problema",
    options=["Regresión", "Clasificación"]
)

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

st.subheader("Selección actual")
sel_col1, sel_col2, sel_col3 = st.columns(3)
with sel_col1:
    st.metric("Target", target_col)
with sel_col2:
    st.metric("Predictores elegidos", len(selected_features))
with sel_col3:
    st.metric("Filas disponibles", f"{df[selected_features + [target_col]].dropna().shape[0]:,}")

with st.expander("Ver predictores seleccionados", expanded=False):
    st.write(selected_features)


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

if problem_type == "Regresión" and numeric_features:
    st.subheader("Relación visual con el target")
    x_axis_feature = st.selectbox(
        "Elige un predictor para graficar contra el target",
        options=numeric_features,
        index=0,
    )
    scatter_df = data[[x_axis_feature, target_col]].sample(
        min(2000, len(data)),
        random_state=int(random_state),
    )
    st.scatter_chart(scatter_df, x=x_axis_feature, y=target_col)


# -----------------------------
# Preprocesamiento
# -----------------------------
try:
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", one_hot_encoder, categorical_features)
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

st.subheader("Resultado de RFE: variables seleccionadas")

results_visual = results.copy()
results_visual["estado"] = np.where(results_visual["seleccionada"], "✅ Seleccionada", "No seleccionada")
results_visual["puntuacion_visual"] = results_visual["ranking"].max() - results_visual["ranking"] + 1
results_visual["color"] = np.where(results_visual["seleccionada"], "#10b981", "#d1d5db")

summary_col1, summary_col2, summary_col3 = st.columns(3)
with summary_col1:
    st.metric("Variables transformadas", len(results_visual))
with summary_col2:
    st.metric("Seleccionadas por RFE", int(results_visual["seleccionada"].sum()))
with summary_col3:
    st.metric("No seleccionadas", int((~results_visual["seleccionada"]).sum()))

st.markdown(
    """
    <span class="legend-box legend-selected"></span><b>Verde:</b> variable seleccionada por RFE &nbsp;&nbsp;
    <span class="legend-box legend-not-selected"></span><b>Gris:</b> variable no seleccionada
    """,
    unsafe_allow_html=True,
)

st.markdown("#### Variables seleccionadas")
if selected_transformed_features:
    chips_html = "".join(
        f'<span class="variable-chip chip-selected">{feature}</span>'
        for feature in selected_transformed_features
    )
    st.markdown(chips_html, unsafe_allow_html=True)
else:
    st.info("No hay variables seleccionadas.")

st.markdown("#### Ranking visual con colores")
plot_df = results_visual.sort_values(["seleccionada", "puntuacion_visual"], ascending=[False, False]).head(30)
plot_df = plot_df.sort_values("puntuacion_visual", ascending=True)

fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot_df))))
ax.barh(
    plot_df["variable_transformada"],
    plot_df["puntuacion_visual"],
    color=plot_df["color"],
)
ax.set_xlabel("Puntuación visual: mayor valor = mejor ranking RFE")
ax.set_ylabel("Variable transformada")
ax.set_title("Variables seleccionadas en verde")
ax.grid(axis="x", alpha=0.25)
st.pyplot(fig)

st.caption("Se muestran hasta 30 variables transformadas, priorizando las seleccionadas y las mejor rankeadas.")

with st.expander("Ver tabla completa con colores", expanded=True):
    table_view = results_visual[["variable_transformada", "estado", "ranking", "puntuacion_visual"]]

    def highlight_selected(row):
        if row["estado"] == "✅ Seleccionada":
            return ["background-color: #d1fae5; color: #065f46; font-weight: bold"] * len(row)
        return ["background-color: #f9fafb; color: #6b7280"] * len(row)

    st.dataframe(
        table_view.style.apply(highlight_selected, axis=1),
        use_container_width=True,
    )

with st.expander("Ver variables no seleccionadas", expanded=False):
    not_selected_features = results_visual.loc[
        ~results_visual["seleccionada"],
        "variable_transformada"
    ].tolist()
    if not_selected_features:
        chips_html = "".join(
            f'<span class="variable-chip chip-not-selected">{feature}</span>'
            for feature in not_selected_features
        )
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.success("Todas las variables transformadas fueron seleccionadas.")


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

    st.subheader("Predicción vs valor real")
    pred_df = pd.DataFrame({
        "real": np.array(y_test),
        "predicho": np.array(y_pred),
    }).sample(min(2000, len(y_test)), random_state=int(random_state))

    st.scatter_chart(pred_df, x="real", y="predicho")

    st.subheader("Errores del modelo")
    residuals_df = pd.DataFrame({"residuo": pred_df["real"] - pred_df["predicho"]})
    st.bar_chart(residuals_df["residuo"].value_counts(bins=30).sort_index())

else:
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", round(accuracy, 4))

    with col2:
        st.metric("F1 weighted", round(f1, 4))

    st.subheader("Matriz de confusión")
    labels = np.unique(np.concatenate([np.array(y_test), np.array(y_pred)]))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(cm)
    ax.set_title("Matriz de confusión")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(image, ax=ax)
    st.pyplot(fig)

    with st.expander("Ver classification report", expanded=False):
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
