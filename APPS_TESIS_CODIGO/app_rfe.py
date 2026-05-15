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


# =========================================================
# Configuración general
# =========================================================
st.set_page_config(
    page_title="Selección de variables con RFE",
    layout="wide",
)

SELECTED_COLOR = "#10b981"      # verde
NOT_SELECTED_COLOR = "#d1d5db"  # gris
TEXT_DARK = "#111827"
TEXT_MUTED = "#6b7280"


st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        color: #4b5563;
        font-size: 1rem;
        margin-bottom: 1.25rem;
    }
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
    .section-note {
        color: #6b7280;
        font-size: 0.92rem;
        margin-top: -0.4rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Selección visual de variables con RFE</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Por defecto se usa California Housing. También puedes cargar un archivo Parquet, elegir target y predictores con checkboxes, ejecutar RFE y revisar los resultados con gráficos claros.</div>',
    unsafe_allow_html=True,
)


# =========================================================
# Funciones auxiliares
# =========================================================
@st.cache_data
def load_california_housing() -> pd.DataFrame:
    """Carga California Housing como dataset por defecto."""
    california = fetch_california_housing(as_frame=True)
    return california.frame.copy()


def clean_feature_name(feature_name: str) -> str:
    """Elimina prefijos técnicos generados por ColumnTransformer."""
    feature_name = str(feature_name)
    for prefix in ("num__", "cat__", "remainder__"):
        if feature_name.startswith(prefix):
            return feature_name.replace(prefix, "", 1)
    return feature_name


def make_one_hot_encoder() -> OneHotEncoder:
    """Crea OneHotEncoder compatible con distintas versiones de scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def render_chips(items, selected: bool = True) -> None:
    """Muestra una lista como chips visuales."""
    chip_class = "chip-selected" if selected else "chip-not-selected"
    if not items:
        st.info("No hay variables para mostrar.")
        return

    chips_html = "".join(
        f'<span class="variable-chip {chip_class}">{item}</span>'
        for item in items
    )
    st.markdown(chips_html, unsafe_allow_html=True)


def draw_rfe_ranking_chart(results_visual: pd.DataFrame) -> None:
    """Grafica ranking RFE en barras horizontales sin emojis."""
    plot_df = results_visual.sort_values(
        ["seleccionada", "ranking", "variable"],
        ascending=[False, True, True],
    ).head(30).copy()

    plot_df["etiqueta"] = np.where(
        plot_df["seleccionada"],
        "Seleccionada - " + plot_df["variable"].astype(str),
        "No seleccionada - " + plot_df["variable"].astype(str),
    )

    fig_height = max(5, 0.45 * len(plot_df))
    fig, ax = plt.subplots(figsize=(13, fig_height))

    bars = ax.barh(
        plot_df["etiqueta"],
        plot_df["puntuacion_visual"],
        color=plot_df["color"],
    )

    ax.invert_yaxis()
    ax.set_xlabel("Puntuación visual: mayor valor = mejor posición en RFE")
    ax.set_ylabel("Variables")
    ax.set_title("Ranking RFE horizontal")
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="y", labelsize=9)

    max_score = plot_df["puntuacion_visual"].max()
    for bar, ranking in zip(bars, plot_df["ranking"]):
        width = bar.get_width()
        ax.text(
            width + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"ranking {int(ranking)}",
            va="center",
            fontsize=9,
            color=TEXT_DARK,
        )

    ax.set_xlim(0, max_score + 2)
    fig.tight_layout()
    st.pyplot(fig)


def highlight_selected(row):
    """Aplica colores a la tabla de resultados."""
    if row["estado"] == "Seleccionada":
        return ["background-color: #d1fae5; color: #065f46; font-weight: bold"] * len(row)
    return ["background-color: #f9fafb; color: #6b7280"] * len(row)


# =========================================================
# Carga de datos
# =========================================================
st.sidebar.header("Datos")

uploaded_file = st.sidebar.file_uploader(
    "Opcional: carga tu archivo .parquet",
    type=["parquet"],
)

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

if df.empty:
    st.error("El dataset está vacío.")
    st.stop()

st.subheader("Resumen del dataset")
summary_col1, summary_col2, summary_col3 = st.columns(3)
with summary_col1:
    st.metric("Filas", f"{df.shape[0]:,}")
with summary_col2:
    st.metric("Columnas", f"{df.shape[1]:,}")
with summary_col3:
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


# =========================================================
# Configuración del modelo
# =========================================================
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
        key=f"target_{col}",
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
        key=f"feature_{col}",
    )
    if checked:
        selected_features.append(col)

if not selected_features:
    st.warning("Selecciona al menos una variable predictora.")
    st.stop()

problem_type = st.sidebar.selectbox(
    "Tipo de problema",
    options=["Regresión", "Clasificación"],
)

n_features_to_select = st.sidebar.slider(
    "Número de variables a seleccionar",
    min_value=1,
    max_value=len(selected_features),
    value=min(5, len(selected_features)),
)

model_choice = st.sidebar.selectbox(
    "Modelo base para RFE",
    options=["Regresión lineal / logística", "Random Forest"],
)

test_size = st.sidebar.slider(
    "Porcentaje para prueba",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05,
)

random_state = st.sidebar.number_input(
    "Random state",
    min_value=0,
    value=42,
    step=1,
)


# =========================================================
# Preparación de datos
# =========================================================
data = df[selected_features + [target_col]].copy().dropna()

if data.empty:
    st.error("Después de eliminar valores faltantes, no quedan filas disponibles.")
    st.stop()

X = data[selected_features]
y = data[target_col]

numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

st.subheader("Selección actual")
selection_col1, selection_col2, selection_col3, selection_col4 = st.columns(4)
with selection_col1:
    st.metric("Target", target_col)
with selection_col2:
    st.metric("Predictores", len(selected_features))
with selection_col3:
    st.metric("Numéricas", len(numeric_features))
with selection_col4:
    st.metric("Categóricas", len(categorical_features))

with st.expander("Ver predictores seleccionados", expanded=False):
    render_chips(selected_features, selected=True)

st.subheader("Tipos de variables detectadas")
type_col1, type_col2 = st.columns(2)
with type_col1:
    st.write("**Variables numéricas**")
    st.write(numeric_features if numeric_features else "Ninguna")
with type_col2:
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


# =========================================================
# Preprocesamiento y modelo
# =========================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", make_one_hot_encoder(), categorical_features),
    ],
    remainder="drop",
)

if problem_type == "Regresión":
    estimator = (
        LinearRegression()
        if model_choice == "Regresión lineal / logística"
        else RandomForestRegressor(n_estimators=200, random_state=int(random_state), n_jobs=-1)
    )
else:
    estimator = (
        LogisticRegression(max_iter=1000, solver="liblinear")
        if model_choice == "Regresión lineal / logística"
        else RandomForestClassifier(n_estimators=200, random_state=int(random_state), n_jobs=-1)
    )

run_button = st.button("Ejecutar RFE", type="primary")
if not run_button:
    st.stop()


# =========================================================
# Entrenamiento y RFE
# =========================================================
try:
    stratify = y if problem_type == "Clasificación" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=int(random_state),
        stratify=stratify,
    )
except Exception as e:
    st.error(f"Error al dividir los datos: {e}")
    st.stop()

try:
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names_raw = preprocessor.get_feature_names_out()
    feature_names = [clean_feature_name(name) for name in feature_names_raw]
except Exception as e:
    st.error(f"Error en el preprocesamiento: {e}")
    st.stop()

if n_features_to_select > len(feature_names):
    st.warning(
        "Después del preprocesamiento hay menos variables transformadas que las solicitadas. "
        "Se ajustará automáticamente el número de variables a seleccionar."
    )
    n_features_to_select = len(feature_names)

try:
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
    rfe.fit(X_train_transformed, y_train)
except Exception as e:
    st.error(f"Error ejecutando RFE: {e}")
    st.stop()


# =========================================================
# Resultados de selección
# =========================================================
results = pd.DataFrame(
    {
        "variable": feature_names,
        "seleccionada": rfe.support_,
        "ranking": rfe.ranking_,
    }
).sort_values(["ranking", "variable"])

results_visual = results.copy()
results_visual["estado"] = np.where(results_visual["seleccionada"], "Seleccionada", "No seleccionada")
results_visual["puntuacion_visual"] = results_visual["ranking"].max() - results_visual["ranking"] + 1
results_visual["color"] = np.where(results_visual["seleccionada"], SELECTED_COLOR, NOT_SELECTED_COLOR)

selected_features_rfe = results_visual.loc[results_visual["seleccionada"], "variable"].tolist()
not_selected_features_rfe = results_visual.loc[~results_visual["seleccionada"], "variable"].tolist()

st.subheader("Resultado de RFE")
st.markdown(
    '<div class="section-note">Las variables seleccionadas aparecen en verde. Las no seleccionadas aparecen en gris. El gráfico no usa emojis para mantener una lectura más limpia.</div>',
    unsafe_allow_html=True,
)

rfe_col1, rfe_col2, rfe_col3 = st.columns(3)
with rfe_col1:
    st.metric("Variables transformadas", len(results_visual))
with rfe_col2:
    st.metric("Seleccionadas", int(results_visual["seleccionada"].sum()))
with rfe_col3:
    st.metric("No seleccionadas", int((~results_visual["seleccionada"]).sum()))

st.markdown(
    """
    <span class="legend-box legend-selected"></span><b>Verde:</b> seleccionada por RFE &nbsp;&nbsp;
    <span class="legend-box legend-not-selected"></span><b>Gris:</b> no seleccionada
    """,
    unsafe_allow_html=True,
)

st.markdown("#### Variables seleccionadas")
render_chips(selected_features_rfe, selected=True)

st.markdown("#### Ranking horizontal")
draw_rfe_ranking_chart(results_visual)
st.caption(
    "El gráfico muestra primero las variables seleccionadas y luego las no seleccionadas mejor posicionadas. "
    "Los nombres fueron limpiados para ocultar prefijos técnicos como num__ y cat__."
)

with st.expander("Ver tabla completa con colores", expanded=True):
    table_view = results_visual[["variable", "estado", "ranking", "puntuacion_visual"]]
    st.dataframe(table_view.style.apply(highlight_selected, axis=1), use_container_width=True)

with st.expander("Ver variables no seleccionadas", expanded=False):
    render_chips(not_selected_features_rfe, selected=False)


# =========================================================
# Modelo final y evaluación
# =========================================================
try:
    X_train_selected = rfe.transform(X_train_transformed)
    X_test_selected = rfe.transform(X_test_transformed)

    final_model = estimator
    final_model.fit(X_train_selected, y_train)
    y_pred = final_model.predict(X_test_selected)
except Exception as e:
    st.error(f"Error entrenando el modelo final: {e}")
    st.stop()

st.subheader("Evaluación del modelo con variables seleccionadas")

if problem_type == "Regresión":
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        st.metric("RMSE", round(rmse, 4))
    with eval_col2:
        st.metric("R²", round(r2, 4))

    st.subheader("Predicción vs valor real")
    pred_df = pd.DataFrame(
        {
            "real": np.array(y_test),
            "predicho": np.array(y_pred),
        }
    ).sample(min(2000, len(y_test)), random_state=int(random_state))
    st.scatter_chart(pred_df, x="real", y="predicho")

    st.subheader("Distribución de errores")
    residuals_df = pd.DataFrame({"residuo": pred_df["real"] - pred_df["predicho"]})
    st.bar_chart(residuals_df["residuo"].value_counts(bins=30).sort_index())
else:
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        st.metric("Accuracy", round(accuracy, 4))
    with eval_col2:
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
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("Ver classification report", expanded=False):
        st.text(classification_report(y_test, y_pred))


# =========================================================
# Descarga de resultados
# =========================================================
csv = results.drop(columns=[]).to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descargar ranking de variables",
    data=csv,
    file_name="rfe_ranking_variables.csv",
    mime="text/csv",
)
