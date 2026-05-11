import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math


st.set_page_config(
    page_title="CatBoost Regressor",
    layout="wide"
)

st.title("CatBoost Regressor")
st.write(
    "Sube un archivo CSV o Parquet, elige la variable target, selecciona predictores "
    "y entrena un modelo CatBoost de regresión."
)


# Colormap estilo SHAP: azul bajo, violeta medio, rosado alto
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap_blue_pink",
    ["#008BFB", "#7B2CBF", "#FF0051"]
)


@st.cache_data
def cargar_california_housing():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    return df


def cargar_archivo(uploaded_file):
    nombre = uploaded_file.name.lower()

    if nombre.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if nombre.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)

    raise ValueError("Formato no soportado. Usa CSV o Parquet.")


def calcular_metricas(y_real, y_predicho, nombre_set):
    mae = mean_absolute_error(y_real, y_predicho)
    mse = mean_squared_error(y_real, y_predicho)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_predicho)

    return {
        "dataset": nombre_set,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse
    }


def convertir_categoricas_a_codigos(X):
    X_plot = X.copy()
    mapas_categorias = {}

    for col in X_plot.columns:
        if not pd.api.types.is_numeric_dtype(X_plot[col]):
            cat = pd.Categorical(X_plot[col])
            X_plot[col] = cat.codes
            mapas_categorias[col] = list(cat.categories)

    return X_plot, mapas_categorias


def seleccionar_variable_interaccion(variable_principal, variables, shap_df):
    candidatas = [v for v in variables if v != variable_principal]

    if len(candidatas) == 0:
        return variable_principal

    importancia_shap = (
        shap_df[candidatas]
        .abs()
        .mean()
        .sort_values(ascending=False)
    )

    return importancia_shap.index[0]


def jitter_beeswarm(x, ancho=0.32, bins=40, seed=42):
    """
    Genera jitter vertical tipo beeswarm simple.
    Agrupa los SHAP values en bins y reparte los puntos alrededor del centro.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)

    if len(x) == 0:
        return np.array([])

    if np.nanmax(x) == np.nanmin(x):
        return rng.normal(0, ancho / 5, size=len(x))

    bins_edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    bin_id = np.digitize(x, bins_edges) - 1

    jitter = np.zeros(len(x))

    for b in np.unique(bin_id):
        idx = np.where(bin_id == b)[0]
        n = len(idx)

        if n <= 1:
            jitter[idx] = 0
        else:
            posiciones = np.linspace(-ancho, ancho, n)
            rng.shuffle(posiciones)
            jitter[idx] = posiciones

    jitter += rng.normal(0, ancho * 0.04, size=len(x))

    return jitter


def shap_summary_plot_estilo(
    shap_df,
    X_plot,
    shap_importancia,
    max_display,
    titulo="SHAP summary plot - Test"
):
    variables = shap_importancia["variable"].head(max_display).tolist()
    variables_reverso = list(reversed(variables))

    fig, ax = plt.subplots(
        figsize=(11, max(5, len(variables) * 0.55))
    )

    ultimo_scatter = None

    for i, variable in enumerate(variables_reverso):
        valores_shap = shap_df[variable].values
        valores_feature = X_plot[variable].values

        y_base = np.full(len(valores_shap), i)
        y_jitter = y_base + jitter_beeswarm(
            valores_shap,
            ancho=0.28,
            bins=35,
            seed=100 + i
        )

        ultimo_scatter = ax.scatter(
            valores_shap,
            y_jitter,
            c=valores_feature,
            cmap=SHAP_CMAP,
            s=16,
            alpha=0.85,
            linewidths=0
        )

    ax.axvline(0, color="gray", linewidth=1.5)

    ax.set_yticks(range(len(variables_reverso)))
    ax.set_yticklabels(variables_reverso, fontsize=12)

    ax.set_xlabel("SHAP value (impact on model output)", fontsize=13)
    ax.set_ylabel("")
    ax.set_title(titulo, fontsize=14)

    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    cbar = fig.colorbar(ultimo_scatter, ax=ax, pad=0.02)
    cbar.set_label("Feature value", rotation=270, labelpad=18, fontsize=12)
    cbar.set_ticks([])
    cbar.ax.text(
        1.8,
        1.00,
        "High",
        transform=cbar.ax.transAxes,
        ha="left",
        va="center",
        fontsize=11
    )
    cbar.ax.text(
        1.8,
        0.00,
        "Low",
        transform=cbar.ax.transAxes,
        ha="left",
        va="center",
        fontsize=11
    )

    fig.tight_layout()
    return fig


def dependence_plot_estilo_shap(
    ax,
    X_original,
    X_plot,
    shap_df,
    variable_principal,
    variable_color,
    mapas_categorias
):
    x = X_plot[variable_principal].values
    y = shap_df[variable_principal].values
    c = X_plot[variable_color].values

    rng = np.random.default_rng(123)

    if pd.api.types.is_numeric_dtype(X_original[variable_principal]):
        x_plot = x
    else:
        x_plot = x + rng.normal(0, 0.05, size=len(x))

    scatter = ax.scatter(
        x_plot,
        y,
        c=c,
        cmap=SHAP_CMAP,
        alpha=0.85,
        s=18,
        linewidths=0
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(variable_principal, fontsize=12)
    ax.set_ylabel(f"SHAP value for\n{variable_principal}", fontsize=12)
    ax.set_title(
        f"{variable_principal} coloreado por {variable_color}",
        fontsize=13
    )

    ax.grid(alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if variable_principal in mapas_categorias:
        categorias_x = mapas_categorias[variable_principal]
        ax.set_xticks(range(len(categorias_x)))
        ax.set_xticklabels(categorias_x, rotation=45, ha="right")

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)

    if variable_color in mapas_categorias:
        categorias_color = mapas_categorias[variable_color]
        ticks = list(range(len(categorias_color)))
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(categorias_color)
        cbar.set_label(variable_color, rotation=270, labelpad=18)
    else:
        cbar.set_label(variable_color, rotation=270, labelpad=18)
        cbar.ax.text(
            1.8,
            1.00,
            "High",
            transform=cbar.ax.transAxes,
            ha="left",
            va="center",
            fontsize=10
        )
        cbar.ax.text(
            1.8,
            0.00,
            "Low",
            transform=cbar.ax.transAxes,
            ha="left",
            va="center",
            fontsize=10
        )

    return ax


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


if df.shape[1] < 2:
    st.error("El dataset debe tener al menos dos columnas: una target y una predictora.")
    st.stop()


columnas = df.columns.tolist()

target_default = "MedHouseVal" if "MedHouseVal" in columnas else columnas[-1]

target = st.sidebar.selectbox(
    "Elige la variable target",
    columnas,
    index=columnas.index(target_default)
)

predictoras_disponibles = [col for col in columnas if col != target]

predictoras = st.sidebar.multiselect(
    "Elige las variables predictoras",
    predictoras_disponibles,
    default=predictoras_disponibles
)

if len(predictoras) == 0:
    st.warning("Selecciona al menos una variable predictora.")
    st.stop()


st.sidebar.subheader("División de datos")

test_size = st.sidebar.slider(
    "Porcentaje para test",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05
)

valid_size = st.sidebar.slider(
    "Porcentaje para validación",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05
)

if test_size + valid_size >= 0.90:
    st.error("La suma de test y validación debe dejar suficiente proporción para entrenamiento.")
    st.stop()


st.sidebar.subheader("Parámetros del modelo")

random_state = st.sidebar.number_input(
    "Random state",
    min_value=0,
    value=42,
    step=1
)

iterations = st.sidebar.slider(
    "Iteraciones máximas",
    min_value=100,
    max_value=5000,
    value=2000,
    step=100
)

learning_rate = st.sidebar.slider(
    "Learning rate",
    min_value=0.001,
    max_value=0.300,
    value=0.050,
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

early_stopping_rounds = st.sidebar.slider(
    "Early stopping rounds",
    min_value=10,
    max_value=300,
    value=50,
    step=10
)


st.sidebar.subheader("Opciones de gráficos")

top_n_shap = st.sidebar.slider(
    "Variables para SHAP dependence plots",
    min_value=1,
    max_value=len(predictoras),
    value=len(predictoras),
    step=1
)

max_display_summary = st.sidebar.slider(
    "Variables para SHAP summary plot",
    min_value=1,
    max_value=len(predictoras),
    value=len(predictoras),
    step=1
)


df_model = df[[target] + predictoras].copy()

for col in df_model.columns:
    if df_model[col].dtype == "object":
        df_model[col] = df_model[col].astype("category")

filas_antes = df_model.shape[0]
df_model = df_model.dropna()
filas_despues = df_model.shape[0]

if filas_despues < filas_antes:
    st.info(f"Se eliminaron {filas_antes - filas_despues} filas con valores nulos.")

if df_model.shape[0] < 10:
    st.error("Después de eliminar nulos quedan muy pocas filas para entrenar el modelo.")
    st.stop()


X = df_model[predictoras]
y = df_model[target]

if not pd.api.types.is_numeric_dtype(y):
    st.error("La variable target debe ser numérica para usar CatBoostRegressor.")
    st.stop()


cat_features = [
    col for col in X.columns
    if str(X[col].dtype) == "category"
]

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=int(random_state)
)

valid_relative_size = valid_size / (1 - test_size)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp,
    y_temp,
    test_size=valid_relative_size,
    random_state=int(random_state)
)


max_test_index = X_test.shape[0]

if max_test_index <= 20:
    n_index_plot = max_test_index
else:
    n_index_plot = st.sidebar.slider(
        "Observaciones para Real vs Predicho vs Index",
        min_value=20,
        max_value=max_test_index,
        value=min(200, max_test_index),
        step=10
    )


st.subheader("Configuración seleccionada")

col_a, col_b, col_c, col_d = st.columns(4)

col_a.metric("Target", target)
col_b.metric("Predictores", len(predictoras))
col_c.metric("Filas usadas", df_model.shape[0])
col_d.metric("Categóricas", len(cat_features))

col_e, col_f, col_g = st.columns(3)

col_e.metric("Train", X_train.shape[0])
col_f.metric("Validación", X_valid.shape[0])
col_g.metric("Test", X_test.shape[0])

with st.expander("Ver variables seleccionadas"):
    st.write("**Target:**", target)
    st.write("**Predictoras:**", predictoras)
    st.write("**Variables categóricas detectadas:**", cat_features if cat_features else "Ninguna")


if st.sidebar.button("Entrenar CatBoost"):
    with st.spinner("Entrenando modelo CatBoost con validación y early stopping..."):
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=int(random_state),
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=True,
            verbose=False
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cat_features if len(cat_features) > 0 else None
        )

        y_pred_train = model.predict(X_train)
        y_pred_valid = model.predict(X_valid)
        y_pred_test = model.predict(X_test)

    metricas_train = calcular_metricas(y_train, y_pred_train, "Train")
    metricas_valid = calcular_metricas(y_valid, y_pred_valid, "Validación")
    metricas_test = calcular_metricas(y_test, y_pred_test, "Test")

    tabla_metricas = pd.DataFrame([
        metricas_train,
        metricas_valid,
        metricas_test
    ])

    tabla_metricas = tabla_metricas[["dataset", "R2", "RMSE", "MAE", "MSE"]]

    r2 = metricas_test["R2"]
    rmse = metricas_test["RMSE"]
    mae = metricas_test["MAE"]
    mse = metricas_test["MSE"]

    st.success("Modelo entrenado correctamente.")

    st.subheader("Métricas principales en test")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("R²", f"{r2:,.4f}")
    col2.metric("RMSE", f"{rmse:,.4f}")
    col3.metric("MAE", f"{mae:,.4f}")
    col4.metric("MSE", f"{mse:,.4f}")

    st.subheader("Tabla de métricas")

    st.dataframe(
        tabla_metricas.style.format({
            "R2": "{:.4f}",
            "RMSE": "{:.4f}",
            "MAE": "{:.4f}",
            "MSE": "{:.4f}"
        }),
        use_container_width=True
    )

    mejor_iteracion = model.get_best_iteration()

    st.write(f"**Mejor iteración:** {mejor_iteracion}")

    if mejor_iteracion == iterations - 1:
        st.info(
            "La mejor iteración coincide con la última iteración disponible. "
            "Eso significa que el early stopping no se activó antes del límite máximo. "
            "Puedes aumentar las iteraciones o ajustar learning rate, depth o early stopping rounds."
        )

    st.subheader("Gráfico: Real vs Predicho")

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_test, y_pred_test, alpha=0.6)

    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--"
    )

    texto_metricas = (
        f"R² = {r2:.4f}\n"
        f"RMSE = {rmse:.4f}\n"
        f"MAE = {mae:.4f}\n"
        f"MSE = {mse:.4f}"
    )

    ax.text(
        0.05,
        0.95,
        texto_metricas,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    ax.set_xlabel("Valor real")
    ax.set_ylabel("Valor predicho")
    ax.set_title("Real vs Predicho - Test")

    st.pyplot(fig)

    st.subheader("Gráfico: Real vs Predicho vs Index")

    resultados = pd.DataFrame({
        "real": y_test.values,
        "predicho": y_pred_test,
        "residuo": y_test.values - y_pred_test,
        "residuo_absoluto": np.abs(y_test.values - y_pred_test)
    }).reset_index(drop=True)

    resultados["index"] = resultados.index

    resultados_index_plot = resultados.head(n_index_plot)

    fig_index, ax_index = plt.subplots(figsize=(12, 5))

    ax_index.plot(
        resultados_index_plot["index"],
        resultados_index_plot["real"],
        label="Real"
    )

    ax_index.plot(
        resultados_index_plot["index"],
        resultados_index_plot["predicho"],
        label="Predicho"
    )

    ax_index.set_xlabel("Index")
    ax_index.set_ylabel(target)
    ax_index.set_title(
        f"Real vs Predicho vs Index - Test | Mostrando {n_index_plot} de {len(resultados)} observaciones"
    )
    ax_index.legend()

    st.pyplot(fig_index)

    st.subheader("Residuos")

    col_res1, col_res2 = st.columns(2)

    with col_res1:
        fig_res_scatter, ax_res_scatter = plt.subplots(figsize=(7, 5))

        ax_res_scatter.scatter(y_pred_test, resultados["residuo"], alpha=0.6)
        ax_res_scatter.axhline(0, linestyle="--")

        ax_res_scatter.set_xlabel("Valor predicho")
        ax_res_scatter.set_ylabel("Residuo")
        ax_res_scatter.set_title("Residuos vs Predicho - Test")

        st.pyplot(fig_res_scatter)

    with col_res2:
        fig_res_hist, ax_res_hist = plt.subplots(figsize=(7, 5))

        ax_res_hist.hist(resultados["residuo"], bins=30)

        ax_res_hist.set_xlabel("Residuo")
        ax_res_hist.set_ylabel("Frecuencia")
        ax_res_hist.set_title("Distribución de residuos - Test")

        st.pyplot(fig_res_hist)

    st.subheader("Curva de aprendizaje")

    evals_result = model.get_evals_result()

    train_rmse = evals_result["learn"]["RMSE"]
    valid_rmse = evals_result["validation"]["RMSE"]

    curva = pd.DataFrame({
        "iteracion": np.arange(1, len(train_rmse) + 1),
        "RMSE train": train_rmse,
        "RMSE validación": valid_rmse
    })

    fig_learning, ax_learning = plt.subplots(figsize=(10, 5))

    ax_learning.plot(curva["iteracion"], curva["RMSE train"], label="RMSE train")
    ax_learning.plot(curva["iteracion"], curva["RMSE validación"], label="RMSE validación")

    ax_learning.set_xlabel("Iteración")
    ax_learning.set_ylabel("RMSE")
    ax_learning.set_title("Curva de aprendizaje")
    ax_learning.legend()

    st.pyplot(fig_learning)

    with st.expander("Ver datos de la curva de aprendizaje"):
        st.dataframe(curva, use_container_width=True)

    st.subheader("Importancia de variables")

    importancias = pd.DataFrame({
        "variable": X.columns,
        "importancia": model.get_feature_importance()
    }).sort_values("importancia", ascending=False)

    st.dataframe(importancias, use_container_width=True)

    importancias_plot = importancias.sort_values("importancia", ascending=True)

    fig2, ax2 = plt.subplots(figsize=(10, max(5, len(importancias_plot) * 0.35)))

    ax2.barh(importancias_plot["variable"], importancias_plot["importancia"])
    ax2.set_xlabel("Importancia")
    ax2.set_ylabel("Variable")
    ax2.set_title("Importancia de variables")

    st.pyplot(fig2)

    st.subheader("SHAP values")

    with st.spinner("Calculando SHAP values en test..."):
        test_pool = Pool(
            X_test,
            y_test,
            cat_features=cat_features if len(cat_features) > 0 else None
        )

        shap_values_full = model.get_feature_importance(
            test_pool,
            type="ShapValues"
        )

    shap_values = shap_values_full[:, :-1]
    shap_base_value = shap_values_full[:, -1]

    shap_df = pd.DataFrame(
        shap_values,
        columns=predictoras
    )

    X_test_plot, mapas_categorias = convertir_categoricas_a_codigos(X_test)

    shap_importancia = pd.DataFrame({
        "variable": predictoras,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    st.write("**Importancia global basada en SHAP:**")

    st.dataframe(shap_importancia, use_container_width=True)

    shap_plot = shap_importancia.sort_values("mean_abs_shap", ascending=True)

    fig_shap_bar, ax_shap_bar = plt.subplots(
        figsize=(10, max(5, len(shap_plot) * 0.35))
    )

    ax_shap_bar.barh(shap_plot["variable"], shap_plot["mean_abs_shap"])
    ax_shap_bar.set_xlabel("Mean |SHAP value|")
    ax_shap_bar.set_ylabel("Variable")
    ax_shap_bar.set_title("Importancia global SHAP")

    st.pyplot(fig_shap_bar)

    st.subheader("SHAP summary plot")

    fig_shap_summary = shap_summary_plot_estilo(
        shap_df=shap_df,
        X_plot=X_test_plot,
        shap_importancia=shap_importancia,
        max_display=max_display_summary,
        titulo="SHAP summary plot - Test"
    )

    st.pyplot(fig_shap_summary)

    with st.expander("Ver SHAP values por observación"):
        shap_mostrar = shap_df.copy()
        shap_mostrar["base_value"] = shap_base_value
        shap_mostrar["predicho"] = y_pred_test
        shap_mostrar["real"] = y_test.values
        st.dataframe(shap_mostrar.head(200), use_container_width=True)

    st.subheader("SHAP dependence plot individual")

    variable_dependence = st.selectbox(
        "Variable principal para SHAP dependence plot",
        options=shap_importancia["variable"].tolist(),
        index=0
    )

    variable_color_default = seleccionar_variable_interaccion(
        variable_dependence,
        predictoras,
        shap_df
    )

    variable_color = st.selectbox(
        "Variable usada como color/interacción",
        options=predictoras,
        index=predictoras.index(variable_color_default)
    )

    fig_dep_ind, ax_dep_ind = plt.subplots(figsize=(9, 6))

    dependence_plot_estilo_shap(
        ax=ax_dep_ind,
        X_original=X_test,
        X_plot=X_test_plot,
        shap_df=shap_df,
        variable_principal=variable_dependence,
        variable_color=variable_color,
        mapas_categorias=mapas_categorias
    )

    fig_dep_ind.tight_layout()

    st.pyplot(fig_dep_ind)

    st.subheader("SHAP dependence plots en matriz")

    variables_top_shap = shap_importancia["variable"].head(top_n_shap).tolist()

    n_cols = 2
    n_rows = math.ceil(len(variables_top_shap) / n_cols)

    fig_dep, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(16, max(6, n_rows * 5))
    )

    if n_rows == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    for i, variable in enumerate(variables_top_shap):
        ax_dep = axes[i]

        variable_interaccion = seleccionar_variable_interaccion(
            variable,
            predictoras,
            shap_df
        )

        dependence_plot_estilo_shap(
            ax=ax_dep,
            X_original=X_test,
            X_plot=X_test_plot,
            shap_df=shap_df,
            variable_principal=variable,
            variable_color=variable_interaccion,
            mapas_categorias=mapas_categorias
        )

    for j in range(len(variables_top_shap), len(axes)):
        axes[j].axis("off")

    fig_dep.tight_layout()

    st.pyplot(fig_dep)

    st.subheader("Predicciones en test")

    st.dataframe(resultados.head(100), use_container_width=True)

    csv_resultados = resultados.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar predicciones en CSV",
        data=csv_resultados,
        file_name="predicciones_catboost.csv",
        mime="text/csv"
    )

    csv_metricas = tabla_metricas.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar métricas en CSV",
        data=csv_metricas,
        file_name="metricas_catboost.csv",
        mime="text/csv"
    )

    csv_shap = shap_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar SHAP values en CSV",
        data=csv_shap,
        file_name="shap_values_catboost.csv",
        mime="text/csv"
    )

else:
    st.info("Configura las variables en el panel lateral y presiona **Entrenar CatBoost**.")
