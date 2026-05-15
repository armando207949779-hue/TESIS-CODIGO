# app.py

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor
)


# ============================================================
# Librerías opcionales
# ============================================================

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# ============================================================
# Configuración general
# ============================================================

st.set_page_config(
    page_title="Comparador de Modelos de Regresión",
    layout="wide"
)

RANDOM_STATE = 42


# ============================================================
# Carga de datos
# ============================================================

@st.cache_data
def load_default_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={"MedHouseVal": "target"}, inplace=True)
    return df


# ============================================================
# Clasificación de modelos
# ============================================================

def classify_model(name):
    if name == "Decision Tree":
        return "Árbol simple"

    elif name in {
        "Random Forest",
        "Extra Trees",
        "Bagging + Decision Tree"
    }:
        return "Ensamble de árboles"

    elif name in {
        "Gradient Boosting",
        "Hist Gradient Boosting",
        "AdaBoost + Decision Tree",
        "CatBoost",
        "XGBoost",
        "LightGBM"
    }:
        return "Boosting con árboles"

    else:
        return "Otro modelo"


# ============================================================
# Modelos
# ============================================================

def get_models():
    models = {
        # 1
        "Linear Regression": LinearRegression(),

        # 2
        "Ridge Regression": Ridge(
            alpha=1.0,
            random_state=RANDOM_STATE
        ),

        # 3
        "Lasso Regression": Lasso(
            alpha=0.001,
            max_iter=10_000,
            random_state=RANDOM_STATE
        ),

        # 4
        "ElasticNet": ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=10_000,
            random_state=RANDOM_STATE
        ),

        # 5
        "KNN Regressor": KNeighborsRegressor(
            n_neighbors=10,
            weights="distance"
        ),

        # 6
        "SVR": SVR(
            kernel="rbf",
            C=10,
            epsilon=0.1
        ),

        # 7
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE
        ),

        # 8
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=5,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        # 9
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=5,
            max_features="sqrt",
            bootstrap=False,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        # 10
        "Bagging + Decision Tree": BaggingRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=RANDOM_STATE
            ),
            n_estimators=250,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        # 11
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=RANDOM_STATE
        ),

        # 12
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            early_stopping=True,
            random_state=RANDOM_STATE
        ),

        # 13
        "AdaBoost + Decision Tree": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=4,
                min_samples_leaf=8,
                random_state=RANDOM_STATE
            ),
            n_estimators=200,
            learning_rate=0.05,
            random_state=RANDOM_STATE
        )
    }

    # 14
    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False
        )

    # 15
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    # 16
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )

    return models


# ============================================================
# Pipeline
# ============================================================

def build_pipeline(model):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model)
        ]
    )


# ============================================================
# Evaluación
# ============================================================

def evaluate_model(name, model, X_train, X_test, y_train, y_test, cv_splits):
    pipeline = build_pipeline(model)

    cv = KFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    }

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    overfit_gap = train_r2 - test_r2

    return {
        "Modelo": name,
        "Familia": classify_model(name),
        "CV MAE": -cv_results["test_MAE"].mean(),
        "CV RMSE": -cv_results["test_RMSE"].mean(),
        "CV R2": cv_results["test_R2"].mean(),
        "Train R2": train_r2,
        "Test R2": test_r2,
        "Test MAE": test_mae,
        "Test RMSE": test_rmse,
        "Brecha Overfitting": overfit_gap,
        "Pipeline": pipeline
    }


# ============================================================
# Interfaz principal
# ============================================================

st.title("Comparador de Modelos de Machine Learning para Regresión")

st.write(
    """
    Esta aplicación usa por defecto el dataset **California Housing**.
    También permite subir un archivo **Parquet**, seleccionar variables predictoras
    mediante checkboxes y comparar modelos de regresión usando métricas de desempeño.
    """
)


# ============================================================
# Sidebar: datos
# ============================================================

st.sidebar.header("1. Datos")

uploaded_file = st.sidebar.file_uploader(
    "Sube un archivo Parquet",
    type=["parquet"]
)

if uploaded_file is not None:
    try:
        df = pd.read_parquet(uploaded_file)
        st.sidebar.success("Archivo Parquet cargado correctamente.")
    except Exception as e:
        st.sidebar.error(f"No se pudo leer el archivo Parquet: {e}")
        st.stop()
else:
    df = load_default_data()
    st.sidebar.info("Usando California Housing por defecto.")


# ============================================================
# Vista previa
# ============================================================

st.subheader("Vista previa de los datos")

st.dataframe(df.head(), use_container_width=True)

col_a, col_b = st.columns(2)
col_a.metric("Filas", df.shape[0])
col_b.metric("Columnas", df.shape[1])


# ============================================================
# Validación de columnas numéricas
# ============================================================

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_columns) < 2:
    st.error("El dataset debe tener al menos dos columnas numéricas.")
    st.stop()


# ============================================================
# Sidebar: variable objetivo
# ============================================================

st.sidebar.header("2. Variable objetivo")

default_target = "target" if "target" in numeric_columns else numeric_columns[-1]

target_col = st.sidebar.selectbox(
    "Selecciona la variable objetivo",
    numeric_columns,
    index=numeric_columns.index(default_target)
)


# ============================================================
# Sidebar: variables predictoras
# ============================================================

st.sidebar.header("3. Variables predictoras")

available_features = [
    col for col in numeric_columns
    if col != target_col
]

selected_features = []

st.sidebar.write("Selecciona las variables a usar:")

for col in available_features:
    checked = st.sidebar.checkbox(
        col,
        value=True
    )

    if checked:
        selected_features.append(col)


# ============================================================
# Sidebar: configuración
# ============================================================

st.sidebar.header("4. Configuración")

test_size = st.sidebar.slider(
    "Porcentaje para test",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05
)

cv_splits = st.sidebar.slider(
    "Número de folds para validación cruzada",
    min_value=3,
    max_value=10,
    value=5,
    step=1
)

confirm = st.sidebar.button("Confirmar y comparar modelos")


# ============================================================
# Estado inicial
# ============================================================

if not confirm:
    st.info("Configura las opciones en la barra lateral y presiona **Confirmar y comparar modelos**.")

    st.subheader("Modelos configurados")

    default_model_list = [
        ["Linear Regression", "Otro modelo"],
        ["Ridge Regression", "Otro modelo"],
        ["Lasso Regression", "Otro modelo"],
        ["ElasticNet", "Otro modelo"],
        ["KNN Regressor", "Otro modelo"],
        ["SVR", "Otro modelo"],
        ["Decision Tree", "Árbol simple"],
        ["Random Forest", "Ensamble de árboles"],
        ["Extra Trees", "Ensamble de árboles"],
        ["Bagging + Decision Tree", "Ensamble de árboles"],
        ["Gradient Boosting", "Boosting con árboles"],
        ["Hist Gradient Boosting", "Boosting con árboles"],
        ["AdaBoost + Decision Tree", "Boosting con árboles"],
        ["CatBoost", "Boosting con árboles"],
        ["XGBoost", "Boosting con árboles"],
        ["LightGBM", "Boosting con árboles"]
    ]

    st.dataframe(
        pd.DataFrame(default_model_list, columns=["Modelo", "Familia"]),
        use_container_width=True
    )

    st.stop()


# ============================================================
# Validaciones antes de entrenar
# ============================================================

if len(selected_features) == 0:
    st.error("Debes seleccionar al menos una variable predictora.")
    st.stop()

data = df[selected_features + [target_col]].copy()
data = data.dropna(subset=[target_col])

if data.shape[0] < 30:
    st.error("El dataset tiene muy pocas filas después de limpiar la variable objetivo.")
    st.stop()

X = data[selected_features]
y = data[target_col]

if y.nunique() < 2:
    st.error("La variable objetivo debe tener más de un valor único.")
    st.stop()

if cv_splits >= len(data):
    st.error("El número de folds no puede ser mayor o igual al número de filas.")
    st.stop()


# ============================================================
# Split train/test
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=RANDOM_STATE
)


# ============================================================
# Obtener modelos
# ============================================================

models = get_models()

st.subheader("Modelos evaluados")

st.write(f"Se evaluarán **{len(models)} modelos de regresión**.")

model_table = pd.DataFrame(
    {
        "Nº": range(1, len(models) + 1),
        "Modelo": list(models.keys()),
        "Familia": [classify_model(name) for name in models.keys()]
    }
)

st.dataframe(model_table, use_container_width=True)


# ============================================================
# Aviso si faltan librerías opcionales
# ============================================================

missing_optional = []

if not CATBOOST_AVAILABLE:
    missing_optional.append("CatBoost")

if not XGBOOST_AVAILABLE:
    missing_optional.append("XGBoost")

if not LIGHTGBM_AVAILABLE:
    missing_optional.append("LightGBM")

if missing_optional:
    st.warning(
        "No se evaluarán algunos modelos porque faltan librerías instaladas: "
        + ", ".join(missing_optional)
        + ". Para tener los 16 modelos, instala: catboost xgboost lightgbm."
    )


# ============================================================
# Entrenamiento
# ============================================================

st.subheader("Entrenamiento y comparación")

results = []
progress = st.progress(0)

for i, (name, model) in enumerate(models.items(), start=1):
    try:
        result = evaluate_model(
            name=name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            cv_splits=cv_splits
        )

        results.append(result)

    except Exception as e:
        st.warning(f"El modelo **{name}** falló: {e}")

    progress.progress(i / len(models))

if len(results) == 0:
    st.error("Ningún modelo pudo entrenarse correctamente.")
    st.stop()


# ============================================================
# Resultados
# ============================================================

results_df = pd.DataFrame(results)

pipelines = {
    row["Modelo"]: row["Pipeline"]
    for row in results
}

display_df = results_df.drop(columns=["Pipeline"]).copy()


# Score final:
# Menor CV RMSE es mejor.
# Menor brecha de sobreajuste es mejor.
# Mayor CV R2 es mejor.

display_df["Score final"] = (
    display_df["CV RMSE"].rank(ascending=True) +
    display_df["Brecha Overfitting"].abs().rank(ascending=True) +
    display_df["CV R2"].rank(ascending=False)
)

display_df = display_df.sort_values(
    by=["Score final", "CV RMSE", "Brecha Overfitting"],
    ascending=[True, True, True]
)


# ============================================================
# Resumen general
# ============================================================

st.subheader("Resumen por familia de modelos")

family_counts = (
    display_df["Familia"]
    .value_counts()
    .reset_index()
)

family_counts.columns = ["Familia", "Cantidad"]

st.dataframe(family_counts, use_container_width=True)


# ============================================================
# Ranking general
# ============================================================

st.subheader("Ranking general")

format_cols = {
    "CV MAE": "{:.4f}",
    "CV RMSE": "{:.4f}",
    "CV R2": "{:.4f}",
    "Train R2": "{:.4f}",
    "Test R2": "{:.4f}",
    "Test MAE": "{:.4f}",
    "Test RMSE": "{:.4f}",
    "Brecha Overfitting": "{:.4f}",
    "Score final": "{:.2f}"
}

st.dataframe(
    display_df.style.format(format_cols),
    use_container_width=True
)


# ============================================================
# Mejor modelo
# ============================================================

best_row = display_df.iloc[0]
best_model_name = best_row["Modelo"]
best_pipeline = pipelines[best_model_name]

st.success(f"Mejor modelo según score final: **{best_model_name}**")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Mejor modelo", best_model_name)
col2.metric("Familia", best_row["Familia"])
col3.metric("CV RMSE", f"{best_row['CV RMSE']:.4f}")
col4.metric("CV R2", f"{best_row['CV R2']:.4f}")
col5.metric("Brecha", f"{best_row['Brecha Overfitting']:.4f}")


# ============================================================
# Rankings por familia
# ============================================================

st.subheader("Ranking por familia")

families = display_df["Familia"].unique().tolist()

selected_family = st.selectbox(
    "Selecciona una familia de modelos",
    families
)

family_df = display_df[display_df["Familia"] == selected_family]

st.dataframe(
    family_df.style.format(format_cols),
    use_container_width=True
)


# ============================================================
# Comparación visual
# ============================================================

st.subheader("Comparación visual de métricas")

chart_metric = st.selectbox(
    "Métrica para graficar",
    [
        "CV RMSE",
        "CV MAE",
        "CV R2",
        "Test RMSE",
        "Test MAE",
        "Test R2",
        "Brecha Overfitting",
        "Score final"
    ]
)

chart_df = display_df[["Modelo", chart_metric]].copy()
chart_df = chart_df.set_index("Modelo")

st.bar_chart(chart_df)


# ============================================================
# Predicciones del mejor modelo
# ============================================================

st.subheader("Predicciones del mejor modelo")

y_pred = best_pipeline.predict(X_test)

pred_df = pd.DataFrame(
    {
        "Valor real": y_test.values,
        "Predicción": y_pred,
        "Error": y_test.values - y_pred,
        "Error absoluto": np.abs(y_test.values - y_pred)
    }
)

st.dataframe(pred_df.head(50), use_container_width=True)


# ============================================================
# Gráfico real vs predicción
# ============================================================

st.subheader("Valor real vs predicción")

scatter_df = pred_df[["Valor real", "Predicción"]].copy()

st.scatter_chart(scatter_df)


# ============================================================
# Descarga de resultados
# ============================================================

st.subheader("Descargar resultados")

csv = display_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descargar comparación en CSV",
    data=csv,
    file_name="comparacion_modelos_regresion.csv",
    mime="text/csv"
)


# ============================================================
# Explicación de métricas
# ============================================================

st.subheader("Interpretación de métricas")

st.write(
    """
    **CV RMSE**: error promedio en validación cruzada. Menor es mejor.

    **CV MAE**: error absoluto promedio en validación cruzada. Menor es mejor.

    **CV R2**: capacidad explicativa promedio en validación cruzada. Mayor es mejor.

    **Train R2** y **Test R2**: permiten revisar si el modelo aprende bien y si generaliza.

    **Brecha Overfitting**: diferencia entre `Train R2` y `Test R2`.
    Una brecha alta puede indicar sobreajuste.

    **Score final**: ranking combinado usando `CV RMSE`, `CV R2` y la brecha de sobreajuste.
    Menor score final es mejor.
    """
)


# ============================================================
# Variables usadas
# ============================================================

st.subheader("Variables usadas")

st.write("Variable objetivo:")
st.code(target_col)

st.write("Variables predictoras:")
st.write(selected_features)
