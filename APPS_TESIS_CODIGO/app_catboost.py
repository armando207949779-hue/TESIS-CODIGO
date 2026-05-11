"""
CatBoost Regressor - App Streamlit

App para entrenar un modelo CatBoost de regresión, explorar resultados con SHAP
y exportar todo lo necesario para usarlo posteriormente como función objetivo en
algoritmos de optimización (por ejemplo scipy.optimize.differential_evolution).
"""

from __future__ import annotations

import io
import json
import math
import platform
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor, Pool, __version__ as catboost_version
from matplotlib.colors import LinearSegmentedColormap
from sklearn import __version__ as sklearn_version
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# ======================================================================
# 1. CONFIGURACIÓN GLOBAL Y ESTÉTICA
# ======================================================================

st.set_page_config(
    page_title="CatBoost Regressor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta de colores coherente para toda la app
COLORS = {
    "primary":   "#2E86AB",   # azul
    "secondary": "#E63946",   # rojo
    "tertiary":  "#06A77D",   # verde
    "highlight": "#F77F00",   # naranja
    "neutral":   "#6C757D",   # gris
    "text":      "#1F2937",
    "grid":      "#E5E7EB",
}

# Colormap estilo SHAP (azul -> morado -> rosado/rojo)
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap_blue_pink",
    ["#008BFB", "#7B2CBF", "#FF0051"],
)

# Estilo global de matplotlib
plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     COLORS["text"],
    "axes.labelcolor":    COLORS["text"],
    "axes.titlesize":     14,
    "axes.titleweight":   "semibold",
    "axes.titlepad":      14,
    "axes.labelsize":     12,
    "axes.labelpad":      8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     1.0,
    "xtick.color":        COLORS["text"],
    "ytick.color":        COLORS["text"],
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "grid.color":         COLORS["grid"],
    "grid.linestyle":     "-",
    "grid.linewidth":     0.7,
    "grid.alpha":         0.7,
    "legend.frameon":     False,
    "legend.fontsize":    10,
    "font.family":        "DejaVu Sans",
})


# ======================================================================
# 2. ARCHIVOS EMBEBIDOS PARA EXPORTACIÓN
# ======================================================================

PREDICT_PY = r'''"""
Wrapper para cargar el modelo CatBoost exportado y usarlo en optimización.

Uso típico con differential evolution:

    from scipy.optimize import differential_evolution
    from predict import load_predictor

    predictor = load_predictor(".")
    bounds = predictor.bounds()

    result = differential_evolution(
        lambda x: -predictor.predict_single(x),   # maximizar => minimizar negativo
        bounds=bounds,
        seed=42,
    )
    print(predictor.decode(result.x), -result.fun)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


class ModelPredictor:
    """Encapsula modelo + schema para usarlo en optimización."""

    def __init__(self, model_dir: str | Path = "."):
        self.model_dir = Path(model_dir)

        with open(self.model_dir / "feature_schema.json", "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        with open(self.model_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.model = CatBoostRegressor()
        self.model.load_model(str(self.model_dir / "model.cbm"))

        self.feature_names = self.schema["feature_names"]
        self.cat_features = self.schema["cat_features"]
        self.target = self.metadata["target"]

    # ---- Bounds y defaults para el optimizador ----

    def bounds(self) -> list[tuple[float, float]]:
        """Retorna [(min, max), ...] en el mismo orden que feature_names.

        Para variables categóricas se retorna (0, n_categorias - 1):
        el optimizador trabajará con un índice y `predict_single` lo
        redondea y mapea a la categoría real internamente.
        """
        b = []
        for name in self.feature_names:
            info = self.schema["features"][name]
            if info["type"] == "categorical":
                b.append((0.0, float(len(info["categories"]) - 1)))
            else:
                b.append((float(info["min"]), float(info["max"])))
        return b

    def defaults(self) -> np.ndarray:
        """Valores por defecto (media para numéricas, índice de la moda para categóricas)."""
        d = []
        for name in self.feature_names:
            info = self.schema["features"][name]
            if info["type"] == "categorical":
                d.append(float(info["default_index"]))
            else:
                d.append(float(info["mean"]))
        return np.asarray(d)

    # ---- Conversión x (np.array) <-> dataframe ----

    def _row_from_vector(self, x: np.ndarray) -> dict:
        row = {}
        for i, name in enumerate(self.feature_names):
            info = self.schema["features"][name]
            if info["type"] == "categorical":
                idx = int(round(float(x[i])))
                idx = max(0, min(idx, len(info["categories"]) - 1))
                row[name] = info["categories"][idx]
            else:
                row[name] = float(x[i])
        return row

    def decode(self, x: np.ndarray) -> dict:
        """Devuelve el x como dict legible (con categorías como string)."""
        return self._row_from_vector(np.asarray(x))

    # ---- Predicciones ----

    def predict_single(self, x: np.ndarray) -> float:
        df = pd.DataFrame([self._row_from_vector(np.asarray(x))])
        return float(self.model.predict(df)[0])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        rows = [self._row_from_vector(np.asarray(x)) for x in X]
        df = pd.DataFrame(rows)
        return self.model.predict(df)

    def predict_dict(self, d: dict) -> float:
        df = pd.DataFrame([d])
        return float(self.model.predict(df)[0])


def load_predictor(model_dir: str | Path = ".") -> ModelPredictor:
    return ModelPredictor(model_dir)


if __name__ == "__main__":
    import sys

    p = load_predictor(sys.argv[1] if len(sys.argv) > 1 else ".")
    print("Target:           ", p.target)
    print("Features:         ", p.feature_names)
    print("Categóricas:      ", p.cat_features)
    print("Bounds:           ", p.bounds())
    print("Defaults:         ", p.defaults())
    print("Predicción default:", p.predict_single(p.defaults()))
'''

OPTIMIZE_EXAMPLE_PY = r'''"""
Ejemplo: optimizar el modelo CatBoost exportado con differential evolution.

Por defecto MAXIMIZA la predicción del target. Si quieres minimizar, quita el
signo menos en `objetivo`.
"""
from scipy.optimize import differential_evolution

from predict import load_predictor


def main():
    predictor = load_predictor(".")
    bounds = predictor.bounds()

    def objetivo(x):
        # differential_evolution minimiza, así que negamos para maximizar
        return -predictor.predict_single(x)

    result = differential_evolution(
        objetivo,
        bounds=bounds,
        seed=42,
        maxiter=300,
        popsize=20,
        tol=1e-7,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        workers=1,
    )

    print("=" * 60)
    print("Resultado de la optimización")
    print("=" * 60)
    print(f"Target predicho: {-result.fun:.6f}")
    print(f"Iteraciones:     {result.nit}")
    print(f"Convergió:       {result.success}")
    print("\nValores óptimos por variable:")
    decoded = predictor.decode(result.x)
    for name, val in decoded.items():
        print(f"  {name:30s} = {val}")


if __name__ == "__main__":
    main()
'''

REQUIREMENTS_TXT = """catboost>=1.2
numpy>=1.23
pandas>=1.5
scipy>=1.10
scikit-learn>=1.2
optuna>=3.0
"""

README_MD = """# CatBoost model export

Carpeta autocontenida con un modelo CatBoost entrenado y todo lo necesario para
usarlo como función objetivo en algoritmos de optimización
(`scipy.optimize.differential_evolution`, `optuna`, etc.).

## Archivos

| Archivo | Contenido |
|---|---|
| `model.cbm` | Modelo CatBoost serializado |
| `metadata.json` | Hiperparámetros, métricas, fecha, versiones |
| `feature_schema.json` | Tipo, bounds, categorías y stats de cada variable |
| `feature_stats.csv` | Estadísticas detalladas de cada predictor |
| `metrics.csv` | Métricas en train / validación / test |
| `feature_importance.csv` | Importancia CatBoost |
| `shap_importance.csv` | Importancia global SHAP |
| `predictions_test.csv` | Predicciones del set de test |
| `predict.py` | Wrapper `ModelPredictor` para inferencia |
| `optimize_example.py` | Ejemplo listo para correr con DE |
| `requirements.txt` | Dependencias mínimas |

## Uso rápido

```bash
pip install -r requirements.txt
python optimize_example.py
```

## Uso programático

```python
from predict import load_predictor
from scipy.optimize import differential_evolution

p = load_predictor(".")
result = differential_evolution(
    lambda x: -p.predict_single(x),
    bounds=p.bounds(),
    seed=42,
)
print(p.decode(result.x), -result.fun)
```

`predict_single(x)` recibe un vector con un valor por variable, en el mismo
orden que `feature_names`. Para categóricas se espera un índice (float que se
redondea); los bounds que entrega `bounds()` ya están preparados para esto.
"""


# ======================================================================
# 3. HELPERS DE DATOS Y MÉTRICAS
# ======================================================================

@st.cache_data(show_spinner=False)
def cargar_california_housing() -> pd.DataFrame:
    data = fetch_california_housing(as_frame=True)
    return data.frame.copy()


def cargar_archivo(uploaded_file) -> pd.DataFrame:
    nombre = uploaded_file.name.lower()
    if nombre.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if nombre.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    raise ValueError("Formato no soportado. Usa CSV o Parquet.")


def calcular_metricas(y_real, y_pred, nombre_set: str) -> dict:
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_real, y_pred)
    return {
        "dataset": nombre_set,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
    }


def convertir_categoricas_a_codigos(X: pd.DataFrame):
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
    if not candidatas:
        return variable_principal
    importancia = shap_df[candidatas].abs().mean().sort_values(ascending=False)
    return importancia.index[0]


# ======================================================================
# 4. HELPERS DE GRÁFICOS
# ======================================================================

def pyplot_show(fig):
    """Renderiza la figura en Streamlit y la cierra para liberar memoria."""
    st.pyplot(fig)
    plt.close(fig)


def jitter_beeswarm(x, ancho=0.32, bins=40, seed=42):
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


def shap_summary_plot_estilo(shap_df, X_plot, shap_importancia, max_display,
                             titulo="SHAP summary plot - Test"):
    variables = shap_importancia["variable"].head(max_display).tolist()
    variables_reverso = list(reversed(variables))

    fig, ax = plt.subplots(figsize=(11, max(5, len(variables) * 0.55)))

    ultimo_scatter = None
    for i, variable in enumerate(variables_reverso):
        valores_shap = shap_df[variable].values
        valores_feature = X_plot[variable].values

        y_base = np.full(len(valores_shap), i)
        y_jitter = y_base + jitter_beeswarm(valores_shap, ancho=0.28,
                                            bins=35, seed=100 + i)

        ultimo_scatter = ax.scatter(
            valores_shap, y_jitter,
            c=valores_feature, cmap=SHAP_CMAP,
            s=16, alpha=0.85, linewidths=0,
        )

    ax.axvline(0, color=COLORS["neutral"], linewidth=1.2)
    ax.set_yticks(range(len(variables_reverso)))
    ax.set_yticklabels(variables_reverso, fontsize=11)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_ylabel("")
    ax.set_title(titulo)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["left"].set_visible(False)

    cbar = fig.colorbar(ultimo_scatter, ax=ax, pad=0.02)
    cbar.set_label("Feature value", rotation=270, labelpad=18, fontsize=11)
    cbar.set_ticks([])
    cbar.ax.text(1.8, 1.00, "High", transform=cbar.ax.transAxes,
                 ha="left", va="center", fontsize=10)
    cbar.ax.text(1.8, 0.00, "Low", transform=cbar.ax.transAxes,
                 ha="left", va="center", fontsize=10)

    fig.tight_layout()
    return fig


def dependence_plot_estilo_shap(ax, X_original, X_plot, shap_df,
                                variable_principal, variable_color,
                                mapas_categorias):
    x = X_plot[variable_principal].values
    y = shap_df[variable_principal].values
    c = X_plot[variable_color].values

    rng = np.random.default_rng(123)
    if pd.api.types.is_numeric_dtype(X_original[variable_principal]):
        x_plot = x
    else:
        x_plot = x + rng.normal(0, 0.05, size=len(x))

    scatter = ax.scatter(x_plot, y, c=c, cmap=SHAP_CMAP,
                         alpha=0.85, s=18, linewidths=0)

    ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=1)
    ax.set_xlabel(variable_principal)
    ax.set_ylabel(f"SHAP value for\n{variable_principal}")
    ax.set_title(f"{variable_principal} coloreado por {variable_color}",
                 fontsize=12)
    ax.grid(alpha=0.4)

    if variable_principal in mapas_categorias:
        categorias_x = mapas_categorias[variable_principal]
        ax.set_xticks(range(len(categorias_x)))
        ax.set_xticklabels(categorias_x, rotation=45, ha="right")

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    if variable_color in mapas_categorias:
        categorias_color = mapas_categorias[variable_color]
        cbar.set_ticks(list(range(len(categorias_color))))
        cbar.set_ticklabels(categorias_color)
        cbar.set_label(variable_color, rotation=270, labelpad=18)
    else:
        cbar.set_label(variable_color, rotation=270, labelpad=18)
        cbar.ax.text(1.8, 1.00, "High", transform=cbar.ax.transAxes,
                     ha="left", va="center", fontsize=9)
        cbar.ax.text(1.8, 0.00, "Low", transform=cbar.ax.transAxes,
                     ha="left", va="center", fontsize=9)


# ======================================================================
# 5. EXPORTACIÓN DEL MODELO (carpeta + zip para descargar)
# ======================================================================

def construir_feature_schema(X_train: pd.DataFrame,
                             cat_features: list[str]) -> dict:
    """Construye el schema con bounds, categorías, defaults y estadísticas.

    Esta es la pieza clave para que el modelo se pueda usar después como
    función objetivo en optimización: el optimizador necesita saber el rango
    en el que mover cada variable.
    """
    schema = {
        "feature_names": list(X_train.columns),
        "cat_features": list(cat_features),
        "features": {},
    }

    for col in X_train.columns:
        s = X_train[col]
        if col in cat_features:
            cat = pd.Categorical(s)
            categorias = [str(c) for c in cat.categories]
            moda_idx = int(pd.Series(cat.codes).mode().iloc[0])
            schema["features"][col] = {
                "type": "categorical",
                "categories": categorias,
                "n_categories": len(categorias),
                "default_index": moda_idx,
                "default_value": categorias[moda_idx] if categorias else None,
            }
        else:
            s_num = pd.to_numeric(s, errors="coerce").dropna()
            schema["features"][col] = {
                "type": "numerical",
                "min":    float(s_num.min()),
                "max":    float(s_num.max()),
                "mean":   float(s_num.mean()),
                "median": float(s_num.median()),
                "std":    float(s_num.std()),
                "q25":    float(s_num.quantile(0.25)),
                "q75":    float(s_num.quantile(0.75)),
            }

    return schema


def construir_feature_stats(X_train: pd.DataFrame,
                            cat_features: list[str]) -> pd.DataFrame:
    rows = []
    for col in X_train.columns:
        s = X_train[col]
        if col in cat_features:
            vc = s.astype(str).value_counts()
            rows.append({
                "variable":     col,
                "type":         "categorical",
                "n_unique":     int(s.nunique(dropna=True)),
                "most_common":  str(vc.index[0]) if not vc.empty else None,
                "most_common_freq": int(vc.iloc[0]) if not vc.empty else None,
                "min": None, "max": None, "mean": None, "median": None, "std": None,
            })
        else:
            s_num = pd.to_numeric(s, errors="coerce")
            rows.append({
                "variable":     col,
                "type":         "numerical",
                "n_unique":     int(s.nunique(dropna=True)),
                "most_common":  None,
                "most_common_freq": None,
                "min":    float(s_num.min()),
                "max":    float(s_num.max()),
                "mean":   float(s_num.mean()),
                "median": float(s_num.median()),
                "std":    float(s_num.std()),
            })
    return pd.DataFrame(rows)


def exportar_modelo_a_zip(
    model: CatBoostRegressor,
    X_train, y_train,
    X_test, y_pred_test, y_test,
    predictoras: list[str],
    cat_features: list[str],
    target: str,
    hiperparametros: dict,
    tabla_metricas: pd.DataFrame,
    importancias: pd.DataFrame,
    shap_importancia: pd.DataFrame,
    predicciones_test: pd.DataFrame,
) -> bytes:
    """Construye una carpeta de export en /tmp, la zippea y devuelve los bytes."""

    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir) / "modelo_export"
        export_dir.mkdir()

        # Modelo
        model.save_model(str(export_dir / "model.cbm"))

        # Schemas y stats
        schema = construir_feature_schema(X_train, cat_features)
        with open(export_dir / "feature_schema.json", "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)

        feature_stats = construir_feature_stats(X_train, cat_features)
        feature_stats.to_csv(export_dir / "feature_stats.csv", index=False)

        # Metadata global
        metadata = {
            "created_at":       datetime.now().isoformat(timespec="seconds"),
            "target":           target,
            "feature_names":    predictoras,
            "cat_features":     cat_features,
            "n_train":          int(len(X_train)),
            "n_test":           int(len(X_test)),
            "best_iteration":   int(model.get_best_iteration() or 0),
            "tree_count":       int(model.tree_count_),
            "hyperparameters":  hiperparametros,
            "metrics":          tabla_metricas.to_dict(orient="records"),
            "target_stats": {
                "min":    float(np.min(y_train)),
                "max":    float(np.max(y_train)),
                "mean":   float(np.mean(y_train)),
                "median": float(np.median(y_train)),
                "std":    float(np.std(y_train)),
            },
            "versions": {
                "catboost":  catboost_version,
                "sklearn":   sklearn_version,
                "numpy":     np.__version__,
                "pandas":    pd.__version__,
                "python":    platform.python_version(),
            },
        }
        with open(export_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # CSVs auxiliares
        tabla_metricas.to_csv(export_dir / "metrics.csv", index=False)
        importancias.to_csv(export_dir / "feature_importance.csv", index=False)
        shap_importancia.to_csv(export_dir / "shap_importance.csv", index=False)
        predicciones_test.to_csv(export_dir / "predictions_test.csv", index=False)

        # Wrappers y docs
        (export_dir / "predict.py").write_text(PREDICT_PY, encoding="utf-8")
        (export_dir / "optimize_example.py").write_text(OPTIMIZE_EXAMPLE_PY, encoding="utf-8")
        (export_dir / "requirements.txt").write_text(REQUIREMENTS_TXT, encoding="utf-8")
        (export_dir / "README.md").write_text(README_MD, encoding="utf-8")

        # Zip en memoria
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in export_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(export_dir))
        return buffer.getvalue()


# ======================================================================
# 5b. ENTRENAMIENTO REUSABLE (usado por el botón manual y por Optuna)
# ======================================================================

def entrenar_y_guardar(hiperparametros, X_train, y_train, X_valid, y_valid,
                       X_test, y_test, X, predictoras, cat_features, target):
    """Entrena CatBoost, calcula métricas y guarda todo en session_state."""
    model = CatBoostRegressor(
        **hiperparametros,
        use_best_model=True,
        verbose=False,
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_features if cat_features else None,
    )

    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_valid)
    y_pred_test = model.predict(X_test)

    tabla_metricas = pd.DataFrame([
        calcular_metricas(y_train, y_pred_train, "Train"),
        calcular_metricas(y_valid, y_pred_valid, "Validación"),
        calcular_metricas(y_test, y_pred_test, "Test"),
    ])[["dataset", "R2", "RMSE", "MAE", "MSE"]]

    importancias = pd.DataFrame({
        "variable": X.columns,
        "importancia": model.get_feature_importance(),
    }).sort_values("importancia", ascending=False).reset_index(drop=True)

    evals_result = model.get_evals_result()
    curva = pd.DataFrame({
        "iteracion": np.arange(1, len(evals_result["learn"]["RMSE"]) + 1),
        "RMSE train": evals_result["learn"]["RMSE"],
        "RMSE validación": evals_result["validation"]["RMSE"],
    })

    resultados = pd.DataFrame({
        "real":             y_test.values,
        "predicho":         y_pred_test,
        "residuo":          y_test.values - y_pred_test,
        "residuo_absoluto": np.abs(y_test.values - y_pred_test),
    }).reset_index(drop=True)
    resultados["index"] = resultados.index

    st.session_state.entrenado = True
    st.session_state.shap_calculado = False
    st.session_state.model = model
    st.session_state.X_train = X_train
    st.session_state.y_train = y_train
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.y_pred_test = y_pred_test
    st.session_state.predictoras = predictoras
    st.session_state.cat_features = cat_features
    st.session_state.target = target
    st.session_state.hiperparametros = hiperparametros
    st.session_state.tabla_metricas = tabla_metricas
    st.session_state.importancias = importancias
    st.session_state.curva = curva
    st.session_state.resultados = resultados

    return model, tabla_metricas


# ======================================================================
# 5c. OPTUNA: BÚSQUEDA AUTOMÁTICA DE HIPERPARÁMETROS
# ======================================================================

def correr_optuna(X_train, y_train, X_valid, y_valid, cat_features,
                  n_trials, seed, progress_callback=None):
    """Búsqueda Optuna minimizando RMSE en validación.

    El espacio de búsqueda usa rangos pensados para ser robustos al overfit en
    cualquier dataset, alineados con los defaults de los sliders.
    """

    def objective(trial):
        params = {
            "iterations":          trial.suggest_int("iterations", 500, 3000, step=100),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth":               trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "random_strength":     trial.suggest_float("random_strength", 0.0, 10.0),
            "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 1, 30),
        }
        m = CatBoostRegressor(
            **params,
            early_stopping_rounds=50,
            random_seed=int(seed),
            loss_function="RMSE",
            eval_metric="RMSE",
            use_best_model=True,
            verbose=False,
        )
        m.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cat_features if cat_features else None,
        )
        pred = m.predict(X_valid)
        return float(np.sqrt(mean_squared_error(y_valid, pred)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(seed)),
    )
    callbacks = [progress_callback] if progress_callback is not None else None
    study.optimize(
        objective,
        n_trials=int(n_trials),
        callbacks=callbacks,
        show_progress_bar=False,
    )
    return study


def calcular_mejora_porcentual(baseline_metricas: dict,
                               optimizado_metricas: dict) -> dict:
    """% de mejora por métrica (positivo siempre = mejor)."""
    mejoras = {}
    for k in ["R2", "RMSE", "MAE", "MSE"]:
        b = baseline_metricas[k]
        o = optimizado_metricas[k]
        denom = max(abs(b), 1e-6)
        if k == "R2":
            mejoras[k] = (o - b) / denom * 100  # mayor es mejor
        else:
            mejoras[k] = (b - o) / denom * 100  # menor es mejor
    return mejoras


def plot_optuna_history(study):
    values = [t.value for t in study.trials if t.value is not None]
    if not values:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(1, len(values) + 1)
    running_best = np.minimum.accumulate(values)
    ax.scatter(x, values, alpha=0.55, s=40, color=COLORS["primary"],
               label="Trial", edgecolors="none")
    ax.plot(x, running_best, linewidth=2.5, color=COLORS["secondary"],
            label="Mejor RMSE acumulado")
    ax.set_xlabel("Trial")
    ax.set_ylabel("RMSE validación")
    ax.set_title("Historial de búsqueda (Optuna)")
    ax.grid(alpha=0.35)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_optuna_param_importance(study):
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        return None
    df = pd.DataFrame({
        "parametro":   list(importances.keys()),
        "importancia": list(importances.values()),
    }).sort_values("importancia", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.45)))
    ax.barh(df["parametro"], df["importancia"], color=COLORS["tertiary"])
    ax.set_xlabel("Importancia relativa")
    ax.set_title("Importancia de hiperparámetros (Optuna)")
    ax.grid(axis="x", alpha=0.35)
    fig.tight_layout()
    return fig


# ======================================================================
# 5d. K-FOLD CROSS-VALIDATION
# ======================================================================

def cross_validation_kfold(X, y, cat_features, params,
                           n_splits=5, seed=42, progress=None):
    """K-fold CV con los hiperparámetros indicados. Retorna métricas por fold."""
    kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    rows = []
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model_cv = CatBoostRegressor(**params, verbose=False)
        model_cv.fit(X_tr, y_tr,
                     cat_features=cat_features if cat_features else None)
        pred = model_cv.predict(X_val)
        rows.append(calcular_metricas(y_val, pred, f"Fold {fold_idx}"))
        if progress is not None:
            progress(fold_idx, n_splits)
    return pd.DataFrame(rows)


def plot_kfold_resultados(cv_df, metric="R2"):
    fig, ax = plt.subplots(figsize=(10, 5))
    folds = cv_df["dataset"]
    values = cv_df[metric]
    mean_val = values.mean()
    std_val = values.std()

    bars = ax.bar(folds, values, color=COLORS["primary"],
                  edgecolor="white", linewidth=1.5)
    ax.axhspan(mean_val - std_val, mean_val + std_val,
               color=COLORS["secondary"], alpha=0.12,
               label=f"±1 std = {std_val:.4f}")
    ax.axhline(mean_val, linestyle="--", color=COLORS["secondary"],
               linewidth=2, label=f"Media = {mean_val:.4f}")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(), f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} por fold (K-fold CV)")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# ======================================================================
# 6. SIDEBAR: CARGA, TARGET, PREDICTORES, HIPERPARÁMETROS
# ======================================================================

st.title("CatBoost Regressor")
st.write(
    "Entrena un modelo CatBoost de regresión, explora SHAP y exporta una carpeta "
    "lista para usar el modelo en algoritmos de optimización."
)

# Init de session_state
for key, default in [
    ("entrenado", False),
    ("model", None),
    ("shap_calculado", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


with st.sidebar:
    st.header("1. Datos")

    uploaded_file = st.file_uploader(
        "Sube tu archivo CSV o Parquet",
        type=["csv", "parquet"],
    )

    if uploaded_file is not None:
        try:
            df = cargar_archivo(uploaded_file)
            st.success("Archivo cargado correctamente.")
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            st.stop()
    else:
        df = cargar_california_housing()
        st.info("Usando dataset por defecto: California Housing.")

# ----- Vista previa del dataset -----
st.subheader("Vista previa del dataset")
st.dataframe(df.head(), use_container_width=True)
st.write(f"Filas: **{df.shape[0]:,}** &nbsp;|&nbsp; Columnas: **{df.shape[1]}**")

if df.shape[1] < 2:
    st.error("El dataset debe tener al menos dos columnas (target + 1 predictor).")
    st.stop()


with st.sidebar:
    st.header("2. Variables")

    columnas = df.columns.tolist()
    target_default = "MedHouseVal" if "MedHouseVal" in columnas else columnas[-1]
    target = st.selectbox(
        "Variable target",
        columnas,
        index=columnas.index(target_default),
    )
    predictoras_disponibles = [c for c in columnas if c != target]
    predictoras = st.multiselect(
        "Variables predictoras",
        predictoras_disponibles,
        default=predictoras_disponibles,
    )

    if not predictoras:
        st.warning("Selecciona al menos una variable predictora.")
        st.stop()

    st.header("3. División de datos")
    test_size = st.slider("% para test", 0.10, 0.40, 0.20, 0.05)
    valid_size = st.slider("% para validación", 0.10, 0.40, 0.20, 0.05)
    if test_size + valid_size >= 0.90:
        st.error("Test + validación dejan muy poco para entrenar.")
        st.stop()

    st.header("4. Hiperparámetros")
    st.caption(
        "Defaults pensados para ser **robustos al overfitting** en cualquier "
        "dataset. Ajusta solo si entiendes el efecto."
    )

    random_state = st.number_input(
        "Random state", min_value=0, value=42, step=1,
        help="Semilla para reproducibilidad de los splits y del modelo.",
    )

    iterations = st.slider(
        "Iteraciones máximas", 200, 5000, 2000, 100,
        help=(
            "Rango usual: 1000-5000. Ponlo alto y deja que el early stopping "
            "decida cuándo parar. No es 'mientras más mejor': si no se activa "
            "early stopping, el modelo puede estar sobre-ajustando."
        ),
    )

    learning_rate = st.slider(
        "Learning rate", 0.005, 0.300, 0.030, 0.005, format="%.3f",
        help=(
            "Rango usual: 0.01-0.10. **Menor → más estable y robusto** al "
            "overfitting, pero requiere más iteraciones. Default sugerido: "
            "0.03. Para datasets muy ruidosos prueba 0.01-0.02."
        ),
    )

    depth = st.slider(
        "Profundidad de árbol", 2, 12, 5, 1,
        help=(
            "Rango usual: 4-6. **Bajo (4-6) → robusto.** Alto (>8) memoriza "
            "ruido. Default CatBoost = 6; bajamos a 5 por seguridad."
        ),
    )

    l2_leaf_reg = st.slider(
        "L2 leaf regularization", 1.0, 30.0, 5.0, 0.5,
        help=(
            "Rango usual: 1-10. **Más alto → más regularización** y menos "
            "overfit. Default CatBoost = 3; aquí 5 como base más segura."
        ),
    )

    bagging_temperature = st.slider(
        "Bagging temperature", 0.0, 5.0, 1.0, 0.1,
        help=(
            "Rango usual: 0-1. Intensidad del Bayesian bagging. **Mayor → "
            "más aleatoriedad** y resistencia al overfit, menos precisión. "
            "Default = 1."
        ),
    )

    random_strength = st.slider(
        "Random strength", 0.0, 20.0, 1.0, 0.5,
        help=(
            "Rango usual: 1-10. Aleatoriedad al elegir splits. **Mayor → "
            "más regularización**. Default = 1; sube si hay overfit."
        ),
    )

    min_data_in_leaf = st.slider(
        "Min data in leaf", 1, 100, 5, 1,
        help=(
            "Rango usual: 1-20. Mínimo de muestras por hoja. **Mayor → árboles "
            "más conservadores**. Default CatBoost = 1; aquí 5 como base."
        ),
    )

    early_stopping_rounds = st.slider(
        "Early stopping rounds", 10, 300, 75, 5,
        help=(
            "Rango usual: 50-100. Iteraciones sin mejora en validación antes "
            "de parar. Mayor → más paciencia."
        ),
    )

    st.header("5. Gráficos")
    top_n_shap = st.slider(
        "Variables para SHAP dependence plots",
        1, len(predictoras), len(predictoras), 1,
    )
    max_display_summary = st.slider(
        "Variables para SHAP summary plot",
        1, len(predictoras), len(predictoras), 1,
    )

    st.header("6. Herramientas avanzadas (opcional)")

    with st.expander("🎯 Tuning con Optuna"):
        if not OPTUNA_AVAILABLE:
            st.warning("Optuna no está instalado.\n\n`pip install optuna`")
            n_trials = 0
            run_optuna = False
        else:
            st.caption(
                "Busca automáticamente la mejor combinación de hiperparámetros "
                "minimizando RMSE en validación. Reentrena el modelo con los "
                "mejores parámetros y muestra la **mejora porcentual** vs. la "
                "configuración actual de los sliders."
            )
            n_trials = st.slider(
                "Número de trials", 10, 200, 30, 5,
                help="Más trials = búsqueda más exhaustiva, pero más tiempo. "
                     "30-50 suele bastar para datasets medianos.",
            )
            run_optuna = st.button("Buscar con Optuna",
                                   use_container_width=True)

    with st.expander("🔁 Cross-validation K-fold"):
        st.caption(
            "Estima la **robustez** del modelo entrenando K modelos en "
            "particiones disjuntas. Reporta media ± std por métrica. "
            "Requiere haber entrenado al menos un modelo (reutiliza sus "
            "hiperparámetros)."
        )
        n_splits = st.slider(
            "Número de folds (K)", 3, 10, 5, 1,
            help="K=5 es estándar. K=10 es más robusto pero ~2x más lento.",
        )
        run_kfold = st.button("Ejecutar K-fold CV",
                              use_container_width=True)


# ======================================================================
# 7. PREPARACIÓN DE DATOS Y VALIDACIONES
# ======================================================================

df_model = df[[target] + predictoras].copy()
for col in df_model.columns:
    if df_model[col].dtype == "object":
        df_model[col] = df_model[col].astype("category")

filas_antes = df_model.shape[0]
df_model = df_model.dropna()
filas_despues = df_model.shape[0]
if filas_despues < filas_antes:
    st.info(f"Se eliminaron {filas_antes - filas_despues:,} filas con valores nulos.")

if df_model.shape[0] < 30:
    st.error("Después de eliminar nulos quedan muy pocas filas para entrenar.")
    st.stop()

X = df_model[predictoras]
y = df_model[target]

if not pd.api.types.is_numeric_dtype(y):
    st.error("La variable target debe ser numérica para CatBoostRegressor.")
    st.stop()
if y.nunique() < 2:
    st.error("El target es constante (varianza = 0).")
    st.stop()
if not np.isfinite(y).all():
    st.error("El target tiene valores no finitos (inf / NaN).")
    st.stop()

cat_features = [col for col in X.columns if str(X[col].dtype) == "category"]

# Warning de alta cardinalidad
for col in cat_features:
    if X[col].nunique() > 0.5 * len(X):
        st.warning(
            f"La variable categórica '{col}' tiene cardinalidad muy alta "
            f"({X[col].nunique():,} valores únicos sobre {len(X):,} filas). "
            "Considera excluirla o agruparla."
        )

# Splits
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)
valid_relative_size = valid_size / (1 - test_size)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=valid_relative_size,
    random_state=int(random_state),
)


# ----- Resumen de configuración -----
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

with st.expander("Ver variables y stats"):
    st.write("**Target:**", target)
    st.write("**Predictoras:**", predictoras)
    st.write("**Categóricas detectadas:**", cat_features if cat_features else "Ninguna")
    st.write("**Stats del target (train):**")
    st.dataframe(y_train.describe().to_frame().T, use_container_width=True)


# ======================================================================
# 8. ENTRENAMIENTO (manual, Optuna y K-fold)
# ======================================================================

entrenar = st.sidebar.button("🚀 Entrenar CatBoost", type="primary",
                             use_container_width=True)

# ---- 8a. Entrenamiento manual ----
if entrenar:
    try:
        hiperparametros = {
            "iterations":            int(iterations),
            "learning_rate":         float(learning_rate),
            "depth":                 int(depth),
            "l2_leaf_reg":           float(l2_leaf_reg),
            "bagging_temperature":   float(bagging_temperature),
            "random_strength":       float(random_strength),
            "min_data_in_leaf":      int(min_data_in_leaf),
            "early_stopping_rounds": int(early_stopping_rounds),
            "random_seed":           int(random_state),
            "loss_function":         "RMSE",
            "eval_metric":           "RMSE",
        }
        with st.spinner("Entrenando modelo CatBoost..."):
            entrenar_y_guardar(
                hiperparametros,
                X_train, y_train, X_valid, y_valid,
                X_test, y_test, X, predictoras, cat_features, target,
            )
        # Limpiar resultados de Optuna previos (este es un train manual)
        for k in ["optuna_run", "optuna_baseline_metrics",
                  "optuna_optimized_metrics", "optuna_mejoras",
                  "optuna_best_params", "optuna_study", "optuna_n_trials"]:
            st.session_state.pop(k, None)
    except Exception as e:
        st.error(f"Error durante el entrenamiento: {e}")
        st.exception(e)

# ---- 8b. Optuna: tuning automático ----
if OPTUNA_AVAILABLE and run_optuna:
    try:
        # 1) Baseline con los sliders actuales
        baseline_params = {
            "iterations":            int(iterations),
            "learning_rate":         float(learning_rate),
            "depth":                 int(depth),
            "l2_leaf_reg":           float(l2_leaf_reg),
            "bagging_temperature":   float(bagging_temperature),
            "random_strength":       float(random_strength),
            "min_data_in_leaf":      int(min_data_in_leaf),
            "early_stopping_rounds": int(early_stopping_rounds),
            "random_seed":           int(random_state),
            "loss_function":         "RMSE",
            "eval_metric":           "RMSE",
        }
        with st.spinner("Entrenando baseline con la configuración actual..."):
            baseline_model = CatBoostRegressor(
                **baseline_params, use_best_model=True, verbose=False,
            )
            baseline_model.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
                cat_features=cat_features if cat_features else None,
            )
            baseline_pred = baseline_model.predict(X_test)
            baseline_metrics = calcular_metricas(y_test, baseline_pred, "Baseline")

        # 2) Búsqueda Optuna
        st.info(f"Ejecutando {n_trials} trials con Optuna...")
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def optuna_callback(study, trial):
            done = len(study.trials)
            progress_bar.progress(min(done / float(n_trials), 1.0))
            best = study.best_value if study.best_value is not None else float("inf")
            status_text.text(
                f"Trial {done}/{n_trials} · Mejor RMSE (val) = {best:.4f}"
            )

        study = correr_optuna(
            X_train, y_train, X_valid, y_valid,
            cat_features, n_trials, random_state,
            progress_callback=optuna_callback,
        )
        progress_bar.empty()
        status_text.empty()

        # 3) Reentrenar con mejores params
        best_params = dict(study.best_params)
        best_params.update({
            "early_stopping_rounds": 50,
            "random_seed":           int(random_state),
            "loss_function":         "RMSE",
            "eval_metric":           "RMSE",
        })
        with st.spinner("Reentrenando con los mejores hiperparámetros..."):
            entrenar_y_guardar(
                best_params,
                X_train, y_train, X_valid, y_valid,
                X_test, y_test, X, predictoras, cat_features, target,
            )

        # 4) Comparar baseline vs optimizado
        optimized_row = (
            st.session_state.tabla_metricas
            .query("dataset == 'Test'")
            .iloc[0]
            .to_dict()
        )
        mejoras = calcular_mejora_porcentual(baseline_metrics, optimized_row)

        st.session_state.optuna_run = True
        st.session_state.optuna_baseline_metrics = baseline_metrics
        st.session_state.optuna_optimized_metrics = optimized_row
        st.session_state.optuna_mejoras = mejoras
        st.session_state.optuna_best_params = best_params
        st.session_state.optuna_study = study
        st.session_state.optuna_n_trials = int(n_trials)

        st.success("Optuna terminó. Modelo actualizado con los mejores hiperparámetros.")
    except Exception as e:
        st.error(f"Error durante la búsqueda con Optuna: {e}")
        st.exception(e)

# ---- 8c. K-fold cross-validation ----
if run_kfold:
    if not st.session_state.entrenado:
        st.warning("Primero entrena un modelo (manualmente o con Optuna) "
                   "antes de correr K-fold CV.")
    else:
        try:
            hp = st.session_state.hiperparametros
            # En CV usamos iteraciones fijas (las que early stopping eligió)
            # y omitimos early_stopping_rounds porque no hay valid set interno.
            params_cv = {
                "iterations":          int(st.session_state.model.tree_count_),
                "learning_rate":       hp["learning_rate"],
                "depth":               hp["depth"],
                "l2_leaf_reg":         hp["l2_leaf_reg"],
                "bagging_temperature": hp["bagging_temperature"],
                "random_strength":     hp["random_strength"],
                "min_data_in_leaf":    hp["min_data_in_leaf"],
                "random_seed":         hp["random_seed"],
                "loss_function":       "RMSE",
                "eval_metric":         "RMSE",
            }

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def kfold_callback(fold_idx, total):
                progress_bar.progress(fold_idx / float(total))
                status_text.text(f"Entrenando fold {fold_idx}/{total}...")

            with st.spinner(f"Ejecutando {n_splits}-fold CV..."):
                cv_df = cross_validation_kfold(
                    X, y, cat_features, params_cv,
                    n_splits=n_splits, seed=hp["random_seed"],
                    progress=kfold_callback,
                )

            progress_bar.empty()
            status_text.empty()

            resumen = pd.DataFrame({
                "metrica": ["R2", "RMSE", "MAE", "MSE"],
                "media":   [cv_df[m].mean() for m in ["R2", "RMSE", "MAE", "MSE"]],
                "std":     [cv_df[m].std()  for m in ["R2", "RMSE", "MAE", "MSE"]],
                "min":     [cv_df[m].min()  for m in ["R2", "RMSE", "MAE", "MSE"]],
                "max":     [cv_df[m].max()  for m in ["R2", "RMSE", "MAE", "MSE"]],
            })

            st.session_state.kfold_run = True
            st.session_state.kfold_df = cv_df
            st.session_state.kfold_resumen = resumen
            st.session_state.kfold_n_splits = int(n_splits)
            st.success(f"K-fold CV terminado ({n_splits} folds).")
        except Exception as e:
            st.error(f"Error en K-fold CV: {e}")
            st.exception(e)


# ======================================================================
# 9. RESULTADOS (solo si está entrenado)
# ======================================================================

if not st.session_state.entrenado:
    st.info("Configura las variables en el panel lateral y presiona **Entrenar CatBoost**.")
    st.stop()

# Lectura desde session_state
model           = st.session_state.model
X_train         = st.session_state.X_train
y_train         = st.session_state.y_train
X_test          = st.session_state.X_test
y_test          = st.session_state.y_test
y_pred_test     = st.session_state.y_pred_test
predictoras_ss  = st.session_state.predictoras
cat_features_ss = st.session_state.cat_features
target_ss       = st.session_state.target
hiperparametros = st.session_state.hiperparametros
tabla_metricas  = st.session_state.tabla_metricas
importancias    = st.session_state.importancias
curva           = st.session_state.curva
resultados      = st.session_state.resultados

metricas_test_row = tabla_metricas.query("dataset == 'Test'").iloc[0]
r2, rmse, mae, mse = (metricas_test_row["R2"], metricas_test_row["RMSE"],
                      metricas_test_row["MAE"], metricas_test_row["MSE"])

st.success("Modelo entrenado correctamente.")

# ----- Métricas principales -----
st.subheader("Métricas en test")
col1, col2, col3, col4 = st.columns(4)
col1.metric("R²", f"{r2:,.4f}")
col2.metric("RMSE", f"{rmse:,.4f}")
col3.metric("MAE", f"{mae:,.4f}")
col4.metric("MSE", f"{mse:,.4f}")

# Métrica de gap train-test (señal de overfit)
r2_train = tabla_metricas.query("dataset == 'Train'").iloc[0]["R2"]
gap = r2_train - r2
if gap > 0.15:
    st.warning(
        f"Gap R² train-test = {gap:.3f}. El modelo puede estar sobreajustando. "
        "Prueba a aumentar `l2_leaf_reg`, `min_data_in_leaf`, o bajar `depth`."
    )

st.subheader("Tabla de métricas")
st.dataframe(
    tabla_metricas.style.format({
        "R2": "{:.4f}", "RMSE": "{:.4f}", "MAE": "{:.4f}", "MSE": "{:.4f}"
    }),
    use_container_width=True,
)

mejor_iteracion = model.get_best_iteration()
st.write(f"**Mejor iteración:** {mejor_iteracion} / {hiperparametros['iterations']}")
if mejor_iteracion is not None and mejor_iteracion >= hiperparametros["iterations"] - 1:
    st.info(
        "La mejor iteración coincide con el máximo: early stopping no se activó. "
        "Considera aumentar `iterations` o ajustar `learning_rate`."
    )


# ----- Optuna: comparación baseline vs optimizado -----
if st.session_state.get("optuna_run", False):
    st.subheader("🎯 Mejora con Optuna")

    baseline = st.session_state.optuna_baseline_metrics
    optimized = st.session_state.optuna_optimized_metrics
    mejoras = st.session_state.optuna_mejoras

    st.caption(
        f"Comparación de métricas en **test**: configuración inicial de los "
        f"sliders vs. mejores hiperparámetros encontrados tras "
        f"{st.session_state.optuna_n_trials} trials. Δ% positivo = mejora."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²",  f"{optimized['R2']:.4f}",   f"{mejoras['R2']:+.2f}%")
    c2.metric("RMSE", f"{optimized['RMSE']:.4f}", f"{mejoras['RMSE']:+.2f}%")
    c3.metric("MAE",  f"{optimized['MAE']:.4f}",  f"{mejoras['MAE']:+.2f}%")
    c4.metric("MSE",  f"{optimized['MSE']:.4f}",  f"{mejoras['MSE']:+.2f}%")

    comp_df = pd.DataFrame({
        "Métrica":     ["R²", "RMSE", "MAE", "MSE"],
        "Baseline":    [baseline[k] for k in ["R2", "RMSE", "MAE", "MSE"]],
        "Optimizado":  [optimized[k] for k in ["R2", "RMSE", "MAE", "MSE"]],
        "Δ absoluto":  [optimized[k] - baseline[k] for k in ["R2", "RMSE", "MAE", "MSE"]],
        "Mejora %":    [mejoras[k] for k in ["R2", "RMSE", "MAE", "MSE"]],
    })
    st.dataframe(
        comp_df.style.format({
            "Baseline":   "{:.4f}",
            "Optimizado": "{:.4f}",
            "Δ absoluto": "{:+.4f}",
            "Mejora %":   "{:+.2f}%",
        }),
        use_container_width=True,
    )

    with st.expander("Mejores hiperparámetros encontrados"):
        st.json(st.session_state.optuna_best_params)

    col_oh, col_oi = st.columns(2)
    with col_oh:
        fig_hist = plot_optuna_history(st.session_state.optuna_study)
        if fig_hist is not None:
            pyplot_show(fig_hist)
    with col_oi:
        fig_pi = plot_optuna_param_importance(st.session_state.optuna_study)
        if fig_pi is not None:
            pyplot_show(fig_pi)
        else:
            st.caption("Importancia de parámetros no disponible (pocos trials).")


# ----- K-fold cross-validation -----
if st.session_state.get("kfold_run", False):
    st.subheader(f"🔁 Cross-validation ({st.session_state.kfold_n_splits} folds)")

    cv_df = st.session_state.kfold_df
    resumen = st.session_state.kfold_resumen

    st.caption(
        "Métricas evaluadas en cada fold (hold-out). Una std baja indica que "
        "el modelo es **estable** ante distintas particiones."
    )

    media_r2 = resumen.query("metrica == 'R2'").iloc[0]
    media_rmse = resumen.query("metrica == 'RMSE'").iloc[0]
    media_mae = resumen.query("metrica == 'MAE'").iloc[0]
    media_mse = resumen.query("metrica == 'MSE'").iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R² medio",  f"{media_r2['media']:.4f}",  f"± {media_r2['std']:.4f}")
    k2.metric("RMSE medio", f"{media_rmse['media']:.4f}", f"± {media_rmse['std']:.4f}")
    k3.metric("MAE medio",  f"{media_mae['media']:.4f}",  f"± {media_mae['std']:.4f}")
    k4.metric("MSE medio",  f"{media_mse['media']:.4f}",  f"± {media_mse['std']:.4f}")

    col_kt1, col_kt2 = st.columns([1, 1])
    with col_kt1:
        st.write("**Por fold:**")
        st.dataframe(
            cv_df.style.format({"R2": "{:.4f}", "RMSE": "{:.4f}",
                                "MAE": "{:.4f}", "MSE": "{:.4f}"}),
            use_container_width=True,
        )
    with col_kt2:
        st.write("**Resumen:**")
        st.dataframe(
            resumen.style.format({"media": "{:.4f}", "std": "{:.4f}",
                                  "min": "{:.4f}", "max": "{:.4f}"}),
            use_container_width=True,
        )

    metric_plot = st.selectbox(
        "Métrica a graficar",
        options=["R2", "RMSE", "MAE", "MSE"],
        index=0,
    )
    fig_cv = plot_kfold_resultados(cv_df, metric=metric_plot)
    pyplot_show(fig_cv)


# ----- Real vs Predicho -----
st.subheader("Real vs Predicho")
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_test, y_pred_test, alpha=0.6, s=35, color=COLORS["primary"],
           edgecolors="none")

min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
margen = (max_val - min_val) * 0.05
min_plot, max_plot = min_val - margen, max_val + margen
ax.plot([min_plot, max_plot], [min_plot, max_plot],
        linestyle="--", linewidth=2, color=COLORS["secondary"],
        label="Predicción perfecta")

texto = (f"R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nMSE = {mse:.4f}")
ax.text(0.05, 0.95, texto, transform=ax.transAxes, va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  alpha=0.92, edgecolor=COLORS["grid"]))
ax.set_xlim(min_plot, max_plot)
ax.set_ylim(min_plot, max_plot)
ax.set_xlabel("Valor real")
ax.set_ylabel("Valor predicho")
ax.set_title("Real vs Predicho - Test")
ax.grid(alpha=0.35)
ax.legend(loc="lower right")
fig.tight_layout()
pyplot_show(fig)


# ----- Real vs Predicho vs Index -----
st.subheader("Real vs Predicho vs Index")
max_test_index = X_test.shape[0]
n_index_plot = st.slider(
    "Observaciones a mostrar",
    min_value=min(20, max_test_index),
    max_value=max_test_index,
    value=min(200, max_test_index),
    step=10,
)

resultados_idx = resultados.head(n_index_plot)
fig_idx, ax_idx = plt.subplots(figsize=(12, 5))
ax_idx.plot(resultados_idx["index"], resultados_idx["real"],
            label="Real", linewidth=2, color=COLORS["primary"])
ax_idx.plot(resultados_idx["index"], resultados_idx["predicho"],
            label="Predicho", linewidth=2, color=COLORS["secondary"])
ax_idx.set_xlabel("Index")
ax_idx.set_ylabel(target_ss)
ax_idx.set_title(f"Real vs Predicho - {n_index_plot:,} de {len(resultados):,} obs")
ax_idx.grid(alpha=0.35)
ax_idx.legend()
fig_idx.tight_layout()
pyplot_show(fig_idx)


# ----- Residuos -----
st.subheader("Residuos")
col_r1, col_r2 = st.columns(2)
with col_r1:
    fig_rs, ax_rs = plt.subplots(figsize=(7, 5))
    ax_rs.scatter(y_pred_test, resultados["residuo"],
                  alpha=0.55, color=COLORS["primary"], edgecolors="none")
    ax_rs.axhline(0, linestyle="--", linewidth=2, color=COLORS["secondary"])
    ax_rs.set_xlabel("Valor predicho")
    ax_rs.set_ylabel("Residuo")
    ax_rs.set_title("Residuos vs Predicho")
    ax_rs.grid(alpha=0.35)
    fig_rs.tight_layout()
    pyplot_show(fig_rs)

with col_r2:
    fig_rh, ax_rh = plt.subplots(figsize=(7, 5))
    ax_rh.hist(resultados["residuo"], bins=30,
               color=COLORS["primary"], edgecolor="white", linewidth=0.7)
    ax_rh.axvline(0, linestyle="--", linewidth=2, color=COLORS["secondary"])
    ax_rh.set_xlabel("Residuo")
    ax_rh.set_ylabel("Frecuencia")
    ax_rh.set_title("Distribución de residuos")
    ax_rh.grid(alpha=0.35)
    fig_rh.tight_layout()
    pyplot_show(fig_rh)


# ----- Curva de aprendizaje -----
st.subheader("Curva de aprendizaje")
mejor_iteracion_plot = (mejor_iteracion + 1) if mejor_iteracion is not None else len(curva)
mejor_rmse_valid = curva["RMSE validación"].iloc[mejor_iteracion_plot - 1]

fig_lr, ax_lr = plt.subplots(figsize=(11, 5.5))
ax_lr.plot(curva["iteracion"], curva["RMSE train"],
           label="RMSE train", linewidth=2, color=COLORS["primary"])
ax_lr.plot(curva["iteracion"], curva["RMSE validación"],
           label="RMSE validación", linewidth=2, color=COLORS["secondary"])
ax_lr.axvline(mejor_iteracion_plot, linestyle="--", linewidth=2,
              color=COLORS["highlight"],
              label=f"Mejor iteración: {mejor_iteracion_plot}")
ax_lr.scatter([mejor_iteracion_plot], [mejor_rmse_valid],
              s=90, zorder=5, color=COLORS["highlight"])

y_offset = np.std(curva["RMSE validación"]) * 0.8 or mejor_rmse_valid * 0.05
ax_lr.annotate(
    f"Early stopping\nIter {mejor_iteracion_plot}\nRMSE = {mejor_rmse_valid:.4f}",
    xy=(mejor_iteracion_plot, mejor_rmse_valid),
    xytext=(mejor_iteracion_plot, mejor_rmse_valid + y_offset),
    arrowprops=dict(arrowstyle="->", lw=1.5, color=COLORS["text"]),
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
              alpha=0.9, edgecolor=COLORS["grid"]),
)
ax_lr.set_xlabel("Iteración")
ax_lr.set_ylabel("RMSE")
ax_lr.set_title("Curva de aprendizaje con early stopping")
ax_lr.grid(alpha=0.35)
ax_lr.legend()
fig_lr.tight_layout()
pyplot_show(fig_lr)

with st.expander("Datos de la curva"):
    st.dataframe(curva, use_container_width=True)


# ----- Importancia de variables -----
st.subheader("Importancia de variables (CatBoost)")
st.dataframe(importancias, use_container_width=True)

importancias_plot = importancias.sort_values("importancia", ascending=True)
fig_imp, ax_imp = plt.subplots(figsize=(10, max(5, len(importancias_plot) * 0.35)))
ax_imp.barh(importancias_plot["variable"], importancias_plot["importancia"],
            color=COLORS["primary"])
ax_imp.set_xlabel("Importancia")
ax_imp.set_title("Importancia de variables")
ax_imp.grid(axis="x", alpha=0.35)
fig_imp.tight_layout()
pyplot_show(fig_imp)


# ======================================================================
# 10. SHAP (cálculo on-demand, persistido)
# ======================================================================

st.subheader("SHAP values")

if not st.session_state.shap_calculado:
    if st.button("Calcular SHAP values (puede tardar)"):
        try:
            with st.spinner("Calculando SHAP values en test..."):
                test_pool = Pool(
                    X_test, y_test,
                    cat_features=cat_features_ss if cat_features_ss else None,
                )
                shap_values_full = model.get_feature_importance(
                    test_pool, type="ShapValues"
                )

            shap_values = shap_values_full[:, :-1]
            shap_base_value = shap_values_full[:, -1]
            shap_df = pd.DataFrame(shap_values, columns=predictoras_ss)
            X_test_plot, mapas_categorias = convertir_categoricas_a_codigos(X_test)
            shap_importancia = pd.DataFrame({
                "variable": predictoras_ss,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

            st.session_state.shap_df = shap_df
            st.session_state.shap_base_value = shap_base_value
            st.session_state.X_test_plot = X_test_plot
            st.session_state.mapas_categorias = mapas_categorias
            st.session_state.shap_importancia = shap_importancia
            st.session_state.shap_calculado = True
            st.rerun()
        except Exception as e:
            st.error(f"Error calculando SHAP: {e}")
            st.exception(e)
    else:
        st.info("Presiona el botón para calcular los SHAP values.")

if st.session_state.shap_calculado:
    shap_df          = st.session_state.shap_df
    shap_base_value  = st.session_state.shap_base_value
    X_test_plot      = st.session_state.X_test_plot
    mapas_categorias = st.session_state.mapas_categorias
    shap_importancia = st.session_state.shap_importancia

    st.write("**Importancia global basada en SHAP:**")
    st.dataframe(shap_importancia, use_container_width=True)

    shap_plot = shap_importancia.sort_values("mean_abs_shap", ascending=True)
    fig_sb, ax_sb = plt.subplots(figsize=(10, max(5, len(shap_plot) * 0.35)))
    ax_sb.barh(shap_plot["variable"], shap_plot["mean_abs_shap"],
               color=COLORS["tertiary"])
    ax_sb.set_xlabel("Mean |SHAP value|")
    ax_sb.set_title("Importancia global SHAP")
    ax_sb.grid(axis="x", alpha=0.35)
    fig_sb.tight_layout()
    pyplot_show(fig_sb)

    st.subheader("SHAP summary plot")
    fig_ss = shap_summary_plot_estilo(
        shap_df=shap_df, X_plot=X_test_plot,
        shap_importancia=shap_importancia,
        max_display=max_display_summary,
        titulo="SHAP summary plot - Test",
    )
    pyplot_show(fig_ss)

    with st.expander("Ver SHAP values por observación"):
        shap_mostrar = shap_df.copy()
        shap_mostrar["base_value"] = shap_base_value
        shap_mostrar["predicho"]   = y_pred_test
        shap_mostrar["real"]       = y_test.values
        st.dataframe(shap_mostrar.head(200), use_container_width=True)

    st.subheader("SHAP dependence plot individual")
    variable_dependence = st.selectbox(
        "Variable principal",
        options=shap_importancia["variable"].tolist(),
        index=0,
    )
    variable_color_default = seleccionar_variable_interaccion(
        variable_dependence, predictoras_ss, shap_df,
    )
    variable_color = st.selectbox(
        "Variable de color / interacción",
        options=predictoras_ss,
        index=predictoras_ss.index(variable_color_default),
    )
    fig_di, ax_di = plt.subplots(figsize=(9, 6))
    dependence_plot_estilo_shap(
        ax=ax_di, X_original=X_test, X_plot=X_test_plot,
        shap_df=shap_df, variable_principal=variable_dependence,
        variable_color=variable_color, mapas_categorias=mapas_categorias,
    )
    fig_di.tight_layout()
    pyplot_show(fig_di)

    st.subheader("SHAP dependence plots en matriz")
    variables_top = shap_importancia["variable"].head(top_n_shap).tolist()
    n_cols = 2
    n_rows = math.ceil(len(variables_top) / n_cols)
    fig_dep, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(16, max(6, n_rows * 5)))
    if n_rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, variable in enumerate(variables_top):
        var_inter = seleccionar_variable_interaccion(
            variable, predictoras_ss, shap_df,
        )
        dependence_plot_estilo_shap(
            ax=axes[i], X_original=X_test, X_plot=X_test_plot,
            shap_df=shap_df, variable_principal=variable,
            variable_color=var_inter, mapas_categorias=mapas_categorias,
        )
    for j in range(len(variables_top), len(axes)):
        axes[j].axis("off")
    fig_dep.tight_layout()
    pyplot_show(fig_dep)


# ======================================================================
# 11. PREDICCIÓN SOBRE DATOS NUEVOS
# ======================================================================

st.subheader("Predicción sobre datos nuevos")

archivo_nuevo = st.file_uploader(
    "Sube un CSV o Parquet con las mismas variables predictoras",
    type=["csv", "parquet"],
    key="archivo_nuevo",
)

if archivo_nuevo is not None:
    try:
        df_nuevo = cargar_archivo(archivo_nuevo)
        faltan = [c for c in predictoras_ss if c not in df_nuevo.columns]
        if faltan:
            st.error(f"Faltan columnas en el archivo: {faltan}")
        else:
            X_nuevo = df_nuevo[predictoras_ss].copy()
            for col in cat_features_ss:
                X_nuevo[col] = X_nuevo[col].astype("category")
            with st.spinner("Generando predicciones..."):
                pred_nuevo = model.predict(X_nuevo)
            df_nuevo_out = df_nuevo.copy()
            df_nuevo_out[f"prediccion_{target_ss}"] = pred_nuevo
            st.dataframe(df_nuevo_out.head(100), use_container_width=True)
            st.download_button(
                "Descargar predicciones",
                df_nuevo_out.to_csv(index=False).encode("utf-8"),
                "predicciones_nuevas.csv",
                "text/csv",
            )
    except Exception as e:
        st.error(f"Error procesando archivo nuevo: {e}")


# ======================================================================
# 12. DESCARGAS Y EXPORTACIÓN PARA OPTIMIZACIÓN
# ======================================================================

st.subheader("Descargas")

col_d1, col_d2, col_d3 = st.columns(3)
with col_d1:
    st.download_button(
        "📊 Predicciones (CSV)",
        resultados.to_csv(index=False).encode("utf-8"),
        "predicciones_catboost.csv",
        "text/csv",
        use_container_width=True,
    )
with col_d2:
    st.download_button(
        "📈 Métricas (CSV)",
        tabla_metricas.to_csv(index=False).encode("utf-8"),
        "metricas_catboost.csv",
        "text/csv",
        use_container_width=True,
    )
with col_d3:
    if st.session_state.shap_calculado:
        st.download_button(
            "🎯 SHAP values (CSV)",
            st.session_state.shap_df.to_csv(index=False).encode("utf-8"),
            "shap_values_catboost.csv",
            "text/csv",
            use_container_width=True,
        )
    else:
        st.caption("Calcula SHAP para descargar los valores")


st.subheader("📦 Exportar paquete completo para optimización")
st.write(
    "Genera un `.zip` con el modelo, schema de features (bounds y categorías), "
    "metadata, métricas, importancias, `predict.py` y un `optimize_example.py` "
    "listo para usar con `scipy.optimize.differential_evolution`."
)

if st.button("Generar paquete de exportación", type="primary"):
    try:
        with st.spinner("Empaquetando modelo y artefactos..."):
            shap_importancia_export = (
                st.session_state.shap_importancia
                if st.session_state.shap_calculado
                else pd.DataFrame({"variable": predictoras_ss,
                                   "mean_abs_shap": [np.nan] * len(predictoras_ss)})
            )
            zip_bytes = exportar_modelo_a_zip(
                model=model,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_pred_test=y_pred_test, y_test=y_test,
                predictoras=predictoras_ss,
                cat_features=cat_features_ss,
                target=target_ss,
                hiperparametros=hiperparametros,
                tabla_metricas=tabla_metricas,
                importancias=importancias,
                shap_importancia=shap_importancia_export,
                predicciones_test=resultados,
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.success("Paquete generado.")
        st.download_button(
            "⬇️ Descargar paquete (.zip)",
            zip_bytes,
            f"catboost_export_{timestamp}.zip",
            "application/zip",
            use_container_width=True,
        )
        st.caption(
            "Dentro del zip: `model.cbm`, `feature_schema.json`, `metadata.json`, "
            "`metrics.csv`, `feature_importance.csv`, `shap_importance.csv`, "
            "`predictions_test.csv`, `feature_stats.csv`, `predict.py`, "
            "`optimize_example.py`, `requirements.txt`, `README.md`."
        )
    except Exception as e:
        st.error(f"Error generando el paquete: {e}")
        st.exception(e)
