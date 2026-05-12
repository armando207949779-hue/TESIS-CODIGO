from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostRegressor
from scipy.optimize import differential_evolution


# ============================================================
# Configuración
# ============================================================

st.set_page_config(
    page_title="Optimizador flexible CatBoost",
    layout="wide",
)

st.title("Optimizador flexible para modelo CatBoost exportado")

st.write(
    "Sube el paquete `.zip` exportado desde tu app CatBoost. "
    "Luego podrás maximizar, minimizar o buscar un valor específico del target, "
    "aplicando restricciones sobre los predictores."
)


# ============================================================
# Predictor flexible
# ============================================================

class FlexibleModelPredictor:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)

        with open(self.model_dir / "feature_schema.json", "r", encoding="utf-8") as f:
            self.schema = json.load(f)

        with open(self.model_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.model = CatBoostRegressor()
        self.model.load_model(str(self.model_dir / "model.cbm"))

        self.feature_names = self.schema["feature_names"]
        self.cat_features = self.schema.get("cat_features", [])
        self.target = self.metadata["target"]

    def default_current_values(self) -> dict:
        values = {}

        for name in self.feature_names:
            info = self.schema["features"][name]

            if info["type"] == "categorical":
                values[name] = info.get("default_value", info["categories"][0])
            else:
                values[name] = float(info.get("mean", 0.0))

        return values

    def predict_dict(self, values: dict) -> float:
        row = {}

        for name in self.feature_names:
            info = self.schema["features"][name]

            if info["type"] == "categorical":
                row[name] = str(values[name])
            else:
                row[name] = float(values[name])

        df = pd.DataFrame([row])
        return float(self.model.predict(df)[0])

    def vector_to_dict(self, x: np.ndarray, opt_config: dict) -> dict:
        row = {}

        for i, name in enumerate(self.feature_names):
            cfg = opt_config[name]
            info = self.schema["features"][name]

            if cfg["fixed"]:
                row[name] = cfg["fixed_value"]
                continue

            if info["type"] == "categorical":
                allowed = cfg["allowed_categories"]
                idx = int(round(float(x[i])))
                idx = max(0, min(idx, len(allowed) - 1))
                row[name] = allowed[idx]
            else:
                row[name] = float(x[i])

        return row

    def vector_bounds(self, opt_config: dict) -> list[tuple[float, float]]:
        bounds = []

        for name in self.feature_names:
            cfg = opt_config[name]
            info = self.schema["features"][name]

            if cfg["fixed"]:
                value = cfg["fixed_vector_value"]
                bounds.append((value, value))
                continue

            if info["type"] == "categorical":
                n = len(cfg["allowed_categories"])
                bounds.append((0.0, float(max(n - 1, 0))))
            else:
                bounds.append((float(cfg["lower"]), float(cfg["upper"])))

        return bounds


# ============================================================
# Utilidades
# ============================================================

def safe_pct_change(new, old):
    try:
        new = float(new)
        old = float(old)

        if abs(old) < 1e-12:
            return np.nan

        return (new - old) / abs(old) * 100
    except Exception:
        return np.nan


def direction_arrow(new, old):
    try:
        new = float(new)
        old = float(old)

        if new > old:
            return "↑"
        if new < old:
            return "↓"
        return "→"
    except Exception:
        if str(new) == str(old):
            return "→"
        return "↔"


def target_improvement_pct(current_pred, best_pred, mode, target_value=None):
    current_pred = float(current_pred)
    best_pred = float(best_pred)

    if mode == "Maximizar target":
        if abs(current_pred) < 1e-12:
            return np.nan
        return (best_pred - current_pred) / abs(current_pred) * 100

    if mode == "Minimizar target":
        if abs(current_pred) < 1e-12:
            return np.nan
        return (current_pred - best_pred) / abs(current_pred) * 100

    current_error = abs(current_pred - target_value)
    best_error = abs(best_pred - target_value)

    if abs(current_error) < 1e-12:
        return 0.0

    return (current_error - best_error) / abs(current_error) * 100


def style_comparison_table(df):
    def color_gain(value):
        try:
            value = float(value)
        except Exception:
            return ""

        if value > 0:
            return "color: #0a7f28; font-weight: 700;"
        if value < 0:
            return "color: #c1121f; font-weight: 700;"
        return "color: #6c757d; font-weight: 700;"

    return df.style.applymap(
        color_gain,
        subset=["ganancia_%"],
    )


def build_convergence_chart(history_df):
    fig = px.line(
        history_df,
        x="iteracion_global",
        y="mejor_valor_objetivo",
        color="corrida",
        markers=False,
        title="Convergencia por corrida",
    )

    fig.update_layout(
        xaxis_title="Iteración",
        yaxis_title="Mejor valor objetivo interno",
        height=450,
    )

    return fig


def build_parallel_plot(results_df, predictor):
    df = results_df.copy()

    if df.shape[0] < 2:
        return None

    dimensions = []

    for name in predictor.feature_names:
        if name not in df.columns:
            continue

        info = predictor.schema["features"][name]

        if info["type"] == "categorical":
            categories = sorted(df[name].astype(str).unique().tolist())
            mapping = {cat: i for i, cat in enumerate(categories)}
            encoded = df[name].astype(str).map(mapping)

            dimensions.append(
                dict(
                    label=name,
                    values=encoded,
                    tickvals=list(mapping.values()),
                    ticktext=list(mapping.keys()),
                )
            )
        else:
            dimensions.append(
                dict(
                    label=name,
                    values=df[name].astype(float),
                )
            )

    dimensions.append(
        dict(
            label=f"Predicción {predictor.target}",
            values=df["prediccion"].astype(float),
        )
    )

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df["prediccion"].astype(float),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=predictor.target),
            ),
            dimensions=dimensions,
        )
    )

    fig.update_layout(
        title="Parallel plot de soluciones óptimas",
        height=550,
    )

    return fig


def build_radar_chart(current_values, best_values, predictor):
    numeric_features = []

    for name in predictor.feature_names:
        info = predictor.schema["features"][name]
        if info["type"] == "numerical":
            numeric_features.append(name)

    if len(numeric_features) < 3:
        return None

    labels = []
    current_scaled = []
    best_scaled = []

    for name in numeric_features:
        info = predictor.schema["features"][name]
        min_v = float(info["min"])
        max_v = float(info["max"])
        denom = max(max_v - min_v, 1e-12)

        cur = float(current_values[name])
        opt = float(best_values[name])

        labels.append(name)
        current_scaled.append((cur - min_v) / denom)
        best_scaled.append((opt - min_v) / denom)

    labels_closed = labels + [labels[0]]
    current_closed = current_scaled + [current_scaled[0]]
    best_closed = best_scaled + [best_scaled[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=current_closed,
            theta=labels_closed,
            fill="toself",
            name="Actual",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=best_closed,
            theta=labels_closed,
            fill="toself",
            name="Óptimo",
        )
    )

    fig.update_layout(
        title="Gráfico radial: valores actuales vs óptimos",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )
        ),
        height=550,
    )

    return fig


# ============================================================
# Carga del ZIP
# ============================================================

uploaded_zip = st.file_uploader(
    "Sube el `.zip` exportado desde la app principal",
    type=["zip"],
)

if uploaded_zip is None:
    st.info("Sube el paquete exportado para comenzar.")
    st.stop()

tmpdir = tempfile.TemporaryDirectory()
model_dir = Path(tmpdir.name) / "modelo_export"
model_dir.mkdir(parents=True, exist_ok=True)

try:
    with zipfile.ZipFile(uploaded_zip, "r") as zf:
        zf.extractall(model_dir)
except Exception as e:
    st.error(f"No se pudo extraer el ZIP: {e}")
    st.stop()

required_files = [
    "model.cbm",
    "feature_schema.json",
    "metadata.json",
]

missing = [f for f in required_files if not (model_dir / f).exists()]

if missing:
    st.error(f"El ZIP no es válido. Faltan archivos: {missing}")
    st.stop()

try:
    predictor = FlexibleModelPredictor(model_dir)
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

st.success("Modelo cargado correctamente.")

c1, c2, c3 = st.columns(3)
c1.metric("Target", predictor.target)
c2.metric("Predictores", len(predictor.feature_names))
c3.metric("Categóricas", len(predictor.cat_features))


# ============================================================
# Configuración del objetivo
# ============================================================

st.sidebar.header("Objetivo de optimización")

optimization_mode = st.sidebar.radio(
    "Modo",
    [
        "Maximizar target",
        "Minimizar target",
        "Buscar valor específico del target",
    ],
)

target_value = None

if optimization_mode == "Buscar valor específico del target":
    target_value = st.sidebar.number_input(
        f"Valor objetivo para {predictor.target}",
        value=0.0,
        step=0.1,
    )

st.sidebar.header("Corridas")

n_runs = st.sidebar.slider(
    "Número de corridas independientes",
    min_value=1,
    max_value=30,
    value=5,
    step=1,
)

base_seed = st.sidebar.number_input(
    "Seed base",
    min_value=0,
    value=42,
    step=1,
)

maxiter = st.sidebar.slider(
    "Máximo de iteraciones por corrida",
    min_value=50,
    max_value=2000,
    value=400,
    step=50,
)

popsize = st.sidebar.slider(
    "Population size",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
)

tol = st.sidebar.select_slider(
    "Tolerancia",
    options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    value=1e-6,
)

polish = st.sidebar.checkbox(
    "Polish final",
    value=True,
)


# ============================================================
# Valores actuales y restricciones
# ============================================================

st.subheader("Valores actuales y restricciones por predictor")

default_current = predictor.default_current_values()

opt_config = {}
current_values = {}

with st.expander("Configurar predictores", expanded=True):
    for name in predictor.feature_names:
        info = predictor.schema["features"][name]

        st.markdown(f"### {name}")

        col_a, col_b, col_c, col_d = st.columns([1.2, 1.2, 1.2, 1])

        if info["type"] == "categorical":
            categories = info["categories"]

            with col_a:
                current_value = st.selectbox(
                    "Valor actual",
                    options=categories,
                    index=categories.index(default_current[name])
                    if default_current[name] in categories
                    else 0,
                    key=f"current_{name}",
                )

            with col_b:
                allowed = st.multiselect(
                    "Categorías permitidas",
                    options=categories,
                    default=categories,
                    key=f"allowed_{name}",
                )

            if not allowed:
                st.warning(f"Debes permitir al menos una categoría para {name}.")
                st.stop()

            with col_c:
                fixed = st.checkbox(
                    "Fijar predictor",
                    value=False,
                    key=f"fixed_{name}",
                )

            with col_d:
                if fixed:
                    fixed_value = st.selectbox(
                        "Valor fijo",
                        options=allowed,
                        index=allowed.index(current_value)
                        if current_value in allowed
                        else 0,
                        key=f"fixed_value_{name}",
                    )
                else:
                    fixed_value = current_value

            current_values[name] = current_value

            fixed_vector_value = float(
                allowed.index(fixed_value)
                if fixed_value in allowed
                else 0
            )

            opt_config[name] = {
                "fixed": fixed,
                "fixed_value": fixed_value,
                "fixed_vector_value": fixed_vector_value,
                "allowed_categories": allowed,
            }

        else:
            min_original = float(info["min"])
            max_original = float(info["max"])
            mean_original = float(info.get("mean", (min_original + max_original) / 2))

            with col_a:
                current_value = st.number_input(
                    "Valor actual",
                    value=float(mean_original),
                    key=f"current_{name}",
                )

            with col_b:
                lower = st.number_input(
                    "Mínimo permitido",
                    value=float(min_original),
                    key=f"lower_{name}",
                )

            with col_c:
                upper = st.number_input(
                    "Máximo permitido",
                    value=float(max_original),
                    key=f"upper_{name}",
                )

            with col_d:
                fixed = st.checkbox(
                    "Fijar predictor",
                    value=False,
                    key=f"fixed_{name}",
                )

            if lower > upper:
                st.error(f"En {name}, el mínimo permitido no puede ser mayor que el máximo.")
                st.stop()

            if fixed:
                fixed_value = current_value
                fixed_vector_value = float(current_value)
            else:
                fixed_value = current_value
                fixed_vector_value = float(current_value)

            current_values[name] = float(current_value)

            opt_config[name] = {
                "fixed": fixed,
                "fixed_value": float(fixed_value),
                "fixed_vector_value": float(fixed_vector_value),
                "lower": float(lower),
                "upper": float(upper),
            }


# ============================================================
# Predicción actual
# ============================================================

try:
    current_prediction = predictor.predict_dict(current_values)
except Exception as e:
    st.error(f"No se pudo calcular la predicción actual: {e}")
    st.stop()

st.subheader("Estado actual")

m1, m2 = st.columns(2)
m1.metric("Predicción actual", f"{current_prediction:,.6f}")
m2.metric("Target", predictor.target)

with st.expander("Ver valores actuales"):
    current_df = pd.DataFrame({
        "variable": list(current_values.keys()),
        "valor_actual": list(current_values.values()),
    })
    st.dataframe(current_df, use_container_width=True)


# ============================================================
# Bounds
# ============================================================

bounds = predictor.vector_bounds(opt_config)

bounds_rows = []

for i, name in enumerate(predictor.feature_names):
    info = predictor.schema["features"][name]
    cfg = opt_config[name]

    if info["type"] == "categorical":
        bounds_rows.append({
            "variable": name,
            "tipo": "categorical",
            "fijo": cfg["fixed"],
            "restricción": ", ".join(map(str, cfg["allowed_categories"])),
        })
    else:
        bounds_rows.append({
            "variable": name,
            "tipo": "numerical",
            "fijo": cfg["fixed"],
            "mínimo": cfg["lower"],
            "máximo": cfg["upper"],
        })

st.subheader("Resumen de restricciones")
st.dataframe(pd.DataFrame(bounds_rows), use_container_width=True)


# ============================================================
# Optimización
# ============================================================

st.subheader("Ejecutar optimización")

run_optimization = st.button("Ejecutar optimización", type="primary")

if run_optimization:
    progress_bar = st.progress(0)
    status_text = st.empty()

    all_results = []
    convergence_rows = []

    total_iterations_estimated = maxiter * n_runs
    global_iteration = 0

    best_global = None
    best_global_objective = np.inf
    best_global_result = None

    for run_idx in range(n_runs):
        seed = int(base_seed) + run_idx
        run_label = f"Corrida {run_idx + 1}"

        run_history = {
            "iteration": 0,
            "best_objective": np.inf,
        }

        def objective(x):
            values = predictor.vector_to_dict(x, opt_config)
            pred = predictor.predict_dict(values)

            if optimization_mode == "Maximizar target":
                return -pred

            if optimization_mode == "Minimizar target":
                return pred

            return abs(pred - float(target_value))

        def callback(xk, convergence):
            nonlocal global_iteration

            global_iteration += 1

            try:
                current_obj = objective(xk)
                run_history["best_objective"] = min(
                    run_history["best_objective"],
                    float(current_obj),
                )
            except Exception:
                pass

            pct = min(global_iteration / total_iterations_estimated, 1.0)
            progress_bar.progress(int(pct * 100))

            status_text.write(
                f"Progreso: **{pct * 100:,.1f}%** | "
                f"{run_label} | "
                f"Iteración aproximada: {run_history['iteration'] + 1}/{maxiter}"
            )

            convergence_rows.append({
                "corrida": run_label,
                "iteracion_corrida": run_history["iteration"] + 1,
                "iteracion_global": global_iteration,
                "mejor_valor_objetivo": run_history["best_objective"],
            })

            run_history["iteration"] += 1

            return False

        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                seed=seed,
                maxiter=int(maxiter),
                popsize=int(popsize),
                tol=float(tol),
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=bool(polish),
                workers=1,
                updating="immediate",
                callback=callback,
            )

            decoded = predictor.vector_to_dict(result.x, opt_config)
            pred = predictor.predict_dict(decoded)
            obj = float(objective(result.x))

            row = {
                "corrida": run_label,
                "seed": seed,
                "prediccion": pred,
                "objetivo_interno": obj,
                "convergio": bool(result.success),
                "iteraciones": int(result.nit),
                "mensaje": str(result.message),
            }

            row.update(decoded)
            all_results.append(row)

            if obj < best_global_objective:
                best_global_objective = obj
                best_global = decoded
                best_global_result = row

        except Exception as e:
            st.error(f"Error en {run_label}: {e}")

    progress_bar.progress(100)
    status_text.success("Optimización terminada.")

    if not all_results:
        st.error("No se obtuvo ninguna solución válida.")
        st.stop()

    results_df = pd.DataFrame(all_results)
    convergence_df = pd.DataFrame(convergence_rows)

    best_prediction = float(best_global_result["prediccion"])

    st.success("Optimización completada.")

    # ========================================================
    # Métricas principales
    # ========================================================

    st.subheader("Mejor solución encontrada")

    improvement = target_improvement_pct(
        current_prediction,
        best_prediction,
        optimization_mode,
        target_value,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicción actual", f"{current_prediction:,.6f}")
    k2.metric("Mejor predicción", f"{best_prediction:,.6f}")
    k3.metric("Ganancia target", f"{improvement:,.2f}%")
    k4.metric("Corridas", n_runs)

    if optimization_mode == "Buscar valor específico del target":
        e1, e2 = st.columns(2)
        e1.metric("Valor objetivo", f"{target_value:,.6f}")
        e2.metric("Error absoluto final", f"{abs(best_prediction - target_value):,.6f}")

    # ========================================================
    # Tabla actual vs óptimo
    # ========================================================

    comparison_rows = []

    for name in predictor.feature_names:
        actual = current_values[name]
        optimal = best_global[name]

        pct = safe_pct_change(optimal, actual)
        arrow = direction_arrow(optimal, actual)

        comparison_rows.append({
            "variable": name,
            "tipo": predictor.schema["features"][name]["type"],
            "valor_actual": actual,
            "valor_optimo": optimal,
            "direccion": arrow,
            "ganancia_%": pct,
        })

    comparison_rows.append({
        "variable": predictor.target,
        "tipo": "target",
        "valor_actual": current_prediction,
        "valor_optimo": best_prediction,
        "direccion": direction_arrow(best_prediction, current_prediction),
        "ganancia_%": improvement,
    })

    comparison_df = pd.DataFrame(comparison_rows)

    st.subheader("Tabla comparativa: actual vs óptimo")

    st.dataframe(
        style_comparison_table(comparison_df),
        use_container_width=True,
    )

    csv_comparison = comparison_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Descargar tabla comparativa CSV",
        data=csv_comparison,
        file_name="comparacion_actual_vs_optimo.csv",
        mime="text/csv",
    )

    # ========================================================
    # Resultados por corrida
    # ========================================================

    st.subheader("Resultados de múltiples corridas")

    st.dataframe(results_df, use_container_width=True)

    csv_results = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Descargar resultados por corrida CSV",
        data=csv_results,
        file_name="resultados_corridas.csv",
        mime="text/csv",
    )

    # ========================================================
    # Robustez
    # ========================================================

    st.subheader("Robustez de la solución")

    pred_mean = results_df["prediccion"].mean()
    pred_std = results_df["prediccion"].std(ddof=1) if len(results_df) > 1 else 0.0
    pred_cv = pred_std / abs(pred_mean) * 100 if abs(pred_mean) > 1e-12 else np.nan

    if np.isnan(pred_cv):
        robustness_score = np.nan
    else:
        robustness_score = max(0.0, 100.0 - pred_cv)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Media predicción", f"{pred_mean:,.6f}")
    r2.metric("Desviación estándar", f"{pred_std:,.6f}")
    r3.metric("Coef. variación", f"{pred_cv:,.2f}%")
    r4.metric("Robustez", f"{robustness_score:,.2f}%")

    st.caption(
        "Interpretación: una desviación estándar baja entre corridas indica que "
        "la solución es más estable. La robustez se calcula como 100 - coeficiente "
        "de variación porcentual."
    )

    numeric_summary_rows = []

    for name in predictor.feature_names:
        info = predictor.schema["features"][name]

        if info["type"] == "numerical":
            numeric_summary_rows.append({
                "variable": name,
                "media": results_df[name].astype(float).mean(),
                "std": results_df[name].astype(float).std(ddof=1)
                if len(results_df) > 1
                else 0.0,
                "min": results_df[name].astype(float).min(),
                "max": results_df[name].astype(float).max(),
            })

    if numeric_summary_rows:
        st.write("**Estabilidad de predictores numéricos entre corridas:**")
        robustness_df = pd.DataFrame(numeric_summary_rows)
        st.dataframe(robustness_df, use_container_width=True)

    # ========================================================
    # Gráfico de convergencia
    # ========================================================

    if not convergence_df.empty:
        st.subheader("Gráfico de convergencia")
        fig_conv = build_convergence_chart(convergence_df)
        st.plotly_chart(fig_conv, use_container_width=True)

    # ========================================================
    # Parallel plot
    # ========================================================

    st.subheader("Parallel plot")

    fig_parallel = build_parallel_plot(results_df, predictor)

    if fig_parallel is not None:
        st.plotly_chart(fig_parallel, use_container_width=True)
    else:
        st.info("Se necesitan al menos 2 corridas para mostrar el parallel plot.")

    # ========================================================
    # Radar chart
    # ========================================================

    st.subheader("Gráfico radial")

    fig_radar = build_radar_chart(current_values, best_global, predictor)

    if fig_radar is not None:
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info(
            "El gráfico radial requiere al menos 3 predictores numéricos."
        )

    # ========================================================
    # Mejor solución en JSON
    # ========================================================

    with st.expander("Ver mejor solución en formato JSON"):
        st.json(best_global)

    with st.expander("Ver configuración usada"):
        st.json({
            "modo": optimization_mode,
            "target_value": target_value,
            "n_runs": n_runs,
            "base_seed": base_seed,
            "maxiter": maxiter,
            "popsize": popsize,
            "tol": tol,
            "polish": polish,
        })
