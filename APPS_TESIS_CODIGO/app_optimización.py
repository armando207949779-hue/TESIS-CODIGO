# app_optimizer.py
from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor
from scipy.optimize import differential_evolution


# ============================================================
# Predictor compatible con el paquete exportado
# ============================================================

class ModelPredictor:
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

    def bounds(self) -> list[tuple[float, float]]:
        b = []

        for name in self.feature_names:
            info = self.schema["features"][name]

            if info["type"] == "categorical":
                b.append((0.0, float(len(info["categories"]) - 1)))
            else:
                b.append((float(info["min"]), float(info["max"])))

        return b

    def defaults(self) -> np.ndarray:
        d = []

        for name in self.feature_names:
            info = self.schema["features"][name]

            if info["type"] == "categorical":
                d.append(float(info["default_index"]))
            else:
                d.append(float(info["mean"]))

        return np.asarray(d, dtype=float)

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
        return self._row_from_vector(np.asarray(x, dtype=float))

    def predict_single(self, x: np.ndarray) -> float:
        row = self._row_from_vector(np.asarray(x, dtype=float))
        df = pd.DataFrame([row])
        return float(self.model.predict(df)[0])


# ============================================================
# Streamlit app
# ============================================================

st.set_page_config(
    page_title="Optimización de modelo CatBoost",
    layout="wide",
)

st.title("Optimizar modelo CatBoost exportado")
st.write(
    "Sube el `.zip` exportado desde tu app CatBoost y ejecuta una optimización "
    "sobre las variables del modelo."
)

uploaded_zip = st.file_uploader(
    "Sube el paquete `.zip` exportado",
    type=["zip"],
)

if uploaded_zip is None:
    st.info("Sube primero el archivo `.zip` exportado desde la app principal.")
    st.stop()


# ============================================================
# Extraer ZIP temporalmente
# ============================================================

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
    st.error(
        "El ZIP no parece ser un paquete válido. "
        f"Faltan estos archivos: {missing}"
    )
    st.stop()


# ============================================================
# Cargar predictor
# ============================================================

try:
    predictor = ModelPredictor(model_dir)
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()


st.success("Modelo cargado correctamente.")

col1, col2, col3 = st.columns(3)
col1.metric("Target", predictor.target)
col2.metric("Variables", len(predictor.feature_names))
col3.metric("Categóricas", len(predictor.cat_features))


# ============================================================
# Mostrar variables y bounds
# ============================================================

bounds = predictor.bounds()
defaults = predictor.defaults()

rows = []

for i, name in enumerate(predictor.feature_names):
    info = predictor.schema["features"][name]

    if info["type"] == "categorical":
        rows.append({
            "variable": name,
            "tipo": "categorical",
            "min": 0,
            "max": len(info["categories"]) - 1,
            "default": info.get("default_value"),
            "detalle": ", ".join(map(str, info["categories"][:10])),
        })
    else:
        rows.append({
            "variable": name,
            "tipo": "numerical",
            "min": info["min"],
            "max": info["max"],
            "default": info["mean"],
            "detalle": "",
        })

bounds_df = pd.DataFrame(rows)

st.subheader("Variables optimizables")
st.dataframe(bounds_df, use_container_width=True)


# ============================================================
# Configuración de optimización
# ============================================================

st.sidebar.header("Configuración")

modo = st.sidebar.radio(
    "Objetivo",
    ["Maximizar predicción", "Minimizar predicción"],
)

maxiter = st.sidebar.slider(
    "Max iteraciones",
    min_value=50,
    max_value=1000,
    value=300,
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
    options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
    value=1e-6,
)

seed = st.sidebar.number_input(
    "Seed",
    min_value=0,
    value=42,
    step=1,
)

polish = st.sidebar.checkbox(
    "Polish final",
    value=True,
)

st.sidebar.caption(
    "Para maximizar, la app minimiza el negativo de la predicción."
)


# ============================================================
# Probar predicción default
# ============================================================

st.subheader("Predicción con valores default")

try:
    default_pred = predictor.predict_single(defaults)
    st.metric("Predicción default", f"{default_pred:,.6f}")

    with st.expander("Ver valores default"):
        st.json(predictor.decode(defaults))

except Exception as e:
    st.error(f"No se pudo calcular la predicción default: {e}")


# ============================================================
# Ejecutar optimización
# ============================================================

st.subheader("Optimización")

run = st.button("Ejecutar optimización", type="primary")

if run:
    try:
        progress = st.empty()

        def objective(x):
            pred = predictor.predict_single(x)

            if modo == "Maximizar predicción":
                return -pred

            return pred

        with st.spinner("Optimizando..."):
            result = differential_evolution(
                objective,
                bounds=bounds,
                seed=int(seed),
                maxiter=int(maxiter),
                popsize=int(popsize),
                tol=float(tol),
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=bool(polish),
                workers=1,
                updating="immediate",
            )

        decoded = predictor.decode(result.x)
        best_prediction = predictor.predict_single(result.x)

        st.success("Optimización terminada.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Mejor predicción", f"{best_prediction:,.6f}")
        c2.metric("Iteraciones", result.nit)
        c3.metric("Convergió", "Sí" if result.success else "No")

        st.write("**Mensaje del optimizador:**")
        st.code(str(result.message))

        st.subheader("Valores óptimos encontrados")

        opt_df = pd.DataFrame({
            "variable": list(decoded.keys()),
            "valor_optimo": list(decoded.values()),
        })

        st.dataframe(opt_df, use_container_width=True)

        csv = opt_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Descargar valores óptimos CSV",
            data=csv,
            file_name="valores_optimos.csv",
            mime="text/csv",
        )

        with st.expander("Vector bruto usado por el optimizador"):
            st.write(result.x)

    except Exception as e:
        st.error(f"Error durante la optimización: {e}")
        st.exception(e)
