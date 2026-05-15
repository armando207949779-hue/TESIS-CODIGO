# app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Clustering - California Housing",
    page_icon="🏘️",
    layout="wide"
)


@st.cache_data
def load_california_housing():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return df


def load_uploaded_parquet(uploaded_file):
    return pd.read_parquet(uploaded_file)


def get_numeric_columns(df):
    return df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()


def run_clustering(X_scaled, algorithm, n_clusters, eps, min_samples):
    if algorithm == "K-Means":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(X_scaled)

    elif algorithm == "DBSCAN":
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )
        labels = model.fit_predict(X_scaled)

    elif algorithm == "Jerárquico":
        model = AgglomerativeClustering(
            n_clusters=n_clusters
        )
        labels = model.fit_predict(X_scaled)

    return labels


def plot_clusters(X_pca, labels):
    fig, ax = plt.subplots(figsize=(8, 5))

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        alpha=0.7
    )

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Visualización de clusters con PCA")

    plt.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)


st.title("Machine Learning - Clustering")
st.write(
    "Esta app usa **California Housing** por defecto, "
    "pero también permite subir un archivo **Parquet** para aplicar clustering."
)

st.sidebar.header("Datos")

uploaded_file = st.sidebar.file_uploader(
    "Sube un archivo Parquet",
    type=["parquet"]
)

if uploaded_file is not None:
    try:
        df = load_uploaded_parquet(uploaded_file)
        st.sidebar.success("Archivo Parquet cargado correctamente.")
        data_source = "Archivo Parquet subido"
    except Exception as e:
        st.sidebar.error(f"Error al leer el archivo Parquet: {e}")
        df = load_california_housing()
        data_source = "California Housing por defecto"
else:
    df = load_california_housing()
    data_source = "California Housing por defecto"


st.subheader("Fuente de datos")
st.info(data_source)

st.subheader("Vista previa de los datos")
st.dataframe(df.head())

st.subheader("Información del dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Filas", df.shape[0])

with col2:
    st.metric("Columnas", df.shape[1])

with col3:
    st.metric("Valores nulos", int(df.isnull().sum().sum()))


numeric_columns = get_numeric_columns(df)

if len(numeric_columns) < 2:
    st.error("El dataset debe tener al menos 2 columnas numéricas para aplicar clustering.")
    st.stop()


st.sidebar.header("Configuración del modelo")

selected_columns = st.sidebar.multiselect(
    "Selecciona columnas numéricas",
    options=numeric_columns,
    default=numeric_columns[: min(5, len(numeric_columns))]
)

if len(selected_columns) < 2:
    st.warning("Selecciona al menos 2 columnas numéricas.")
    st.stop()


algorithm = st.sidebar.selectbox(
    "Algoritmo de clustering",
    ["K-Means", "DBSCAN", "Jerárquico"]
)

n_clusters = st.sidebar.slider(
    "Número de clusters",
    min_value=2,
    max_value=10,
    value=3
)

eps = st.sidebar.slider(
    "eps para DBSCAN",
    min_value=0.1,
    max_value=5.0,
    value=0.8,
    step=0.1
)

min_samples = st.sidebar.slider(
    "min_samples para DBSCAN",
    min_value=2,
    max_value=20,
    value=5
)


X = df[selected_columns].copy()

X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna()

if X.empty:
    st.error("No quedan datos válidos después de eliminar nulos o infinitos.")
    st.stop()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

labels = run_clustering(
    X_scaled=X_scaled,
    algorithm=algorithm,
    n_clusters=n_clusters,
    eps=eps,
    min_samples=min_samples
)

result_df = X.copy()
result_df["Cluster"] = labels


st.subheader("Resultados del clustering")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Algoritmo", algorithm)

with col2:
    st.metric("Registros usados", X.shape[0])

with col3:
    st.metric("Clusters encontrados", len(set(labels)))


st.write("Distribución de registros por cluster:")
cluster_counts = result_df["Cluster"].value_counts().sort_index()
st.dataframe(cluster_counts.rename("Cantidad"))


valid_labels = set(labels)

if algorithm == "DBSCAN" and -1 in valid_labels:
    st.warning(
        "DBSCAN marcó algunos puntos como ruido. "
        "Estos aparecen con el cluster -1."
    )


try:
    unique_clusters = set(labels)

    if len(unique_clusters) > 1:
        if not (algorithm == "DBSCAN" and len(unique_clusters) == 2 and -1 in unique_clusters):
            score = silhouette_score(X_scaled, labels)
            st.metric("Silhouette Score", round(score, 4))
        else:
            st.info("No se puede calcular correctamente Silhouette Score con solo ruido y un cluster.")
    else:
        st.info("No se puede calcular Silhouette Score con un solo cluster.")

except Exception as e:
    st.info(f"No se pudo calcular Silhouette Score: {e}")


st.subheader("Visualización de clusters")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plot_clusters(X_pca, labels)


st.subheader("Datos con cluster asignado")
st.dataframe(result_df.head(100))


csv = result_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descargar resultados en CSV",
    data=csv,
    file_name="clustering_resultados.csv",
    mime="text/csv"
)


st.subheader("Promedios por cluster")

cluster_summary = result_df.groupby("Cluster").mean(numeric_only=True)
st.dataframe(cluster_summary)
