import streamlit as st
import joblib
import os
from pathlib import Path

# Configuración de rutas
current_dir = Path(__file__).parent

try:
    # Carga de modelos con nombres correctos
    def load_model(name):
        path = current_dir / name
        if not path.exists():
            st.error(f"❌ Archivo no encontrado: {path}")
            st.stop()
        return joblib.load(path)

    # Cargar modelos con los nombres exactos
    model_tree = load_model("modelo-clas-tree.pkl")  # Árbol de clasificación
    model_knn = load_model("modelo-reg-tree-knn-nn.pkl")[1]  # KNN
    model_rl = load_model("modelo-clas-tree-RL.pkl")[1]  # Regresión Logística

    # Interfaz de usuario
    st.title("🛡️ Predicción de Riesgo Vehicular")
    edad = st.slider("Edad del conductor", 18, 100, 30)
    tipo = st.selectbox("Tipo de vehículo", ["Familiar", "Deportivo", "Combi", "Minivan"])
    
    if st.button("🔮 Predecir riesgo"):
        # Preparar inputs
        inputs = [[
            edad,
            1 if tipo == "Combi" else 0,
            1 if tipo == "Familiar" else 0,
            1 if tipo == "Minivan" else 0,
            1 if tipo == "Deportivo" else 0
        ]]
        
        # Predicción con Árbol de Decisión (ejemplo)
        riesgo = model_tree[0].predict(inputs)[0]
        nivel_riesgo = model_tree[1].inverse_transform([riesgo])[0]
        
        st.success(f"✅ Nivel de riesgo predicho: {nivel_riesgo.upper()}")

except Exception as e:
    st.error(f"⚠️ Error: {str(e)}")