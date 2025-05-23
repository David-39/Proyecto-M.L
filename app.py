import streamlit as st
import joblib
import os

# Cargar SOLO el modelo necesario
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "modelo-reg-tree-knn-nn.pkl")
    return joblib.load(model_path)  # Carga el modelo

modelo = load_model()

# Interfaz mínima
st.title("🚗 Predicción de Riesgo")
edad = st.slider("Edad", 18, 100)
tipo = st.selectbox("Tipo de vehículo", ["Familiar", "Deportivo", "Combi", "Minivan"])

if st.button("Predecir"):
    # Preparar inputs (ajusta según tu modelo)
    inputs = [[edad, 1 if tipo == "Combi" else 0, 
               1 if tipo == "Familiar" else 0, 
               1 if tipo == "Minivan" else 0, 
               1 if tipo == "Deportivo" else 0]]
    
    # Predicción (ajusta según tu modelo)
    riesgo = modelo.predict(inputs)[0]
    st.success(f"Riesgo: {'Alto' if riesgo == 2 else 'Medio' if riesgo == 1 else 'Bajo'}")