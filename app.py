from sklearn.tree import plot_tree
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import MinMaxScaler

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Riesgo de Vehículos",
    page_icon="🚗",
    layout="wide"
)

# Título de la aplicación
st.title("🚗 Predicción de Riesgo de Vehículos")
st.markdown("""
Esta aplicación predice el nivel de riesgo de un vehículo basado en sus características.
""")

# Cargar los modelos
@st.cache_resource
def load_models():
    try:
        # Cargar modelo de árbol de decisión
        with open('modelo-clas-tree.pkl', 'rb') as f:
            model_tree, label_encoder, variables = pickle.load(f)
        
        # Cargar modelo KNN y Red Neuronal
        with open('modelo-clas-tree-knn-nn.pkl', 'rb') as f:
            model_tree2, model_knn, model_nn, label_encoder2, variables2, scaler = pickle.load(f)
        
        # Cargar modelo de Regresión Logística
        with open('modelo-clas-tree-RL.pkl', 'rb') as f:
            model_tree3, model_lr, label_encoder3, variables3, scaler2 = pickle.load(f)
        
        return {
            'tree': model_tree,
            'knn': model_knn,
            'nn': model_nn,
            'lr': model_lr,
            'label_encoder': label_encoder,
            'variables': variables,
            'scaler': scaler
        }
    except Exception as e:
        st.error(f"Error al cargar los modelos: {str(e)}")
        return None

models = load_models()

if models is None:
    st.stop()

# Sidebar para entrada de datos
st.sidebar.header("📋 Datos del Vehículo")

# Función para obtener datos del usuario
def get_user_input():
    age = st.sidebar.slider('Edad del conductor', 18, 80, 30)
    
    cartype = st.sidebar.selectbox('Tipo de vehículo', 
                                 ['Familiar', 'Deportivo', 'Combi', 'Minivan'])
    
    # Convertir a variables dummy
    cartype_combi = 1 if cartype == 'Combi' else 0
    cartype_family = 1 if cartype == 'Familiar' else 0
    cartype_minivan = 1 if cartype == 'Minivan' else 0
    cartype_sport = 1 if cartype == 'Deportivo' else 0
    
    # Crear dataframe con los datos del usuario
    user_data = {
        'age': age,
        'cartype_combi': cartype_combi,
        'cartype_family': cartype_family,
        'cartype_minivan': cartype_minivan,
        'cartype_sport': cartype_sport
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()

# Mostrar datos de entrada
st.subheader("📊 Datos ingresados")
st.write(user_input)

# Preprocesamiento
try:
    # Normalizar la edad
    user_input_normalized = user_input.copy()
    user_input_normalized['age'] = models['scaler'].transform(user_input[['age']])
    
    # Modelos disponibles
    model_options = {
        'Árbol de Decisión': 'tree',
        'K-Vecinos más Cercanos (KNN)': 'knn',
        'Red Neuronal': 'nn',
        'Regresión Logística': 'lr'
    }
    
    selected_model_name = st.selectbox("Seleccione el modelo a utilizar:", list(model_options.keys()))
    selected_model = models[model_options[selected_model_name]]
    
    # Predicción
    if st.button('Predecir Riesgo'):
        with st.spinner('Realizando predicción...'):
            try:
                # Realizar predicción
                prediction = selected_model.predict(user_input_normalized)
                prediction_proba = selected_model.predict_proba(user_input_normalized)
                
                # Decodificar la predicción
                risk_level = models['label_encoder'].inverse_transform(prediction)[0]
                
                # Mostrar resultados
                st.subheader("🔮 Resultado de la Predicción")
                st.markdown(f"**Nivel de riesgo predicho:** {risk_level}")
                
                # Mostrar probabilidades
                st.subheader("📈 Probabilidades por Categoría")
                proba_df = pd.DataFrame({
                    'Categoría': models['label_encoder'].classes_,
                    'Probabilidad': prediction_proba[0]
                })
                st.bar_chart(proba_df.set_index('Categoría'))
                
                # Sección de análisis del modelo
                st.subheader("📊 Análisis del Modelo")
                
                # Mostrar importancia de características (si es árbol de decisión)
                if selected_model_name == 'Árbol de Decisión':
                    st.markdown("**Importancia de las características:**")
                    feature_importance = pd.DataFrame({
                        'Característica': models['variables'],
                        'Importancia': selected_model.feature_importances_
                    }).sort_values('Importancia', ascending=False)
                    st.write(feature_importance)
                    
                    # Mostrar árbol de decisión
                    st.markdown("**Estructura del árbol de decisión:**")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    plot_tree(selected_model, 
                            feature_names=models['variables'], 
                            class_names=models['label_encoder'].classes_, 
                            filled=True, 
                            rounded=True, 
                            ax=ax)
                    st.pyplot(fig)
                
                # Mostrar curva ROC (requeriría datos de prueba)
                # Mostrar matriz de confusión (requeriría datos de prueba)
                
            except Exception as e:
                st.error(f"Error al realizar la predicción: {str(e)}")
    
except Exception as e:
    st.error(f"Error en el procesamiento: {str(e)}")

# Sección de información sobre los modelos
st.sidebar.header("ℹ️ Acerca de los Modelos")
st.sidebar.markdown("""
Esta aplicación utiliza cuatro modelos de machine learning:
1. **Árbol de Decisión**: Modelo basado en reglas de decisión.
2. **KNN**: Basado en similitud con casos conocidos.
3. **Red Neuronal**: Modelo complejo con capas ocultas.
4. **Regresión Logística**: Modelo lineal para clasificación.
""")

# Notas finales
st.markdown("---")
st.markdown("""
**Nota:** Los resultados son predicciones basadas en modelos de machine learning y deben ser interpretados por un experto.
""")