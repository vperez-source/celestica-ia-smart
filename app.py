import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import io

st.set_page_config(page_title="Celestica IA", layout="wide")

# --- CABECERA Y EXPLICACIÓN ---
st.title("🛡️ Celestica IA: Smart-Trace Analyzer")
st.markdown("""
Esta herramienta usa un algoritmo **Isolation Forest**. 
Cada vez que cambias la sensibilidad, la IA **re-aprende** qué es normal y qué es ruido.
""")
st.markdown("---")

# --- BARRA LATERAL (CONTROLES) ---
with st.sidebar:
    st.header("🧠 Control del Cerebro IA")
    
    # ESTE ES EL CONTROL DE ENTRENAMIENTO
    contamination = st.slider(
        "Sensibilidad de la IA (% de Ruido estimado)", 
        min_value=1, max_value=20, value=5, 
        help="Si subes este valor, la IA será más agresiva borrando datos. Si lo bajas, será más permisiva."
    ) / 100
    
    st.divider()
    st.header("⚙️ Parámetros de Planta")
    h_turno = st.number_input("Horas Turno", value=8)
    m_descanso = st.number_input("Minutos Descanso", value=45)
    eficiencia = st.slider("Eficiencia Objetivo %", 50, 100, 75) / 100

# --- CARGA DE ARCHIVO (Solo XLSX) ---
uploaded_file = st.file_uploader("Sube tu archivo (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        # 1. Lectura Limpia
        df = pd.read_excel(uploaded_file)
        
        # 2. Limpieza básica de columnas
        df.columns = df.columns.astype(str).str.strip()
        
        # Corrección de fecha (Error 2025)
        col_target = 'In DateTime'
        if col_target in df.columns:
            df[col_target] = pd.to_datetime(df[col_target], errors='coerce')
            df.loc[df[col_target].dt.year < 100, col_target] += pd.offsets.DateOffset(years=2000)
            df = df.dropna(subset=[col_target]).sort_values(col_target)
            
            # 3. Preparación de datos para la IA
            # Calculamos el tiempo entre piezas (Gap)
            df['gap_mins'] = df.groupby('Station')[col_target].diff().dt.total_seconds() / 60
            df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())
            
            # --- AQUÍ OCURRE LA MAGIA (ENTRENAMIENTO) ---
            # El modelo se crea y entrena de nuevo con el parámetro que tú le das
            model = IsolationForest(contamination=contamination, random_state=42)
            
            # La IA predice: 1 = Normal, -1 = Anomalía
            df['IA_Status'] = model.fit_predict(df[['gap_mins']])
            
            # Separamos los datos
            df_clean = df[df['IA_Status'] == 1]     # Lo que la IA aceptó
            df_noise = df[df['IA_Status'] == -1]    # Lo que la IA rechazó
            
            # --- CÁLCULOS FINALES ---
            media_ct = df_clean['gap_mins'].mean()
            capacidad = ((h_turno*60 - m_descanso)/media_ct) * eficiencia
            
            # --- VISUALIZACIÓN ---
            
            # Métricas
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cycle Time (IA)", f"{media_ct:.2f} min")
            c2.metric("Capacidad Real", f"{int(capacidad)} uds")
            c3.metric("Datos Aceptados", len(df_clean), delta="Datos Reales")
            c4.metric("Datos Eliminados", len(df_noise), delta_color="inverse", delta="Considerado Ruido")

            # Gráfico Interactivo
            st.subheader("👁️ Visualización del Criterio de la IA")
            st.caption("Los puntos ROJOS son lo que la IA ha aprendido que es 'Ruido' según tu sensibilidad.")
            
            fig = px.scatter(df, x=col_target, y='gap_mins', 
                           color=df['IA_Status'].astype(str),
                           color_discrete_map={'1':'#2ecc71', '-1':'#e74c3c'}, # Verde y Rojo
                           title=f"Entrenamiento con Sensibilidad al {int(contamination*100)}%")
            st.plotly_chart(fig, use_container_width=True)

            # Botón descarga
            st.download_button("📥 Descargar Datos Limpios", df_clean.to_csv(index=False).encode('utf-8'), "celestica_ia_clean.csv")
            
        else:
            st.error("No se encontró la columna 'In DateTime'. Revisa el Excel.")

    except Exception as e:
        st.error(f"Error al procesar: {e}")
