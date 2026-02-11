import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica AI Auto", layout="wide", page_icon="🛡️")
st.title("🛡️ Celestica IA: Advance Tracking Analyzer")

with st.sidebar:
    st.header("⚙️ Configuración")
    contamination = st.slider("Sensibilidad IA (% Ruido)", 1, 25, 5) / 100
    st.divider()
    h_turno = st.number_input("Horas Turno", value=8)
    m_descanso = st.number_input("Minutos Descanso", value=45)
    eficiencia = st.slider("Eficiencia %", 50, 100, 75) / 100

# --- FUNCIÓN DE LECTURA BLINDADA ---
@st.cache_data(ttl=3600) # Guardamos en caché para que sea rápido
def load_data(file):
    try:
        # Intento 1: Calamine (El mejor para Celestica)
        return pd.read_excel(file, engine='calamine')
    except:
        try:
            # Intento 2: OpenPyXL
            file.seek(0)
            return pd.read_excel(file, engine='openpyxl')
        except:
            return None

# --- CEREBRO DE AUTO-MAPEO (La magia) ---
def normalizar_columnas(df):
    """Renombra las columnas automáticamente para que la IA entienda cualquier archivo."""
    df.columns = df.columns.astype(str).str.strip()
    
    # Diccionario de sinónimos (Lo que busca la IA)
    mapa = {
        'FECHA': ['In DateTime', 'Date', 'Time', 'Fecha', 'Hora', 'Timestamp'],
        'ESTACION': ['Station', 'Operation', 'Work Center', 'Estacion', 'Maquina', 'Process'],
        'USUARIO': ['User', 'Operator', 'Name', 'Usuario', 'Empleado', 'Worker']
    }
    
    col_fecha, col_estacion, col_usuario = None, None, None

    # Buscamos la columna de FECHA
    for posible in mapa['FECHA']:
        match = next((c for c in df.columns if posible.lower() in c.lower()), None)
        if match:
            col_fecha = match
            break
            
    # Buscamos la columna de ESTACIÓN
    for posible in mapa['ESTACION']:
        match = next((c for c in df.columns if posible.lower() in c.lower()), None)
        if match:
            col_estacion = match
            break

    # Buscamos la columna de USUARIO
    for posible in mapa['USUARIO']:
        match = next((c for c in df.columns if posible.lower() in c.lower()), None)
        if match:
            col_usuario = match
            break
            
    return df, col_fecha, col_estacion, col_usuario

# --- INTERFAZ PRINCIPAL ---
uploaded_file = st.file_uploader("Sube tu archivo (Excel o Texto)", type=["xlsx", "xls", "txt"])

if uploaded_file:
    df_raw = load_data(uploaded_file)

    if df_raw is not None:
        # 1. AUTO-DETECTAR COLUMNAS
        df, col_f, col_s, col_u = normalizar_columnas(df_raw)

        if not col_f or not col_s:
            st.error("❌ No pude detectar automáticamente las columnas de Fecha o Estación.")
            st.write("Columnas encontradas:", list(df.columns))
            st.stop()
            
        # Feedback discreto (para que sepas qué cogió)
        st.caption(f"✅ Auto-Mapeo: Fecha='{col_f}' | Estación='{col_s}' | Usuario='{col_u}'")

        # 2. PROCESAMIENTO
        try:
            df[col_f] = pd.to_datetime(df[col_f], errors='coerce')
            df.loc[df[col_f].dt.year < 100, col_f] += pd.offsets.DateOffset(years=2000)
            df = df.dropna(subset=[col_f]).sort_values(col_f)

            # Cálculo de Gaps
            df['gap_mins'] = df.groupby(col_s)[col_f].diff().dt.total_seconds() / 60
            df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())

            # 3. IA (ISOLATION FOREST)
            model = IsolationForest(contamination=contamination, random_state=42)
            df['IA_Status'] = model.fit_predict(df[['gap_mins']])
            
            df_clean = df[df['IA_Status'] == 1].copy()
            df_noise = df[df['IA_Status'] == -1]

            # 4. RESULTADOS (KPIs)
            media = df_clean['gap_mins'].mean()
            capacidad = ((h_turno*60 - m_descanso)/media) * eficiencia

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("⏱️ Cycle Time", f"{media:.2f} min")
            k2.metric("📦 Capacidad", f"{int(capacidad)} uds")
            k3.metric("✅ Piezas OK", len(df_clean))
            k4.metric("🗑️ Ruido", len(df_noise), delta_color="inverse")

            st.markdown("---")

            # 5. RANKING AUTOMÁTICO
            if col_u:
                st.subheader("🏆 Ranking de Productividad")
                user_stats = df_clean.groupby(col_u)['gap_mins'].agg(['count', 'mean', 'std']).reset_index()
                user_stats.columns = ['Operario', 'Piezas', 'Velocidad (min)', 'Estabilidad']
                user_stats = user_stats.sort_values('Piezas', ascending=False)

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.dataframe(user_stats.style.background_gradient(subset=['Piezas'], cmap='Greens'), use_container_width=True)
                with c2:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=user_stats['Operario'], y=user_stats['Piezas'], marker_color='#2ecc71'))
                    fig.update_layout(title="Producción por Operario")
                    st.plotly_chart(fig, use_container_width=True)

            # 6. GRÁFICO FINAL
            st.subheader("🔍 Mapa de Anomalías")
            fig_scatter = px.scatter(df, x=col_f, y='gap_mins', color=df['IA_Status'].astype(str),
                                   color_discrete_map={'1': '#2ecc71', '-1': '#e74c3c'},
                                   title="Detección de Anomalías (Rojo)")
            st.plotly_chart(fig_scatter, use_container_width=True)

        except Exception as e:
            st.error(f"Error en el cálculo: {e}")
    else:
        st.error("No se pudo leer el archivo. Prueba formato .txt")
