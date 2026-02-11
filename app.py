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

# --- LECTURA ---
@st.cache_data(ttl=3600)
def load_data(file):
    try:
        return pd.read_excel(file, engine='calamine', header=None) # Leemos SIN cabecera para buscarla luego
    except:
        try:
            file.seek(0)
            return pd.read_excel(file, engine='openpyxl', header=None)
        except:
            try:
                file.seek(0)
                dfs = pd.read_html(file, header=None)
                if len(dfs) > 0: return dfs[0]
            except:
                try:
                    file.seek(0)
                    return pd.read_csv(file, sep='\t', encoding='latin-1', header=None)
                except:
                    return None

# --- CAZADOR DE CABECERAS (NUEVO) ---
def encontrar_cabecera_y_normalizar(df):
    df = df.astype(str) # Convertimos todo a texto para buscar
    
    # Palabras clave que DEBEN estar en la fila de cabecera
    keywords_fecha = ['date', 'time', 'fecha', 'hora', 'timestamp']
    keywords_estacion = ['station', 'operation', 'work', 'estacion', 'maquina', 'process']
    
    start_row = -1
    
    # 1. Buscamos en las primeras 20 filas la fila que tenga ambos conceptos
    for i in range(min(20, len(df))):
        fila = df.iloc[i].str.lower().tolist()
        has_date = any(k in str(val) for val in fila for k in keywords_fecha)
        has_station = any(k in str(val) for val in fila for k in keywords_estacion)
        
        if has_date and has_station:
            start_row = i
            break
    
    # Si no encontramos nada, devolvemos error
    if start_row == -1:
        return None, None, None, None

    # 2. Promocionamos esa fila a Cabecera
    df.columns = df.iloc[start_row]
    df = df[start_row + 1:].reset_index(drop=True)
    
    # 3. Mapeo de columnas (Sinónimos)
    df.columns = df.columns.astype(str).str.strip()
    
    mapa = {
        'FECHA': keywords_fecha + ['in datetime'],
        'ESTACION': keywords_estacion,
        'USUARIO': ['user', 'operator', 'name', 'usuario', 'empleado', 'created by']
    }
    
    col_f, col_s, col_u = None, None, None

    for col in df.columns:
        col_lower = col.lower()
        if not col_f and any(k in col_lower for k in mapa['FECHA']): col_f = col
        if not col_s and any(k in col_lower for k in mapa['ESTACION']): col_s = col
        if not col_u and any(k in col_lower for k in mapa['USUARIO']): col_u = col

    return df, col_f, col_s, col_u

# --- APP ---
uploaded_file = st.file_uploader("Sube tu archivo", type=["xlsx", "xls", "txt"])

if uploaded_file:
    with st.spinner("⏳ Escaneando archivo en busca de datos..."):
        df_raw = load_data(uploaded_file)

        if df_raw is not None:
            # 1. BUSCAR CABECERA AUTOMÁTICAMENTE
            df, col_f, col_s, col_u = encontrar_cabecera_y_normalizar(df_raw)

            if df is None or not col_f or not col_s:
                st.error("❌ No pude encontrar dónde empiezan los datos.")
                st.write("👀 Primeras 5 filas crudas (para depurar):")
                st.write(df_raw.head())
                st.stop()
                
            st.success(f"✅ Datos encontrados. Columnas detectadas: Fecha='{col_f}' | Estación='{col_s}'")

            # 2. PROCESAMIENTO
            try:
                df[col_f] = pd.to_datetime(df[col_f], errors='coerce')
                df.loc[df[col_f].dt.year < 100, col_f] += pd.offsets.DateOffset(years=2000)
                df = df.dropna(subset=[col_f]).sort_values(col_f)

                if df.empty:
                    st.error("⚠️ No hay fechas válidas.")
                    st.stop()

                # Gaps
                df['gap_mins'] = df.groupby(col_s)[col_f].diff().dt.total_seconds() / 60
                df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())

                # IA
                model = IsolationForest(contamination=contamination, random_state=42)
                df['IA_Status'] = model.fit_predict(df[['gap_mins']])
                
                df_clean = df[df['IA_Status'] == 1].copy()
                df_noise = df[df['IA_Status'] == -1]

                # KPIs
                media = df_clean['gap_mins'].mean()
                capacidad = ((h_turno*60 - m_descanso)/media) * eficiencia

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Cycle Time", f"{media:.2f} min")
                k2.metric("Capacidad", f"{int(capacidad)} uds")
                k3.metric("OK", len(df_clean))
                k4.metric("Ruido", len(df_noise), delta_color="inverse")

                st.markdown("---")

                # GRÁFICAS
                if col_u:
                    st.subheader("🏆 Ranking Operarios")
                    user_stats = df_clean.groupby(col_u)['gap_mins'].agg(['count', 'mean']).reset_index()
                    user_stats.columns = ['Operario', 'Piezas', 'Velocidad']
                    user_stats = user_stats.sort_values('Piezas', ascending=False)
                    
                    c1, c2 = st.columns([1,1])
                    with c1:
                        st.dataframe(user_stats.style.background_gradient(subset=['Piezas'], cmap='Greens'), use_container_width=True)
                    with c2:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=user_stats['Operario'], y=user_stats['Piezas'], name='Piezas', marker_color='#2ecc71', yaxis='y'))
                        fig.add_trace(go.Scatter(x=user_stats['Operario'], y=user_stats['Velocidad'], name='Velocidad', marker_color='#e74c3c', yaxis='y2'))
                        fig.update_layout(yaxis2=dict(overlaying='y', side='right'), title="Piezas vs Velocidad")
                        st.plotly_chart(fig, use_container_width=True)

                st.subheader("Mapa IA")
                fig = px.scatter(df, x=col_f, y='gap_mins', color=df['IA_Status'].astype(str), 
                               color_discrete_map={'1': '#2ecc71', '-1': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error cálculo: {e}")
        else:
            st.error("No se pudo leer.")
