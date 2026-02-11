import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- 1. CONFIGURACIÓN E INTERFAZ ---
st.set_page_config(page_title="Celestica Smart Tracker AI", layout="wide", page_icon="🏭")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🏭 Celestica AI: Smart Tracker & Heartbeat Analyzer")
st.info("Algoritmo de **Imputación de Carga** activo: Distribuyendo tiempos de preparación en lotes de Spectrum/SOAC.")

with st.sidebar:
    st.header("⚙️ Parámetros de Planta")
    h_turno = st.number_input("Horas de Turno", value=8)
    oee_target = st.slider("Eficiencia Objetivo (OEE) %", 50, 100, 85) / 100
    st.divider()
    st.markdown("### 🧠 Lógica IA")
    st.caption("El sistema detecta automáticamente la 'Moda' (pico de rendimiento) para filtrar paradas y ruido de sistema.")

# --- 2. MOTOR DE INGESTIÓN BLINDADA (Fase A) ---
def parse_xml_2003(file):
    try:
        content = file.getvalue().decode('latin-1', errors='ignore')
        soup = BeautifulSoup(content, 'xml')
        data = []
        for row in soup.find_all(['Row', 'ss:Row']):
            cells = [cell.get_text(strip=True) for cell in row.find_all(['Cell', 'ss:Cell'])]
            if any(cells): data.append(cells)
        return pd.DataFrame(data)
    except Exception: return None

@st.cache_data(ttl=3600)
def load_and_clean_data(file):
    # Intentar XML Legacy primero
    df = parse_xml_2003(file)
    if df is None or df.empty:
        try:
            file.seek(0)
            df = pd.read_excel(file, engine='calamine', header=None)
        except:
            file.seek(0)
            df = pd.read_csv(file, sep='\t', encoding='latin-1', header=None)
    
    if df is None or df.empty: return None, None

    # Mapeo Semántico (Fase B)
    df = df.astype(str)
    start_row = -1
    for i in range(min(50, len(df))):
        row_lower = df.iloc[i].str.lower().tolist()
        if any(x in str(v) for v in row_lower for x in ['date', 'time', 'station', 'product']):
            start_row = i
            break
    
    if start_row == -1: return None, None

    df.columns = df.iloc[start_row]
    df = df[start_row + 1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()

    # Normalización de Columnas Críticas
    cols_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'date' in cl or 'time' in cl: cols_map['Fecha'] = c
        elif 'product' in cl or 'item' in cl: cols_map['Producto'] = c
        elif 'family' in cl: cols_map['Familia'] = c
        elif 'user' in cl or 'operator' in cl: cols_map['Usuario'] = c
        elif 'station' in cl or 'oper' in cl: cols_map['Estacion'] = c

    # Fallbacks de seguridad
    if 'Fecha' not in cols_map: return None, None
    if 'Producto' not in cols_map: df['Producto'] = 'N/A'; cols_map['Producto'] = 'Producto'
    if 'Familia' not in cols_map: df['Familia'] = 'N/A'; cols_map['Familia'] = 'Familia'
    if 'Usuario' not in cols_map: df['Usuario'] = 'VALUODC1'; cols_map['Usuario'] = 'Usuario'
    
    return df, cols_map

# --- 3. LÓGICA DE IMPUTACIÓN Y ESTADÍSTICA (Fase C y D) ---
def analyze_heartbeat(df, cols):
    c_fec = cols['Fecha']
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce')
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)

    # 1. Agrupación por Segundo (Batch Detection)
    batches = df.groupby(c_fec).size().reset_index(name='Piezas_Batch')
    
    # 2. Imputación de Carga: $$CT_{unitario} = \frac{\Delta T_{pre-lote}}{N_{piezas}}$$
    batches['Gap_Sec'] = batches[c_fec].diff().dt.total_seconds().fillna(0)
    batches['CT_Unitario_Sec'] = batches['Gap_Sec'] / batches['Piezas_Batch']

    # 3. Filtro de Realidad (Ignorar paradas > 45 min y tiempos de 0)
    valid_data = batches[(batches['CT_Unitario_Sec'] > 0.5) & (batches['CT_Unitario_Sec'] < 2700)]['CT_Unitario_Sec']

    if len(valid_data) < 10: return 0, batches, 0

    # 4. Kernel Density Estimation (Búsqueda de la Moda)
    kde = gaussian_kde(valid_data)
    x_range = np.linspace(valid_data.min(), valid_data.max(), 1000)
    y_dens = kde(x_range)
    mode_sec = x_range[np.argmax(y_dens)]
    
    return mode_sec / 60, batches, mode_sec

# --- 4. FLUJO PRINCIPAL ---
uploaded_file = st.file_uploader("📤 Arrastra aquí el reporte de Spectrum/SOAC (.xls, .xml, .xlsx)", type=["xls", "xml", "xlsx", "csv"])

if uploaded_file:
    with st.spinner("🧠 Analizando patrones de latido (Heartbeat)..."):
        df_raw, cols = load_and_clean_data(uploaded_file)
        
        if df_raw is not None:
            ct_real_min, df_batches, modo_s = analyze_heartbeat(df_raw, cols)
            
            if ct_real_min > 0:
                # KPIs
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("⏱️ TC Real (Moda)", f"{ct_real_min:.2f} min")
                
                capacidad = (h_turno * 60 / ct_real_min) * oee_target
                c2.metric("📦 Capacidad Turno", f"{int(capacidad)} uds")
                c3.metric("📊 Total Piezas", f"{len(df_raw)}")
                
                # Cálculo de tiempo de inactividad detectado
                total_time_file = (df_raw[cols['Fecha']].max() - df_raw[cols['Fecha']].min()).total_seconds() / 3600
                c4.metric("⏳ Horas en Archivo", f"{total_time_file:.1f}h")

                st.divider()

                # VISUALIZACIONES
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.subheader("📈 Distribución de Ritmos Detectados")
                    # Mostrar el histograma de densidad para validar la Moda
                    fig_hist = px.histogram(df_batches[df_batches['CT_Unitario_Sec'] < (modo_s * 5)], 
                                          x="CT_Unitario_Sec", nbins=50, 
                                          title="Frecuencia de Tiempos de Ciclo (Suelo de Ruido Filtrado)",
                                          color_discrete_sequence=['#1f77b4'],
                                          labels={'CT_Unitario_Sec': 'Segundos por Pieza'})
                    fig_hist.add_vline(x=modo_s, line_dash="dash", line_color="red", 
                                     annotation_text=f"Ritmo de Crucero: {modo_s:.1f}s")
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col_right:
                    st.subheader("📦 Volumen por Familia")
                    fam_data = df_raw[cols['Familia']].value_counts().reset_index()
                    fig_pie = px.pie(fam_data, values='count', names=cols['Familia'], hole=0.4,
                                   color_discrete_sequence=px.colors.sequential.RdBu)
                    st.plotly_chart(fig_pie, use_container_width=True)

                # TABLA DE AUDITORÍA
                with st.expander("🔍 Ver Auditoría de Lotes e Imputación"):
                    st.dataframe(df_batches.sort_values(cols['Fecha'], ascending=False).head(100), use_container_width=True)
            else:
                st.error("No se han podido detectar suficientes intervalos de tiempo para calcular un ciclo realista.")
        else:
            st.error("Error al procesar el archivo. Asegúrate de que contiene columnas de 'Date' y 'Station'.")
