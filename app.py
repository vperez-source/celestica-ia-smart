import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- 1. CONFIGURACIÓN VISUAL (ESTILO ANTERIOR) ---
st.set_page_config(page_title="Celestica AI Smart-Tracker", layout="wide", page_icon="🏭")
st.title("🏭 Celestica IA: Smart Heartbeat Analyzer")
st.markdown("<style>.stMetric { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }</style>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuración")
    st.info("La IA busca el pico de la distribución Gamma para obtener el tiempo de ciclo puro.")
    h_turno = st.number_input("Horas Turno", value=8)
    eficiencia_oee = st.slider("Eficiencia Objetivo %", 50, 100, 85) / 100

# --- 2. MOTOR DE INGESTIÓN (TANQUE XML) ---
def parse_xml_legacy(file):
    try:
        content = file.getvalue().decode('latin-1', errors='ignore')
        soup = BeautifulSoup(content, 'lxml-xml')
        data = []
        for row in soup.find_all(['Row', 'ss:Row']):
            cells = [c.get_text(strip=True) for c in row.find_all(['Cell', 'ss:Cell'])]
            if any(cells): data.append(cells)
        return pd.DataFrame(data)
    except: return None

@st.cache_data(ttl=3600)
def load_data(file):
    df = parse_xml_legacy(file)
    if df is None or df.empty:
        try:
            file.seek(0)
            df = pd.read_excel(file, header=None)
        except: return None, None

    # Mapeo Semántico
    df = df.astype(str)
    start_row = -1
    for i in range(min(50, len(df))):
        row_str = " ".join(df.iloc[i].astype(str)).lower()
        if 'date' in row_str or 'time' in row_str:
            start_row = i; break
    if start_row == -1: return None, None
    df.columns = df.iloc[start_row]
    df = df[start_row + 1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()

    # Identificación de columnas críticas
    cols = {}
    for c in df.columns:
        cl = c.lower()
        if not cols.get('Fecha') and any(x in cl for x in ['date', 'time', 'fecha', 'timestamp']): cols['Fecha'] = c
        if not cols.get('Serial') and any(x in cl for x in ['serial', 'sn', 'unitid']): cols['Serial'] = c
        if not cols.get('Product') and any(x in cl for x in ['product', 'item', 'part']): cols['Product'] = c
    return df, cols

# --- 3. CEREBRO IA: CÁLCULO DE MÁXIMA DENSIDAD ---
def analyze_industrial_rhythm(df, cols):
    c_fec = cols['Fecha']
    c_sn = cols.get('Serial')
    
    # A. Limpieza de fechas y duplicados por Serial Number
    df[c_fec] = pd.to_datetime(df[c_fec], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn:
        df = df.drop_duplicates(subset=[c_sn], keep='first')
    
    # B. De-batching y Imputación
    batches = df.groupby(c_fec).size().reset_index(name='piezas')
    batches['gap'] = batches[c_fec].diff().dt.total_seconds().fillna(0)
    
    # REGLA DE ORO: Si el gap es > 15 min, es una parada, no tiempo de preparación.
    # Limitamos el gap máximo a repartir para no inflar el TC.
    batches['gap_capped'] = batches['gap'].apply(lambda x: x if x < 900 else 60) # Cap a 60s si es parada larga
    batches['tc_unitario'] = batches['gap_capped'] / batches['piezas']
    
    # C. Filtro de Realidad Física (Solo datos entre 5s y 10min)
    valid_data = batches[(batches['tc_unitario'] > 5) & (batches['tc_unitario'] < 600)]['tc_unitario']
    
    if len(valid_data) < 5: return 0, batches, 0

    # D. MODA POR KDE (Búsqueda del pico de la montaña)
    kde = gaussian_kde(valid_data)
    x = np.linspace(valid_data.min(), valid_data.max(), 1000)
    y = kde(x)
    mode_seconds = x[np.argmax(y)]
    
    return mode_seconds / 60, batches, mode_seconds

# --- 4. INTERFAZ Y DASHBOARD ---
uploaded_file = st.file_uploader("Subir archivo de Spectrum/SOAC", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🧠 Depurando ruido y buscando ritmo real..."):
        df, cols = load_data(uploaded_file)
        
        if df is not None and cols.get('Fecha'):
            tc_teo_min, batches, modo_s = analyze_industrial_rhythm(df, cols)
            
            if tc_teo_min > 0:
                st.success(f"✅ Análisis completado con éxito.")
                
                # KPIs PRINCIPALES
                k1, k2, k3 = st.columns(3)
                k1.metric("⏱️ TC TEÓRICO (Moda)", f"{tc_teo_min:.2f} min", 
                          help="El ritmo más frecuente detectado, equivalente al 'Peak Performance'.")
                
                cap_8h = (h_turno * 60 / tc_teo_min) * eficiencia_oee
                k2.metric("📦 Capacidad Est. Turno", f"{int(cap_8h)} uds", 
                          help=f"Basado en un OEE del {eficiencia_oee*100}%")
                
                k3.metric("📊 Muestras Únicas", f"{len(df)} piezas")

                st.divider()

                # GRÁFICA DE DISTRIBUCIÓN
                st.subheader("📈 Mapa de Densidad de Ritmo")
                st.caption("La zona de máxima altura representa el tiempo de ciclo real sin paradas.")
                
                # Preparamos histograma depurado
                fig_data = batches[batches['tc_unitario'] < (modo_s * 4)]
                fig = px.histogram(fig_data, x="tc_unitario", nbins=60, 
                                 title="Frecuencia de Ciclos (Segundos)",
                                 color_discrete_sequence=['#2ecc71'],
                                 labels={'tc_unitario': 'Segundos por unidad'})
                fig.add_vline(x=modo_s, line_dash="dash", line_color="#e74c3c", line_width=4, 
                             annotation_text=f"TIEMPO REAL: {modo_s:.1f}s")
                st.plotly_chart(fig, use_container_width=True)

                # TABLA POR PRODUCTO
                if cols.get('Product'):
                    st.subheader("📦 Rendimiento por Producto")
                    prod_stats = df.groupby(cols['Product']).size().reset_index(name='Unidades')
                    prod_stats['Tiempo Est. (horas)'] = (prod_stats['Unidades'] * tc_teo_min) / 60
                    st.dataframe(prod_stats.sort_values('Unidades', ascending=False), use_container_width=True)
            else:
                st.error("No se pudo detectar un patrón de tiempo lógico. ¿Las fechas son correctas?")
        else:
            st.error("No se encontró la cabecera 'Date' o 'In DateTime'.")
