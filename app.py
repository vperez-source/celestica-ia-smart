import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN DE LA INTERFAZ ---
st.set_page_config(page_title="Celestica Theoretical Master", layout="wide", page_icon="🛡️")
st.title("🛡️ Celestica IA: Análisis de Frontera Teórica (Unique SN)")
st.markdown("""
**Protocolo de Depuración:** 1. Limpieza de dobles escaneos mediante **Serial Number**.
2. Reparto de carga por lotes (De-batching).
3. Cálculo de la **Frontera Teórica** (Ritmo de flujo sin ineficiencias).
""")

with st.sidebar:
    st.header("⚙️ Ingeniería de Datos")
    p_excelencia = st.slider("Percentil Teórico (Frontera)", 5, 50, 20, 
                             help="Define el ritmo de excelencia (el mejor X% de la producción).")
    st.divider()
    h_turno = st.number_input("Horas Turno", value=8)

# --- 1. MOTOR DE INGESTIÓN (XML/XLS) ---
def parse_xml_tanque(file):
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
def load_and_map_data(file):
    df = parse_xml_tanque(file)
    if df is None or df.empty:
        try:
            file.seek(0)
            df = pd.read_excel(file, header=None)
        except: return None, {}

    df = df.astype(str)
    start_row = -1
    for i in range(min(100, len(df))):
        row_str = " ".join(df.iloc[i].astype(str)).lower()
        if 'date' in row_str or 'time' in row_str:
            start_row = i; break
    
    if start_row == -1: return None, {}

    df.columns = df.iloc[start_row]
    df = df[start_row + 1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()

    # Mapeo Semántico incluyendo Serial Number
    cols = {}
    for c in df.columns:
        cl = c.lower()
        if not cols.get('Fecha') and any(x in cl for x in ['date', 'time', 'fecha']): cols['Fecha'] = c
        if not cols.get('Serial') and any(x in cl for x in ['serial', 'sn', 'unitid']): cols['Serial'] = c
        if not cols.get('Producto') and any(x in cl for x in ['product', 'item']): cols['Producto'] = c
        if not cols.get('Familia') and 'family' in cl: cols['Familia'] = c
    
    return df, cols

# --- 2. ALGORITMO DE FRONTERA CON DEDUPLICACIÓN ---
def analyze_theoretical_flow(df, cols, p_target):
    c_fec = cols['Fecha']
    c_sn = cols.get('Serial')
    
    # A. Conversión y Limpieza
    df[c_fec] = pd.to_datetime(df[c_fec], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[c_fec])
    
    # B. DEDUPLICACIÓN POR SERIAL NUMBER (Elimina doble escaneo)
    original_count = len(df)
    if c_sn:
        # Nos quedamos con la primera vez que se vio el número de serie
        df = df.sort_values(c_fec).drop_duplicates(subset=[c_sn], keep='first')
    unique_count = len(df)
    
    # C. CÁLCULO DE HEARTBEAT (Reparto por lote)
    batches = df.groupby(c_fec).size().reset_index(name='piezas')
    batches['gap'] = batches[c_fec].diff().dt.total_seconds().fillna(0)
    batches['tc_unitario'] = batches['gap'] / batches['piezas']
    
    # Filtro técnico (mínimo 1s físico y máximo 30 min)
    valid_data = batches[(batches['tc_unitario'] > 1) & (batches['tc_unitario'] < 1800)]['tc_unitario']
    
    if valid_data.empty: return 0, 0, batches, original_count, unique_count

    # D. FRONTERA LOG-NORMAL (Percentil de Excelencia)
    tc_teorico_seg = np.percentile(valid_data, p_target)
    tc_real_mediana_seg = valid_data.median()
    
    return tc_teorico_seg / 60, tc_real_mediana_seg / 60, batches, original_count, unique_count

# --- 3. DASHBOARD PRINCIPAL ---
uploaded_file = st.file_uploader("Subir Archivo de Spectrum/SOAC", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🕵️ Deduplicando números de serie y calculando frontera..."):
        df, cols = load_and_map_data(uploaded_file)
        
        if df is not None and cols.get('Fecha'):
            tc_teo, tc_real, batches, total_logs, total_unicos = analyze_theoretical_flow(df, cols, p_excelencia)
            
            if tc_teo > 0:
                st.success(f"✅ Análisis completado: {total_logs - total_unicos} dobles escaneos eliminados.")
                
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("⏱️ TC TEÓRICO", f"{tc_teo:.2f} min", help="Ritmo objetivo eliminando desperdicio.")
                k2.metric("⏱️ TC REAL", f"{tc_real:.2f} min", delta=f"{((tc_real/tc_teo)-1)*100:.1f}% Ineficiencia", delta_color="inverse")
                k3.metric("📦 Unidades Únicas", total_unicos)
                k4.metric("🗑️ Ruido SN", total_logs - total_unicos)

                st.divider()

                # --- VISUALIZACIÓN DE DENSIDAD ---
                st.subheader("📊 Radiografía de la Distribución (Gamma/Log-Normal)")
                st.caption("El 'Pico de Excelencia' (Línea Roja) es tu tiempo de ciclo puro sin interferencias.")
                
                # Gráfico depurado
                fig_data = batches[(batches['tc_unitario'] > 0) & (batches['tc_unitario'] < tc_real*120)]
                fig = px.histogram(fig_data, x="tc_unitario", nbins=80, 
                                 marginal="box", # Añadimos diagrama de caja arriba
                                 color_discrete_sequence=['#2ecc71'],
                                 labels={'tc_unitario': 'Segundos por unidad única'})
                
                fig.add_vline(x=tc_teo*60, line_dash="dash", line_color="red", line_width=4, annotation_text="FRONTERA TEÓRICA")
                fig.add_vline(x=tc_real*60, line_dash="dot", line_color="blue", line_width=2, annotation_text="REALIDAD ACTUAL")
                
                st.plotly_chart(fig, use_container_width=True)

                # --- TABLA DE CAPACIDAD ---
                st.subheader("📈 Proyección de Capacidad")
                cap_teorica = int((h_turno * 60) / tc_teo)
                cap_real = int((h_turno * 60) / tc_real)
                
                st.write(f"Con un ritmo teórico de **{tc_teo:.2f} min**, la capacidad nominal es de **{cap_teorica} unidades** por turno.")
                st.progress(cap_real / cap_teorica)
                st.caption(f"Aprovechamiento actual: {(cap_real/cap_teorica)*100:.1f}% de la capacidad teórica.")

            else:
                st.error("Los datos procesados no permiten establecer una frontera clara. Verifica el formato de fecha.")
        else:
            st.error("No se detectó la columna 'In DateTime' o 'Date'.")
