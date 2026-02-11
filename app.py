import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica AI Frontier", layout="wide", page_icon="🏭")
st.title("🏭 Celestica IA: Análisis de Ciclo Teórico Realista")

with st.sidebar:
    st.header("⚙️ Ingeniería de Procesos")
    h_turno = st.number_input("Horas Turno Totales", value=8.0)
    st.divider()
    st.markdown("### 🛡️ Filtros Anti-Ruido")
    min_fisico = st.slider("Mínimo Físico (segundos)", 10, 120, 45, 
                           help="Ninguna pieza puede tardar menos de esto. Evita que el TC baje a 0.")
    st.info("La IA ignorará ráfagas por debajo de este tiempo para el cálculo del Teórico.")

# --- 2. LECTOR DE DATOS ---
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
def load_and_map(file):
    df = parse_xml_tanque(file)
    if df is None or df.empty:
        try:
            file.seek(0)
            df = pd.read_excel(file, header=None)
        except: return None, None

    df = df.astype(str)
    header_idx = -1
    for i in range(min(100, len(df))):
        row_str = " ".join(df.iloc[i].astype(str)).lower()
        if 'date' in row_str or 'time' in row_str:
            header_idx = i; break
    if header_idx == -1: return None, None

    df.columns = df.iloc[header_idx]
    df = df[header_idx + 1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()

    cols = {
        'Fecha': next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None),
        'SN': next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None),
        'Product': next((c for c in df.columns if any(x in c.lower() for x in ['product', 'item'])), 'Producto'),
        'Family': next((c for c in df.columns if any(x in c.lower() for x in ['family', 'familia'])), 'Familia')
    }
    return df, cols

# --- 3. MOTOR DE CÁLCULO DE PRECISIÓN ---
def calcular_frontera_limpia(df, cols, min_sec):
    c_fec = cols['Fecha']
    c_sn = cols['SN']
    
    # A. Limpieza de base: Fechas y Unicidad de Serial Number
    df[c_fec] = pd.to_datetime(df[c_fec], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn:
        df = df.drop_duplicates(subset=[c_sn], keep='first')
    
    # B. Agrupación por lotes de sistema (Batching)
    batches = df.groupby(c_fec).size().reset_index(name='piezas_lote')
    batches['gap_bruto'] = batches[c_fec].diff().dt.total_seconds().fillna(0)
    
    # C. IMPUTACIÓN CON FILTRO DE "SUELO FÍSICO"
    # Calculamos el tiempo unitario: gap / piezas
    batches['tc_unitario'] = batches['gap_bruto'] / batches['piezas_lote']
    
    # D. DEPURA EL RUIDO: 
    # 1. Ignoramos lo que sea menor al "Mínimo Físico" (ruido de ráfaga)
    # 2. Ignoramos paradas > 20 min (no es tiempo de ciclo, es ineficiencia)
    valid_data = batches[(batches['tc_unitario'] >= min_sec) & (batches['tc_unitario'] < 1200)]['tc_unitario']
    
    if len(valid_data) < 5:
        # Fallback si el filtro es muy agresivo: usar mediana de todo lo mayor a 0
        valid_data = batches[batches['tc_unitario'] > 5]['tc_unitario']
        if valid_data.empty: return 0, 0, 0, batches

    # E. CÁLCULO DE LA MODA (Pico de la montaña Gamma)
    # Usamos KDE para encontrar el ritmo de "Flow"
    kde = gaussian_kde(valid_data)
    x_range = np.linspace(valid_data.min(), valid_data.max(), 1000)
    y_dens = kde(x_range)
    tc_teorico_seg = x_range[np.argmax(y_dens)]
    
    # TC REAL: Es el promedio de los tiempos que están en la zona productiva
    tc_real_seg = valid_data.median()
    
    return tc_teorico_seg / 60, tc_real_seg / 60, len(df), batches, tc_teorico_seg

# --- 4. DASHBOARD ---
uploaded_file = st.file_uploader("📤 Sube el archivo de Spectrum/SOAC", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🕵️ Filtrando ráfagas y buscando frontera física..."):
        df, cols = load_and_map(uploaded_file)
        
        if df is not None and cols['Fecha']:
            tc_teo, tc_real, total_piezas, df_batches, modo_s = calcular_frontera_limpia(df, cols, min_fisico)
            
            if tc_teo > 0:
                st.success("✅ Análisis de Ingeniería Completado")
                
                # KPIs (Diseño Limpio)
                k1, k2, k3 = st.columns(3)
                k1.metric("⏱️ TC TEÓRICO (Target)", f"{tc_teo:.2f} min", 
                          help=f"Ritmo puro detectado ({modo_s:.1f}s). Ignora ráfagas de sistema.")
                k2.metric("⏱️ TC REAL (Mediana)", f"{tc_real:.2f} min", 
                          delta=f"{((tc_real/tc_teo)-1)*100:.1f}% Desvío", delta_color="inverse")
                
                capacidad_teorica = (h_turno * 60) / tc_teo
                k3.metric("📦 Capacidad (100%)", f"{int(capacidad_teorica)} uds", 
                          help="Capacidad máxima si la línea trabajara siempre al ritmo teórico.")

                st.divider()

                # GRÁFICA DE FRECUENCIA
                st.subheader("📊 Distribución del Ritmo de Trabajo")
                st.caption(f"La IA ha detectado que el ritmo más repetido es de {modo_s:.1f} segundos.")
                
                fig_data = df_batches[(df_batches['tc_unitario'] > 0) & (df_batches['tc_unitario'] < tc_real * 180)]
                fig = px.histogram(fig_data, x="tc_unitario", nbins=100, 
                                 title="Densidad de Tiempos Unitarios (Segundos)",
                                 color_discrete_sequence=['#3498db'])
                fig.add_vline(x=modo_s, line_dash="dash", line_color="red", line_width=4, annotation_text="PICO TEÓRICO")
                st.plotly_chart(fig, use_container_width=True)

                # TABLA PRODUCTO
                st.subheader("📋 Desglose por Familia y Producto")
                resumen = df.groupby([cols['Family'], cols['Product']]).size().reset_index(name='Unidades')
                resumen['Tiempo Est. (h)'] = (resumen['Unidades'] * tc_teo) / 60
                st.dataframe(resumen.sort_values('Unidades', ascending=False), use_container_width=True)

            else:
                st.error("No se pudo detectar un flujo lógico. Prueba a bajar el 'Mínimo Físico' en la barra lateral.")
        else:
            st.error("No se encontró la columna de fecha.")
