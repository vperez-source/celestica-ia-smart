import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Self-Adaptive AI", layout="wide", page_icon="🤖")
st.title("🤖 Celestica IA: Smart-Tracker Autoadaptativo")
st.markdown("""
**Modo Inteligente:** La IA detecta automáticamente el ruido de sistema y las paradas. 
Busca la 'Frontera de Eficiencia' (tu ritmo de 110s) analizando la densidad de la distribución Gamma.
""")

# --- 1. MOTOR DE INGESTIÓN ---
def leer_xml_celestica(file):
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
    df = leer_xml_celestica(file)
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

# --- 2. CEREBRO IA: FILTRADO DE DENSIDAD AUTO-ADAPTATIVO ---
def calcular_ciclo_ia(df, cols):
    c_fec = cols['Fecha']
    c_sn = cols['SN']
    
    # A. Limpieza Base
    df[c_fec] = pd.to_datetime(df[c_fec], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn:
        df = df.drop_duplicates(subset=[c_sn], keep='first')
    
    # B. Cálculo de Gaps e Imputación Proporcional
    batches = df.groupby(c_fec).size().reset_index(name='piezas_lote')
    batches['gap_bruto'] = batches[c_fec].diff().dt.total_seconds().fillna(0)
    batches['tc_unitario'] = batches['gap_bruto'] / batches['piezas_lote']
    
    # C. DETECCIÓN AUTOMÁTICA DE RUIDO (IA)
    # Filtramos ráfagas de red (< 2s) y paradas masivas (> 1h) para el análisis inicial
    raw_times = batches[(batches['tc_unitario'] > 2) & (batches['tc_unitario'] < 3600)]['tc_unitario'].values
    
    if len(raw_times) < 10: return 0, 0, 0, batches, 0

    # D. LOCALIZACIÓN DE LA MODA (Pico de la Montaña)
    # Usamos Gaussian KDE para encontrar el ritmo más frecuente (donde están tus 110s)
    kde = gaussian_kde(raw_times)
    # Buscamos el pico en un rango lógico para procesos humanos
    x_test = np.linspace(raw_times.min(), raw_times.max(), 1000)
    y_test = kde(x_test)
    tc_teorico_seg = x_test[np.argmax(y_test)]
    
    # E. REFINAMIENTO DE "FRONTERA"
    # El TC Teórico es el pico, pero el TC Real de Turno debe ser la media de los datos 
    # que están dentro de la campana de ese pico, ignorando la cola larga.
    # Definimos la zona de "Producción Real" como +/- 50% alrededor del pico.
    zona_productiva = batches[
        (batches['tc_unitario'] > tc_teorico_seg * 0.5) & 
        (batches['tc_unitario'] < tc_teorico_seg * 2.5)
    ]['tc_unitario']
    
    tc_real_seg = zona_productiva.mean() if not zona_productiva.empty else tc_teorico_seg
    
    return tc_teorico_seg / 60, tc_real_seg / 60, len(df), batches, tc_teorico_seg

# --- 3. DASHBOARD ---
uploaded_file = st.file_uploader("📤 Sube el archivo (Spectrum/SOAC)", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🕵️ La IA está localizando el ritmo de producción real..."):
        df, cols = load_and_map(uploaded_file)
        
        if df is not None and cols['Fecha']:
            tc_teo, tc_real, total_piezas, df_batches, modo_s = calcular_ciclo_ia(df, cols)
            
            if tc_teo > 0:
                st.success("✅ Análisis Completado: Ritmo de Flujo Detectado.")
                
                # KPIs
                k1, k2, k3 = st.columns(3)
                k1.metric("⏱️ TC TEÓRICO (Flow)", f"{tc_teo:.2f} min", 
                          help=f"Ritmo puro detectado en el pico de la distribución ({modo_s:.1f}s).")
                k2.metric("⏱️ TC REAL (Turno)", f"{tc_real:.2f} min", 
                          delta=f"{((tc_real/tc_teo)-1)*100:.1f}% Ineficiencia", delta_color="inverse")
                
                # Capacidad en 8 horas basada en el teórico
                cap_teorica = (8 * 60) / tc_teo
                k3.metric("📦 Capacidad Nominal", f"{int(cap_teorica)} uds", 
                          help="Capacidad máxima si se mantuviera el ritmo de flujo detectado.")

                st.divider()

                # GRÁFICA DE FRECUENCIA
                st.subheader("📊 Distribución del Ritmo Real vs. Paradas")
                st.caption(f"La IA ha ignorado la 'cola' de ineficiencias y se ha centrado en el pico de **{modo_s:.1f} segundos**.")
                
                # Histograma centrado en la zona de trabajo
                fig_data = df_batches[(df_batches['tc_unitario'] > 0) & (df_batches['tc_unitario'] < tc_teo * 300)]
                fig = px.histogram(fig_data, x="tc_unitario", nbins=100, 
                                 title="Frecuencia de Ciclos Unitarios (Segundos)",
                                 color_discrete_sequence=['#3498db'])
                fig.add_vline(x=modo_s, line_dash="dash", line_color="red", line_width=4, annotation_text="PICO REAL")
                st.plotly_chart(fig, use_container_width=True)

                # TABLA PRODUCTO
                st.subheader("📋 Resumen de Producción")
                resumen = df.groupby([cols['Family'], cols['Product']]).size().reset_index(name='Unidades')
                resumen['Tiempo Estimado (h)'] = (resumen['Unidades'] * tc_teo) / 60
                st.dataframe(resumen.sort_values('Unidades', ascending=False), use_container_width=True)

            else:
                st.error("No se pudo detectar un patrón de producción consistente.")
        else:
            st.error("No se detectó columna de fecha.")
