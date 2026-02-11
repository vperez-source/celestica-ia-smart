import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica AI Intel", layout="wide", page_icon="🧠")
st.title("🧠 Celestica IA: Inteligencia Estadística de Procesos")
st.markdown("""
**Análisis de Distribución Gamma:** El sistema descompone los tiempos de ciclo para encontrar el 
ritmo de flujo real, separando el ruido de sistema y las paradas de larga duración.
""")

with st.sidebar:
    st.header("🏭 Parámetros Globales")
    h_turno = st.number_input("Horas Turno", value=8.0)
    st.divider()
    st.info("La IA está configurada para buscar el 'ritmo de crucero' humano (Moda).")

# --- LECTORES ---
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

# --- CEREBRO: FILTRADO POR DENSIDAD LOG-NORMAL ---
def calcular_metricas_avanzadas(df, cols):
    c_fec = cols['Fecha']
    c_sn = cols['SN']
    
    # 1. Limpieza y Deduplicación
    df[c_fec] = pd.to_datetime(df[c_fec], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn:
        df = df.drop_duplicates(subset=[c_sn], keep='first')
    
    # 2. De-batching: Reparto de tiempos por segundo
    batches = df.groupby(c_fec).size().reset_index(name='piezas_lote')
    batches['gap'] = batches[c_fec].diff().dt.total_seconds().fillna(0)
    batches['tc_unitario'] = batches['gap'] / batches['piezas_lote']
    
    # 3. FILTRO DE REALIDAD (IA):
    # Ignoramos lo menor a 5s (ruido sistema) y lo mayor a 20min (parada clara)
    # Estos límites se usan para ENCONTRAR la montaña, no para borrar datos.
    mask_estudio = (batches['tc_unitario'] > 5) & (batches['tc_unitario'] < 1200)
    data_points = batches[mask_estudio]['tc_unitario'].values
    
    if len(data_points) < 5: return None

    # 4. BUSQUEDA DE LA MODA (Pico de Rendimiento)
    # Usamos KDE sobre el logaritmo para mayor precisión en la frontera
    log_data = np.log(data_points)
    kde = gaussian_kde(log_data)
    x_range = np.linspace(log_data.min(), log_data.max(), 1000)
    y_dens = kde(x_range)
    # El pico en escala logarítmica reconvertido a segundos
    tc_teorico_seg = np.exp(x_range[np.argmax(y_dens)])
    
    # 5. ESTADÍSTICAS COMPARATIVAS
    tc_mediana_seg = np.median(data_points)
    tc_media_seg = np.mean(data_points)
    
    return {
        'teorico': tc_teorico_seg / 60,
        'mediana': tc_mediana_seg / 60,
        'media': tc_media_seg / 60,
        'datos': batches,
        'n_piezas': len(df)
    }

# --- INTERFAZ ---
uploaded_file = st.file_uploader("📤 Sube el reporte Spectrum/SOAC", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🕵️ Analizando la 'Firma Temporal' de la línea..."):
        df, cols = load_and_map(uploaded_file)
        
        if df is not None and cols['Fecha']:
            res = calcular_metricas_avanzadas(df, cols)
            
            if res:
                st.success("✅ Análisis de Patrones Completado")
                
                # KPIs (Triángulo de la Verdad)
                k1, k2, k3, k4 = st.columns(4)
                # El TEÓRICO es la Moda (ritmo puro)
                k1.metric("⏱️ TC TEÓRICO (Moda)", f"{res['teorico']:.2f} min", help="El ritmo más frecuente de los operarios.")
                # El REAL es la Mediana (incluye variabilidad normal)
                k2.metric("⏱️ TC REAL (Mediana)", f"{res['mediana']:.2f} min")
                # Capacidad basada en el TEÓRICO
                cap_teo = (h_turno * 60) / res['teorico']
                k3.metric("📦 Capacidad Nominal", f"{int(cap_teo)} uds")
                k4.metric("📊 Total Unidades", res['n_piezas'])

                st.divider()

                # GRÁFICA DE VALIDACIÓN
                st.subheader("📊 Distribución del Ritmo de Flujo")
                st.caption("La zona de mayor densidad indica dónde se estabiliza la producción.")
                
                # Histograma centrado en la zona de interés
                fig_data = res['datos'][(res['datos']['tc_unitario'] > 0) & (res['datos']['tc_unitario'] < res['mediana']*180)]
                fig = px.histogram(fig_data, x="tc_unitario", nbins=100, 
                                 title="Histograma de Segundos por Pieza",
                                 color_discrete_sequence=['#95a5a6'])
                
                # Añadir las tres líneas para que veas la diferencia
                fig.add_vline(x=res['teorico']*60, line_color="#e74c3c", line_width=4, annotation_text="MODA (Teórico)")
                fig.add_vline(x=res['mediana']*60, line_color="#3498db", line_width=2, annotation_text="Mediana")
                fig.add_vline(x=res['media']*60, line_color="#f1c40f", line_width=1, annotation_text="Media")
                
                st.plotly_chart(fig, use_container_width=True)

                # TABLA POR PRODUCTO
                st.subheader("📋 Detalle por Familia/Producto")
                df_prod = df.groupby([cols['Family'], cols['Product']]).size().reset_index(name='Unidades')
                df_prod['Horas Teóricas'] = (df_prod['Unidades'] * res['teorico']) / 60
                st.dataframe(df_prod.sort_values('Unidades', ascending=False), use_container_width=True)

            else:
                st.error("No se han podido extraer patrones consistentes. ¿El archivo tiene suficientes datos?")
