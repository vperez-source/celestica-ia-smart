import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- 1. CONFIGURACIÓN E INTERFAZ ---
st.set_page_config(page_title="Celestica Frontier AI", layout="wide", page_icon="🏭")
st.title("🏭 Celestica IA: Analizador de Capacidad y Ciclos")

with st.sidebar:
    st.header("⚙️ Parámetros de Ingeniería")
    h_turno = st.number_input("Horas Turno Totales", value=8.0)
    st.divider()
    st.info("La IA detecta automáticamente paradas > 15 min como tiempo no productivo.")

# --- 2. LECTOR DE ALTA COMPATIBILIDAD ---
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
def load_and_clean(file):
    df = leer_xml_celestica(file)
    if df is None or df.empty:
        try:
            file.seek(0)
            df = pd.read_excel(file, header=None)
        except: return None, None

    # Mapeo Semántico de Columnas
    df = df.astype(str)
    header_idx = -1
    for i in range(min(50, len(df))):
        row_str = " ".join(df.iloc[i].astype(str)).lower()
        if 'date' in row_str or 'time' in row_str:
            header_idx = i; break
    if header_idx == -1: return None, None

    df.columns = df.iloc[header_idx]
    df = df[header_idx + 1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()

    # Identificación de columnas clave
    cols = {
        'Fecha': next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None),
        'SN': next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None),
        'Product': next((c for c in df.columns if any(x in c.lower() for x in ['product', 'item'])), 'Producto'),
        'Family': next((c for c in df.columns if any(x in c.lower() for x in ['family', 'familia'])), 'Familia')
    }
    return df, cols

# --- 3. MOTOR DE DEPURACIÓN Y CÁLCULO DE FRONTERA ---
def calcular_metricas_reales(df, cols):
    c_fec = cols['Fecha']
    c_sn = cols['SN']
    
    # A. Limpieza estricta de duplicados por SN
    df[c_fec] = pd.to_datetime(df[c_fec], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn:
        df = df.drop_duplicates(subset=[c_sn], keep='first')
    
    # B. Lógica de De-batching (Reparto de ráfagas)
    # Agrupamos por segundo para identificar cuántas piezas entraron a la vez
    batches = df.groupby(c_fec).size().reset_index(name='piezas_lote')
    batches['gap_total'] = batches[c_fec].diff().dt.total_seconds().fillna(0)
    
    # C. Limpieza de "Silencios" (Si el gap es > 15 min, lo capamos a 60s para no corromper la media)
    # Esto asume que parones largos NO son tiempo de ciclo, sino ineficiencia.
    batches['gap_limpio'] = batches['gap_total'].apply(lambda x: x if x < 900 else 60)
    
    # Tiempo unitario imputado (segundos)
    batches['tc_unitario'] = batches['gap_limpio'] / batches['piezas_lote']
    
    # Filtro de ruidos extremos (< 2s)
    valid_data = batches[batches['tc_unitario'] > 2]['tc_unitario']
    
    if valid_data.empty: return 0, 0, 0, 0

    # D. CÁLCULO DE LOS 3 PILARES
    # 1. TC TEÓRICO: Percentil 15 (La frontera donde el proceso vuela)
    tc_teorico_seg = np.percentile(valid_data, 15)
    
    # 2. TC REAL: Mediana (El ritmo constante del turno)
    tc_real_seg = valid_data.median()
    
    return tc_teorico_seg / 60, tc_real_seg / 60, len(df), batches

# --- 4. DASHBOARD DE RESULTADOS ---
uploaded_file = st.file_uploader("📤 Sube el reporte de trazabilidad (.xls, .xml, .xlsx)", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🤖 Depurando ruido de red y calculando frontera teórica..."):
        df, cols = load_and_clean(uploaded_file)
        
        if df is not None and cols['Fecha']:
            tc_teo, tc_real, total_piezas, df_batches = calcular_metricas_reales(df, cols)
            
            if tc_teo > 0:
                st.success("✅ Análisis de Capacidad Finalizado")
                
                # METRICAS PRINCIPALES
                m1, m2, m3 = st.columns(3)
                m1.metric("⏱️ TC TEÓRICO (Ideal)", f"{tc_teo:.2f} min", 
                          help="Tiempo de ciclo puro sin interferencias (Frontera P15).")
                m2.metric("⏱️ TC REAL (Turno)", f"{tc_real:.2f} min", 
                          delta=f"{((tc_real/tc_teo)-1)*100:.1f}% Variabilidad", delta_color="inverse")
                
                capacidad_teorica = (h_turno * 60) / tc_teo
                m3.metric("📦 Capacidad Turno", f"{int(capacidad_teorica)} uds", 
                          help=f"Capacidad máxima teórica en {h_turno} horas.")

                st.divider()

                # GRAFICA DE DISTRIBUCIÓN GAMMA
                st.subheader("📊 Análisis de Distribución de Ritmos")
                st.caption("La montaña roja es el TC Teórico al que debes aspirar. La azul es tu realidad actual.")
                
                # Limpiamos datos para la gráfica
                fig_data = df_batches[(df_batches['tc_unitario'] > 0) & (df_batches['tc_unitario'] < tc_real * 150)]
                
                fig = px.histogram(fig_data, x="tc_unitario", nbins=100, 
                                 title="Histograma de Tiempos Unitarios (Segundos)",
                                 color_discrete_sequence=['#95a5a6'])
                
                fig.add_vline(x=tc_teo*60, line_color="#e74c3c", line_width=4, annotation_text="TEÓRICO")
                fig.add_vline(x=tc_real*60, line_color="#3498db", line_width=3, annotation_text="REAL")
                
                st.plotly_chart(fig, use_container_width=True)

                # TABLA DE PRODUCTIVIDAD
                st.subheader("📋 Resumen por Familia de Producto")
                resumen = df.groupby([cols['Family'], cols['Product']]).size().reset_index(name='Unidades')
                resumen['Tiempo Est. (h)'] = (resumen['Unidades'] * tc_real) / 60
                st.dataframe(resumen.sort_values('Unidades', ascending=False), use_container_width=True)

            else:
                st.error("No hay datos suficientes para establecer un ritmo lógico.")
        else:
            st.error("No se encontró la columna de fecha en el archivo.")
