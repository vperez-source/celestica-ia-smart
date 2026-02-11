import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Smart-Tracker Ultra", layout="wide", page_icon="🛡️")
st.title("🛡️ Celestica IA: Smart-Tracker (Versión Blindada)")

# --- INTERFAZ DE USUARIO ---
with st.sidebar:
    st.header("⚙️ Ajustes de Proceso")
    h_turno = st.number_input("Horas Turno", value=8)
    oee_target = st.slider("Eficiencia Objetivo %", 50, 100, 85) / 100
    st.divider()
    st.info("Esta versión incluye un motor de rescate si los datos son muy irregulares.")

# --- LECTOR XML RESILIENTE ---
def leer_xml_profesional(file):
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
    df = leer_xml_profesional(file)
    if df is None or df.empty:
        try:
            file.seek(0)
            df = pd.read_excel(file, header=None)
        except: return None, None

    df = df.astype(str)
    start_row = -1
    for i in range(min(100, len(df))):
        row_str = " ".join(df.iloc[i].astype(str)).lower()
        if any(x in row_str for x in ['date', 'time', 'fecha', 'productid']):
            start_row = i; break
            
    if start_row == -1: return None, None

    df.columns = df.iloc[start_row]
    df = df[start_row + 1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()

    col_fec = next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha', 'timestamp'])), None)
    col_sn = next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None)
    col_prod = next((c for c in df.columns if any(x in c.lower() for x in ['product', 'item', 'partid'])), None)
    
    return df, {'Fecha': col_fec, 'Serial': col_sn, 'Producto': col_prod}

# --- CEREBRO: MOTOR DE CÁLCULO DUAL ---
def analizar_ritmo_seguro(df, cols):
    c_fec = cols['Fecha']
    c_sn = cols['Serial']
    
    # 1. Limpieza de Fechas (Múltiples formatos)
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    
    # 2. Deduplicación por Serial Number
    if c_sn:
        df = df.drop_duplicates(subset=[c_sn], keep='first')
    
    total_piezas = len(df)
    if total_piezas < 2: return 0, 0, None, total_piezas

    # 3. Cálculo de Gaps e Imputación
    batches = df.groupby(c_fec).size().reset_index(name='piezas')
    batches['gap'] = batches[c_fec].diff().dt.total_seconds().fillna(0)
    
    # Imputación proporcional (repartimos el tiempo de espera)
    batches['tc_unitario'] = batches['gap'] / batches['piezas']
    
    # 4. INTENTO A: MODA ESTADÍSTICA (Pico de Rendimiento)
    # Filtramos para buscar el "ritmo de flujo" (entre 0.5s y 15min)
    valid_data = batches[(batches['tc_unitario'] > 0.5) & (batches['tc_unitario'] < 900)]['tc_unitario']
    
    metodo = "Moda IA (Pico)"
    try:
        if len(valid_data) > 10:
            kde = gaussian_kde(valid_data)
            x = np.linspace(valid_data.min(), valid_data.max(), 1000)
            y = kde(x)
            tc_segundos = x[np.argmax(y)]
        else:
            # INTENTO B: MEDIANA DE FLUJO
            tc_segundos = valid_data.median() if not valid_data.empty else 0
            metodo = "Mediana de Flujo"
            
        # INTENTO C: RESCATE GLOBAL (Si todo falla o da valores absurdos)
        if tc_segundos < 0.1:
            duracion_total = (df[c_fec].max() - df[c_fec].min()).total_seconds()
            tc_segundos = duracion_total / total_piezas
            metodo = "Promedio Global (Rescate)"
    except:
        duracion_total = (df[c_fec].max() - df[c_fec].min()).total_seconds()
        tc_segundos = duracion_total / total_piezas
        metodo = "Promedio Global (Rescate)"

    return tc_segundos / 60, tc_segundos, metodo, total_piezas

# --- FLUJO DE UI ---
uploaded_file = st.file_uploader("Subir reporte Spectrum/SOAC", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🚀 Procesando datos..."):
        df_raw, cols = load_data(uploaded_file)
        
        if df_raw is not None and cols['Fecha']:
            tc_min, tc_seg, metodo, n_piezas = analizar_ritmo_seguro(df_raw, cols)
            
            if tc_min > 0:
                st.success(f"✅ Análisis completado usando método: **{metodo}**")
                
                # DASHBOARD
                k1, k2, k3 = st.columns(3)
                k1.metric("⏱️ Cycle Time Real", f"{tc_min:.2f} min", help=f"Equivalente a {tc_seg:.1f} segundos por pieza.")
                
                capacidad = (h_turno * 60 / tc_min) * oee_target
                k2.metric("📦 Capacidad Turno", f"{int(capacidad)} uds", help=f"Basado en {oee_target*100}% de OEE.")
                k3.metric("📊 Piezas Únicas", n_piezas)

                st.divider()

                # GRÁFICA DE TENDENCIA
                st.subheader("📈 Estabilidad del Ritmo de Producción")
                df_raw['Hora'] = df_raw[cols['Fecha']].dt.hour
                hourly_prod = df_raw.groupby('Hora').size().reset_index(name='Piezas')
                
                fig = px.bar(hourly_prod, x='Hora', y='Piezas', title="Piezas procesadas por hora", color='Piezas', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)

                # TABLA POR PRODUCTO
                if cols['Producto']:
                    st.subheader("📋 Desglose por Producto")
                    prod_df = df_raw.groupby(cols['Producto']).size().reset_index(name='Unidades')
                    prod_df['Tiempo Estimado (min)'] = prod_df['Unidades'] * tc_min
                    st.dataframe(prod_df.sort_values('Unidades', ascending=False), use_container_width=True)
            else:
                st.error("No se ha podido calcular el tiempo. Verifica que la columna de fecha tenga datos válidos.")
        else:
            st.error("No se encontró la estructura de columnas necesaria. Asegúrate de que el archivo tenga una columna de Fecha/Hora.")
