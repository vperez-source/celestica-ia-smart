import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Smart-Tracker PRO", layout="wide", page_icon="🎯")
st.title("🎯 Celestica IA: Diagnóstico y Análisis de Ciclos")

with st.sidebar:
    st.header("⚙️ Parámetros")
    h_turno = st.number_input("Horas de Turno", value=8)
    oee_target = st.slider("Eficiencia %", 50, 100, 85) / 100
    st.divider()
    st.warning("⚠️ Si el archivo es muy grande, espera a que la barra superior termine de cargar.")

# --- FASE A: INGESTIÓN ULTRA-ROBUSTA ---
def parse_xml_flexible(file):
    try:
        content = file.getvalue().decode('latin-1', errors='ignore')
        if "Workbook" not in content and "<?xml" not in content: return None
        soup = BeautifulSoup(content, 'lxml-xml')
        data = []
        rows = soup.find_all(['Row', 'ss:Row', 'row'])
        for row in rows:
            cells = [c.get_text(strip=True) for c in row.find_all(['Cell', 'ss:Cell', 'cell'])]
            if any(cells): data.append(cells)
        return pd.DataFrame(data)
    except: return None

@st.cache_data(ttl=3600)
def load_data(file):
    # Intentar XML Legacy
    df = parse_xml_flexible(file)
    if df is None or df.empty:
        try:
            file.seek(0)
            df = pd.read_excel(file, header=None)
        except:
            try:
                file.seek(0)
                df = pd.read_csv(file, sep='\t', encoding='latin-1', header=None)
            except: return None, None

    # FASE B: MAPEO SEMÁNTICO REFORZADO
    df = df.astype(str)
    start_row = -1
    # Buscamos la fila que tiene los nombres de las columnas reales
    for i in range(min(100, len(df))):
        row_str = " ".join(df.iloc[i].astype(str)).lower()
        if any(x in row_str for x in ['date', 'time', 'station', 'productid']):
            start_row = i
            break
    
    if start_row == -1: return None, None

    df.columns = df.iloc[start_row]
    df = df[start_row + 1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()

    # Diccionario de búsqueda
    target_cols = {}
    for c in df.columns:
        cl = c.lower()
        if not target_cols.get('Fecha') and any(x in cl for x in ['date', 'time', 'fecha', 'timestamp']): target_cols['Fecha'] = c
        if not target_cols.get('Producto') and any(x in cl for x in ['productid', 'item', 'part']): target_cols['Producto'] = c
        if not target_cols.get('Familia') and 'family' in cl: target_cols['Familia'] = c
        if not target_cols.get('Usuario') and any(x in cl for x in ['user', 'operator', 'name']): target_cols['Usuario'] = c
    
    return df, target_cols

# --- FASE C: LÓGICA DE IMPUTACIÓN (HEARTBEAT) ---
def analyze_heartbeat(df, cols):
    c_fec = cols.get('Fecha')
    if not c_fec: return 0, None, 0

    # Forzamos conversión de fecha
    df[c_fec] = pd.to_datetime(df[c_fec], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)

    if df.empty: return 0, None, 0

    # Agrupamos por segundo
    batches = df.groupby(c_fec).size().reset_index(name='Piezas')
    batches['Gap_Sec'] = batches[c_fec].diff().dt.total_seconds().fillna(0)
    
    # Imputación: Tiempo de espera / Piezas del lote
    batches['CT_Unitario'] = batches['Gap_Sec'] / batches['Piezas']

    # RELAJAMOS FILTROS: Ignoramos 0s (ráfaga pura) y > 1 hora (comida)
    valid = batches[(batches['CT_Unitario'] > 0.1) & (batches['CT_Unitario'] < 3600)]['CT_Unitario']

    if len(valid) < 5:
        # Si falla el KDE, intentamos una mediana simple para no dar error
        if not batches[batches['CT_Unitario'] > 0].empty:
            mediana = batches[batches['CT_Unitario'] > 0]['CT_Unitario'].median()
            return mediana / 60, batches, mediana
        return 0, batches, 0

    # FASE D: MODA (KDE)
    kde = gaussian_kde(valid)
    x = np.linspace(valid.min(), valid.max(), 1000)
    y = kde(x)
    modo_s = x[np.argmax(y)]
    
    return modo_s / 60, batches, modo_s

# --- INTERFAZ PRINCIPAL ---
uploaded_file = st.file_uploader("Subir Archivo de Trazabilidad", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("⏳ Procesando archivo de 15MB..."):
        df, cols = load_data(uploaded_file)
        
        if df is not None and cols.get('Fecha'):
            # SECCIÓN DE DIAGNÓSTICO
            with st.expander("🔍 Ver Diagnóstico de Columnas"):
                st.write("**Columnas Detectadas:**", cols)
                st.write("**Vista Previa de Datos:**")
                st.dataframe(df.head(5))

            ct, batches, modo_s = analyze_heartbeat(df, cols)
            
            if ct > 0:
                st.success(f"✅ Análisis completado. Ritmo detectado: {ct:.2f} min/ud")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ Cycle Time Real", f"{ct:.2f} min")
                c2.metric("📦 Capacidad Turno", f"{int((h_turno * 60 / ct) * oee_target)}")
                c3.metric("📊 Piezas Analizadas", len(df))

                # Gráfico
                fig = px.histogram(batches[batches['CT_Unitario'] < 600], x="CT_Unitario", 
                                 nbins=50, title="Frecuencia de Ritmos de Trabajo")
                fig.add_vline(x=modo_s, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("La IA no pudo detectar ciclos. Revisa en 'Diagnóstico' si la fecha se lee correctamente.")
        else:
            st.error("No se encontró la cabecera del archivo. Asegúrate de que el archivo tenga una fila con 'In DateTime' o 'Date'.")
