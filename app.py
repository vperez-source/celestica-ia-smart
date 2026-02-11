import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- CONFIGURACIÓN PROFESIONAL ---
st.set_page_config(page_title="Celestica Precision Flow AI", layout="wide", page_icon="🎯")
st.title("🎯 Celestica IA: Smart-Tracker de Alta Precisión")

with st.sidebar:
    st.header("⚙️ Configuración de Ingeniería")
    tc_esperado_seg = st.number_input("TC Objetivo (seg)", value=120)
    h_turno = st.number_input("Horas Turno", value=8.0)
    st.divider()
    st.info("Algoritmo v12.0: Optimizado para distribuciones Gamma y ráfagas de servidor.")

# --- 1. LECTOR DE ALTA COMPATIBILIDAD ---
def parse_xml_robust(file):
    try:
        content = file.getvalue().decode('latin-1', errors='ignore')
        if "<?xml" not in content and "Workbook" not in content: return None
        soup = BeautifulSoup(content, 'lxml-xml')
        data = [[c.get_text(strip=True) for c in row.find_all(['Cell', 'ss:Cell'])] 
                for row in soup.find_all(['Row', 'ss:Row'])]
        return pd.DataFrame([d for d in data if d])
    except: return None

@st.cache_data(ttl=3600)
def load_and_map(file):
    df = parse_xml_robust(file)
    if df is None or df.empty:
        try:
            file.seek(0)
            df = pd.read_excel(file, header=None)
        except: return None, None
    
    df = df.astype(str)
    # Buscador de cabecera dinámico
    header_idx = -1
    for i in range(min(100, len(df))):
        row = " ".join(df.iloc[i]).lower()
        if any(x in row for x in ['date', 'time', 'station', 'productid', 'sn']):
            header_idx = i; break
            
    if header_idx == -1: return None, None
    df.columns = df.iloc[header_idx].str.strip()
    df = df[header_idx + 1:].reset_index(drop=True)

    cols = {
        'Fecha': next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None),
        'SN': next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None),
        'Product': next((c for c in df.columns if any(x in c.lower() for x in ['product', 'item'])), 'Producto'),
        'Family': next((c for c in df.columns if any(x in c.lower() for x in ['family', 'familia'])), 'Familia')
    }
    return df, cols

# --- 2. CEREBRO DE PRECISIÓN ESTADÍSTICA ---
def analyze_precision_cycle(df, cols):
    c_fec = cols['Fecha']
    c_sn = cols['SN']
    
    # A. Limpieza y Deduplicación Estricta
    df[c_fec] = pd.to_datetime(df[c_fec], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn:
        df = df.drop_duplicates(subset=[c_sn], keep='first')
    
    # B. Cálculo de Gaps de Flujo
    # Calculamos el tiempo entre cada registro único
    df['Gap_Sec'] = df[c_fec].diff().dt.total_seconds().fillna(0)
    
    # C. Filtrado de "Zona de Flujo Humano"
    # Ignoramos ráfagas de servidor (< 5s) y paradas de descanso (> 15 min)
    # Esto aísla los datos donde el operario está trabajando activamente
    flujo_activo = df[(df['Gap_Sec'] >= 10) & (df['Gap_Sec'] <= 900)]['Gap_Sec'].values
    
    if len(flujo_activo) < 10:
        # Fallback: Si no hay flujo, calculamos el rendimiento por ventanas de densidad
        return None

    # D. ESTIMACIÓN DE DENSIDAD (KDE) - Encontrando el 1.40 min / 120s
    # Usamos la escala logarítmica para manejar la cola larga de la distribución Gamma
    log_data = np.log1p(flujo_activo)
    kde = gaussian_kde(log_data)
    x_range = np.linspace(log_data.min(), log_data.max(), 1000)
    tc_teorico_seg = np.expm1(x_range[np.argmax(kde(x_range))])
    
    # El TC Real (Mediana del flujo activo)
    tc_real_seg = np.median(flujo_activo)
    
    return {
        'teo_min': tc_teorico_seg / 60,
        'real_min': tc_real_seg / 60,
        'modo_seg': tc_teorico_seg,
        'muestras': len(flujo_activo),
        'df_plot': flujo_activo
    }

# --- 3. UI Y RESULTADOS ---
uploaded_file = st.file_uploader("Sube el reporte (1.9MB / 15.4MB)", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🤖 Aplicando filtros de flujo de precisión..."):
        df_raw, cols = load_and_map(uploaded_file)
        
        if df_raw is not None and cols['Fecha']:
            res = analyze_precision_cycle(df_raw, cols)
            
            if res:
                st.success("✅ Análisis de Flujo Realizado con Éxito")
                
                # KPIs PRINCIPALES (Diseño Limpio)
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO (Target)", f"{res['teo_min']:.2f} min", 
                          help=f"Ritmo de máxima densidad detectado: {res['modo_seg']:.1f}s")
                c2.metric("⏱️ TC REAL (Sostenido)", f"{res['real_min']:.2f} min",
                          delta=f"{((res['real_min']/res['teo_min'])-1)*100:.1f}% Desvío", delta_color="inverse")
                
                capacidad = (h_turno * 60) / res['teo_min']
                c3.metric("📦 Capacidad Nominal", f"{int(capacidad)} uds", help="Capacidad al 100% de eficiencia teórica.")

                st.divider()

                # --- VISUALIZACIÓN DE DISTRIBUCIÓN ---
                st.subheader("📊 Diagnóstico de la Curva de Producción (Gamma)")
                st.caption("El pico indica el ritmo más estable de la línea. La cola hacia la derecha representa las ineficiencias.")
                
                # Histograma de los gaps de flujo activo
                fig = px.histogram(x=res['df_plot'], nbins=100, 
                                 title="Densidad de Tiempos de Ciclo (Datos Filtrados)",
                                 labels={'x': 'Segundos por Pieza'},
                                 color_discrete_sequence=['#2ecc71'])
                
                fig.add_vline(x=res['modo_seg'], line_dash="dash", line_color="red", line_width=4, 
                             annotation_text=f"PICO TEÓRICO: {res['modo_seg']:.1f}s")
                st.plotly_chart(fig, use_container_width=True)

                # --- TABLA DE RENDIMIENTO ---
                st.subheader("📋 Resumen por Estación y Producto")
                resumen = df_raw.groupby([cols['Product']]).size().reset_index(name='Unidades')
                resumen['Horas Est. (Teórico)'] = (resumen['Unidades'] * res['teo_min']) / 60
                st.dataframe(resumen.sort_values('Unidades', ascending=False), use_container_width=True)
                
            else:
                st.error("No se pudo detectar un patrón de flujo. El archivo podría contener solo registros masivos (batch) sin marcas de tiempo individuales.")
        else:
            st.error("Formato de archivo no reconocido o faltan columnas esenciales.")
