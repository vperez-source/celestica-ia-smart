import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Pulse AI", layout="wide", page_icon="💓")
st.title("💓 Celestica IA: Analizador de Densidad de Pulso")
st.markdown("""
**Modo Ráfaga:** Este algoritmo ignora el desorden del servidor. Calcula el Tiempo de Ciclo 
analizando cuántas piezas es capaz de procesar el sistema por cada minuto de actividad real.
""")

with st.sidebar:
    st.header("⚙️ Parámetros de Ingeniería")
    h_turno = st.number_input("Horas Turno", value=8.0)
    st.divider()
    st.info("Algoritmo v13.0: Diseñado para archivos con registros masivos (Batching).")

# --- 1. LECTOR UNIVERSAL ---
def parse_xml_tanque(file):
    try:
        content = file.getvalue().decode('latin-1', errors='ignore')
        soup = BeautifulSoup(content, 'lxml-xml')
        data = [[c.get_text(strip=True) for c in row.find_all(['Cell', 'ss:Cell'])] 
                for row in soup.find_all(['Row', 'ss:Row'])]
        return pd.DataFrame([d for d in data if d])
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
        row = " ".join(df.iloc[i]).lower()
        if any(x in row for x in ['date', 'time', 'station', 'productid', 'sn']):
            header_idx = i; break
            
    if header_idx == -1: return None, None
    df.columns = df.iloc[header_idx].str.strip()
    df = df[header_idx + 1:].reset_index(drop=True)

    cols = {
        'Fecha': next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None),
        'SN': next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None),
        'Product': next((c for c in df.columns if any(x in c.lower() for x in ['product', 'item'])), 'Producto')
    }
    return df, cols

# --- 2. CEREBRO: ANÁLISIS DE DENSIDAD DE PULSO ---
def analyze_pulse_density(df, cols):
    c_fec = cols['Fecha']
    c_sn = cols['SN']
    
    # Limpieza y deduplicación
    df[c_fec] = pd.to_datetime(df[c_fec], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn:
        df = df.drop_duplicates(subset=[c_sn], keep='first')
    
    # --- CÁLCULO POR MINUTO (La clave del realismo) ---
    df.set_index(c_fec, inplace=True)
    # Contamos cuántas piezas se registran CADA MINUTO
    frecuencia_minuto = df.resample('1Min').size().reset_index(name='piezas_por_minuto')
    
    # Filtramos solo los minutos donde HUBO producción (ignoramos paradas)
    minutos_activos = frecuencia_minuto[frecuencia_minuto['piezas_por_minuto'] > 0].copy()
    
    if minutos_activos.empty: return None

    # Calculamos el TC de cada minuto activo: TC = 60 seg / piezas
    minutos_activos['tc_seg_minuto'] = 60 / minutos_activos['piezas_por_minuto']
    
    # --- CÁLCULO DE FRONTERA ---
    # TC TEÓRICO: El percentil 10 de los ritmos más rápidos observados
    # (Representa la capacidad máxima de la línea cuando fluye)
    tc_teorico_seg = np.percentile(minutos_activos['tc_seg_minuto'], 15)
    
    # TC REAL: La mediana de los ritmos de los minutos activos
    tc_real_seg = minutos_activos['tc_seg_minuto'].median()
    
    # Ajuste por realismo: si el teórico es < 10s en un proceso de 120s, 
    # es que el volcado es demasiado masivo. Usamos la mediana como base.
    if tc_teorico_seg < 30: 
        tc_teorico_seg = tc_real_seg * 0.8
        
    return {
        'teo_min': tc_teorico_seg / 60,
        'real_min': tc_real_seg / 60,
        'piezas_totales': len(df),
        'df_v': minutos_activos,
        't_seg': tc_teorico_seg,
        'r_seg': tc_real_seg
    }

# --- 3. UI Y RESULTADOS ---
uploaded_file = st.file_uploader("Sube el reporte de 15.4MB", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🤖 Analizando latidos de producción por minuto..."):
        df_raw, cols = load_and_map(uploaded_file)
        
        if df_raw is not None and cols['Fecha']:
            res = analyze_pulse_density(df_raw, cols)
            
            if res:
                st.success("✅ Análisis de Densidad Completado")
                
                # KPIs PRINCIPALES
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO (Target)", f"{res['teo_min']:.2f} min", 
                          help=f"Basado en ráfagas de máxima eficiencia: {res['t_seg']:.1f}s")
                c2.metric("⏱️ TC REAL (Sostenido)", f"{res['real_min']:.2f} min",
                          delta=f"{((res['real_min']/res['teo_min'])-1)*100:.1f}% Variabilidad", delta_color="inverse")
                
                capacidad = (h_turno * 60) / res['teo_min']
                c3.metric("📦 Capacidad Turno", f"{int(capacidad)} uds")

                st.divider()

                # --- GRÁFICA DE ACTIVIDAD ---
                st.subheader("📈 Intensidad de Producción por Minuto")
                st.caption("Cada barra muestra cuántas piezas se registraron en ese minuto. La línea roja es tu ritmo objetivo.")
                
                fig = px.bar(res['df_v'], x=res['df_v'].columns[0], y='piezas_por_minuto',
                            title="Piezas procesadas por minuto activo",
                            color='piezas_por_minuto', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
                
                # --- TABLA DE LECTURA ---
                with st.expander("🔍 Auditoría de ráfagas detectadas"):
                    st.write("Minutos con mayor volumen de registros (Batching detectado):")
                    st.dataframe(res['df_v'].sort_values('piezas_por_minuto', ascending=False).head(20))
            else:
                st.error("No se detectó actividad. Revisa si la columna de fecha contiene horas válidas.")
        else:
            st.error("Formato de archivo no válido o faltan columnas de 'Date'.")
