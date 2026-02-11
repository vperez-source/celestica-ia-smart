import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Active Window AI", layout="wide", page_icon="⚡")
st.title("⚡ Celestica IA: Detector de Flujo Real (Anti-Sincronización)")

with st.sidebar:
    st.header("⚙️ Parámetros de Ingeniería")
    tc_esperado_seg = st.number_input("TC Objetivo (seg)", value=120)
    h_turno = st.number_input("Horas Turno", value=8.0)
    st.divider()
    st.info("Esta versión analiza 'Ventanas de Producción' para ignorar los parones del servidor.")

# --- 1. LECTOR ROBUSTO ---
def parse_xml_tanque(file):
    try:
        content = file.getvalue().decode('latin-1', errors='ignore')
        soup = BeautifulSoup(content, 'lxml-xml')
        data = [[c.get_text(strip=True) for c in row.find_all(['Cell', 'ss:Cell'])] 
                for row in soup.find_all(['Row', 'ss:Row'])]
        return pd.DataFrame([d for d in data if d])
    except: return None

@st.cache_data(ttl=3600)
def load_data(file):
    df = parse_xml_tanque(file)
    if df is None or df.empty:
        try:
            file.seek(0)
            df = pd.read_excel(file, header=None)
        except: return None, None
    
    df = df.astype(str)
    for i in range(min(50, len(df))):
        row = " ".join(df.iloc[i]).lower()
        if any(x in row for x in ['date', 'time', 'station', 'productid', 'sn']):
            df.columns = df.iloc[i].str.strip()
            return df[i+1:].reset_index(drop=True), i
    return None, None

# --- 2. CEREBRO: ANÁLISIS DE VENTANA ACTIVA ---
def analyze_active_throughput(df, h_turno):
    c_fec = next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None)
    c_sn = next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None)
    
    if not c_fec: return None

    # Limpieza
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn: df = df.drop_duplicates(subset=[c_sn], keep='first')

    # A. CREAR VENTANAS DE 15 MINUTOS
    df.set_index(c_fec, inplace=True)
    # Contamos cuántas piezas entran en cada bloque de 15 min
    throughput = df.resample('15Min').size().reset_index(name='piezas')
    
    # B. FILTRAR VENTANAS ACTIVAS
    # Consideramos que la línea está "trabajando" si hay al menos 3 piezas en 15 min
    # (Ajuste dinámico según el TC esperado: si esperas 120s, deberían salir ~7 piezas)
    ventanas_activas = throughput[throughput['piezas'] >= 3].copy()
    
    if ventanas_activas.empty:
        return None

    # C. CALCULAR TC POR VENTANA
    # TC = (15 min * 60 seg) / número de piezas
    ventanas_activas['tc_seg'] = 900 / ventanas_activas['piezas']
    
    # D. RESULTADOS: TEÓRICO vs REAL
    # El TEÓRICO es el percentil 10 de tus mejores ventanas (tu Peak Performance)
    tc_teorico_seg = np.percentile(ventanas_activas['tc_seg'], 15)
    # El REAL es la mediana de todas las ventanas activas
    tc_real_seg = ventanas_activas['tc_seg'].median()
    
    return {
        'teorico_min': tc_teorico_seg / 60,
        'real_min': tc_real_seg / 60,
        'piezas_totales': len(df),
        'df_v': ventanas_activas,
        'tc_t_seg': tc_teorico_seg,
        'tc_r_seg': tc_real_seg
    }

# --- 3. UI Y DASHBOARD ---
uploaded_file = st.file_uploader("Sube el archivo de 15.4MB", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🤖 Analizando densidad de flujo en ventanas de tiempo..."):
        df_raw, _ = load_data(uploaded_file)
        
        if df_raw is not None:
            res = analyze_active_throughput(df_raw, h_turno)
            
            if res:
                st.success("✅ Análisis de Ventana Activa Completado")
                
                # KPIs PRINCIPALES
                k1, k2, k3 = st.columns(3)
                k1.metric("⏱️ TC TEÓRICO (Flow)", f"{res['teorico_min']:.2f} min", 
                          help=f"Ritmo basado en tus mejores ventanas de 15 min ({res['tc_t_seg']:.1f}s)")
                k2.metric("⏱️ TC REAL (Activo)", f"{res['real_min']:.2f} min",
                          delta=f"{((res['real_min']/res['teorico_min'])-1)*100:.1f}% Ineficiencia", delta_color="inverse")
                
                capacidad = (h_turno * 60) / res['teorico_min']
                k3.metric("📦 Capacidad Nominal", f"{int(capacidad)} uds")

                st.divider()

                # EXPLICACIÓN TÉCNICA DEL DESVÍO
                if res['real_min'] > (tc_esperado_seg / 60) * 2:
                    with st.warning("⚠️ Diagnóstico: Tiempo Detectado Superior al Real"):
                        st.write(f"La IA detecta que incluso en las ventanas más rápidas, el sistema solo registra {res['df_v']['piezas'].max()} piezas cada 15 min.")
                        st.write("Esto sucede si el sistema Spectrum no registra los datos en tiempo real. **El TC calculado refleja el ritmo de entrada al sistema, no el de montaje manual.**")

                # GRÁFICA DE RENDIMIENTO POR VENTANA
                st.subheader("📈 Evolución del Ritmo (Ventanas de 15 min)")
                fig = px.line(res['df_v'], x=res['df_v'].columns[0], y='tc_seg', 
                             title="Segundos por pieza a lo largo del día",
                             labels={'tc_seg': 'Segundos / Pieza'})
                fig.add_hline(y=res['tc_t_seg'], line_dash="dash", line_color="red", annotation_text="Teórico")
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("No se pudo detectar flujo activo. ¿El archivo tiene registros repartidos en el tiempo?")
