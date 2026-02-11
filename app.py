import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica AI Self-Explainer", layout="wide", page_icon="🕵️")
st.title("🕵️ Celestica IA: Smart-Tracker & Diagnostic Engine")

with st.sidebar:
    st.header("⚙️ Baseline de Ingeniería")
    tc_esperado_seg = st.number_input("TC Objetivo Esperado (seg)", value=110)
    h_turno = st.number_input("Horas Turno", value=8.0)
    st.divider()
    st.info("Si el resultado se desvía del objetivo, la IA generará una explicación técnica.")

# --- 1. LECTOR DE ALTA PRECISIÓN ---
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
    # Buscador de cabeceras avanzado
    for i in range(min(100, len(df))):
        row = " ".join(df.iloc[i].astype(str)).lower()
        if any(x in row for x in ['date', 'time', 'station', 'productid', 'sn']):
            df.columns = df.iloc[i].str.strip()
            return df[i+1:].reset_index(drop=True), i
    return None, None

# --- 2. MOTOR DE CÁLCULO Y EXPLICACIÓN ---
def analyze_with_explanation(df, tc_obj_seg):
    c_fec = next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None)
    c_sn = next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None)
    
    # Limpieza
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn: df = df.drop_duplicates(subset=[c_sn], keep='first')
    
    # Imputación de ráfagas
    batches = df.groupby(c_fec).size().reset_index(name='piezas')
    batches['gap'] = batches[c_fec].diff().dt.total_seconds().fillna(0)
    batches['tc_unitario'] = batches['gap'] / batches['piezas']
    
    # --- FILTRO DE FLUJO (INDENTACIÓN CORREGIDA) ---
    # Buscamos rastro de vida entre 1s y 30min
    frontera_data = batches[(batches['tc_unitario'] >= 1) & (batches['tc_unitario'] <= 1800)]['tc_unitario']

    # MODO RESCATE: Si hay muy pocos datos de flujo
    if len(frontera_data) < 2:
        duracion_total = (df[c_fec].max() - df[c_fec].min()).total_seconds()
        tc_emergencia = duracion_total / len(df) if len(df) > 0 else 0
        return {
            'teorico': tc_emergencia / 60,
            'real': tc_emergencia / 60,
            'modo_seg': tc_emergencia,
            'explicacion': ["⚠️ Datos Colapsados: Se ha usado el promedio total por falta de flujo constante."],
            'df_b': batches
        }, None

    # Cálculo de la Moda (Pico de la Montaña Gamma)
    try:
        kde = gaussian_kde(frontera_data)
        x_range = np.linspace(frontera_data.min(), frontera_data.max(), 1000)
        tc_moda_seg = x_range[np.argmax(kde(x_range))]
    except:
        tc_moda_seg = frontera_data.median()
    
    tc_mediana_seg = frontera_data.median()
    
    # --- MOTOR DE EXPLICACIÓN ---
    razones = []
    ratio_desvio = tc_moda_seg / tc_obj_seg
    
    if ratio_desvio > 2:
        razones.append(f"⚠️ El TC es {ratio_desvio:.1f}x mayor al objetivo.")
        gaps_grandes = batches[batches['gap'] > 300]['gap'].sum()
        total_time = (df[c_fec].max() - df[c_fec].min()).total_seconds()
        pct_inactividad = (gaps_grandes / total_time) * 100 if total_time > 0 else 0
        
        if pct_inactividad > 40:
            razones.append(f"🔍 Causa: Alta inactividad ({pct_inactividad:.1f}% del tiempo son paros > 5 min).")
        
        batching_level = batches['piezas'].mean()
        if batching_level > 5:
            razones.append(f"🔍 Causa: Batching alto ({batching_level:.1f} piezas/seg). El sistema registra en bloque.")

    return {
        'teorico': tc_moda_seg / 60,
        'real': tc_mediana_seg / 60,
        'modo_seg': tc_moda_seg,
        'explicacion': razones,
        'df_b': batches
    }, None

# --- 3. UI ---
uploaded_file = st.file_uploader("Sube el archivo de Spectrum/SOAC", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🤖 Analizando y auditando registros..."):
        df_raw, _ = load_and_map(uploaded_file)
        
        if df_raw is not None:
            res, err = analyze_with_explanation(df_raw, tc_esperado_seg)
            
            if err:
                st.error(err)
            else:
                st.success("✅ Análisis Completado")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO (Moda)", f"{res['teorico']:.2f} min", 
                          help=f"Ritmo más frecuente: {res['modo_seg']:.1f}s")
                c2.metric("⏱️ TC REAL (Mediana)", f"{res['real']:.2f} min")
                
                if res['teorico'] > 0:
                    cap = (h_turno * 60) / res['teorico']
                    c3.metric("📦 Capacidad Nominal", f"{int(cap)} uds")
                else:
                    c3.metric("📦 Capacidad Nominal", "0 uds")

                if res['explicacion']:
                    with st.expander("📝 Diagnóstico de la IA", expanded=True):
                        for r in res['explicacion']:
                            st.write(r)

                st.subheader("📊 Distribución de la Firma Temporal")
                # Limitar gráfico para que sea legible
                max_x = res['modo_seg'] * 5 if res['modo_seg'] > 0 else 600
                fig = px.histogram(res['df_b'][res['df_b']['tc_unitario'] < max_x], 
                                 x="tc_unitario", nbins=100, 
                                 title="Frecuencia de Ritmos Detectados",
                                 color_discrete_sequence=['#2ecc71'])
                fig.add_vline(x=res['modo_seg'], line_dash="dash", line_color="red", 
                             annotation_text=f"Pico: {res['modo_seg']:.1f}s")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No se pudo procesar el archivo. Revisa las cabeceras.")
