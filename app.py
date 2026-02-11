import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Binned AI", layout="wide", page_icon="📊")
st.title("📊 Celestica IA: Detector de Ritmo por Tramo de Frecuencia")

with st.sidebar:
    st.header("⚙️ Configuración")
    h_turno = st.number_input("Horas Turno", value=8.0)
    st.info("v21.0: Sistema de búsqueda por cubetas (Bins). Ignora el ruido de red por exclusión directa.")

# --- 1. MOTOR DE CARGA ---
@st.cache_data(ttl=3600)
def load_data(file):
    fname = file.name.lower()
    df = None
    try:
        if fname.endswith(('.xml', '.xls')):
            content = file.getvalue().decode('latin-1', errors='ignore')
            soup = BeautifulSoup(content, 'lxml-xml')
            data = [[c.get_text(strip=True) for c in row.find_all(['Cell', 'ss:Cell'])] 
                    for row in soup.find_all(['Row', 'ss:Row'])]
            df = pd.DataFrame([d for d in data if d])
        else:
            file.seek(0)
            df = pd.read_csv(file, sep=None, engine='python', header=None)
    except: return None, {}

    if df is None or df.empty: return None, {}

    df = df.astype(str)
    header_idx = 0
    for i in range(min(100, len(df))):
        row_str = " ".join(df.iloc[i]).lower()
        if any(x in row_str for x in ['date', 'time', 'fecha', 'sn', 'serial']):
            header_idx = i; break
    
    df.columns = df.iloc[header_idx].str.strip()
    df = df[header_idx + 1:].reset_index(drop=True)

    cols = {
        'Fecha': next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None),
        'SN': next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None),
        'Producto': next((c for c in df.columns if any(x in c.lower() for x in ['product', 'item'])), "Producto"),
        'Operacion': next((c for c in df.columns if any(x in c.lower() for x in ['station', 'oper', 'step'])), "Operación")
    }
    return df, cols

# --- 2. CEREBRO: BINNED MODE DETECTION ---
def analyze_binned_efficiency(df, cols):
    c_fec = cols['Fecha']
    c_sn = cols['SN']
    
    # Preparación
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn and c_sn in df.columns:
        df = df.drop_duplicates(subset=[c_sn], keep='first')

    # Cálculo de Gaps (segundos)
    df['Gap'] = df[c_fec].diff().dt.total_seconds().fillna(0)
    
    # 1. EXCLUSIÓN TOTAL DEL RUIDO (0-20s) Y PARADAS (>20min)
    # Buscamos la vida inteligente entre 20s y 1200s
    zona_viva = df[(df['Gap'] > 20) & (df['Gap'] <= 1200)].copy()
    
    if len(zona_viva) < 3:
        return None

    # 2. MÉTODO DE CUBETAS (Binned Mode)
    # Dividimos en cubetas de 5 segundos para encontrar el pico humano
    bins = np.arange(20, 1205, 5)
    zona_viva['Cubeta'] = pd.cut(zona_viva['Gap'], bins=bins)
    
    # Buscamos la cubeta más frecuente (la montaña de producción)
    ranking_cubetas = zona_viva.groupby('Cubeta', observed=True).size().reset_index(name='Frecuencia')
    ranking_cubetas = ranking_cubetas.sort_values('Frecuencia', ascending=False)
    
    if ranking_cubetas.empty: return None

    # El TC Teórico es el punto medio de la cubeta ganadora
    cubeta_ganadora = ranking_cubetas.iloc[0]['Cubeta']
    tc_teorico_seg = cubeta_ganadora.mid
    
    # TC Real: Mediana de la zona viva (incluye variabilidad del operario)
    tc_real_seg = zona_viva['Gap'].median()
    
    return {
        'teo': tc_teorico_seg / 60,
        'real': tc_real_seg / 60,
        't_seg': tc_teorico_seg,
        'r_seg': tc_real_seg,
        'df_plot': zona_viva,
        'producto': df[cols['Producto']].iloc[0] if cols['Producto'] in df else "N/A",
        'operacion': df[cols['Operacion']].iloc[0] if cols['Operacion'] in df else "N/A"
    }

# --- 3. UI ---
uploaded_file = st.file_uploader("Sube el archivo (15.4MB / 1.9MB)", type=["xls", "xml", "xlsx", "csv", "txt"])

if uploaded_file:
    with st.spinner("🕵️ Escaneando cubetas de productividad..."):
        df_raw, cols_map = load_data(uploaded_file)
        
        if df_raw is not None and cols_map['Fecha']:
            res = analyze_binned_efficiency(df_raw, cols_map)
            
            if res:
                st.success(f"✅ Análisis Finalizado: {res['operacion']} | {res['producto']}")
                
                # KPIs PRINCIPALES
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO (Pico)", f"{res['teo']:.2f} min", 
                          help=f"Ritmo más frecuente en la zona de producción: {res['t_seg']:.1f}s")
                c2.metric("⏱️ TC REAL (Sostenido)", f"{res['real']:.2f} min",
                          delta=f"{((res['real']/res['teo'])-1)*100:.1f}% Variabilidad", delta_color="inverse")
                
                capacidad = (h_turno * 60) / res['teo']
                c3.metric("📦 Capacidad Nominal", f"{int(capacidad)} uds")

                st.divider()

                # GRÁFICA DE CUBETAS
                st.subheader("📊 Mapa de Frecuencia de Producción (Excluyendo Ruido)")
                st.caption("Esta gráfica muestra solo los datos entre 20s y 600s para localizar tu ritmo real.")
                
                fig = px.histogram(res['df_plot'][res['df_plot']['Gap'] < 600], x="Gap", nbins=60, 
                                 title="Distribución de Tiempos de Ciclo Reales",
                                 color_discrete_sequence=['#2ecc71'])
                fig.add_vline(x=res['t_seg'], line_dash="dash", line_color="red", line_width=4, 
                             annotation_text="Pico Detectado")
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("🔍 Ver Auditoría de Tiempos Raw"):
                    st.write("Muestra de los últimos gaps detectados (segundos):")
                    st.dataframe(res['df_plot'][['Gap']].tail(20))
            else:
                st.error("No se detectó el ritmo de producción. El archivo solo contiene registros con menos de 20 segundos de diferencia.")
        else:
            st.error("Columnas no detectadas.")
