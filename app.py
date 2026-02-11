import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica AI Physical Logic", layout="wide", page_icon="⚙️")
st.title("⚙️ Celestica IA: Analizador de Capacidad (Filtro Físico)")
st.markdown("""
**Depuración de Ráfagas:** Esta versión detecta la 'montaña de producción' real, 
ignorando automáticamente los registros de sistema que falsean el Tiempo de Ciclo.
""")

with st.sidebar:
    st.header("🏭 Ajustes de Ingeniería")
    h_turno = st.number_input("Horas Turno", value=8.0)
    st.divider()
    st.info("La IA busca el ritmo más frecuente por encima de los 30 segundos para evitar ruidos de red.")

# --- 1. LECTOR UNIVERSAL ---
def parse_xml_robust(file):
    try:
        content = file.getvalue().decode('latin-1', errors='ignore')
        soup = BeautifulSoup(content, 'lxml-xml')
        data = [[c.get_text(strip=True) for c in row.find_all(['Cell', 'ss:Cell', 'cell'])] 
                for row in soup.find_all(['Row', 'ss:Row', 'row'])]
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
    for i in range(min(100, len(df))):
        row_str = " ".join(df.iloc[i]).lower()
        if any(x in row_str for x in ['date', 'time', 'station', 'productid', 'sn']):
            df.columns = df.iloc[i].str.strip()
            return df[i+1:].reset_index(drop=True), i
    return None, None

# --- 2. CEREBRO IA: BÚSQUEDA DEL PUNTO MEDIO ---
def analyze_production_peak(df):
    c_fec = next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None)
    c_sn = next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None)
    
    if not c_fec: return None

    # Limpieza
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn:
        df = df.drop_duplicates(subset=[c_sn], keep='first')
    
    # Imputación de ráfagas (Batching)
    batches = df.groupby(c_fec).size().reset_index(name='piezas')
    batches['gap'] = batches[c_fec].diff().dt.total_seconds().fillna(0)
    batches['tc_unitario'] = batches['gap'] / batches['piezas']
    
    # --- EL FILTRO DE INTELIGENCIA (Evitar el 0.02) ---
    # Solo consideramos "producción real" los tiempos entre 30s y 20min.
    zona_real = batches[(batches['tc_unitario'] >= 30) & (batches['tc_unitario'] <= 1200)]['tc_unitario']
    
    if len(zona_real) < 5:
        # Fallback: Si no hay datos en esa zona, promediamos el tiempo total
        total_sec = (df[c_fec].max() - df[c_fec].min()).total_seconds()
        tc_manual = (total_sec / len(df)) if len(df) > 0 else 0
        return {'teo': tc_manual/60, 'real': tc_manual/60, 'piezas': len(df), 'df_b': batches, 'modo': tc_manual}

    # CÁLCULO DE LA MODA (Pico de la Montaña)
    kde = gaussian_kde(zona_real)
    x_range = np.linspace(zona_real.min(), zona_real.max(), 1000)
    tc_teorico_seg = x_range[np.argmax(kde(x_range))]
    
    # CÁLCULO DE LA MEDIANA (Punto Medio Realista)
    # Filtramos la zona productiva alrededor del pico (+/- 60%)
    ritmo_sostenido = zona_real[(zona_real > tc_teorico_seg * 0.4) & (zona_real < tc_teorico_seg * 2.0)]
    tc_real_seg = ritmo_sostenido.median() if not ritmo_sostenido.empty else tc_teorico_seg

    return {
        'teo': tc_teorico_seg / 60,
        'real': tc_real_seg / 60,
        'piezas': len(df),
        'df_b': batches,
        'modo': tc_teorico_seg
    }

# --- 3. UI Y RESULTADOS ---
uploaded_file = st.file_uploader("Sube el archivo (1.9MB / 15MB)", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🤖 Localizando el punto medio de producción..."):
        df_raw, _ = load_and_map(uploaded_file)
        
        if df_raw is not None:
            res = analyze_production_peak(df_raw)
            if res:
                # KPIs PRINCIPALES
                c1, c2, c3 = st.columns(3)
                # El TEÓRICO es el ritmo de flujo puro (Pico)
                c1.metric("⏱️ TC TEÓRICO (Flujo)", f"{res['teo']:.2f} min", 
                          help=f"Ritmo de excelencia detectado: {res['modo']:.1f} segundos.")
                # El REAL es la mediana de la zona productiva (Punto medio)
                c2.metric("⏱️ TC REAL (Punto Medio)", f"{res['real']:.2f} min", 
                          delta=f"{((res['real']/res['teo'])-1)*100:.1f}% Desvío", delta_color="inverse")
                
                capacidad = (h_turno * 60) / res['teo']
                c3.metric("📦 Capacidad Turno", f"{int(capacidad)} uds")

                st.divider()

                # GRÁFICA DE DENSIDAD
                st.subheader("📊 Distribución de la Producción Real")
                st.caption(f"La IA ha ignorado el ruido de ráfagas y ha detectado el pico en **{res['modo']:.1f} segundos**.")
                
                fig_plot = res['df_b'][(res['df_b']['tc_unitario'] >= 10) & (res['df_b']['tc_unitario'] < res['modo'] * 4)]
                fig = px.histogram(fig_plot, x="tc_unitario", nbins=50, 
                                 title="Histograma de Ritmos de Trabajo (Segundos)",
                                 color_discrete_sequence=['#2ecc71'])
                fig.add_vline(x=res['modo'], line_dash="dash", line_color="red", line_width=4, annotation_text="PUNTO TEÓRICO")
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("🔍 Auditoría de Lotes"):
                    st.dataframe(res['df_b'].sort_values('piezas', ascending=False).head(50))
