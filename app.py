import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Flow Master AI", layout="wide", page_icon="🧬")
st.title("🧬 Celestica IA: Smart Heartbeat & Theoretical Capacity")

with st.sidebar:
    st.header("⚙️ Configuración de IA")
    st.info("Esta versión detecta y reparte automáticamente el tiempo de las ráfagas (Batching).")
    h_turno = st.number_input("Horas Turno Disponibles", value=8.0)
    st.divider()
    st.caption("v8.0 - Inmune a errores de flujo cero.")

# --- 1. MOTOR DE LECTURA UNIVERSAL ---
def parse_xml_robust(file):
    try:
        content = file.getvalue().decode('latin-1', errors='ignore')
        soup = BeautifulSoup(content, 'lxml-xml')
        data = [[c.get_text(strip=True) for c in row.find_all(['Cell', 'ss:Cell', 'cell'])] 
                for row in soup.find_all(['Row', 'ss:Row', 'row'])]
        return pd.DataFrame([d for d in data if d])
    except: return None

@st.cache_data(ttl=3600)
def load_data(file):
    df = parse_xml_robust(file)
    if df is None or df.empty:
        try:
            file.seek(0)
            df = pd.read_excel(file, header=None)
        except: return None, None
    
    df = df.astype(str)
    # Buscador dinámico de cabeceras (Fase B)
    for i in range(min(100, len(df))):
        row_str = " ".join(df.iloc[i]).lower()
        if any(x in row_str for x in ['date', 'time', 'station', 'productid', 'sn', 'serial']):
            df.columns = df.iloc[i].str.strip()
            return df[i+1:].reset_index(drop=True), i
    return None, None

# --- 2. CEREBRO IA: IMPUTACIÓN Y FRONTERA (Fase C y D) ---
def analyze_industrial_flow(df):
    # Identificar columnas
    c_fec = next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None)
    c_sn = next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None)
    
    if not c_fec: return None

    # Limpieza y deduplicación por Serial Number
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn:
        df = df.drop_duplicates(subset=[c_sn], keep='first')
    
    # --- LÓGICA DE IMPUTACIÓN (Heartbeat) ---
    # 1. Agrupar por segundo para ver ráfagas
    batches = df.groupby(c_fec).size().reset_index(name='piezas_en_segundo')
    
    # 2. Calcular el gap con el SEGUNDO anterior
    batches['gap_previo'] = batches[c_fec].diff().dt.total_seconds().fillna(0)
    
    # 3. Imputar tiempo: $$CT_{unitario} = \frac{Gap}{Piezas}$$
    # Si el gap es > 20 min (1200s), lo limitamos a 60s para el "Teórico" (es una parada)
    batches['gap_limpio'] = batches['gap_previo'].apply(lambda x: x if x < 1200 else 60)
    batches['tc_imputado'] = batches['gap_limpio'] / batches['piezas_en_segundo']
    
    # 4. Filtro de Realismo Humano
    # Nos quedamos con datos > 0.5s para evitar errores matemáticos
    data_points = batches[batches['tc_imputado'] > 0.5]['tc_imputado'].values
    
    if len(data_points) < 2:
        # Fallback total: Tiempo total del archivo / Total piezas
        total_time = (df[c_fec].max() - df[c_fec].min()).total_seconds()
        tc_manual = (total_time / len(df)) if len(df) > 0 else 0
        return {'teo': tc_manual/60, 'real': tc_manual/60, 'piezas': len(df), 'df_b': batches, 'modo': tc_manual}

    # --- ESTADÍSTICA DE FRONTERA ---
    # Usamos el Percentil 10 para el Teórico y la Mediana para el Real
    tc_teorico_seg = np.percentile(data_points, 10) # El 10% más rápido es el "Teórico"
    tc_real_seg = np.median(data_points)
    
    # Si el teórico da absurdamente bajo (< 30s) en un proceso que sabes que es de 110s,
    # es porque hay mucho de-batching. Usamos la MODA.
    try:
        kde = gaussian_kde(data_points)
        x_range = np.linspace(data_points.min(), data_points.max(), 500)
        tc_teorico_seg = x_range[np.argmax(kde(x_range))]
    except: pass

    return {
        'teo': tc_teorico_seg / 60,
        'real': tc_real_seg / 60,
        'piezas': len(df),
        'df_b': batches,
        'modo': tc_teorico_seg
    }

# --- 3. UI Y DASHBOARD ---
uploaded_file = st.file_uploader("Sube el archivo de 1.9MB (o cualquier tamaño)", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🤖 Procesando ráfagas de datos..."):
        df_raw, _ = load_data(uploaded_file)
        
        if df_raw is not None:
            res = analyze_industrial_flow(df_raw)
            
            if res:
                # KPIs PRINCIPALES (Diseño limpio)
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO (Target)", f"{res['teo']:.2f} min", 
                          help=f"Ritmo de excelencia detectado: {res['modo']:.1f} segundos.")
                c2.metric("⏱️ TC REAL (Sostenido)", f"{res['real']:.2f} min", 
                          delta=f"{((res['real']/res['teo'])-1)*100:.1f}% Desvío", delta_color="inverse")
                
                capacidad = (h_turno * 60) / res['teo']
                c3.metric("📦 Capacidad (100%)", f"{int(capacidad)} uds")

                st.divider()

                # GRÁFICA DE DISTRIBUCIÓN
                st.subheader("📊 Distribución Gamma de Producción")
                st.caption("El pico de la montaña representa tu ritmo de crucero real.")
                
                # Gráfico de los tiempos imputados
                df_plot = res['df_b'][res['df_b']['tc_imputado'] < (res['modo'] * 5)]
                fig = px.histogram(df_plot, x="tc_imputado", nbins=60, 
                                 title="Histograma de Ritmos Unitarios (Segundos)",
                                 color_discrete_sequence=['#3498db'])
                fig.add_vline(x=res['modo'], line_dash="dash", line_color="red", line_width=4, annotation_text="MODA")
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("🔍 Ver datos depurados (Top 50 registros)"):
                    st.write(f"Total registros únicos procesados: {res['piezas']}")
                    st.dataframe(res['df_b'].sort_values('piezas_en_segundo', ascending=False).head(50))
            else:
                st.error("No se pudo extraer información temporal. Revisa el formato del archivo.")
