import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Process Intelligence", layout="wide", page_icon="⚙️")
st.title("⚙️ Celestica IA: Smart Tracker & Batch Analyzer")
st.markdown("### Interpretación inteligente de ráfagas de datos y tiempos de espera")

# --- 1. LECTOR UNIVERSAL ---
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
        'Producto': next((c for c in df.columns if any(x in c.lower() for x in ['product', 'item', 'part'])), "Producto"),
        'Operacion': next((c for c in df.columns if any(x in c.lower() for x in ['station', 'oper', 'step'])), "Operación")
    }
    return df, cols

# --- 2. CEREBRO: LÓGICA DE REPARTO DE CARGA ---
def analyze_smart_flow(df, cols):
    c_fec = cols['Fecha']
    c_sn = cols['SN']
    
    # Limpieza Inicial
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    
    # Eliminamos duplicados reales (mismo SN en el mismo proceso)
    if c_sn and c_sn in df.columns:
        df = df.drop_duplicates(subset=[c_sn], keep='first')

    # --- PASO A: IDENTIFICAR LOTES ---
    # Agrupamos por marca de tiempo exacta para identificar qué entró en ráfaga
    lotes = df.groupby(c_fec).size().reset_index(name='piezas_en_lote')
    
    # --- PASO B: CALCULAR EL TIEMPO PREVIO AL LOTE ---
    lotes['silencio_previo'] = lotes[c_fec].diff().dt.total_seconds().fillna(0)
    
    # --- PASO C: IMPUTAR TIEMPO POR UNIDAD ---
    # Si entran 10 piezas tras 1000s de silencio, cada una costó 100s.
    lotes['tc_imputado'] = lotes['silencio_previo'] / lotes['piezas_en_lote']
    
    # --- PASO D: FILTRO DE VALOR AÑADIDO (Criterio de Ingeniería) ---
    # 1. Ignoramos lo menor a 5s (sigue siendo ruido de sistema, imposible para un humano).
    # 2. Ignoramos lo mayor a 1800s (30 min: paradas, comidas, productos olvidados).
    flujo_real = lotes[(lotes['tc_imputado'] >= 5) & (lotes['tc_imputado'] <= 1800)].copy()
    
    if flujo_real.empty:
        # Fallback: Si no hay flujo limpio, tomamos una muestra del 15% central de los datos brutos
        tc_manual = (df[c_fec].max() - df[c_fec].min()).total_seconds() / len(df)
        return {'teo': tc_manual/60, 'real': tc_manual/60, 'resumen': 'Ajuste Global por falta de flujo'}

    # TC TEÓRICO (Frontera de Eficiencia): Percentil 25 de los tiempos imputados
    # Es el ritmo que el operario mantiene cuando el lote fluye bien.
    tc_teorico_seg = np.percentile(flujo_real['tc_imputado'], 25)
    
    # TC REAL (Mediana): El punto medio de los tiempos imputados
    tc_real_seg = flujo_real['tc_imputado'].median()
    
    return {
        'teo': tc_teorico_seg / 60,
        'real': tc_real_seg / 60,
        't_seg': tc_teorico_seg,
        'r_seg': tc_real_seg,
        'df_lotes': lotes,
        'df_flujo': flujo_real,
        'producto': df[cols['Producto']].iloc[0] if cols['Producto'] in df else "N/A",
        'operacion': df[cols['Operacion']].iloc[0] if cols['Operacion'] in df else "N/A"
    }

# --- 3. UI Y RESULTADOS ---
uploaded_file = st.file_uploader("Sube el archivo (XLS, TXT, CSV)", type=["xls", "xml", "xlsx", "csv", "txt"])

if uploaded_file:
    with st.spinner("🤖 Aplicando inteligencia de reparto de carga..."):
        df_raw, cols_map = load_data(uploaded_file)
        
        if df_raw is not None and cols_map['Fecha']:
            res = analyze_smart_flow(df_raw, cols_map)
            
            if res:
                # IDENTIDAD
                st.success(f"📌 Operación: **{res['operacion']}** | Producto: **{res['producto']}**")
                
                # KPIs (Diseño Limpio)
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO (Target)", f"{res['teo']:.2f} min", 
                          help=f"Ritmo de excelencia basado en el mejor 25% de los lotes: {res['t_seg']:.1f}s")
                c2.metric("⏱️ TC REAL (Sostenido)", f"{res['real']:.2f} min",
                          delta=f"{((res['real']/res['teo'])-1)*100:.1f}% Variabilidad", delta_color="inverse")
                
                h_turno = st.sidebar.number_input("Horas Turno", value=8.0)
                capacidad = (h_turno * 60) / res['teo']
                c3.metric("📦 Capacidad Nominal", f"{int(capacidad)} uds")

                st.divider()

                # GRÁFICA DE DISTRIBUCIÓN
                st.subheader("📊 Distribución del Tiempo Imputado")
                st.markdown("Esta gráfica muestra el tiempo real por pieza tras repartir la carga de los lotes.")
                
                fig = px.histogram(res['df_flujo'], x="tc_imputado", nbins=100, 
                                 title="Histograma de Ritmos (5s - 30min)",
                                 labels={'tc_imputado': 'Segundos por Unidad'},
                                 color_discrete_sequence=['#3498db'])
                fig.add_vline(x=res['t_seg'], line_dash="dash", line_color="red", line_width=4, annotation_text="Teórico")
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("🔍 Auditoría: ¿Cómo se calculó esto?"):
                    st.write("La IA identificó los 'silencios' de Spectrum y repartió ese tiempo entre las piezas que entraron de golpe.")
                    st.dataframe(res['df_lotes'].sort_values('piezas_en_lote', ascending=False).head(15))
            else:
                st.error("No se pudo procesar el flujo de datos.")
