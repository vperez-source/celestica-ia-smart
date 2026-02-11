import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Smart-Tracker Pro", layout="wide", page_icon="⚙️")
st.title("⚙️ Celestica IA: Smart-Tracker (Value-Added Analysis)")

# --- 1. MOTOR DE CARGA MULTIFORMATO ---
@st.cache_data(ttl=3600)
def load_data_robust(file):
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

    # Buscador de cabeceras dinámico
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

# --- 2. CEREBRO DE CÁLCULO (INMUNE A KEYERROR) ---
def analyze_manufacturing_flow(df, cols):
    c_fec = cols['Fecha']
    c_sn = cols['SN']
    
    # 1. Limpieza estricta
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn and c_sn in df.columns:
        df = df.drop_duplicates(subset=[c_sn], keep='first')

    # 2. Identificar el Producto y Operación dominante ANTES del cálculo
    prod_label = df[cols['Producto']].mode()[0] if cols['Producto'] in df.columns else "Desconocido"
    oper_label = df[cols['Operacion']].mode()[0] if cols['Operacion'] in df.columns else "Desconocida"

    # 3. Lógica de Lote (De-batching)
    lotes = df.groupby(c_fec).size().reset_index(name='piezas')
    lotes['gap'] = lotes[c_fec].diff().dt.total_seconds().fillna(0)
    lotes['tc_imputado'] = lotes['gap'] / lotes['piezas']
    
    # 4. Filtrado de Valor Añadido (Ignorar ráfagas < 10s y paradas > 30m)
    v_added = lotes[(lotes['tc_imputado'] >= 10) & (lotes['tc_imputado'] <= 1800)].copy()
    
    if v_added.empty:
        # Fallback de seguridad: siempre devuelve las mismas llaves para evitar KeyError
        total_sec = (df[c_fec].max() - df[c_fec].min()).total_seconds()
        tc_promedio = (total_sec / len(df)) if len(df) > 0 else 0
        return {
            'teo': tc_promedio / 60, 'real': tc_promedio / 60, 't_seg': tc_promedio,
            'producto': prod_label, 'operacion': oper_label, 'df_v': lotes, 'status': 'Calculado por Promedio Total'
        }

    # 5. Cálculo de Métricas (Percentiles sobre valor añadido)
    tc_teorico_seg = np.percentile(v_added['tc_imputado'], 20) # El 20% más eficiente
    tc_real_seg = v_added['tc_imputado'].median() # El ritmo constante real

    return {
        'teo': tc_teorico_seg / 60,
        'real': tc_real_seg / 60,
        't_seg': tc_teorico_seg,
        'r_seg': tc_real_seg,
        'producto': prod_label,
        'operacion': oper_label,
        'df_v': v_added,
        'status': 'Análisis de Flujo Activo'
    }

# --- 3. UI Y DASHBOARD ---
uploaded_file = st.file_uploader("Sube el archivo (XLS, TXT, CSV)", type=["xls", "xml", "xlsx", "csv", "txt"])

if uploaded_file:
    with st.spinner("🤖 Aplicando lógica de ingeniería de métodos..."):
        df_raw, cols_map = load_data_robust(uploaded_file)
        
        if df_raw is not None and cols_map['Fecha']:
            res = analyze_manufacturing_flow(df_raw, cols_map)
            
            # IDENTIFICACIÓN CLARA (Fuera del bloque de éxito para seguridad)
            st.success(f"📌 Operación: **{res['operacion']}** | Producto: **{res['producto']}**")
            st.info(f"Metodología: {res['status']}")

            # KPIs
            c1, c2, c3 = st.columns(3)
            c1.metric("⏱️ TC TEÓRICO", f"{res['teo']:.2f} min", help=f"Punto de máxima eficiencia detectado: {res['t_seg']:.1f}s")
            c2.metric("⏱️ TC REAL", f"{res['real']:.2f} min", 
                      delta=f"{((res['real']/res['teo'])-1)*100:.1f}% Desvío", delta_color="inverse")
            
            h_turno = st.sidebar.number_input("Horas Turno", value=8.0)
            capacidad = (h_turno * 60) / res['teo']
            c3.metric("📦 Capacidad (8h)", f"{int(capacidad)} uds")

            st.divider()

            # GRÁFICA DE DISTRIBUCIÓN
            st.subheader("📊 Distribución de Tiempos de Valor Añadido")
            fig = px.histogram(res['df_v'][res['df_v']['tc_imputado'] < 600], x="tc_imputado", nbins=50, 
                             title="Frecuencia de Ritmos Reales (Segundos)",
                             color_discrete_sequence=['#2ecc71'])
            fig.add_vline(x=res['t_seg'], line_dash="dash", line_color="red", line_width=4, annotation_text="Teórico")
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("🔍 Ver datos del sistema (Lotes detectados)"):
                st.dataframe(df_raw.head(20))
        else:
            st.error("No se pudo leer la estructura del archivo. Revisa las columnas.")
