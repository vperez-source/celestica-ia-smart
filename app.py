import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Process Intelligence", layout="wide", page_icon="⚙️")
st.title("⚙️ Celestica IA: Smart-Tracker de Valor Añadido")

# --- 1. MOTOR DE CARGA ROBUSTO ---
@st.cache_data(ttl=3600)
def load_data_universal(file):
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

# --- 2. CEREBRO DE INGENIERÍA: REPARTO DE LOTES ---
def analyze_pulse_flow(df, cols):
    c_fec = cols['Fecha']
    c_sn = cols['SN']
    
    # Limpieza
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn and c_sn in df.columns:
        df = df.drop_duplicates(subset=[c_sn], keep='first')

    # Identificar nombres para el reporte
    prod_name = df[cols['Producto']].iloc[0] if cols['Producto'] in df.columns else "N/A"
    oper_name = df[cols['Operacion']].iloc[0] if cols['Operacion'] in df.columns else "N/A"

    # LÓGICA DE REPARTO DE LOTE
    # 1. Agrupar por segundo para detectar ráfagas
    lotes = df.groupby(c_fec).size().reset_index(name='piezas_lote')
    # 2. Calcular tiempo desde el lote anterior
    lotes['gap_previo'] = lotes[c_fec].diff().dt.total_seconds().fillna(0)
    # 3. Repartir el tiempo: $$TC_{repartido} = \frac{Gap}{Piezas}$$
    lotes['tc_repartido'] = lotes['gap_previo'] / lotes['piezas_lote']
    
    # 4. FILTRAR VALOR AÑADIDO (Tu 15% de datos)
    # Ignoramos el ruido de 0-1s y las paradas de más de 30 min (1800s)
    flujo_real = lotes[(lotes['tc_repartido'] >= 10) & (lotes['tc_repartido'] <= 1800)].copy()

    if flujo_real.empty:
        # Fallback seguro con todas las llaves necesarias para evitar KeyError
        return {
            'teo': 0, 'real': 0, 't_seg': 0, 'r_seg': 0,
            'prod': prod_name, 'oper': oper_name, 'grafica': pd.DataFrame()
        }

    # TC Teórico: El mejor ritmo del pasillo (Percentil 25)
    tc_teorico_seg = np.percentile(flujo_real['tc_repartido'], 25)
    # TC Real: La mediana del pasillo
    tc_real_seg = flujo_real['tc_repartido'].median()

    return {
        'teo': tc_teorico_seg / 60,
        'real': tc_real_seg / 60,
        't_seg': tc_teorico_seg,
        'r_seg': tc_real_seg,
        'prod': prod_name,
        'oper': oper_name,
        'grafica': flujo_real
    }

# --- 3. UI ---
uploaded_file = st.file_uploader("Sube el archivo de 15.4MB (XLS, TXT, CSV)", type=["xls", "xml", "xlsx", "csv", "txt"])

if uploaded_file:
    with st.spinner("🤖 Analizando lotes y detectando valor añadido..."):
        df_raw, cols_map = load_data_universal(uploaded_file)
        
        if df_raw is not None and cols_map['Fecha']:
            res = analyze_pulse_flow(df_raw, cols_map)
            
            if res['t_seg'] > 0:
                # PANEL DE IDENTIDAD
                st.success(f"📋 **Operación:** {res['oper']} | **Producto:** {res['prod']}")
                
                # KPIs
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO", f"{res['teo']:.2f} min", 
                          help=f"Ritmo de excelencia (P25): {res['t_seg']:.1f}s")
                c2.metric("⏱️ TC REAL", f"{res['real']:.2f} min",
                          delta=f"{((res['real']/res['teo'])-1)*100:.1f}% Desvío", delta_color="inverse")
                
                h_turno = st.sidebar.number_input("Horas Turno", value=8.0)
                capacidad = (h_turno * 60) / res['teo']
                c3.metric("📦 Capacidad (8h)", f"{int(capacidad)} uds")

                st.divider()

                # GRÁFICA DEL 15% REAL
                st.subheader("📊 Distribución del Tiempo de Transformación")
                st.markdown("Esta gráfica muestra los tiempos tras repartir la carga de los lotes y eliminar paradas.")
                
                fig = px.histogram(res['grafica'], x="tc_repartido", nbins=50, 
                                 title="Ritmos Unitarios Reconstruidos (Segundos)",
                                 color_discrete_sequence=['#2ecc71'])
                fig.add_vline(x=res['t_seg'], line_dash="dash", line_color="red", line_width=4)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se detectó un flujo de valor añadido. Los datos indican que todas las piezas entraron sin tiempo de espera previo.")
