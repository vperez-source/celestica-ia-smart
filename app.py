import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Engineering Tool", layout="wide", page_icon="⚙️")
st.title("⚙️ Celestica IA: Smart-Tracker de Valor Añadido")

# --- 1. MOTOR DE CARGA ROBUSTO ---
@st.cache_data(ttl=3600)
def load_data_pro(file):
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

    # Buscador de cabeceras
    df = df.astype(str)
    header_idx = 0
    for i in range(min(60, len(df))):
        row_str = " ".join(df.iloc[i]).lower()
        if any(x in row_str for x in ['date', 'time', 'fecha', 'sn', 'serial']):
            header_idx = i; break
    
    df.columns = df.iloc[header_idx].str.strip()
    df = df[header_idx + 1:].reset_index(drop=True)

    cols = {
        'Fecha': next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None),
        'SN': next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None),
        'Producto': next((c for c in df.columns if any(x in c.lower() for x in ['product', 'item', 'part'])), "Producto"),
        'Operacion': next((c for c in df.columns if any(x in c.lower() for x in ['station', 'oper', 'step', 'workcenter'])), "Operación")
    }
    return df, cols

# --- 2. CEREBRO: FILTRO DE VALOR REAL ---
def analyze_real_added_value(df, cols):
    c_fec = cols['Fecha']
    c_sn = cols['SN']
    
    # 1. Preparación y limpieza de fechas
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn and c_sn in df.columns:
        df = df.drop_duplicates(subset=[c_sn], keep='first')

    # 2. Extraer Producto y Operación (Dominantes)
    prod_label = df[cols['Producto']].mode()[0] if cols['Producto'] in df.columns else "N/A"
    oper_label = df[cols['Operacion']].mode()[0] if cols['Operacion'] in df.columns else "N/A"

    # 3. Lógica de Gaps (Tiempo entre registros)
    df['Gap'] = df[c_fec].diff().dt.total_seconds().fillna(0)
    
    # --- APLICACIÓN DEL CRITERIO 80/15/5 ---
    # Ignoramos el 80% de ruido (gaps < 20 segundos)
    # Ignoramos el 5% de paradas (gaps > 25 minutos / 1500s)
    datos_produccion = df[(df['Gap'] >= 20) & (df['Gap'] <= 1500)]['Gap']
    
    if len(datos_produccion) < 3:
        # Modo de rescate: Si no hay gaps, es un lote masivo. 
        # Dividimos el tiempo total entre piezas.
        total_time = (df[c_fec].max() - df[c_fec].min()).total_seconds()
        tc_manual = total_time / len(df) if len(df) > 0 else 0
        return {
            'teo': tc_manual/60, 'real': tc_manual/60, 'status': 'Cálculo por Promedio de Lote',
            'prod': prod_label, 'oper': oper_label, 'grafica': [tc_manual]
        }

    # TC Teórico: El mejor ritmo dentro de ese 15% de datos (Percentil 15)
    tc_teorico_seg = np.percentile(datos_produccion, 15)
    # TC Real: La mediana del pasillo de producción
    tc_real_seg = datos_produccion.median()

    return {
        'teo': tc_teorico_seg / 60,
        'real': tc_real_seg / 60,
        't_seg': tc_teorico_seg,
        'prod': prod_label,
        'oper': oper_label,
        'grafica': datos_produccion,
        'status': 'Análisis de Flujo Sostenido'
    }

# --- 3. UI Y RESULTADOS ---
uploaded_file = st.file_uploader("Sube el archivo de Celestica (XLS, TXT, CSV)", type=["xls", "xml", "xlsx", "csv", "txt"])

if uploaded_file:
    with st.spinner("🤖 Filtrando ruido y localizando tiempo de ciclo real..."):
        df_raw, cols_map = load_data_pro(uploaded_file)
        
        if df_raw is not None and cols_map['Fecha']:
            res = analyze_real_added_value(df_raw, cols_map)
            
            if res:
                # PANEL DE IDENTIDAD
                st.success(f"📋 **Operación:** {res['oper']} | **Producto:** {res['prod']}")
                
                # KPIs PRINCIPALES
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO (Target)", f"{res['teo']:.2f} min", 
                          help=f"Ritmo de eficiencia detectado: {res['t_seg']:.1f} segundos.")
                c2.metric("⏱️ TC REAL (Mediana)", f"{res['real']:.2f} min")
                
                h_turno = st.sidebar.number_input("Horas Turno", value=8.0)
                capacidad = (h_turno * 60) / res['teo']
                c3.metric("📦 Capacidad (8h)", f"{int(capacidad)} uds")

                st.divider()

                # GRÁFICA DE DISTRIBUCIÓN
                st.subheader("📊 Distribución de Tiempos de Producción")
                st.markdown("Esta gráfica muestra **solo tu 15% de datos reales**, habiendo eliminado el ruido de red y los parones largos.")
                
                fig = px.histogram(res['grafica'], x=res['grafica'], nbins=50, 
                                 title="Histograma de Ritmo Manual (Segundos)",
                                 labels={'x': 'Segundos por Pieza'},
                                 color_discrete_sequence=['#2ecc71'])
                fig.add_vline(x=res['t_seg'], line_dash="dash", line_color="red", line_width=4)
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("🔍 Auditoría de Datos Raw"):
                    st.dataframe(df_raw.head(20))
            else:
                st.error("No se pudo extraer información. Revisa el formato del archivo.")
