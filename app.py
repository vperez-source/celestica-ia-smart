import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Frontier AI", layout="wide", page_icon="📈")
st.title("📈 Celestica IA: Análisis de Frontera de Eficiencia")
st.markdown("""
**Criterio Econométrico:** Este modelo ignora las pausas de inactividad y el ruido de ráfagas (batching) 
calculando el ritmo de flujo en 'ventanas de densidad'.
""")

# --- 1. MOTOR DE CARGA ---
@st.cache_data(ttl=3600)
def load_and_map(file):
    fname = file.name.lower()
    df = None
    try:
        if fname.endswith(('.xml', '.xls')):
            content = file.getvalue().decode('latin-1', errors='ignore')
            if "<?xml" in content or "Workbook" in content:
                soup = BeautifulSoup(content, 'lxml-xml')
                data = [[c.get_text(strip=True) for c in row.find_all(['Cell', 'ss:Cell'])] 
                        for row in soup.find_all(['Row', 'ss:Row'])]
                df = pd.DataFrame([d for d in data if d])
            else:
                file.seek(0)
                df = pd.read_excel(file, header=None)
        else:
            file.seek(0)
            df = pd.read_csv(file, sep=None, engine='python', header=None)
    except: return None, {}

    if df is None or df.empty: return None, {}

    # Buscador de cabeceras
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

# --- 2. CEREBRO: ESTIMADOR DE FRONTERA ROBUSTA ---
def analyze_econometric_flow(df, cols):
    c_fec = cols['Fecha']
    c_sn = cols['SN']
    
    # Limpieza de datos
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn and c_sn in df.columns:
        df = df.drop_duplicates(subset=[c_sn], keep='first')

    # A. CREACIÓN DE VENTANAS DE ACTIVIDAD (Binning de 5 minutos)
    # Esto "empaqueta" el batching masivo y lo convierte en piezas/minuto
    df.set_index(c_fec, inplace=True)
    window_size = '5Min'
    throughput = df.resample(window_size).size().reset_index(name='piezas')
    
    # B. FILTRO DE INACTIVIDAD (Criterio Econométrico)
    # Solo analizamos periodos donde el sistema registró actividad real (> 0 piezas)
    # Esto elimina las pausas grandes automáticamente.
    actividad = throughput[throughput['piezas'] > 0].copy()
    
    if actividad.empty: return None

    # C. CÁLCULO DEL TIEMPO DE CICLO UNITARIO POR VENTANA
    # (5 min * 60 seg) / número de piezas en ese bloque
    actividad['tc_ventana_seg'] = 300 / actividad['piezas']
    
    # D. DETERMINACIÓN DE LAS MÉTRICAS (Uso de Percentiles)
    # El TEÓRICO es el P20 (el ritmo del mejor 20% de tus ventanas activas)
    # Es mucho más estable que la moda y más ambicioso que la mediana.
    tc_teorico_seg = np.percentile(actividad['tc_ventana_seg'], 20)
    
    # El REAL es la Mediana (el centro de tu capacidad actual de flujo)
    tc_real_seg = actividad['tc_ventana_seg'].median()
    
    return {
        'teo': tc_teorico_seg / 60,
        'real': tc_real_seg / 60,
        't_seg': tc_teorico_seg,
        'r_seg': tc_real_seg,
        'df_v': actividad,
        'producto': df[cols['Producto']].iloc[0] if cols['Producto'] in df else "N/A",
        'operacion': df[cols['Operacion']].iloc[0] if cols['Operacion'] in df else "N/A"
    }

# --- 3. UI ---
uploaded_file = st.file_uploader("Sube el archivo (XLS, TXT, CSV)", type=["xls", "xml", "xlsx", "csv", "txt"])

if uploaded_file:
    with st.spinner("🤖 Aplicando lógica de frontera de eficiencia..."):
        df_raw, cols_map = load_and_map(uploaded_file)
        
        if df_raw is not None and cols_map['Fecha']:
            res = analyze_econometric_flow(df_raw, cols_map)
            
            if res:
                # HEADER INFORMATIVO
                st.success(f"📌 {res['operacion']} | {res['producto']}")
                
                # KPIs (Diseño Limpio)
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO (Best Practice)", f"{res['teo']:.2f} min", 
                          help=f"Representa el ritmo alcanzado en tus periodos más eficientes ({res['t_seg']:.1f}s).")
                c2.metric("⏱️ TC REAL (Mediana Flujo)", f"{res['real']:.2f} min",
                          delta=f"{((res['real']/res['teo'])-1)*100:.1f}% Gap de Eficiencia", delta_color="inverse")
                
                capacidad = (8 * 60) / res['teo']
                c3.metric("📦 Capacidad Nominal (8h)", f"{int(capacidad)} uds")

                st.divider()

                # GRÁFICA DE CONTROL DE FLUJO
                st.subheader("📊 Estabilidad de la Producción (Ventanas Activas)")
                st.caption("Los puntos muestran el tiempo de ciclo en cada bloque de 5 minutos de trabajo.")
                
                fig = px.scatter(res['df_v'], x=res['df_v'].columns[0], y='tc_ventana_seg', 
                                title="Evolución del Ritmo Unitario",
                                labels={'tc_ventana_seg': 'Segundos / Pieza'},
                                color='piezas', color_continuous_scale='Portland')
                
                fig.add_hline(y=res['t_seg'], line_dash="dash", line_color="red", line_width=3, annotation_text="Teórico")
                fig.add_hline(y=res['r_seg'], line_dash="dot", line_color="blue", annotation_text="Mediana")
                
                st.plotly_chart(fig, use_container_width=True)

                # TABLA DE AUDITORÍA
                with st.expander("🔍 Auditoría de Ventanas"):
                    st.write("Datos agrupados por ventanas de 5 minutos (solo periodos productivos):")
                    st.dataframe(res['df_v'].sort_values('piezas', ascending=False), use_container_width=True)
            else:
                st.error("No se detectó actividad productiva.")
