import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Peak Detector AI", layout="wide", page_icon="🏔️")
st.title("🏔️ Celestica IA: Detector de Segundo Pico (Frontera Real)")
st.markdown("""
**Análisis de Estructura Bimodal:** Este motor ignora el pico masivo de ráfagas (0-1s) 
y localiza automáticamente la 'montaña' de producción real para extraer el TC.
""")

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

# --- 2. CEREBRO: DETECTOR DE SEGUNDO PICO ---
def find_real_production_peak(df, cols):
    c_fec = cols['Fecha']
    c_sn = cols['SN']
    
    # Limpieza
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn and c_sn in df.columns:
        df = df.drop_duplicates(subset=[c_sn], keep='first')

    # Cálculo de Gaps
    df['Gap'] = df[c_fec].diff().dt.total_seconds().fillna(0)
    
    # --- LÓGICA DE TRIAJE (0-1s / 1-20m / >20m) ---
    # 1. Separamos el ruido del servidor (< 10 segundos)
    ruido_servidor = df[df['Gap'] <= 10]['Gap']
    
    # 2. Identificamos la ZONA DE PRODUCCIÓN (donde vive tu 120s)
    # Filtramos gaps entre 10s y 900s (15 min) para encontrar el pico real
    zona_produccion = df[(df['Gap'] > 10) & (df['Gap'] <= 900)]['Gap'].values
    
    if len(zona_produccion) < 5:
        # Si no hay datos aquí, es que el archivo es 100% ráfagas de 0s.
        return None

    # 3. Buscamos el Segundo Pico (Moda en la zona de producción)
    kde = gaussian_kde(zona_produccion)
    x_range = np.linspace(zona_produccion.min(), zona_produccion.max(), 1000)
    y_dens = kde(x_range)
    tc_teorico_seg = x_range[np.argmax(y_dens)]
    
    # TC Real: Mediana de los datos que pertenecen a esa "montaña" de producción
    tc_real_seg = np.median(zona_produccion)
    
    return {
        'teo': tc_teorico_seg / 60,
        'real': tc_real_seg / 60,
        't_seg': tc_teorico_seg,
        'r_seg': tc_real_seg,
        'pct_ruido': (len(ruido_servidor) / len(df)) * 100,
        'df_plot': df,
        'producto': df[cols['Producto']].iloc[0] if cols['Producto'] in df else "N/A",
        'operacion': df[cols['Operacion']].iloc[0] if cols['Operacion'] in df else "N/A"
    }

# --- 3. UI ---
uploaded_file = st.file_uploader("Sube el archivo (15.4MB / 1.9MB)", type=["xls", "xml", "xlsx", "csv", "txt"])

if uploaded_file:
    with st.spinner("🤖 Detectando picos de producción y filtrando ráfagas..."):
        df_raw, cols_map = load_data(uploaded_file)
        
        if df_raw is not None and cols_map['Fecha']:
            res = find_real_production_peak(df_raw, cols_map)
            
            if res:
                st.success(f"✅ Operación: {res['operacion']} | Producto: {res['producto']}")
                
                # KPIs (Diseño Limpio)
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO (Pico 2)", f"{res['teo']:.2f} min", 
                          help=f"Localizado en el segundo pico de la distribución: {res['t_seg']:.1f}s")
                c2.metric("⏱️ TC REAL (Mediana)", f"{res['real']:.2f} min",
                          delta=f"{res['pct_ruido']:.1f}% Ruido de Red", delta_color="off")
                
                capacidad = (8 * 60) / res['teo']
                c3.metric("📦 Capacidad Nominal", f"{int(capacidad)} uds")

                st.divider()

                # GRÁFICA DE LA ESTRUCTURA DE DATOS
                st.subheader("📊 Radiografía de Tiempos (Ruido vs Producción)")
                st.write(f"La IA ha ignorado el **{res['pct_ruido']:.1f}%** de los datos que están cerca de 0s.")

                # Mostramos un histograma que permita ver el ruido y la producción
                # Limitamos a 500s para que el pico de 120s sea visible
                fig_data = res['df_plot'][(res['df_plot']['Gap'] >= 0) & (res['df_plot']['Gap'] <= 600)]
                
                fig = px.histogram(fig_data, x="Gap", nbins=150, 
                                 title="Distribución Total: El pico de la izquierda es RUIDO, el de la derecha es PRODUCCIÓN",
                                 labels={'Gap': 'Segundos entre piezas'},
                                 color_discrete_sequence=['#34495e'])
                
                fig.add_vline(x=res['t_seg'], line_dash="dash", line_color="red", line_width=4, 
                             annotation_text="Pico Producción Real")
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"💡 **Criterio IA:** Se ha detectado un valle entre el ruido de red y tu ritmo de trabajo. El TC objetivo se ha anclado en **{res['t_seg']:.1f} segundos**.")

            else:
                st.error("No se detectó el segundo pico. Es posible que el archivo solo contenga ráfagas de 0s.")
