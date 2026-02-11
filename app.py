import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- CONFIGURACIÓN DE INTERFAZ ---
st.set_page_config(page_title="Celestica Engineering AI", layout="wide", page_icon="⚙️")
st.title("⚙️ Celestica IA: Smart-Tracker con Contexto de Operación")

with st.sidebar:
    st.header("⚙️ Baseline de Ingeniería")
    h_turno = st.number_input("Horas Turno", value=8.0)
    st.divider()
    st.info("Algoritmo v16.0: Detección automática de Producto y Estación de trabajo.")

# --- 1. MOTOR DE CARGA MULTI-FORMATO ---
@st.cache_data(ttl=3600)
def load_any_file(file):
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
        elif fname.endswith('.csv'):
            file.seek(0)
            df = pd.read_csv(file, sep=None, engine='python', header=None)
        elif fname.endswith('.txt'):
            file.seek(0)
            df = pd.read_csv(file, sep='\t', header=None)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return None, {}

    if df is None or df.empty: return None, {}

    # BUSCADOR DINÁMICO DE CABECERAS Y COLUMNAS
    df = df.astype(str)
    header_idx = -1
    for i in range(min(50, len(df))):
        row_str = " ".join(df.iloc[i]).lower()
        if any(x in row_str for x in ['date', 'time', 'fecha', 'sn', 'serial']):
            header_idx = i
            break
            
    if header_idx == -1: return None, {}
    
    df.columns = df.iloc[header_idx].str.strip()
    df = df[header_idx + 1:].reset_index(drop=True)

    # Mapeo de Identidad (Producto y Operación)
    cols = {
        'Fecha': next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None),
        'SN': next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None),
        'Producto': next((c for c in df.columns if any(x in c.lower() for x in ['product', 'item', 'part'])), "N/A"),
        'Operacion': next((c for c in df.columns if any(x in c.lower() for x in ['station', 'oper', 'step', 'process', 'workcenter'])), "N/A")
    }
    return df, cols

# --- 2. CEREBRO DE CÁLCULO ---
def analyze_with_context(df, cols):
    c_fec = cols['Fecha']
    c_sn = cols['SN']
    
    # Limpieza
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn and c_sn != "N/A": 
        df = df.drop_duplicates(subset=[c_sn], keep='first')
    
    # Cálculo de Gap Humano (Filtro 40s - 600s)
    df['Gap'] = df[c_fec].diff().dt.total_seconds().fillna(0)
    ritmos_validos = df[(df['Gap'] >= 40) & (df['Gap'] <= 600)]['Gap']
    
    if len(ritmos_validos) < 5:
        return None

    # Moda (Teórico) y Mediana (Real)
    kde = gaussian_kde(ritmos_validos)
    x_range = np.linspace(ritmos_validos.min(), ritmos_validos.max(), 1000)
    tc_teorico_seg = x_range[np.argmax(kde(x_range))]
    tc_real_seg = ritmos_validos.median()
    
    # Identificar el Producto y Operación dominante
    prod_main = df[cols['Producto']].mode()[0] if cols['Producto'] in df else "Desconocido"
    oper_main = df[cols['Operacion']].mode()[0] if cols['Operacion'] in df else "Desconocida"

    return {
        'teo': tc_teorico_seg / 60,
        'real': tc_real_seg / 60,
        'producto': prod_main,
        'operacion': oper_main,
        'datos_v': ritmos_validos,
        'total_p': len(df)
    }

# --- 3. UI Y RESULTADOS ---
uploaded_file = st.file_uploader("Sube tu reporte (XLS, XML, CSV o TXT)", type=["xls", "xml", "xlsx", "csv", "txt"])

if uploaded_file:
    with st.spinner("🤖 Identificando producto y analizando ritmos..."):
        df_raw, cols_map = load_any_file(uploaded_file)
        
        if df_raw is not None and cols_map.get('Fecha'):
            res = analyze_with_context(df_raw, cols_map)
            
            if res:
                st.success(f"✅ Análisis completado para la Operación: **{res['operacion']}**")
                
                # PANEL DE IDENTIDAD
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.info(f"📦 **Producto detectado:** {res['producto']}")
                with col_info2:
                    st.info(f"⚙️ **Estación/Operación:** {res['operacion']}")

                st.divider()

                # KPIs PRINCIPALES
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO", f"{res['teo']:.2f} min", help="El ritmo más puro detectado.")
                c2.metric("⏱️ TC REAL", f"{res['real']:.2f} min", 
                          delta=f"{((res['real']/res['teo'])-1)*100:.1f}% Desvío", delta_color="inverse")
                
                capacidad = (h_turno * 60) / res['teo']
                c3.metric("📦 Capacidad Turno", f"{int(capacidad)} uds")

                st.divider()
                
                # GRÁFICA DE DISTRIBUCIÓN
                st.subheader("📊 Radiografía de Ritmos Sostenidos")
                fig = px.histogram(res['datos_v'], x="Gap", nbins=50, 
                                 title=f"Distribución en {res['operacion']} (Segundos)",
                                 labels={'Gap': 'Segundos por Pieza'},
                                 color_discrete_sequence=['#3498db'])
                fig.add_vline(x=res['teo']*60, line_dash="dash", line_color="red", line_width=4, annotation_text="Teórico")
                st.plotly_chart(fig, use_container_width=True)

                # TABLA DETALLADA (Si hay varios productos/estaciones en el mismo archivo)
                with st.expander("🔍 Ver desglose por todas las estaciones encontradas"):
                    resumen = df_raw.groupby([cols_map['Operacion'], cols_map['Producto']]).size().reset_index(name='Total Piezas')
                    st.dataframe(resumen, use_container_width=True)
            else:
                st.error("La IA no detectó un flujo humano (gaps entre 40s y 600s). Revisa si los segundos están presentes en el archivo.")
        else:
            st.error("No se detectó la estructura del archivo. Asegúrate de que las cabeceras estén claras.")
