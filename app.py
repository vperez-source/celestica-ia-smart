import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Frontier AI", layout="wide", page_icon="🚀")
st.title("🚀 Celestica IA: Cálculo de Tiempo de Ciclo Teórico")
st.markdown("""
**Análisis de Frontera de Eficiencia:** Este algoritmo ignora las 'colas' de ineficiencia (Gamma distribution) 
y calcula el ritmo de ejecución ideal basándose en el mejor rendimiento sostenido.
""")

with st.sidebar:
    st.header("⚙️ Parámetros de Análisis")
    p_excelencia = st.slider("Percentil de Excelencia (Teórico)", 5, 50, 25, 
                             help="El percentil 25 representa el ritmo del mejor 25% de las piezas. Es tu 'Tiempo de Ciclo Teórico'.")
    st.divider()
    h_turno = st.number_input("Horas Turno", value=8)

# --- FASE A: INGESTIÓN ROBUSTA (XML 2003) ---
def parse_xml_tanque(file):
    try:
        content = file.getvalue().decode('latin-1', errors='ignore')
        soup = BeautifulSoup(content, 'lxml-xml')
        data = []
        # Buscamos filas de forma masiva
        for row in soup.find_all(['Row', 'ss:Row']):
            cells = [c.get_text(strip=True) for c in row.find_all(['Cell', 'ss:Cell'])]
            if any(cells): data.append(cells)
        return pd.DataFrame(data)
    except: return None

@st.cache_data(ttl=3600)
def load_data(file):
    df = parse_xml_tanque(file)
    if df is None or df.empty:
        try:
            file.seek(0)
            df = pd.read_excel(file, header=None)
        except: return None, None

    # FASE B: MAPEO DINÁMICO
    df = df.astype(str)
    start_row = -1
    for i in range(min(100, len(df))):
        row_str = " ".join(df.iloc[i].astype(str)).lower()
        if 'date' in row_str or 'time' in row_str:
            start_row = i; break
            
    if start_row == -1: return None, None

    df.columns = df.iloc[start_row]
    df = df[start_row + 1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()

    # Identificar columna fecha
    col_fec = next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None)
    return df, col_fec

# --- FASE C: ALGORITMO DE FRONTERA ---
def calcular_frontera_teorica(df, col_fec, p_target):
    # 1. Limpieza y conversión
    df[col_fec] = pd.to_datetime(df[col_fec], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[col_fec]).sort_values(col_fec)
    
    # 2. De-batching (Reparto de carga)
    # Agrupamos por segundo
    batches = df.groupby(col_fec).size().reset_index(name='piezas')
    # Tiempo entre lotes
    batches['gap'] = batches[col_fec].diff().dt.total_seconds().fillna(0)
    # Tiempo unitario imputado
    batches['tc_unitario'] = batches['gap'] / batches['piezas']
    
    # 3. FILTRADO DE RUIDO (Sin sesgar la frontera)
    # Solo eliminamos lo que es físicamente imposible (0 seg) y paradas absurdas (> 1h)
    data_limpia = batches[(batches['tc_unitario'] > 0.1) & (batches['tc_unitario'] < 3600)]['tc_unitario']
    
    if data_limpia.empty: return 0, 0, batches

    # 4. CÁLCULO TEÓRICO (Percentil)
    # En una distribución Gamma, el valor 'teórico' es el límite inferior de la montaña
    tc_teorico_seg = np.percentile(data_limpia, p_target)
    tc_real_medio_seg = data_limpia.median()
    
    return tc_teorico_seg / 60, tc_real_medio_seg / 60, batches

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Subir Archivo (.xls, .xml)", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🔍 Extrayendo frontera de eficiencia..."):
        df, col_fec = load_data(uploaded_file)
        
        if df is not None and col_fec:
            tc_teorico, tc_real, batches = calcular_frontera_teorica(df, col_fec, p_excelencia)
            
            if tc_teorico > 0:
                st.success("✅ Análisis de Capacidad Teórica Finalizado")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO (Target)", f"{tc_teorico:.2f} min", 
                          help="Este es el tiempo de ciclo al que puedes aspirar eliminando ineficiencias.")
                c2.metric("⏱️ TC REAL (Mediana)", f"{tc_real:.2f} min", 
                          delta=f"{((tc_real/tc_teorico)-1)*100:.1f}% Pérdida", delta_color="inverse")
                
                capacidad_teorica = (h_turno * 60) / tc_teorico
                c3.metric("📦 Capacidad Ideal", f"{int(capacidad_teorica)} uds", help="Producción si se mantuviera el ritmo de excelencia.")

                st.divider()

                # --- VISUALIZACIÓN GAMMA ---
                st.subheader("📊 Distribución Gamma de la Producción")
                st.markdown(f"La línea **AZUL** es tu realidad actual. La línea **ROJA** es tu potencial (TC Teórico).")
                
                # Filtramos para el gráfico (solo mostrar hasta 3x el tiempo medio para ver la montaña)
                fig_data = batches[(batches['tc_unitario'] > 0) & (batches['tc_unitario'] < tc_real*180)]
                
                fig = px.histogram(fig_data, x="tc_unitario", nbins=100, 
                                 title="Histograma de Tiempos Unitarios",
                                 labels={'tc_unitario': 'Segundos por Pieza'},
                                 color_discrete_sequence=['#95a5a6'])
                
                fig.add_vline(x=tc_real*60, line_color="#3498db", line_width=3, annotation_text="Media Real")
                fig.add_vline(x=tc_teorico*60, line_color="#e74c3c", line_width=4, annotation_text="OBJETIVO TEÓRICO")
                
                st.plotly_chart(fig, use_container_width=True)

                st.info(f"💡 **Asesoría:** Tu proceso tiene una variabilidad del {((tc_real/tc_teorico)-1)*100:.0f}%. El objetivo es desplazar la montaña hacia la izquierda (la zona roja) mediante la eliminación de micro-paradas.")

        else:
            st.error("No se pudo detectar la columna de fecha. Revisa el formato del archivo.")
