import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde
import io

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Universal AI", layout="wide", page_icon="🏭")
st.title("🏭 Celestica IA: Analizador Universal de Productividad")
st.markdown("""
**Instrucciones para el éxito:** Si guardas tu archivo como **TXT** o **CSV**, asegúrate de que la primera fila 
contenga los nombres de las columnas (Date, Serial Number, etc.) y que la fecha incluya segundos.
""")

# --- 1. MOTOR DE CARGA MULTI-FORMATO ---
@st.cache_data(ttl=3600)
def load_any_file(file):
    fname = file.name.lower()
    df = None
    
    try:
        if fname.endswith('.xml') or fname.endswith('.xls'):
            # Intento de lectura XML 2003
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
            df = pd.read_csv(file, sep='\t', header=None) # Delimitado por tabulaciones
            
    except Exception as e:
        st.error(f"Error técnico al leer el archivo: {e}")
        return None

    if df is None or df.empty: return None

    # BUSCADOR DE CABECERA (Criterio de Ingeniería)
    df = df.astype(str)
    for i in range(min(50, len(df))):
        row_str = " ".join(df.iloc[i]).lower()
        if any(x in row_str for x in ['date', 'time', 'station', 'sn', 'serial']):
            df.columns = df.iloc[i].str.strip()
            return df[i+1:].reset_index(drop=True)
            
    return None

# --- 2. CEREBRO DE INGENIERÍA: FILTRO DE COHERENCIA ---
def analyze_industrial_rhythm(df):
    # Identificar columnas
    c_fec = next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None)
    c_sn = next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None)
    
    if not c_fec: return None, "No se encontró columna de Fecha/Hora."

    # A. Limpieza Profunda
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn:
        df = df.drop_duplicates(subset=[c_sn], keep='first')
    
    # B. Cálculo de Gaps (Tiempo entre piezas)
    df['Gap'] = df[c_fec].diff().dt.total_seconds().fillna(0)
    
    # C. APLICACIÓN DEL CRITERIO FÍSICO (Para evitar alucinaciones)
    # 1. Filtro Humano: El TC real debe estar entre 40s y 600s.
    # 2. Si el gap es < 40s, es "Batching" (ruido de servidor).
    # 3. Si el gap es > 600s, es "Parada" (ineficiencia).
    
    ritmos_validos = df[(df['Gap'] >= 40) & (df['Gap'] <= 600)]['Gap']
    
    if len(ritmos_validos) < 5:
        # EXPLICACIÓN TÉCNICA SI FALLA
        motivo = "El archivo no contiene 'flujo humano'. "
        if df['Gap'].max() < 5:
            motivo += "Causa: Todas las piezas se registraron en el mismo segundo (Batching total)."
        else:
            motivo += "Causa: Hay demasiados huecos grandes (>10 min) entre piezas."
        return None, motivo

    # D. CÁLCULO DE RESULTADOS
    # TC TEÓRICO: Es el pico de la montaña (Moda)
    kde = gaussian_kde(ritmos_validos)
    x_range = np.linspace(ritmos_validos.min(), ritmos_validos.max(), 1000)
    tc_teorico_seg = x_range[np.argmax(kde(x_range))]
    
    # TC REAL: Es la mediana de la actividad
    tc_real_seg = ritmos_validos.median()
    
    return {
        'teo': tc_teorico_seg / 60,
        'real': tc_real_seg / 60,
        'datos_v': ritmos_validos,
        'total_p': len(df)
    }, None

# --- 3. INTERFAZ ---
uploaded_file = st.file_uploader("Sube tu archivo (XLS, XML, CSV o TXT)", type=["xls", "xml", "xlsx", "csv", "txt"])

if uploaded_file:
    with st.spinner("🤖 Procesando con criterio de ingeniería..."):
        df_raw = load_any_file(uploaded_file)
        
        if df_raw is not None:
            res, error_msg = analyze_industrial_rhythm(df_raw)
            
            if res:
                st.success("✅ Análisis completado con éxito")
                
                # KPIs
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO", f"{res['teo']:.2f} min", help="Ritmo de máxima eficiencia detectado.")
                c2.metric("⏱️ TC REAL", f"{res['real']:.2f} min", 
                          delta=f"{((res['real']/res['teo'])-1)*100:.1f}% Ineficiencia", delta_color="inverse")
                c3.metric("📦 Capacidad (8h)", f"{int((8*60)/res['teo'])} uds")

                st.divider()
                
                # GRÁFICA
                st.subheader("📊 Distribución del Ritmo de Trabajo")
                fig = px.histogram(res['datos_v'], x="Gap", nbins=50, 
                                 title="Frecuencia de Tiempos (Zona de Producción Real)",
                                 color_discrete_sequence=['#3498db'])
                fig.add_vline(x=res['teo']*60, line_dash="dash", line_color="red", line_width=4)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"❌ Error de Interpretación: {error_msg}")
                with st.expander("🔍 Ver datos crudos para diagnosticar"):
                    st.write("Primeras 20 filas detectadas:")
                    st.dataframe(df_raw.head(20))
        else:
            st.error("No se pudo detectar la estructura de datos en el archivo.")
