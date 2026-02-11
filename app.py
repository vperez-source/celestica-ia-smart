import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Ultra-Analyzer", layout="wide", page_icon="🎯")
st.title("🎯 Celestica IA: Depuración de Ritmo Real (Moda)")
st.markdown("""
**Lógica Avanzada:** Este algoritmo busca el **pico de frecuencia**. Ignora preparaciones, 
descansos y ráfagas de sistema, centrándose únicamente en el ritmo más repetido por los operarios.
""")

# --- LECTORES ---
def leer_xml_robusto(file):
    try:
        content = file.getvalue().decode('latin-1', errors='ignore')
        soup = BeautifulSoup(content, 'xml')
        datos = []
        rows = soup.find_all(['Row', 'ss:Row', 'row'])
        for row in rows:
            fila = [cell.get_text(strip=True) for cell in row.find_all(['Cell', 'ss:Cell', 'cell'])]
            if any(fila): datos.append(fila)
        return pd.DataFrame(datos)
    except: return None

@st.cache_data(ttl=3600)
def load_data(file):
    try: 
        file.seek(0)
        if "<?xml" in file.read(500).decode('latin-1', errors='ignore'): 
            file.seek(0); return leer_xml_robusto(file)
    except: pass
    try: file.seek(0); return pd.read_excel(file, engine='calamine', header=None)
    except: pass
    try: file.seek(0); return pd.read_csv(file, sep='\t', encoding='latin-1', header=None)
    except: return None

# --- MAPEO SEGURO ---
def mapear_columnas(df):
    if df is None: return None, {}
    df = df.astype(str)
    start = -1
    for i in range(min(50, len(df))):
        row = df.iloc[i].str.lower().tolist()
        if any('date' in str(v) for v in row) and any('station' in str(v) for v in row):
            start = i; break
    if start == -1: return None, {}
    df.columns = df.iloc[start]
    df = df[start+1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()
    cols = {'Fecha': None, 'Producto': None, 'Familia': None, 'Usuario': None}
    for c in df.columns:
        cl = c.lower()
        if not cols['Fecha'] and ('date' in cl or 'time' in cl): cols['Fecha'] = c
        if not cols['Producto'] and ('product' in cl or 'item' in cl): cols['Producto'] = c
        if not cols['Familia'] and ('family' in cl): cols['Familia'] = c
        if not cols['Usuario'] and ('user' in cl or 'operator' in cl): cols['Usuario'] = c
    for k, v in cols.items():
        if v is None:
            df[f'Col_{k}'] = "General"
            cols[k] = f'Col_{k}'
    return df, cols

# --- CEREBRO: CÁLCULO DE MODA POR DENSIDAD (EL MÁS PRECISO) ---
def calcular_ritmo_moda(df, col_fec):
    df[col_fec] = pd.to_datetime(df[col_fec], errors='coerce')
    df = df.dropna(subset=[col_fec]).sort_values(col_fec)
    
    # Gap en segundos
    df['Gap_Sec'] = df[col_fec].diff().dt.total_seconds().fillna(0)
    
    # FILTRO 1: Fuera ruido de sistema (<3s) y paradas (>20min)
    datos_filtrados = df[(df['Gap_Sec'] > 3) & (df['Gap_Sec'] < 1200)]['Gap_Sec']
    
    if len(datos_filtrados) < 5:
        return 0, df, 0

    # FILTRO 2: Eliminar Outliers extremos (Percentiles 5-95)
    p5 = datos_filtrados.quantile(0.05)
    p95 = datos_filtrados.quantile(0.95)
    datos_finales = datos_filtrados[(datos_filtrados >= p5) & (datos_filtrados <= p95)]

    # CÁLCULO DE LA MODA (Pico de densidad)
    # Usamos Gaussian KDE para encontrar donde hay más puntos acumulados
    kde = gaussian_kde(datos_finales)
    x_range = np.linspace(datos_finales.min(), datos_finales.max(), 1000)
    y_densidad = kde(x_range)
    modo_segundos = x_range[np.argmax(y_densidad)]
    
    return (modo_segundos / 60), df, modo_segundos

# --- APP ---
uploaded_file = st.file_uploader("Sube el archivo", type=["xlsx", "xls", "xml", "txt"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    if df_raw is not None:
        df_clean, cols = mapear_columnas(df_raw)
        if cols:
            ct_real_min, df_res, modo_seg = calcular_ritmo_moda(df_clean, cols['Fecha'])
            
            # --- DASHBOARD ---
            st.success("🎯 Análisis de Pico de Rendimiento Completado")
            
            k1, k2, k3 = st.columns(3)
            k1.metric("⏱️ Cycle Time Real (Moda)", f"{ct_real_min:.2f} min/ud", help="La IA ha detectado que este es el ritmo que más se repite en la línea, ignorando paradas y ráfagas.")
            
            capacidad = (480 / ct_real_min) * 0.85 if ct_real_min > 0 else 0
            k2.metric("📦 Capacidad Turno (Moda)", f"{int(capacidad)} uds")
            k3.metric("📊 Datos Procesados", len(df_clean))

            st.divider()

            # --- GRÁFICA DE DENSIDAD (PARA VALIDAR) ---
            st.subheader("📊 Distribución de Ritmos Detectados")
            st.markdown(f"La montaña indica dónde se concentran los operarios. El pico está en **{modo_seg:.1f} segundos**.")
            
            # Histograma depurado
            df_hist = df_res[(df_res['Gap_Sec'] > 5) & (df_res['Gap_Sec'] < (modo_seg * 3))]
            fig = px.histogram(df_hist, x="Gap_Sec", nbins=50, 
                             title="Frecuencia de Tiempos entre Piezas",
                             labels={'Gap_Sec': 'Segundos'},
                             color_discrete_sequence=['#3498db'])
            fig.add_vline(x=modo_seg, line_width=4, line_dash="dash", line_color="#e74c3c", annotation_text="Ritmo Real")
            st.plotly_chart(fig, use_container_width=True)
            
            # Ranking Operarios por Moda (su ritmo más frecuente)
            st.subheader("🏆 Velocidad Frecuente por Operario")
            # Agregamos lógica simple de piezas por usuario
            user_rank = df_clean.groupby(cols['Usuario']).size().reset_index(name='Piezas Totales')
            st.dataframe(user_rank.sort_values('Piezas Totales', ascending=False), use_container_width=True)

        else: st.error("Estructura de columnas no reconocida.")
