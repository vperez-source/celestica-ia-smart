import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Industrial Intelligence", layout="wide", page_icon="🏭")
st.title("🏭 Celestica IA: Análisis de Ciclo por Lotes Reales")

with st.sidebar:
    st.header("⚙️ Ajustes de Planta")
    h_turno = st.number_input("Horas Turno Disponibles", value=8)
    eficiencia = st.slider("Eficiencia Objetivo %", 50, 100, 85) / 100

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

# --- MAPEO ---
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

# --- CEREBRO: LÓGICA DE IMPUTACIÓN POR BATCH ---
def calcular_ciclo_maestro(df, col_fec):
    # 1. Limpieza y Ordenación
    df[col_fec] = pd.to_datetime(df[col_fec], errors='coerce')
    df = df.dropna(subset=[col_fec]).sort_values(col_fec)
    
    # 2. Agrupar por segundo (Identificar Lotes Instantáneos)
    lotes_instantaneos = df.groupby(col_fec).size().reset_index(name='Piezas_Lote')
    
    # 3. Calcular tiempo entre lotes (Tiempo de preparación)
    lotes_instantaneos['Gap_Pre_Lote_Sec'] = lotes_instantaneos[col_fec].diff().dt.total_seconds().fillna(0)
    
    # 4. PRORRATEO: Tiempo de ciclo por pieza dentro del lote
    lotes_instantaneos['CT_Individual'] = lotes_instantaneos['Gap_Pre_Lote_Sec'] / lotes_instantaneos['Piezas_Lote']
    
    # 5. FILTRADO EXPERTO:
    # Ignoramos paradas > 30 min (1800s) para no inflar la media
    # Ignoramos gaps de 0 (el primer lote)
    ritmos_validos = lotes_instantaneos[(lotes_instantaneos['CT_Individual'] > 0) & 
                                       (lotes_instantaneos['CT_Individual'] < 1800)]['CT_Individual']
    
    if len(ritmos_validos) < 5:
        return 0, lotes_instantaneos, 0

    # 6. MODA POR DENSIDAD (KDE)
    # Buscamos el punto de mayor concentración de ritmo
    kde = gaussian_kde(ritmos_validos)
    x = np.linspace(ritmos_validos.min(), ritmos_validos.max(), 1000)
    y = kde(x)
    modo_segundos = x[np.argmax(y)]
    
    return (modo_segundos / 60), lotes_instantaneos, modo_segundos

# --- APP ---
uploaded_file = st.file_uploader("Sube el archivo", type=["xlsx", "xls", "xml", "txt"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    if df_raw is not None:
        df_clean, cols = mapear_columnas(df_raw)
        if cols:
            ct_real, df_res, modo_s = calcular_ciclo_maestro(df_clean, cols['Fecha'])
            
            # --- RESULTADOS ---
            st.success("✅ Análisis de Lotes Finalizado")
            
            k1, k2, k3 = st.columns(3)
            k1.metric("⏱️ Cycle Time Realista", f"{ct_real:.2f} min/ud", help="La IA ha repartido el tiempo de espera entre las piezas de cada lote.")
            
            capacidad = (h_turno * 60 / ct_real) * eficiencia if ct_real > 0 else 0
            k2.metric("📦 Capacidad Turno", f"{int(capacidad)} uds")
            k3.metric("📊 Total Piezas", len(df_clean))

            st.divider()

            # --- VISUALIZACIÓN ---
            st.subheader("📊 Distribución del Ritmo de Trabajo")
            # Mostramos el histograma de los ritmos individuales calculados
            fig = px.histogram(df_res[df_res['CT_Individual'] < (modo_s * 4)], 
                             x="CT_Individual", nbins=50,
                             title="Frecuencia de Ritmos (Prorrateados por Lote)",
                             labels={'CT_Individual': 'Segundos por Pieza'},
                             color_discrete_sequence=['#2ecc71'])
            fig.add_vline(x=modo_s, line_width=4, line_dash="dash", line_color="red", annotation_text="Pico Real")
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("🔍 Ver auditoría de lotes (Cómo se calculó el tiempo)"):
                st.write("Esta tabla muestra cómo la IA dividió el tiempo de espera entre las piezas de cada segundo:")
                st.dataframe(df_res.sort_values('Piezas_Lote', ascending=False).head(50))

        else: st.error("Estructura de archivo no reconocida.")
