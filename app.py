import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Real-Time Analyzer", layout="wide", page_icon="⏱️")
st.title("⏱️ Celestica IA: Analizador de Ritmo de Crucero")
st.markdown("""
**Criterio de Realismo:** Este algoritmo ignora ráfagas de sistema y paradas largas. 
Calcula el **ritmo sostenible** buscando la densidad máxima de producción.
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

# --- DETECTOR DE COLUMNAS ---
def detectar_columnas(df):
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

# --- CEREBRO: FILTRO DE DENSIDAD ---
def calcular_ciclo_realista(df, col_fec):
    # 1. Preparación
    df[col_fec] = pd.to_datetime(df[col_fec], errors='coerce')
    df = df.dropna(subset=[col_fec]).sort_values(col_fec)
    
    # 2. Calcular Gaps en segundos
    df['Gap_Sec'] = df[col_fec].diff().dt.total_seconds().fillna(0)
    
    # 3. FILTRADO DE RUIDO EXTREMO (A LA BAJA Y ALTA)
    # Ignoramos lo que sea < 2 segundos (es ruido de sistema/batch)
    # Ignoramos lo que sea > 30 minutos (es una parada de descanso/comida)
    df_filtrado = df[(df['Gap_Sec'] > 2) & (df['Gap_Sec'] < 1800)].copy()
    
    if df_filtrado.empty:
        # Si no hay gaps, es que todo se registró a la vez. 
        # Usamos el tiempo total del archivo entre el numero de piezas.
        total_sec = (df[col_fec].max() - df[col_fec].min()).total_seconds()
        return (total_sec / len(df)) / 60 if len(df) > 0 else 0, df

    # 4. USO DE LA MEDIANA (Resistente a Outliers)
    # La mediana es el valor que está en el centro. No se deja engañar por 
    # un par de piezas muy lentas o muy rápidas.
    ciclo_medio_seg = df_filtrado['Gap_Sec'].median()
    
    # 5. AJUSTE DE "PROCESO POR LOTES"
    # Si detectamos que hay muchas piezas con el mismo timestamp, 
    # promediamos el gap anterior entre el número de piezas que salieron juntas.
    df['Batch_Size'] = df.groupby(col_fec)[col_fec].transform('count')
    df.loc[df['Batch_Size'] > 1, 'Gap_Sec'] = df['Gap_Sec'] / df['Batch_Size']
    
    return (ciclo_medio_seg / 60), df_filtrado

# --- APP ---
uploaded_file = st.file_uploader("Sube el archivo", type=["xlsx", "xls", "xml", "txt"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    if df_raw is not None:
        df_clean, cols = detectar_columnas(df_raw)
        if cols:
            # CÁLCULO IA
            ct_real, df_viz = calcular_ciclo_realista(df_clean, cols['Fecha'])
            
            # --- DASHBOARD ---
            st.success("✅ Ritmo de Producción Calculado")
            
            k1, k2, k3 = st.columns(3)
            k1.metric("⏱️ Cycle Time Realista", f"{ct_real:.2f} min/ud", help="Calculado usando la mediana de tiempos productivos (2s - 30min).")
            
            # Capacidad Teórica
            capacidad = (480 / ct_real) * 0.85 if ct_real > 0 else 0
            k2.metric("📦 Capacidad Turno (8h)", f"{int(capacidad)} uds")
            k3.metric("📊 Registros Procesados", len(df_clean))

            st.divider()

            # --- VISUALIZACIÓN ---
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.subheader("📈 Estabilidad del Ritmo")
                # Solo graficamos puntos razonables para que la gráfica se vea bien
                fig = px.scatter(df_viz[df_viz['Gap_Sec'] < 600], x=cols['Fecha'], y='Gap_Sec', 
                               title="Tiempo entre piezas (Segundos)",
                               labels={'Gap_Sec': 'Segundos'},
                               color_discrete_sequence=['#2ecc71'])
                fig.add_hline(y=ct_real*60, line_dash="dash", line_color="red", annotation_text="Ritmo Real")
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                st.subheader("🔬 Rendimiento por Familia")
                resumen_fam = df_clean.groupby(cols['Familia']).size().reset_index(name='Piezas')
                resumen_fam['Tiempo Est.'] = resumen_fam['Piezas'] * ct_real
                st.dataframe(resumen_fam.sort_values('Piezas', ascending=False), use_container_width=True)

        else: st.error("No encontré la estructura de datos correcta.")
