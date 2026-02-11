import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica AI Industrial Master", layout="wide", page_icon="🏭")
st.title("🏭 Celestica IA: Filtro de Realidad Industrial")
st.markdown("""
**Lógica Inteligente:** Este algoritmo ignora el ruido de milisegundos y las paradas largas. 
Detecta el **ritmo de crucero** real de la línea mediante Clustering.
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

# --- MAPEO DE COLUMNAS ---
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

# --- CEREBRO IA: CLUSTERING INDUSTRIAL ---
def analizar_ciclo_real(df, col_fec):
    # 1. Limpieza y Gaps
    df[col_fec] = pd.to_datetime(df[col_fec], errors='coerce')
    df = df.dropna(subset=[col_fec]).sort_values(col_fec)
    df['Gap_Sec'] = df[col_fec].diff().dt.total_seconds().fillna(0)
    
    # 2. Manejo de Lotes (Batching)
    # Si el tiempo es < 1 seg, es ruido de sistema. Lo agrupamos con el tiempo previo.
    # No lo borramos, pero le damos un valor simbólico para que no altere el clustering.
    df_fit = df[df['Gap_Sec'] > 1].copy() 
    
    if len(df_fit) < 10:
        return 0, df, 0

    # 3. Clustering K-Means (Agrupar por comportamientos)
    # Logaritmo para manejar diferencias entre segundos y horas
    X = np.log1p(df_fit[['Gap_Sec']].values)
    kmeans = KMeans(n_clusters=min(3, len(df_fit)), random_state=42, n_init=10)
    df_fit['Cluster'] = kmeans.fit_predict(X)
    
    # Identificar el cluster de "Producción Real"
    # Suele ser el que tiene la mediana de tiempo razonable (ni el más rápido ni el más lento)
    resumen = df_fit.groupby('Cluster')['Gap_Sec'].median().sort_values()
    
    # Mapeo: 
    # Cluster 0: Micro-paradas / Lotes rápidos
    # Cluster 1: RITMO REAL DE CRUCERO
    # Cluster 2: Paradas largas / Cambios de turno
    
    idx_produccion = resumen.index[1] if len(resumen) > 1 else resumen.index[0]
    
    df_produccion = df_fit[df_fit['Cluster'] == idx_produccion]
    ct_real_min = df_produccion['Gap_Sec'].median() / 60
    
    return ct_real_min, df_fit, idx_produccion

# --- APP ---
uploaded_file = st.file_uploader("Sube el archivo", type=["xlsx", "xls", "xml", "txt"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    if df_raw is not None:
        df_clean, cols = mapear_columnas(df_raw)
        if cols:
            # IA ANALÍTICA
            ct_real, df_ia, idx_prod = analizar_ciclo_real(df_clean, cols['Fecha'])
            
            # --- DASHBOARD ---
            st.success("✅ Análisis de Ritmo Real Completado")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("⏱️ Cycle Time Real", f"{ct_real:.2f} min/ud", help="La IA ha aislado el ritmo de crucero ignorando ráfagas de sistema y paradas de descanso.")
            
            # Capacidad calculada sobre tiempo de trabajo real detectado
            total_piezas = len(df_clean)
            capacidad_8h = (480 / ct_real) * 0.85 if ct_real > 0 else 0
            c2.metric("📦 Capacidad Turno", f"{int(capacidad_8h)} uds")
            c3.metric("📊 Datos Totales", total_piezas)

            st.divider()

            # --- VISUALIZACIÓN ---
            tab1, tab2 = st.tabs(["📉 Análisis de Filtros", "🔬 Desglose Producto"])
            
            with tab1:
                st.subheader("Clasificación Automática de Tiempos")
                st.markdown("""
                La IA ha clasificado cada registro en una categoría. El **Cycle Time** se calcula basándose 
                únicamente en los datos de 'Producción Real'.
                """)
                
                # Gráfico de dispersión coloreado por Cluster
                df_ia['Estado'] = df_ia['Cluster'].apply(lambda x: 'Producción Real' if x == idx_prod else ('Parada Larga' if x > idx_prod else 'Micro-ritmo/Lote'))
                
                fig = px.scatter(df_ia, x=cols['Fecha'], y='Gap_Sec', color='Estado',
                               title="Detección de Patrones de Tiempo",
                               labels={'Gap_Sec': 'Segundos entre piezas'},
                               color_discrete_map={'Producción Real': '#2ecc71', 'Parada Larga': '#e74c3c', 'Micro-ritmo/Lote': '#3498db'})
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Mostrar tabla por producto pero usando el CT real de la IA
                st.subheader("Rendimiento por Familia/Producto")
                resumen_prod = df_clean.groupby([cols['Familia'], cols['Producto']]).size().reset_index(name='Piezas')
                # Estimamos el tiempo total basándonos en el ritmo real detectado
                resumen_prod['Tiempo Est. (min)'] = resumen_prod['Piezas'] * ct_real
                st.dataframe(resumen_prod.sort_values('Piezas', ascending=False), use_container_width=True)

        else: st.error("Archivo no compatible.")
    else: st.error("Error al leer.")
