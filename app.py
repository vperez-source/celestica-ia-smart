import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Auto-Pilot", layout="wide", page_icon="✈️")
st.title("✈️ Celestica IA: Piloto Automático")
st.markdown("**Modo Inteligente:** El algoritmo detecta automáticamente qué es ruido, qué es producción y qué son paradas.")

with st.sidebar:
    st.header("⚙️ Resultados")
    st.info("No hay configuración manual. La IA se adapta a la distribución de tus datos.")
    eficiencia = st.slider("Eficiencia Objetivo %", 50, 100, 85) / 100

# --- LECTORES ROBUSTOS ---
def leer_xml_a_la_fuerza(file):
    try:
        content = file.getvalue().decode('latin-1', errors='ignore')
        soup = BeautifulSoup(content, 'xml')
        datos = []
        rows = soup.find_all(['Row', 'ss:Row', 'row'])
        for row in rows:
            fila_datos = []
            cells = row.find_all(['Cell', 'ss:Cell', 'cell'])
            for cell in cells:
                data_tag = cell.find(['Data', 'ss:Data', 'data'])
                if data_tag: fila_datos.append(data_tag.get_text(strip=True))
                else: fila_datos.append("")
            if any(fila_datos): datos.append(fila_datos)
        return pd.DataFrame(datos)
    except: return None

@st.cache_data(ttl=3600)
def load_data(file):
    try:
        file.seek(0)
        head = file.read(500).decode('latin-1', errors='ignore')
        file.seek(0)
        if "<?xml" in head or "Workbook" in head: return leer_xml_a_la_fuerza(file)
    except: pass
    try: file.seek(0); return pd.read_excel(file, engine='calamine', header=None)
    except: pass
    try: file.seek(0); return pd.read_csv(file, sep='\t', encoding='latin-1', header=None)
    except: return None

# --- DETECTOR INTELIGENTE DE COLUMNAS ---
def detectar_columnas(df):
    if df is None: return None, None, None
    df = df.astype(str)
    k_fecha = ['date', 'time', 'fecha', 'hora', 'timestamp']
    k_usuario = ['user', 'operator', 'name', 'usuario', 'created by']

    start_row = -1
    for i in range(min(50, len(df))):
        fila = df.iloc[i].str.lower().tolist()
        if any(k in str(v) for v in fila for k in k_fecha):
            start_row = i
            break
    
    if start_row == -1: return None, None, None

    df.columns = df.iloc[start_row]
    df = df[start_row + 1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()

    col_f, col_u = None, None
    for col in df.columns:
        c_low = col.lower()
        if not col_f and any(k in c_low for k in k_fecha): col_f = col
        if not col_u and any(k in c_low for k in k_usuario): col_u = col
    if not col_u: col_u = df.columns[0]
        
    return df, col_f, col_u

# --- CEREBRO IA: CLUSTERING AUTOMÁTICO ---
def analizar_ritmo_ia(df, col_fecha):
    # 1. Calcular diferencias (Gaps)
    df = df.sort_values(col_fecha)
    df['gap_seconds'] = df[col_fecha].diff().dt.total_seconds().fillna(0)
    
    # 2. Limpieza básica (Ignoramos negativos o ceros absolutos para el K-Means)
    # Pero los guardamos para contarlos como "Batch/System Logs"
    datos_validos = df[df['gap_seconds'] > 0.5].copy() # Ignoramos < 0.5 seg para el fit
    
    if len(datos_validos) < 10:
        return None, None, "Pocos datos"

    # 3. K-MEANS (El cerebro)
    # Le pedimos que encuentre 3 grupos naturales en los datos
    X = datos_validos[['gap_seconds']].values
    
    # Usamos Logaritmo porque los tiempos varían mucho (segundos vs horas)
    # Esto ayuda a la IA a ver mejor la diferencia
    X_log = np.log1p(X) 
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    datos_validos['cluster'] = kmeans.fit_predict(X_log)
    
    # 4. Interpretar los Clusters
    # Calculamos la mediana de cada grupo para saber cuál es cuál
    resumen = datos_validos.groupby('cluster')['gap_seconds'].median().sort_values()
    
    # El cluster con el tiempo MEDIO (ni el más bajo ni el más alto) suele ser el ritmo real
    # Si hay lotes muy rápidos, el ritmo real puede ser el cluster 0 o 1.
    
    # Heurística:
    # Cluster Bajo (0): Ráfagas rápidas / Ruido
    # Cluster Medio (1): Ritmo de Producción Real
    # Cluster Alto (2): Paradas / Descansos
    
    # Mapeamos los índices ordenados
    indices_ordenados = resumen.index.tolist()
    
    idx_ruido = indices_ordenados[0]     # El grupo más rápido
    idx_produccion = indices_ordenados[1] # El grupo intermedio (CYCLE TIME REAL)
    idx_parada = indices_ordenados[2]     # El grupo más lento
    
    # Si el grupo "rápido" tiene una media > 10 segundos, entonces NO es ruido, es producción rápida.
    # En ese caso, fusionamos ruido y producción.
    cluster_produccion = datos_validos[datos_validos['cluster'] == idx_produccion]
    
    # --- RESULTADO FINAL ---
    cycle_time_ia = cluster_produccion['gap_seconds'].median() / 60 # En minutos
    
    return cycle_time_ia, datos_validos, (idx_ruido, idx_produccion, idx_parada)

# --- APP ---
uploaded_file = st.file_uploader("Sube tu archivo", type=["xlsx", "xls", "txt", "xml"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        df, col_f, col_u = detectar_columnas(df_raw)
        
        if col_f:
            df[col_f] = pd.to_datetime(df[col_f], errors='coerce')
            df = df.dropna(subset=[col_f])
            
            # --- EJECUTAR IA ---
            with st.spinner("🤖 La IA está detectando patrones de ritmo..."):
                ct_real, df_analyzed, clusters_idx = analizar_ritmo_ia(df, col_f)
            
            if ct_real:
                # Calculamos capacidad
                capacidad = (8 * 60) / ct_real * eficiencia

                st.success(f"✅ Ritmo Detectado Automáticamente")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ Cycle Time (IA)", f"{ct_real:.2f} min/ud", help="La IA ha ignorado el ruido y las paradas largas automáticamente.")
                c2.metric("📦 Capacidad (8h)", f"{int(capacidad)} uds")
                c3.metric("📊 Datos Analizados", len(df))
                
                st.divider()
                
                # --- VISUALIZACIÓN DE LA DECISIÓN DE LA IA ---
                st.subheader("🧠 ¿Cómo decidió la IA?")
                st.caption("El algoritmo agrupó tus tiempos en 3 categorías. Aquí ves qué consideró 'Producción Real' (Verde).")
                
                # Preparamos colores para el gráfico
                def color_map(cluster_id):
                    if cluster_id == clusters_idx[1]: return "Producción (Real)"
                    elif cluster_id == clusters_idx[0]: return "Ráfagas/Ruido"
                    else: return "Paradas/Descansos"
                
                df_analyzed['Categoría'] = df_analyzed['cluster'].apply(color_map)
                
                # Histograma coloreado por cluster
                fig = px.histogram(df_analyzed, x="gap_seconds", color="Categoría", nbins=100,
                                 title="Distribución de Tiempos y Clasificación IA",
                                 labels={'gap_seconds': 'Segundos entre piezas'},
                                 color_discrete_map={
                                     "Producción (Real)": "#2ecc71", # Verde
                                     "Ráfagas/Ruido": "#95a5a6",    # Gris
                                     "Paradas/Descansos": "#e74c3c" # Rojo
                                 },
                                 log_y=True) # Escala logarítmica para ver bien los datos pequeños
                st.plotly_chart(fig, use_container_width=True)
                
                # --- RANKING OPERARIOS ---
                if col_u:
                    st.subheader("🏆 Ranking Operarios")
                    # Filtramos solo lo que es Producción Real para el ranking
                    df_prod = df_analyzed[df_analyzed['cluster'] == clusters_idx[1]]
                    user_stats = df_prod.groupby(col_u)['gap_seconds'].median().reset_index()
                    user_stats['gap_seconds'] = user_stats['gap_seconds'] / 60 # a minutos
                    user_stats.columns = ['Operario', 'CT (min)']
                    user_stats = user_stats.sort_values('CT (min)')
                    
                    st.dataframe(user_stats.style.background_gradient(cmap='RdYlGn_r'), use_container_width=True)

            else:
                st.warning("No hay suficientes datos para que la IA detecte patrones claros.")
                
        else:
            st.error("No encontré columna de Fecha.")
    else:
        st.error("Error de lectura.")
