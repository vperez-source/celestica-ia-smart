import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Batch Master", layout="wide", page_icon="🏭")
st.title("🏭 Celestica IA: Análisis de Procesos por Lotes (Batch)")

with st.sidebar:
    st.header("⚙️ Configuración de Lotes")
    
    # NUEVO CONCEPTO: UMBRAL DE PAUSA
    st.info("ℹ️ En producción por lotes, necesitamos distinguir entre 'Preparar Lote' y 'Irse a comer'.")
    umbral_pausa = st.number_input(
        "¿A partir de cuántos minutos consideras que es un DESCANSO?", 
        min_value=5, value=30, 
        help="Si hay un hueco mayor a este tiempo, no se contará como tiempo de producción, sino como parada."
    )
    
    contamination = st.slider("Sensibilidad Anomalías (% Ruido)", 1, 25, 5) / 100
    st.divider()
    eficiencia = st.slider("Eficiencia Objetivo %", 50, 100, 85) / 100

# --- LECTOR XML ---
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
                if data_tag:
                    fila_datos.append(data_tag.get_text(strip=True))
                else:
                    fila_datos.append("")
            if any(fila_datos):
                datos.append(fila_datos)
        return pd.DataFrame(datos)
    except: return None

# --- CARGA ---
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
    try: file.seek(0); dfs = pd.read_html(file, header=None); return dfs[0] if len(dfs)>0 else None
    except: pass
    try: file.seek(0); return pd.read_csv(file, sep='\t', encoding='latin-1', header=None)
    except: return None

# --- NORMALIZADOR ---
def encontrar_cabecera_y_normalizar(df):
    if df is None: return None, None, None, None
    df = df.astype(str)
    
    k_fecha = ['date', 'time', 'fecha', 'hora', 'timestamp']
    k_estacion = ['station', 'operation', 'work', 'estacion', 'maquina', 'productid']
    k_usuario = ['user', 'operator', 'name', 'usuario', 'created by', 'computername']

    start_row = -1
    for i in range(min(50, len(df))):
        fila = df.iloc[i].str.lower().tolist()
        has_date = any(k in str(v) for v in fila for k in k_fecha)
        has_id = any(k in str(v) for v in fila for k in k_estacion + k_usuario)
        if has_date and has_id:
            start_row = i
            break
            
    if start_row == -1: return None, None, None, None

    df.columns = df.iloc[start_row]
    df = df[start_row + 1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()
    
    col_f, col_s, col_u = None, None, None
    for col in df.columns:
        c_low = col.lower()
        if not col_f and any(k in c_low for k in k_fecha): col_f = col
        if not col_s and any(k in c_low for k in k_estacion): col_s = col
        if not col_u and any(k in c_low for k in k_usuario): col_u = col
        
    return df, col_f, col_s, col_u

# --- APP ---
uploaded_file = st.file_uploader("Sube el archivo de Lotes", type=["xlsx", "xls", "txt", "xml"])

if uploaded_file:
    with st.spinner("⏳ Analizando estructura de lotes..."):
        df_raw = load_data(uploaded_file)
        if df_raw is not None:
            # 1. NORMALIZAR
            df, col_f, col_s, col_u = encontrar_cabecera_y_normalizar(df_raw)

            if not col_f or not col_s:
                st.error("❌ No encontré cabeceras válidas.")
                st.stop()

            # 2. LIMPIEZA
            df[col_f] = pd.to_datetime(df[col_f], errors='coerce')
            df.loc[df[col_f].dt.year < 100, col_f] += pd.offsets.DateOffset(years=2000)
            df = df.dropna(subset=[col_f]).sort_values(col_f)

            # 3. CÁLCULO GAPS
            # Calculamos la diferencia en minutos entre una pieza y la anterior
            df['gap_mins'] = df.groupby(col_s)[col_f].diff().dt.total_seconds() / 60
            
            # --- LÓGICA DE LOTES (LA CLAVE) ---
            # Si el tiempo es mayor que el "Umbral de Pausa" (ej. 30 min), asumimos que NO es tiempo de trabajo.
            # Si es menor (ej. 10 min), asumimos que es tiempo de preparación del lote y SÍ cuenta.
            
            # Filtramos solo los tiempos que consideramos "Productivos"
            df_production = df[df['gap_mins'] < umbral_pausa].copy()
            
            # El tiempo total trabajado es la SUMA de todos esos gap
            tiempo_total_trabajado_mins = df_production['gap_mins'].sum()
            total_piezas = len(df) # Contamos TODAS las piezas
            
            # Cycle Time Ponderado = Tiempo Total / Total Piezas
            if total_piezas > 0:
                cycle_time_ponderado = tiempo_total_trabajado_mins / total_piezas
            else:
                cycle_time_ponderado = 0

            # Capacidad Teórica basada en el ritmo del lote
            # (Horas turno * 60) / Cycle Time Ponderado
            if cycle_time_ponderado > 0:
                capacidad = (8 * 60) / cycle_time_ponderado * eficiencia
            else:
                capacidad = 0

            # --- VISUALIZACIÓN ---
            st.success(f"✅ Lógica de Lotes Aplicada. Tiempo Total Activo: {int(tiempo_total_trabajado_mins)} min para {total_piezas} piezas.")

            k1, k2, k3 = st.columns(3)
            k1.metric("⏱️ Cycle Time (Media Ponderada)", f"{cycle_time_ponderado:.2f} min/ud", help="Tiene en cuenta que procesas por lotes.")
            k2.metric("📦 Capacidad Estimada", f"{int(capacidad)} uds")
            k3.metric("📉 Calidad Dato", f"{len(df)} Regs")

            st.markdown("---")

            # --- GRÁFICA DE DISTRIBUCIÓN DE TIEMPOS ---
            # Esto es vital para entender los lotes: Verás dos picos (pico rápido y pico lento)
            st.subheader("📊 Radiografía del Lote")
            st.caption("Deberías ver dos montañas: A la izquierda los escaneos rápidos (0.1 min) y a la derecha los tiempos de preparación del lote.")
            
            fig_hist = px.histogram(df_production, x="gap_mins", nbins=100, 
                                  title="Distribución de Tiempos: ¿Escaneo rápido o Preparación?",
                                  labels={'gap_mins': 'Minutos entre piezas'},
                                  color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig_hist, use_container_width=True)

            # --- RANKING OPERARIOS ---
            if col_u:
                st.subheader("🏆 Ranking Real por Operario")
                
                # Agrupamos por usuario
                # Sumamos sus tiempos productivos y contamos sus piezas
                # Rellenamos NAs con 0 para poder sumar
                df['gap_prod'] = df['gap_mins'].where(df['gap_mins'] < umbral_pausa, 0)
                
                stats = df.groupby(col_u).agg(
                    Piezas=('gap_mins', 'count'),
                    Tiempo_Total=('gap_prod', 'sum')
                ).reset_index()
                
                stats['Cycle_Time_Real'] = stats['Tiempo_Total'] / stats['Piezas']
                stats = stats.sort_values('Piezas', ascending=False)

                c1, c2 = st.columns([1,1])
                with c1:
                    st.dataframe(stats.style.background_gradient(subset=['Piezas'], cmap='Greens'), use_container_width=True)
                with c2:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=stats[col_u], y=stats['Piezas'], name='Piezas', marker_color='#2ecc71', yaxis='y'))
                    fig.add_trace(go.Scatter(x=stats[col_u], y=stats['Cycle_Time_Real'], name='Min/Pieza Real', marker_color='#e74c3c', yaxis='y2'))
                    
                    fig.update_layout(
                        title="Producción vs Ritmo Real",
                        yaxis=dict(title="Piezas"),
                        yaxis2=dict(title="Min/Pieza (Ponderado)", overlaying='y', side='right'),
                        legend=dict(orientation="h", y=1.1)
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("Error de lectura.")
