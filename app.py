import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Cycle Master", layout="wide", page_icon="⏱️")
st.title("⏱️ Celestica IA: Calculadora de Tiempo de Ciclo Real")

with st.sidebar:
    st.header("⚙️ Ajuste Fino")
    st.info("La IA ignorará automáticamente los tiempos absurdamente cortos (logs del sistema) y los descansos largos.")
    
    # Filtros de sentido común
    min_sec = st.number_input("Ignorar si es menor a (segundos):", 1, 60, 5, help="Filtra registros duplicados del mismo segundo.")
    max_min = st.number_input("Ignorar si es mayor a (minutos):", 10, 120, 60, help="Filtra las horas de comida o cambios de turno.")
    
    st.divider()
    h_turno = st.number_input("Horas Turno", value=8)
    eficiencia = st.slider("Eficiencia %", 50, 100, 85) / 100

# --- LECTOR XML ROBUSTO (Tu archivo lo necesita) ---
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

# --- CARGADOR INTELIGENTE ---
@st.cache_data(ttl=3600)
def load_data(file):
    # 1. XML
    try:
        file.seek(0)
        head = file.read(500).decode('latin-1', errors='ignore')
        file.seek(0)
        if "<?xml" in head or "Workbook" in head: return leer_xml_a_la_fuerza(file)
    except: pass
    # 2. Excel/CSV
    try: file.seek(0); return pd.read_excel(file, engine='calamine', header=None)
    except: pass
    try: file.seek(0); return pd.read_csv(file, sep='\t', encoding='latin-1', header=None)
    except: return None

# --- DETECTOR DE COLUMNAS ---
def detectar_columnas(df):
    if df is None: return None, None, None
    df = df.astype(str)
    
    # Palabras clave
    k_fecha = ['date', 'time', 'fecha', 'hora']
    k_usuario = ['user', 'operator', 'name', 'usuario']

    # Buscamos la fila de cabecera
    start_row = -1
    for i in range(min(50, len(df))):
        fila = df.iloc[i].str.lower().tolist()
        # Si tiene "Date" y alguna otra cosa, es la cabecera
        if any(k in str(v) for v in fila for k in k_fecha):
            start_row = i
            break
    
    if start_row == -1: return None, None, None

    # Asignamos cabecera
    df.columns = df.iloc[start_row]
    df = df[start_row + 1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()

    # Buscamos cuál es cuál
    col_f, col_u = None, None
    all_cols = list(df.columns)
    
    for col in all_cols:
        c_low = col.lower()
        if not col_f and any(k in c_low for k in k_fecha): col_f = col
        if not col_u and any(k in c_low for k in k_usuario): col_u = col
    
    # Si no encuentra usuario, usa la primera columna como referencia
    if not col_u: col_u = all_cols[0]
        
    return df, col_f, col_u

# --- APP ---
uploaded_file = st.file_uploader("Sube el archivo", type=["xlsx", "xls", "txt", "xml"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        df, col_f, col_u = detectar_columnas(df_raw)
        
        if col_f:
            # 1. LIMPIEZA DE FECHAS
            df[col_f] = pd.to_datetime(df[col_f], errors='coerce')
            df = df.dropna(subset=[col_f]).sort_values(col_f) # ORDEN CRONOLÓGICO ESTRICTO

            # 2. CÁLCULO DIRECTO (Flujo Continuo)
            # Calculamos la diferencia con la fila anterior, SIN agrupar por nada.
            # Esto asume que el archivo es una lista secuencial de eventos.
            df['diff_seconds'] = df[col_f].diff().dt.total_seconds()
            
            # 3. FILTRADO IA (Estadístico)
            # Ignoramos lo que sea < 5 seg (ruido) y > 60 min (descansos)
            df_valid = df[
                (df['diff_seconds'] >= min_sec) & 
                (df['diff_seconds'] <= (max_min * 60))
            ].copy()
            
            if df_valid.empty:
                st.error("No hay datos válidos tras el filtrado. Intenta bajar el tiempo mínimo en la barra lateral.")
                st.stop()

            # Convertimos a minutos
            df_valid['ct_min'] = df_valid['diff_seconds'] / 60
            
            # 4. ESTADÍSTICAS ROBUSTAS (La clave para evitar el 0.00)
            # Usamos la MEDIANA, no la media. La media se rompe con un solo error.
            ct_mediana = df_valid['ct_min'].median()
            piezas_totales = len(df_valid)
            capacidad = (h_turno * 60) / ct_mediana * eficiencia

            # --- RESULTADOS ---
            st.success("✅ Análisis Completado")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("⏱️ Cycle Time (Mediana)", f"{ct_mediana:.2f} min/ud", help="Este es el ritmo más habitual de tu línea.")
            c2.metric("📦 Capacidad (Turno)", f"{int(capacidad)} uds")
            c3.metric("📊 Muestras Válidas", piezas_totales)
            c4.metric("🗑️ Registros Ignorados", len(df) - len(df_valid), delta="Ruido/Pausas", delta_color="off")

            st.divider()

            # --- GRÁFICAS ---
            col_g1, col_g2 = st.columns([2, 1])
            
            with col_g1:
                st.subheader("📈 Ritmo de Producción")
                # Histograma para ver la distribución real
                fig = px.histogram(df_valid, x="ct_min", nbins=50, 
                                 title="Distribución de Tiempos (La montaña más alta es tu ritmo real)",
                                 labels={'ct_min': 'Minutos por Pieza'},
                                 color_discrete_sequence=['#2ecc71'])
                # Añadimos una línea vertical en la mediana
                fig.add_vline(x=ct_mediana, line_width=3, line_dash="dash", line_color="red", annotation_text="Ritmo Real")
                st.plotly_chart(fig, use_container_width=True)
            
            with col_g2:
                if col_u:
                    st.subheader("🏆 Operarios")
                    # Ranking simple
                    user_stats = df_valid.groupby(col_u)['ct_min'].median().reset_index().sort_values('ct_min')
                    user_stats.columns = ['Operario', 'CT (min)']
                    st.dataframe(user_stats.style.background_gradient(cmap='RdYlGn_r'), use_container_width=True)

            with st.expander("Ver datos crudos usados para el cálculo"):
                st.dataframe(df_valid[[col_f, 'ct_min'] + ([col_u] if col_u else [])].head(100))

        else:
            st.error("No encontré la columna de Fecha. ¿El archivo está vacío?")
    else:
        st.error("Error al leer el archivo.")
