import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Batch AI", layout="wide", page_icon="📦")
st.title("📦 Celestica IA: Detector Inteligente de Lotes")
st.markdown("""
**Lógica de Batch:** El algoritmo identifica cuándo empieza y termina un lote basándose en los saltos de tiempo. 
Suma el tiempo de preparación + el tiempo de las ráfagas rápidas y calcula la media real.
""")

with st.sidebar:
    st.header("⚙️ Resultados")
    st.info("El sistema detecta automáticamente los cortes de lote.")
    eficiencia = st.slider("Eficiencia Objetivo %", 50, 100, 85) / 100
    h_turno = st.number_input("Horas Turno", 8)

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

# --- DETECTOR DE COLUMNAS ---
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
    if not col_u: col_u = df.columns[0] # Si no hay usuario, usamos la primera col
        
    return df, col_f, col_u

# --- CEREBRO DE LOTES (ALGORITMO DE SESSIONIZING) ---
def procesar_por_lotes(df, col_f, col_u):
    # 1. Preparar datos
    df[col_f] = pd.to_datetime(df[col_f], errors='coerce')
    df = df.dropna(subset=[col_f]).sort_values([col_u, col_f])
    
    # 2. Calcular diferencia de tiempo entre filas del MISMO usuario
    df['diff_seconds'] = df.groupby(col_u)[col_f].diff().dt.total_seconds().fillna(0)
    
    # 3. AUTO-DETECCIÓN DE CORTE DE LOTE
    # ¿Qué se considera "Fin de un lote y principio de otro"?
    # Usamos estadística: Si el tiempo es mayor al percentil 90 (ej. saltos grandes), es un corte.
    # O un valor seguro por defecto: 5 minutos (300 segundos).
    
    # Calculamos un umbral dinámico basado en los datos
    # Ignoramos los ceros para calcular el umbral
    tiempos_reales = df[df['diff_seconds'] > 1]['diff_seconds']
    if not tiempos_reales.empty:
        # El umbral es: O bien 5 minutos, o el percentil 95 de los tiempos (lo que sea mayor)
        # Esto permite que si hay paradas naturales de 2 min, no corte el lote.
        umbral_corte = max(300, tiempos_reales.quantile(0.95))
    else:
        umbral_corte = 300 # Default 5 min

    # 4. ASIGNACIÓN DE IDs DE LOTE
    # Cada vez que diff > umbral, creamos un nuevo ID de lote
    df['Nuevo_Lote'] = (df['diff_seconds'] > umbral_corte) | (df[col_u] != df[col_u].shift())
    df['Lote_ID'] = df['Nuevo_Lote'].cumsum()
    
    # 5. AGREGACIÓN POR LOTE (LA MAGIA)
    # Ahora agrupamos todas las ráfagas (0.1s) con su tiempo de preparación (10 min)
    
    resumen_lotes = df.groupby('Lote_ID').agg(
        Usuario=(col_u, 'first'),
        Hora_Inicio=(col_f, 'min'),
        Hora_Fin=(col_f, 'max'),
        Piezas=('diff_seconds', 'count'), # Número de piezas en el lote
        # El tiempo del lote es la suma de todos los gaps
        # OJO: Incluimos el gap inicial grande (preparación) + gaps pequeños (ráfagas)
        Tiempo_Total_Segundos=('diff_seconds', 'sum')
    ).reset_index()

    # Filtramos lotes erróneos (tiempo 0 o 0 piezas)
    resumen_lotes = resumen_lotes[resumen_lotes['Tiempo_Total_Segundos'] > 0]
    
    # Calculamos Cycle Time del Lote
    resumen_lotes['CT_Promedio_Lote'] = (resumen_lotes['Tiempo_Total_Segundos'] / 60) / resumen_lotes['Piezas']
    
    return resumen_lotes, umbral_corte

# --- APP ---
uploaded_file = st.file_uploader("Sube el archivo", type=["xlsx", "xls", "txt", "xml"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        df, col_f, col_u = detectar_columnas(df_raw)
        
        if col_f:
            with st.spinner("📦 Detectando lotes y ráfagas..."):
                df_lotes, umbral_usado = procesar_por_lotes(df, col_f, col_u)
                
                if df_lotes.empty:
                    st.error("No se pudieron detectar lotes válidos.")
                    st.stop()

                # --- KPIs GLOBALES ---
                # Cycle Time Ponderado: (Suma de todos los tiempos) / (Suma de todas las piezas)
                total_minutos = df_lotes['Tiempo_Total_Segundos'].sum() / 60
                total_piezas = df_lotes['Piezas'].sum()
                
                ct_real_global = total_minutos / total_piezas
                capacidad = (h_turno * 60) / ct_real_global * eficiencia

                st.success(f"✅ Análisis de Lotes Completado")
                st.info(f"💡 Criterio de la IA: Se ha considerado un 'Nuevo Lote' cuando pasan más de {int(umbral_usado/60)} minutos sin actividad.")
                
                k1, k2, k3 = st.columns(3)
                k1.metric("⏱️ Cycle Time Real (Ponderado)", f"{ct_real_global:.2f} min/ud", 
                          help="Calculado sumando tiempos de preparación + ráfagas / total piezas.")
                k2.metric("📦 Capacidad Turno", f"{int(capacidad)} uds")
                k3.metric("📊 Lotes Detectados", len(df_lotes))
                
                st.divider()

                # --- GRÁFICAS ---
                c_chart1, c_chart2 = st.columns(2)
                
                with c_chart1:
                    st.subheader("📦 Tamaño de los Lotes")
                    # Histograma de cuántas piezas suele haber por lote
                    fig = px.histogram(df_lotes, x="Piezas", nbins=20, 
                                     title="Distribución: ¿Cuántas piezas hacen por lote?",
                                     color_discrete_sequence=['#3498db'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                with c_chart2:
                    st.subheader("⏱️ Velocidad por Lote")
                    # Scatter plot: Eje X = Hora, Eje Y = CT del Lote, Tamaño = Cantidad Piezas
                    fig = px.scatter(df_lotes, x="Hora_Inicio", y="CT_Promedio_Lote", 
                                   size="Piezas", color="Usuario",
                                   title="Evolución Temporal (Burbuja grande = Lote grande)",
                                   labels={'CT_Promedio_Lote': 'Minutos/Pieza (Media del Lote)'})
                    st.plotly_chart(fig, use_container_width=True)

                # --- RANKING OPERARIOS (REAL) ---
                if col_u:
                    st.subheader("🏆 Ranking Real (Promedio de Lotes)")
                    # Agrupamos los lotes por usuario
                    ranking = df_lotes.groupby('Usuario').agg(
                        Total_Piezas=('Piezas', 'sum'),
                        Tiempo_Total_Min=('Tiempo_Total_Segundos', lambda x: x.sum() / 60)
                    ).reset_index()
                    
                    ranking['CT_Real'] = ranking['Tiempo_Total_Min'] / ranking['Total_Piezas']
                    ranking = ranking.sort_values('Total_Piezas', ascending=False)
                    
                    st.dataframe(ranking.style.background_gradient(subset=['CT_Real'], cmap='RdYlGn_r'), use_container_width=True)
            
        else:
            st.error("No encontré columna de Fecha.")
    else:
        st.error("Error al leer el archivo.")
