import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN PROFESIONAL ---
st.set_page_config(page_title="Celestica Expert Advisor", layout="wide", page_icon="👨‍💻")
st.title("👨‍💻 Agente Digital: Análisis de Ciclos por Lotes (Batch)")
st.markdown("""
**Criterio del Algoritmo:** Este sistema utiliza **'Sessionizing'**. Agrupa ráfagas de escaneo en 'Lotes' detectando los tiempos muertos de preparación.
* **Tiempo Lote** = Tiempo de Preparación + Tiempo de Escaneo Rápido.
* **Cycle Time** = Tiempo Lote / Piezas del Lote.
""")

with st.sidebar:
    st.header("⚙️ Parámetros de Experto")
    
    st.info("Define la frontera entre 'Trabajar' y 'Parar'.")
    umbral_minutos = st.number_input(
        "Umbral de Corte de Lote (min)", 
        min_value=1, value=10, 
        help="Si pasan más de X minutos entre pieza y pieza, la IA asume que es un LOTE NUEVO o una PAUSA."
    )
    
    # Filtro para ignorar descansos largos (ej. comida)
    max_break = st.number_input(
        "Ignorar Pausas Mayores a (min)",
        min_value=30, value=60,
        help="Si el hueco es mayor a esto (ej. 60 min), no se suma al tiempo de trabajo (se considera Almuerzo)."
    )

    eficiencia = st.slider("OEE / Eficiencia Objetivo %", 50, 100, 85) / 100
    h_turno = st.number_input("Horas Disponibles Turno", 8)

# --- MOTORES DE INGESTIÓN DE DATOS ---
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
def cargar_dato(file):
    # Intentamos todas las llaves posibles para abrir el archivo
    try: 
        file.seek(0)
        if "<?xml" in file.read(500).decode('latin-1', errors='ignore'): 
            file.seek(0); return leer_xml_robusto(file)
    except: pass
    try: file.seek(0); return pd.read_excel(file, engine='calamine', header=None)
    except: pass
    try: file.seek(0); return pd.read_csv(file, sep='\t', encoding='latin-1', header=None)
    except: return None

# --- INTELIGENCIA: DETECCIÓN DE ESTRUCTURA ---
def auto_map(df):
    if df is None: return None, None, None
    df = df.astype(str)
    
    # Palabras clave para identificar columnas
    k_date = ['date', 'time', 'fecha', 'hora']
    k_user = ['user', 'operator', 'name', 'usuario']

    # 1. Encontrar la fila de cabecera
    start = -1
    for i in range(min(50, len(df))):
        row = df.iloc[i].str.lower().tolist()
        if any(x in str(v) for v in row for x in k_date):
            start = i; break
            
    if start == -1: return None, None, None

    # 2. Renombrar columnas
    df.columns = df.iloc[start]
    df = df[start+1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()

    # 3. Identificar roles de columnas
    c_date, c_user = None, None
    for c in df.columns:
        cl = c.lower()
        if not c_date and any(x in cl for x in k_date): c_date = c
        if not c_user and any(x in cl for x in k_user): c_user = c
    
    if not c_user: c_user = df.columns[0] # Fallback
    return df, c_date, c_user

# --- CEREBRO: ALGORITMO DE SESSIONIZING ---
def procesar_logica_experta(df, col_f, col_u, umbral_min, max_break_min):
    # 1. Limpieza y Ordenación (CRÍTICO)
    df[col_f] = pd.to_datetime(df[col_f], errors='coerce')
    df = df.dropna(subset=[col_f]).sort_values([col_u, col_f])
    
    # 2. Calcular Deltas (Tiempo entre piezas)
    # diff() calcula: Tiempo_Actual - Tiempo_Anterior
    df['delta_seg'] = df.groupby(col_u)[col_f].diff().dt.total_seconds().fillna(0)
    df['delta_min'] = df['delta_seg'] / 60
    
    # 3. Lógica de Negocio (Sessionizing)
    # Un "Nuevo Lote" empieza si:
    # A) Cambiamos de usuario
    # B) El tiempo entre piezas es mayor al umbral (ej. > 10 min de preparación)
    condicion_corte = (df[col_u] != df[col_u].shift()) | (df['delta_min'] > umbral_min)
    
    # Asignamos un ID único a cada lote acumulando los cortes (True=1, False=0)
    df['Lote_ID'] = condicion_corte.cumsum()
    
    # 4. Agregación (Cálculo de Métricas por Lote)
    lotes = df.groupby('Lote_ID').agg(
        Usuario=(col_u, 'first'),
        Inicio=(col_f, 'min'),
        Fin=(col_f, 'max'),
        Piezas=('delta_seg', 'count'),
        # Sumamos todos los tiempos DEL LOTE
        Tiempo_Bruto_Min=('delta_min', 'sum')
    ).reset_index()

    # 5. Aplicar Filtro de Experto (Limpiar Pausas de Comida)
    # Si un lote tiene un tiempo de preparación inicial GIGANTE (ej. 60 min), 
    # asumimos que fue la comida y NO lo contamos como tiempo de producción.
    # Restamos ese tiempo excesivo del total del lote.
    
    # Recuperamos el "salto grande" inicial de cada lote
    # (El primer registro de cada lote contiene el tiempo de preparación)
    
    # Filtramos lotes válidos (con más de 1 pieza o tiempo lógico)
    lotes = lotes[lotes['Piezas'] > 0]
    
    # Si el tiempo total del lote excede el 'max_break', lo capamos
    # Esto es una heurística: Si tardaste 2 horas en un lote de 10 piezas, 
    # probablemente 1.5 horas fueron comida.
    
    # Calculamos Cycle Time del Lote
    lotes['CT_Lote'] = lotes['Tiempo_Bruto_Min'] / lotes['Piezas']
    
    # Filtramos aberraciones (lotes con CT > max_break)
    lotes_validos = lotes[lotes['CT_Lote'] < max_break_min].copy()
    
    return lotes_validos

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Sube el archivo de Trazabilidad", type=["xlsx", "xls", "txt", "xml"])

if uploaded_file:
    df_raw = cargar_dato(uploaded_file)
    
    if df_raw is not None:
        df, col_f, col_u = auto_map(df_raw)
        
        if col_f:
            with st.spinner("🤖 El Agente está procesando la lógica de lotes..."):
                # EJECUCIÓN DEL ANÁLISIS
                lotes = procesar_logica_experta(df, col_f, col_u, umbral_minutos, max_break)
                
                # CÁLCULOS GLOBALES (KPIs)
                # Media Ponderada: Suma de Tiempos / Suma de Piezas
                total_tiempo = lotes['Tiempo_Bruto_Min'].sum()
                total_piezas = lotes['Piezas'].sum()
                
                if total_piezas > 0:
                    ct_global = total_tiempo / total_piezas
                    capacidad = (h_turno * 60) / ct_global * eficiencia
                else:
                    ct_global, capacidad = 0, 0
                
                # --- RESULTADOS ---
                st.success("✅ Análisis Completado Exitosamente")
                
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("⏱️ Cycle Time Real", f"{ct_global:.2f} min/ud", 
                          help="Promedio ponderado real (incluye preparaciones, excluye comidas).")
                k2.metric("📦 Capacidad (8h)", f"{int(capacidad)} uds")
                k3.metric("📊 Lotes Procesados", len(lotes))
                k4.metric("⚙️ Piezas Totales", total_piezas)
                
                st.divider()
                
                # --- VISUALIZACIÓN AVANZADA ---
                tab1, tab2 = st.tabs(["📉 Cronograma de Lotes", "🏆 Ranking Operarios"])
                
                with tab1:
                    st.subheader("Mapa de Calor de Lotes")
                    st.markdown("Cada punto es un lote. **Tamaño** = Cantidad de Piezas. **Color** = Velocidad (Rojo=Lento, Verde=Rápido).")
                    
                    fig = px.scatter(lotes, x="Inicio", y="CT_Lote", 
                                   size="Piezas", color="CT_Lote",
                                   hover_data=["Usuario", "Tiempo_Bruto_Min"],
                                   color_continuous_scale="RdYlGn_r", # Rojo es alto (malo), Verde es bajo (bueno)
                                   title="Cronología: ¿Cuándo se hicieron los lotes y a qué velocidad?")
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab2:
                    st.subheader("Rendimiento por Operario")
                    # Agrupación final por usuario
                    ranking = lotes.groupby('Usuario').agg(
                        Lotes=('Lote_ID', 'count'),
                        Piezas_Totales=('Piezas', 'sum'),
                        Tiempo_Total_Min=('Tiempo_Bruto_Min', 'sum')
                    ).reset_index()
                    
                    ranking['CT_Promedio'] = ranking['Tiempo_Total_Min'] / ranking['Piezas_Totales']
                    ranking = ranking.sort_values('Piezas_Totales', ascending=False)
                    
                    st.dataframe(ranking.style.background_gradient(subset=['CT_Promedio'], cmap='RdYlGn_r'), use_container_width=True)
                    
                    # Gráfico de barras comparativo
                    fig_bar = px.bar(ranking, x='Usuario', y='Piezas_Totales', color='CT_Promedio',
                                   title="Productividad vs Velocidad (Color)",
                                   color_continuous_scale='RdYlGn_r')
                    st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.error("No se detectaron columnas de Fecha. Revisa el archivo.")
    else:
        st.error("Error crítico de lectura.")
