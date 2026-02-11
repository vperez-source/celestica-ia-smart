import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN EXPERTA ---
st.set_page_config(page_title="Celestica Spectrum Fix", layout="wide", page_icon="🔧")
st.title("🔧 Celestica IA: Corrección de Lotes Instantáneos")
st.markdown("""
**Corrección Aplicada:** Si las piezas se escanean en el mismo segundo (Tiempo=0), 
el algoritmo imputa el tiempo transcurrido desde el lote anterior como 'Tiempo de Preparación'.
""")

with st.sidebar:
    st.header("⚙️ Ajuste de Lotes")
    st.info("Configura cómo interpretar los silencios entre escaneos.")
    
    umbral_corte = st.number_input(
        "Corte de Lote (minutos):", 
        min_value=5, value=20, 
        help="Si pasa más de este tiempo, se considera un lote nuevo."
    )
    
    max_prep = st.number_input(
        "Máximo Tiempo Preparación (min):", 
        min_value=20, value=60, 
        help="Si el hueco es mayor a esto (ej. 60 min), se considera ALMUERZO y no se suma al tiempo."
    )
    
    eficiencia = st.slider("Eficiencia %", 50, 100, 85) / 100
    h_turno = st.number_input("Horas Turno", 8)

# --- LECTORES BLINDADOS ---
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

# --- AUTO-MAPEO ---
def auto_map(df):
    if df is None: return None, None, None
    df = df.astype(str)
    k_date = ['date', 'time', 'fecha', 'hora']
    k_user = ['user', 'operator', 'name', 'usuario']

    start = -1
    for i in range(min(50, len(df))):
        row = df.iloc[i].str.lower().tolist()
        if any(x in str(v) for v in row for x in k_date):
            start = i; break
            
    if start == -1: return None, None, None

    df.columns = df.iloc[start]
    df = df[start+1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()

    c_date, c_user = None, None
    for c in df.columns:
        cl = c.lower()
        if not c_date and any(x in cl for x in k_date): c_date = c
        if not c_user and any(x in cl for x in k_user): c_user = c
    
    if not c_user: c_user = df.columns[0]
    return df, c_date, c_user

# --- CEREBRO: LÓGICA DE IMPUTACIÓN DE TIEMPO (LA SOLUCIÓN) ---
def procesar_lotes_reales(df, col_f, col_u, corte_min, max_prep_min):
    # 1. Preparar datos
    df[col_f] = pd.to_datetime(df[col_f], errors='coerce')
    df = df.dropna(subset=[col_f]).sort_values(col_f) # ORDEN CRONOLÓGICO STRICTO
    
    # 2. Detectar Cortes
    # Calculamos la diferencia con la fila anterior
    df['diff_min'] = df[col_f].diff().dt.total_seconds().fillna(0) / 60
    
    # Un nuevo lote empieza si hay un salto de tiempo > corte_min
    # O si cambia el usuario (si hubiera usuarios reales)
    df['Nuevo_Lote'] = (df['diff_min'] > corte_min)
    df['Lote_ID'] = df['Nuevo_Lote'].cumsum()
    
    # 3. Agrupar por Lote
    resumen = df.groupby('Lote_ID').agg(
        Inicio_Lote=(col_f, 'min'),
        Fin_Lote=(col_f, 'max'),
        Piezas=('diff_min', 'count'),
        # Usuario mayoritario del lote
        Usuario=(col_u, lambda x: x.mode()[0] if not x.mode().empty else "Desconocido") 
    ).reset_index()

    # 4. CALCULAR TIEMPO REAL DEL LOTE (LA CLAVE)
    # El tiempo del lote no es (Fin - Inicio) porque si es instantáneo da 0.
    # El tiempo es: (Fin de este lote - Fin del lote anterior)
    # Es decir, imputamos el tiempo "hueco" al lote actual.
    
    resumen['Fin_Lote_Anterior'] = resumen['Fin_Lote'].shift(1)
    
    # Para el primer lote, asumimos que tardó lo mismo que la duración interna o 0
    resumen.loc[0, 'Fin_Lote_Anterior'] = resumen.loc[0, 'Inicio_Lote']

    # Tiempo Total = Fin Actual - Fin Anterior
    resumen['Tiempo_Real_Min'] = (resumen['Fin_Lote'] - resumen['Fin_Lote_Anterior']).dt.total_seconds() / 60
    
    # --- FILTRO DE COMIDAS ---
    # Si el tiempo calculado es GIGANTE (ej. 120 min), es que hubo una comida en medio.
    # Lo capamos al máximo permitido (ej. 60 min) o lo marcamos como descanso.
    
    # Si es el primer lote y sale negativo o raro, lo arreglamos
    resumen['Tiempo_Real_Min'] = resumen['Tiempo_Real_Min'].fillna(0)
    
    # Si el tiempo es mayor a max_prep_min, asumimos que solo trabajaron 'corte_min' y el resto fue descanso
    mask_descanso = resumen['Tiempo_Real_Min'] > max_prep_min
    resumen.loc[mask_descanso, 'Tiempo_Real_Min'] = corte_min # Imputamos un tiempo estándar
    
    # Evitamos tiempos 0 absolutos (imputamos 1 segundo mínimo por pieza)
    min_time = resumen['Piezas'] * (1/60) # 1 segundo por pieza
    resumen['Tiempo_Real_Min'] = np.maximum(resumen['Tiempo_Real_Min'], min_time)

    # 5. Cycle Time
    resumen['CT_Real'] = resumen['Tiempo_Real_Min'] / resumen['Piezas']
    
    return resumen

# --- APP ---
uploaded_file = st.file_uploader("Sube el archivo", type=["xlsx", "xls", "txt", "xml"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        df, col_f, col_u = auto_map(df_raw)
        
        if col_f:
            with st.spinner("🔄 Reconstruyendo tiempos de preparación..."):
                try:
                    resumen = procesar_lotes_reales(df, col_f, col_u, umbral_corte, max_prep)
                    
                    if resumen.empty:
                        st.error("No se detectaron datos válidos.")
                        st.stop()

                    # --- BLINDAJE DE DATOS PARA PLOTLY ---
                    # Aseguramos que no haya NaNs ni Infinitos antes de graficar
                    resumen = resumen.replace([np.inf, -np.inf], np.nan).dropna(subset=['CT_Real'])
                    
                    # KPIs GLOBALES
                    total_tiempo = resumen['Tiempo_Real_Min'].sum()
                    total_piezas = resumen['Piezas'].sum()
                    
                    ct_global = total_tiempo / total_piezas if total_piezas > 0 else 0
                    capacidad = (h_turno * 60) / ct_global * eficiencia if ct_global > 0 else 0

                    st.success("✅ Tiempos Recalculados Correctamente")
                    
                    k1, k2, k3 = st.columns(3)
                    k1.metric("⏱️ Cycle Time Real", f"{ct_global:.2f} min/ud", help="Incluye tiempo de preparación entre lotes.")
                    k2.metric("📦 Capacidad (8h)", f"{int(capacidad)} uds")
                    k3.metric("📊 Lotes Detectados", len(resumen))
                    
                    st.divider()

                    # --- GRÁFICA GANTT ---
                    # Usamos un try-except específico para la gráfica
                    try:
                        st.subheader("📅 Cronograma de Lotes (Gantt)")
                        # Creamos una columna de texto para el hover
                        resumen['Info'] = resumen.apply(lambda x: f"{int(x['Piezas'])} uds en {int(x['Tiempo_Real_Min'])} min", axis=1)
                        
                        fig = px.timeline(resumen, 
                                        x_start="Inicio_Lote", 
                                        x_end="Fin_Lote", 
                                        y="Usuario",
                                        color="CT_Real",
                                        hover_name="Info",
                                        title="Lotes Identificados (Color = Velocidad)",
                                        color_continuous_scale="RdYlGn_r",
                                        range_color=[0, resumen['CT_Real'].quantile(0.95)]) # Evitamos que un outlier rompa la escala de color
                        
                        fig.update_yaxes(autorange="reversed")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo generar el Gantt visual (Error de datos), pero los cálculos son correctos. Error: {e}")

                    # --- TABLA ---
                    st.subheader("📋 Detalle de Lotes")
                    st.dataframe(resumen[['Inicio_Lote', 'Usuario', 'Piezas', 'Tiempo_Real_Min', 'CT_Real']].style.background_gradient(subset=['CT_Real'], cmap='RdYlGn_r'), use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error procesando los datos: {e}")
                    st.write("Intenta subir un archivo con más datos o revisa que la columna Fecha sea correcta.")
        else:
            st.error("No encontré columna de fecha.")
    else:
        st.error("Error al leer el archivo.")
