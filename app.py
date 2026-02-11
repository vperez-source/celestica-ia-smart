import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Celestica Pro Analyzer", layout="wide", page_icon="🏭")
st.title("🏭 Celestica IA: Análisis por Producto, Familia y Turno")
st.markdown("""
**Enfoque:** Analizamos `ProductID` y `Family` para entender qué se fabrica. 
Si el usuario es `VALUODC1`, usamos los saltos de tiempo en `In DateTime` para detectar turnos y paradas.
""")

# --- BARRA LATERAL (PARAMETRIZACIÓN) ---
with st.sidebar:
    st.header("⚙️ Definición de Tiempos")
    
    st.info("Ayuda a la IA a entender tus paradas:")
    umbral_lote = st.number_input(
        "Mínimo para Nuevo Lote (min):", 
        value=5, 
        help="Si pasan más de X minutos, asumimos que están preparando un nuevo lote."
    )
    
    umbral_descanso = st.number_input(
        "Mínimo para Descanso/Cambio (min):", 
        value=45, 
        help="Si el parón es mayor a esto, NO se cuenta como trabajo (es comida o cambio de turno)."
    )
    
    eficiencia_target = st.slider("Eficiencia Objetivo %", 50, 100, 85) / 100
    h_turno = st.number_input("Horas Turno Estándar", 8)

# --- 1. MOTOR DE LECTURA BLINDADO (XML/XLS) ---
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
    # Intentamos XML primero (para tus archivos .xls falsos)
    try: 
        file.seek(0)
        if "<?xml" in file.read(500).decode('latin-1', errors='ignore'): 
            file.seek(0); return leer_xml_robusto(file)
    except: pass
    # Intentamos Excel normal
    try: file.seek(0); return pd.read_excel(file, engine='calamine', header=None)
    except: pass
    # Intentamos CSV
    try: file.seek(0); return pd.read_csv(file, sep='\t', encoding='latin-1', header=None)
    except: return None

# --- 2. DETECCIÓN INTELIGENTE DE COLUMNAS ---
def mapear_columnas(df):
    if df is None: return None, {}
    df = df.astype(str)
    
    # Buscamos la fila de cabecera
    start = -1
    keywords = ['date', 'time', 'fecha', 'productid', 'family', 'station', 'user']
    
    for i in range(min(50, len(df))):
        row = df.iloc[i].str.lower().tolist()
        # Si la fila tiene "Date" y "Station", es la cabecera
        if any('date' in str(v) for v in row) and any('station' in str(v) for v in row):
            start = i; break
            
    if start == -1: return None, {}

    # Establecemos cabecera
    df.columns = df.iloc[start]
    df = df[start+1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()

    # Mapeo de nombres reales
    cols = {}
    for c in df.columns:
        cl = c.lower()
        if 'product' in cl and 'id' in cl: cols['Producto'] = c
        elif 'family' in cl: cols['Familia'] = c
        elif 'station' in cl or 'operation' in cl: cols['Estacion'] = c
        elif 'date' in cl or 'time' in cl: cols['Fecha'] = c
        elif 'user' in cl or 'operator' in cl: cols['Usuario'] = c

    # Validaciones mínimas
    if 'Fecha' not in cols: return None, {}
    
    # Si falta alguna no crítica, la rellenamos con "Desconocido"
    if 'Producto' not in cols: 
        df['Desconocido_Prod'] = "Generico"
        cols['Producto'] = 'Desconocido_Prod'
    if 'Familia' not in cols:
        df['Desconocido_Fam'] = "General"
        cols['Familia'] = 'Desconocido_Fam'
    if 'Usuario' not in cols:
        df['Desconocido_User'] = "VALUODC1"
        cols['Usuario'] = 'Desconocido_User'
        
    return df, cols

# --- 3. CEREBRO: PROCESAMIENTO DE TIEMPOS Y LOTES ---
def procesar_celestica(df, cols, umbral_lote, umbral_descanso):
    c_prod, c_fam, c_est, c_fec, c_usr = cols['Producto'], cols['Familia'], cols['Estacion'], cols['Fecha'], cols['Usuario']
    
    # A. Limpieza
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce')
    df = df.dropna(subset=[c_fec]).sort_values(c_fec) # Orden cronológico estricto
    
    # B. Cálculo de Gaps (Diferencia de tiempo con la pieza anterior)
    # Calculamos la diferencia en MINUTOS
    df['Gap_Min'] = df[c_fec].diff().dt.total_seconds().fillna(0) / 60
    
    # C. Lógica de "Usuario API" (VALUODC1) vs "Cambio de Turno"
    # Si el Gap es mayor al umbral de descanso (ej. 45 min), es un NUEVO TURNO/BLOQUE
    df['Nuevo_Bloque'] = df['Gap_Min'] > umbral_descanso
    df['Bloque_ID'] = df['Nuevo_Bloque'].cumsum()
    
    # Creamos un "Nombre de Turno Virtual" para pintar el gráfico
    def nombrar_turno(row):
        hora = row[c_fec].hour
        turno = "Mañana" if 6 <= hora < 14 else "Tarde" if 14 <= hora < 22 else "Noche"
        return f"{turno} (Bloque {row['Bloque_ID']})"
    
    df['Turno_Virtual'] = df.apply(nombrar_turno, axis=1)

    # D. Lógica de Lotes (Batch)
    # Si el Gap es > umbral_lote (ej. 5 min) PERO < umbral_descanso (ej. 45 min)
    # Entonces es TIEMPO DE PREPARACIÓN DE LOTE (Setup Time)
    
    # Asignamos el tiempo:
    # 1. Si Gap > Descanso -> Tiempo = 0 (No contamos la hora de comer como producción)
    # 2. Si Gap > Lote -> Tiempo = Gap (Es tiempo de preparación)
    # 3. Si Gap 0 o pequeño -> Tiempo = Gap (Es tiempo de escaneo rápido)
    
    df['Tiempo_Real_Trabajado'] = df['Gap_Min']
    df.loc[df['Gap_Min'] > umbral_descanso, 'Tiempo_Real_Trabajado'] = 0 # Ignorar comidas
    
    # E. Agrupación por Producto y Familia
    # Sumamos el tiempo trabajado y contamos las piezas
    return df

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Sube tu archivo (Excel/XML)", type=["xlsx", "xls", "xml", "txt"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        df_clean, cols = mapear_columnas(df_raw)
        
        if cols:
            # Procesamos
            df_final = procesar_celestica(df_clean, cols, umbral_lote, umbral_descanso)
            
            # --- KPIS GLOBALES ---
            tiempo_total = df_final['Tiempo_Real_Trabajado'].sum()
            piezas_totales = len(df_final)
            ct_medio_global = tiempo_total / piezas_totales if piezas_totales > 0 else 0
            capacidad = (h_turno * 60) / ct_medio_global * eficiencia_target if ct_medio_global > 0 else 0

            st.success(f"✅ Análisis Completado. Usuario API detectado: Separando turnos por paradas de >{umbral_descanso} min.")
            
            k1, k2, k3 = st.columns(3)
            k1.metric("⏱️ Cycle Time Global", f"{ct_medio_global:.2f} min/ud", help="Media ponderada de todos los productos.")
            k2.metric("📦 Capacidad Estándar", f"{int(capacidad)} uds")
            k3.metric("📊 Total Piezas", piezas_totales)
            
            st.divider()
            
            # --- ANÁLISIS POR FAMILIA Y PRODUCTO (LO QUE PEDISTE) ---
            st.subheader("🔬 Análisis Detallado: Familia & Producto")
            
            # Agrupamos
            c_prod, c_fam = cols['Producto'], cols['Familia']
            
            stats = df_final.groupby([c_fam, c_prod]).agg(
                Piezas=('Tiempo_Real_Trabajado', 'count'),
                Tiempo_Total=('Tiempo_Real_Trabajado', 'sum')
            ).reset_index()
            
            stats['CT_Real'] = stats['Tiempo_Total'] / stats['Piezas']
            stats = stats.sort_values('Piezas', ascending=False)
            
            col_tab, col_graph = st.columns([1, 1])
            
            with col_tab:
                st.write("**Tabla de Tiempos por Producto:**")
                st.dataframe(stats.style.background_gradient(subset=['CT_Real'], cmap='RdYlGn_r'), use_container_width=True)
            
            with col_graph:
                st.write("**Velocidad por Familia:**")
                fig = px.sunburst(stats, path=[c_fam, c_prod], values='Piezas', color='CT_Real',
                                title="Volumen (Tamaño) vs Velocidad (Color)",
                                color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig, use_container_width=True)

            # --- GANTT DE TURNOS (Detección de Parones) ---
            st.subheader("📅 Mapa de Turnos (Detección de Bloques)")
            st.markdown("Aunque el usuario sea 'VALUODC1', aquí ves los bloques de trabajo reales separados por descansos.")
            
            # Agrupamos por bloque para el Gantt
            gantt_data = df_final.groupby('Bloque_ID').agg(
                Inicio=(cols['Fecha'], 'min'),
                Fin=(cols['Fecha'], 'max'),
                Turno=('Turno_Virtual', 'first'),
                Piezas=('Tiempo_Real_Trabajado', 'count'),
                CT_Bloque=('Tiempo_Real_Trabajado', 'mean') # Aprox
            ).reset_index()
            
            # Filtramos bloques vacíos
            gantt_data = gantt_data[gantt_data['Piezas'] > 0]

            fig_gantt = px.timeline(gantt_data, x_start="Inicio", x_end="Fin", y="Turno",
                                  color="CT_Bloque", size="Piezas",
                                  hover_data=["Piezas"],
                                  title="Línea de Tiempo Operativa",
                                  color_continuous_scale='RdYlGn_r')
            fig_gantt.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_gantt, use_container_width=True)

        else:
            st.error("❌ No encontré las columnas necesarias (Date, Station). Revisa el archivo.")
    else:
        st.error("❌ Error de lectura.")
