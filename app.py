import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Pro Blindado", layout="wide", page_icon="🛡️")
st.title("🛡️ Celestica IA: Análisis Robusto (Anti-Errores)")
st.markdown("""
**Modo Seguro:** Si falta alguna columna (Familia, Producto), el sistema la auto-completa para no detener el cálculo.
**Objetivo:** Calcular Cycle Time real separando turnos por los descansos detectados en `In DateTime`.
""")

with st.sidebar:
    st.header("⚙️ Configuración")
    umbral_lote = st.number_input("Gap Mínimo Lote (min):", value=5)
    umbral_descanso = st.number_input("Gap Cambio Turno (min):", value=45)
    eficiencia_target = st.slider("Eficiencia %", 50, 100, 85) / 100
    h_turno = st.number_input("Horas Turno", 8)

# --- LECTURA ---
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

# --- MAPEO INTELIGENTE (LA CORRECCIÓN) ---
def mapear_columnas_seguro(df):
    if df is None: return None, {}
    df = df.astype(str)
    
    # 1. Buscar Cabecera
    start = -1
    keywords = ['date', 'time', 'fecha', 'product', 'family', 'station', 'user']
    
    for i in range(min(50, len(df))):
        row = df.iloc[i].str.lower().tolist()
        # Buscamos una fila que tenga al menos Fecha y Station
        if any('date' in str(v) for v in row) and any('station' in str(v) for v in row):
            start = i; break
            
    if start == -1: return None, {}

    # 2. Asignar nombres
    df.columns = df.iloc[start]
    df = df[start+1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()

    # 3. Diccionario de columnas (Inicializamos a None para evitar errores)
    cols = {
        'Fecha': None, 
        'Usuario': None, 
        'Estacion': None, 
        'Producto': None, 
        'Familia': None
    }

    # 4. Búsqueda Flexible
    for c in df.columns:
        cl = c.lower()
        if not cols['Fecha'] and ('date' in cl or 'time' in cl or 'fecha' in cl): cols['Fecha'] = c
        if not cols['Usuario'] and ('user' in cl or 'operator' in cl or 'usuario' in cl): cols['Usuario'] = c
        if not cols['Estacion'] and ('station' in cl or 'operation' in cl or 'maquina' in cl): cols['Estacion'] = c
        if not cols['Producto'] and ('product' in cl or 'item' in cl or 'part' in cl): cols['Producto'] = c
        if not cols['Familia'] and ('family' in cl or 'familia' in cl): cols['Familia'] = c

    # 5. AUTO-CORRECCIÓN (BLINDAJE)
    # Si no encontró alguna columna, creamos una falsa con valor "Desconocido"
    if not cols['Fecha']: return None, {} # Fecha es obligatoria
    
    if not cols['Producto']: 
        df['Producto_Generico'] = "Producto_Unico"
        cols['Producto'] = 'Producto_Generico'
        
    if not cols['Familia']: 
        df['Familia_Generica'] = "General"
        cols['Familia'] = 'Familia_Generica'
        
    if not cols['Usuario']: 
        df['Usuario_Generico'] = "VALUODC1"
        cols['Usuario'] = 'Usuario_Generico'

    if not cols['Estacion']:
        df['Estacion_Generica'] = "Linea_Principal"
        cols['Estacion'] = 'Estacion_Generica'

    return df, cols

# --- PROCESAMIENTO ---
def procesar(df, cols, umbral_lote, umbral_descanso):
    c_fec = cols['Fecha']
    
    # Limpieza
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce')
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    
    # Gaps
    df['Gap_Min'] = df[c_fec].diff().dt.total_seconds().fillna(0) / 60
    
    # Lógica de Turnos (Cortes grandes)
    df['Nuevo_Turno'] = df['Gap_Min'] > umbral_descanso
    df['Bloque_ID'] = df['Nuevo_Turno'].cumsum()
    
    # Lógica de Tiempo Real
    # Si es descanso (>45min) -> Tiempo = 0
    # Si es lote (>5min y <45min) -> Tiempo = Gap (Preparación)
    # Si es ráfaga (<5min) -> Tiempo = Gap (Ejecución)
    
    df['Tiempo_Real'] = df['Gap_Min']
    df.loc[df['Gap_Min'] > umbral_descanso, 'Tiempo_Real'] = 0 # Anular descansos
    
    # Nombre Turno Virtual
    def get_turno(row):
        h = row[c_fec].hour
        t = "Mañana" if 6<=h<14 else "Tarde" if 14<=h<22 else "Noche"
        return f"{t} (B{row['Bloque_ID']})"
    
    df['Turno_Virtual'] = df.apply(get_turno, axis=1)
    
    return df

# --- APP ---
uploaded_file = st.file_uploader("Sube el archivo", type=["xlsx", "xls", "xml", "txt"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        df_clean, cols = mapear_columnas_seguro(df_raw)
        
        if cols:
            df_final = procesar(df_clean, cols, umbral_lote, umbral_descanso)
            
            # --- RESULTADOS ---
            total_tiempo = df_final['Tiempo_Real'].sum()
            total_piezas = len(df_final)
            
            ct_global = total_tiempo / total_piezas if total_piezas > 0 else 0
            capacidad = (h_turno * 60) / ct_global * eficiencia_target if ct_global > 0 else 0
            
            st.success("✅ Datos Procesados Correctamente")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("⏱️ Cycle Time Global", f"{ct_global:.2f} min/ud")
            c2.metric("📦 Capacidad (8h)", f"{int(capacidad)} uds")
            c3.metric("📊 Piezas Totales", total_piezas)
            
            st.divider()
            
            # --- TABLA POR PRODUCTO/FAMILIA ---
            st.subheader("🔬 Desglose por Producto")
            c_prod, c_fam = cols['Producto'], cols['Familia']
            
            stats = df_final.groupby([c_fam, c_prod]).agg(
                Piezas=('Tiempo_Real', 'count'),
                Tiempo_Total=('Tiempo_Real', 'sum')
            ).reset_index()
            
            stats['CT_Real'] = stats['Tiempo_Total'] / stats['Piezas']
            stats = stats.sort_values('Piezas', ascending=False)
            
            st.dataframe(stats.style.background_gradient(subset=['CT_Real'], cmap='RdYlGn_r'), use_container_width=True)
            
            # --- GANTT ---
            st.subheader("📅 Mapa de Turnos (Bloques de Trabajo)")
            try:
                gantt = df_final.groupby('Bloque_ID').agg(
                    Inicio=(cols['Fecha'], 'min'),
                    Fin=(cols['Fecha'], 'max'),
                    Turno=('Turno_Virtual', 'first'),
                    Piezas=('Tiempo_Real', 'count'),
                    CT=('Tiempo_Real', 'mean')
                ).reset_index()
                
                gantt = gantt[gantt['Piezas'] > 0]
                
                fig = px.timeline(gantt, x_start="Inicio", x_end="Fin", y="Turno", color="CT",
                                title="Turnos Detectados por Inactividad",
                                color_continuous_scale='RdYlGn_r')
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo generar el gráfico: {e}")

        else:
            st.error("❌ No encontré columnas de fecha. Revisa el archivo.")
            st.write("Columnas detectadas:", list(df_clean.columns) if df_clean is not None else "Ninguna")
    else:
        st.error("Error al leer el archivo.")
