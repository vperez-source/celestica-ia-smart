import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN EXPERTA ---
st.set_page_config(page_title="Celestica Spectrum Analyzer", layout="wide", page_icon="🕵️")
st.title("🕵️ Celestica IA: Detector de Turnos (Spectrum/SOAC)")
st.markdown("""
**Problema Detectado:** El usuario `VALUODC1 SPECIAL USER(API)` oculta a los operarios reales.
**Solución IA:** El algoritmo ignora el nombre y **detecta los cambios de turno basándose en los parones de tiempo.**
""")

with st.sidebar:
    st.header("⚙️ Configuración de Cortes")
    
    st.info("Define qué consideras un 'Cambio de Turno' o 'Descanso'.")
    umbral_corte_minutos = st.number_input(
        "Minutos de inactividad para cortar Bloque:", 
        min_value=10, value=30, 
        help="Si la máquina está parada más de X minutos, asumimos que ha cambiado el turno o han ido a comer."
    )
    
    eficiencia = st.slider("Eficiencia Objetivo %", 50, 100, 85) / 100
    h_turno = st.number_input("Horas Turno", 8)

# --- MOTORES DE LECTURA ---
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

# --- DETECTOR INTELIGENTE ---
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

# --- CEREBRO: SEGMENTACIÓN TEMPORAL (LA SOLUCIÓN AL USUARIO API) ---
def procesar_cortes_de_tiempo(df, col_f, col_u, umbral_corte):
    # 1. Limpieza
    df[col_f] = pd.to_datetime(df[col_f], errors='coerce')
    df = df.dropna(subset=[col_f]).sort_values(col_f) # ORDEN ABSOLUTO POR TIEMPO
    
    # 2. Calcular Deltas (Tiempo entre CUALQUIER pieza)
    # Aquí NO agrupamos por usuario, porque el usuario es el genérico 'VALUODC1'
    df['delta_min'] = df[col_f].diff().dt.total_seconds().fillna(0) / 60
    
    # 3. DETECTAR CORTES (Saltos grandes de tiempo)
    # Si pasa más del umbral (ej. 30 min), es un NUEVO BLOQUE DE TRABAJO
    df['Nuevo_Bloque'] = df['delta_min'] > umbral_corte
    
    # 4. Asignar ID de Bloque
    df['Bloque_ID'] = df['Nuevo_Bloque'].cumsum() + 1
    
    # 5. CREAR "USUARIO VIRTUAL"
    # Si el usuario es el genérico, le ponemos nombre basado en la hora del bloque
    usuario_generico_detectado = df[col_u].astype(str).str.contains("VALUODC", case=False).any()
    
    if usuario_generico_detectado:
        # Función para nombrar el bloque según la hora de inicio
        def nombrar_bloque(grupo):
            hora_inicio = grupo[col_f].min()
            nombre_original = grupo[col_u].iloc[0]
            
            # Si es el usuario API, lo renombramos
            if "VALUODC" in str(nombre_original).upper() or "SPECIAL" in str(nombre_original).upper():
                turno = "Mañana" if 6 <= hora_inicio.hour < 14 else "Tarde" if 14 <= hora_inicio.hour < 22 else "Noche"
                return f"Operario_{turno}_{hora_inicio.strftime('%H:%M')}"
            return nombre_original

        # Aplicamos el renombrado agrupando por bloque
        nombres_bloques = df.groupby('Bloque_ID').apply(nombrar_bloque)
        # Mapeamos de vuelta al DF original
        df['Usuario_Virtual'] = df['Bloque_ID'].map(nombres_bloques)
    else:
        df['Usuario_Virtual'] = df[col_u] # Si son usuarios reales, los dejamos

    # 6. Calcular Métricas por Bloque (Sessionizing)
    # Agrupamos por nuestro nuevo Usuario Virtual
    resumen = df.groupby('Usuario_Virtual').agg(
        Inicio=(col_f, 'min'),
        Fin=(col_f, 'max'),
        Piezas=('delta_min', 'count'),
        # Sumamos tiempos (excluyendo el gran salto inicial del bloque)
        # Filtramos los deltas menores al corte para sumar solo tiempo productivo
        Tiempo_Activo_Min=('delta_min', lambda x: x[x < umbral_corte].sum())
    ).reset_index()
    
    # Corregir tiempo activo: Si el tiempo es 0 (solo 1 pieza), le damos un tiempo mínimo
    resumen['Tiempo_Activo_Min'] = resumen['Tiempo_Activo_Min'].replace(0, 1) 
    
    resumen['CT_Real'] = resumen['Tiempo_Activo_Min'] / resumen['Piezas']
    
    return resumen.sort_values('Inicio')

# --- APP ---
uploaded_file = st.file_uploader("Sube el archivo (Spectrum/SOAC)", type=["xlsx", "xls", "txt", "xml"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        df, col_f, col_u = auto_map(df_raw)
        
        if col_f:
            with st.spinner("🤖 Detectando paradas para separar turnos..."):
                
                resumen = procesar_cortes_de_tiempo(df, col_f, col_u, umbral_corte_minutos)
                
                # --- KPIs GLOBALES ---
                total_piezas = resumen['Piezas'].sum()
                # Media Ponderada Global
                if total_piezas > 0:
                    ct_global = resumen['Tiempo_Activo_Min'].sum() / total_piezas
                    capacidad = (h_turno * 60) / ct_global * eficiencia
                else:
                    ct_global, capacidad = 0, 0
                
                st.success("✅ Análisis Temporal Completado")
                if resumen['Usuario_Virtual'].str.contains("Operario_").any():
                    st.warning("⚠️ Se detectó usuario genérico (API). He separado los turnos automáticamente basándome en los descansos.")

                k1, k2, k3 = st.columns(3)
                k1.metric("⏱️ Cycle Time Global", f"{ct_global:.2f} min/ud")
                k2.metric("📦 Capacidad (8h)", f"{int(capacidad)} uds")
                k3.metric("🔄 Bloques Detectados", len(resumen))
                
                st.divider()
                
                # --- VISUALIZACIÓN GANTT (CLAVE PARA VER LOS TURNOS) ---
                st.subheader("📅 Cronograma de Actividad (Gantt)")
                st.markdown("Cada barra es un bloque de trabajo continuo. Los espacios vacíos son las paradas (descansos/cambios).")
                
                fig = px.timeline(resumen, x_start="Inicio", x_end="Fin", y="Usuario_Virtual", 
                                color="CT_Real", 
                                size="Piezas", # Grosor de la barra (truco visual)
                                hover_data=["Piezas", "Tiempo_Activo_Min"],
                                color_continuous_scale="RdYlGn_r",
                                title="Turnos Identificados Automáticamente")
                fig.update_yaxes(autorange="reversed") # Orden cronológico arriba-abajo
                st.plotly_chart(fig, use_container_width=True)
                
                # --- TABLA DE DETALLE ---
                st.subheader("📋 Detalle por Bloque de Trabajo")
                st.dataframe(resumen.style.background_gradient(subset=['CT_Real'], cmap='RdYlGn_r'), use_container_width=True)

        else:
            st.error("No se encontró columna de fecha.")
    else:
        st.error("Error al leer el archivo.")
