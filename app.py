import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Global Cycle", layout="wide", page_icon="⏱️")
st.title("⏱️ Celestica IA: Calculadora Global")

with st.sidebar:
    st.header("⚙️ Ajustes")
    # Ponemos el defecto en 0 para que NO borre nada
    min_sec = st.number_input("Ignorar gaps menores a (segundos):", 0, 60, 0, help="Si es 0, cuenta todo.")
    
    st.divider()
    h_turno = st.number_input("Horas Turno (Teórico)", value=8)
    eficiencia = st.slider("Eficiencia %", 50, 100, 85) / 100

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
                if data_tag: fila_datos.append(data_tag.get_text(strip=True))
                else: fila_datos.append("")
            if any(fila_datos): datos.append(fila_datos)
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
    try: file.seek(0); return pd.read_csv(file, sep='\t', encoding='latin-1', header=None)
    except: return None

# --- DETECTOR CABECERAS ---
def detectar_columnas(df):
    if df is None: return None, None, None
    df = df.astype(str)
    k_fecha = ['date', 'time', 'fecha', 'hora']
    k_usuario = ['user', 'operator', 'name', 'usuario']

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
    all_cols = list(df.columns)
    for col in all_cols:
        c_low = col.lower()
        if not col_f and any(k in c_low for k in k_fecha): col_f = col
        if not col_u and any(k in c_low for k in k_usuario): col_u = col
    if not col_u: col_u = all_cols[0]
        
    return df, col_f, col_u

# --- APP ---
uploaded_file = st.file_uploader("Sube el archivo", type=["xlsx", "xls", "txt", "xml"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        df, col_f, col_u = detectar_columnas(df_raw)
        
        if col_f:
            # 1. ORDENAR CRONOLÓGICAMENTE
            df[col_f] = pd.to_datetime(df[col_f], errors='coerce')
            df = df.dropna(subset=[col_f]).sort_values(col_f)
            
            # 2. MÉTODO GLOBAL (INFALIBLE)
            # Tiempo Total = Última fecha - Primera fecha
            inicio = df[col_f].iloc[0]
            fin = df[col_f].iloc[-1]
            tiempo_total_minutos = (fin - inicio).total_seconds() / 60
            
            # Piezas Totales = Número de filas
            piezas_totales = len(df)
            
            # Cycle Time = Tiempo Total / Piezas
            if piezas_totales > 0 and tiempo_total_minutos > 0:
                cycle_time_global = tiempo_total_minutos / piezas_totales
            else:
                cycle_time_global = 0

            # Capacidad Real
            capacidad = (h_turno * 60) / cycle_time_global * eficiencia if cycle_time_global > 0 else 0

            # --- RESULTADOS ---
            st.success(f"✅ Cálculo Global Exitoso ({inicio.strftime('%H:%M')} - {fin.strftime('%H:%M')})")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("⏱️ Cycle Time Promedio", f"{cycle_time_global:.2f} min/ud", help="Cálculo: (Hora Fin - Hora Inicio) / Total Piezas")
            c2.metric("📦 Capacidad Turno", f"{int(capacidad)} uds")
            c3.metric("📊 Total Procesado", f"{piezas_totales} uds")

            st.divider()

            # --- GRÁFICAS ---
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.subheader("📈 Producción en el Tiempo")
                # Agrupamos por hora para ver cuándo se trabajó
                df['Hora'] = df[col_f].dt.hour
                prod_hora = df.groupby('Hora').size().reset_index(name='Piezas')
                fig = px.bar(prod_hora, x='Hora', y='Piezas', title="Ritmo por Hora", color='Piezas', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
            
            with col_g2:
                if col_u:
                    st.subheader("🏆 Operarios (Volumen)")
                    user_stats = df.groupby(col_u).size().reset_index(name='Piezas').sort_values('Piezas', ascending=False)
                    st.dataframe(user_stats.style.background_gradient(cmap='Greens'), use_container_width=True)

            # --- DEBUGGING (LO QUE TÚ NECESITAS VER) ---
            with st.expander("🔍 Ver por qué daba error antes (Tabla de Tiempos)"):
                df['Diferencia_Segundos'] = df[col_f].diff().dt.total_seconds().fillna(0)
                st.write("Mira la columna 'Diferencia_Segundos'. Si ves muchos ceros, es que es un proceso Batch.")
                st.dataframe(df[[col_f, 'Diferencia_Segundos']].head(50))

        else:
            st.error("No encontré la columna de fecha.")
    else:
        st.error("Error de lectura.")
