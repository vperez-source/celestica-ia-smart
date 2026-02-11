import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup # Esta es la nueva herramienta clave

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica AI Ultimate", layout="wide", page_icon="🛡️")
st.title("🛡️ Celestica IA: Advance Tracking Analyzer")

with st.sidebar:
    st.header("⚙️ Configuración")
    contamination = st.slider("Sensibilidad IA (% Ruido)", 1, 25, 5) / 100
    st.divider()
    h_turno = st.number_input("Horas Turno", value=8)
    m_descanso = st.number_input("Minutos Descanso", value=45)
    eficiencia = st.slider("Eficiencia %", 50, 100, 75) / 100

# --- LECTOR DE RESCATE (BEAUTIFUL SOUP) ---
def leer_xml_a_la_fuerza(file):
    """Usa BeautifulSoup para extraer datos de XMLs rotos o complejos."""
    try:
        # Leemos el archivo como texto plano, ignorando errores de caracteres
        content = file.getvalue().decode('latin-1', errors='ignore')
        
        # Creamos la 'sopa'
        soup = BeautifulSoup(content, 'xml') # Usamos el parser XML de lxml
        
        datos = []
        # Buscamos todas las filas (Row)
        rows = soup.find_all(['Row', 'ss:Row', 'row'])
        
        for row in rows:
            fila_datos = []
            # Buscamos las celdas (Cell) y dentro los datos (Data)
            cells = row.find_all(['Cell', 'ss:Cell', 'cell'])
            for cell in cells:
                data_tag = cell.find(['Data', 'ss:Data', 'data'])
                if data_tag:
                    fila_datos.append(data_tag.get_text(strip=True))
                else:
                    fila_datos.append("") # Celda vacía
            
            if any(fila_datos): # Si la fila no está vacía del todo
                datos.append(fila_datos)
                
        return pd.DataFrame(datos)
    except Exception as e:
        st.error(f"Error en BeautifulSoup: {e}")
        return None

# --- GESTOR DE CARGA ---
@st.cache_data(ttl=3600)
def load_data(file):
    # 1. INTENTO: BeautifulSoup (Para tu archivo XML rebelde)
    try:
        file.seek(0)
        # Leemos los primeros caracteres para ver si es el XML de Microsoft
        head = file.read(500).decode('latin-1', errors='ignore')
        file.seek(0)
        if "<?xml" in head or "Workbook" in head:
            return leer_xml_a_la_fuerza(file)
    except:
        pass

    # 2. INTENTO: Calamine (Excel binario corrupto)
    try:
        file.seek(0)
        return pd.read_excel(file, engine='calamine', header=None)
    except:
        pass

    # 3. INTENTO: HTML (Tablas web)
    try:
        file.seek(0)
        dfs = pd.read_html(file, header=None)
        if len(dfs) > 0: return dfs[0]
    except:
        pass

    # 4. INTENTO: Texto plano
    try:
        file.seek(0)
        return pd.read_csv(file, sep='\t', encoding='latin-1', header=None)
    except:
        return None

# --- NORMALIZADOR DE COLUMNAS ---
def encontrar_cabecera_y_normalizar(df):
    if df is None: return None, None, None, None
    df = df.astype(str) # Todo a texto
    
    # Palabras clave
    k_fecha = ['date', 'time', 'fecha', 'hora', 'timestamp']
    k_estacion = ['station', 'operation', 'work', 'estacion', 'maquina', 'productid']
    k_usuario = ['user', 'operator', 'name', 'usuario', 'created by']

    start_row = -1
    
    # Buscamos la fila de cabecera en las primeras 50 filas
    for i in range(min(50, len(df))):
        fila = df.iloc[i].str.lower().tolist()
        has_date = any(k in str(v) for v in fila for k in k_fecha)
        # Relajamos la condición: con que tenga fecha nos vale para empezar
        if has_date:
            start_row = i
            break
            
    if start_row == -1: return None, None, None, None

    # Cortar
    df.columns = df.iloc[start_row]
    df = df[start_row + 1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()
    
    # Identificar columnas
    col_f, col_s, col_u = None, None, None

    for col in df.columns:
        c_low = col.lower()
        if not col_f and any(k in c_low for k in k_fecha): col_f = col
        if not col_s and any(k in c_low for k in k_estacion): col_s = col
        if not col_u and any(k in c_low for k in k_usuario): col_u = col
        
    return df, col_f, col_s, col_u

# --- APP ---
uploaded_file = st.file_uploader("Sube el archivo XML/Excel", type=["xlsx", "xls", "txt", "xml"])

if uploaded_file:
    with st.spinner("⏳ Extrayendo datos con BeautifulSoup..."):
        df_raw = load_data(uploaded_file)

        if df_raw is not None:
            # 1. NORMALIZAR
            df, col_f, col_s, col_u = encontrar_cabecera_y_normalizar(df_raw)

            if not col_f:
                st.error("❌ Pude leer el archivo, pero no encontré la cabecera 'Date' o 'Time'.")
                st.write("Datos extraídos (primeras 5 filas):", df_raw.head())
                st.stop()
            
            if not col_s:
                st.warning("⚠️ No encontré columna 'Station'. Usando la 1ª columna como agrupación.")
                col_s = df.columns[0]

            st.success(f"✅ Mapeo Exitoso: Fecha='{col_f}' | Estación='{col_s}' | Usuario='{col_u}'")

            # 2. PROCESAR
            try:
                # Limpieza Fechas
                df[col_f] = pd.to_datetime(df[col_f], errors='coerce')
                df.loc[df[col_f].dt.year < 100, col_f] += pd.offsets.DateOffset(years=2000)
                df = df.dropna(subset=[col_f]).sort_values(col_f)

                if df.empty:
                    st.error("No hay fechas válidas.")
                    st.stop()

                # Gaps
                df['gap_mins'] = df.groupby(col_s)[col_f].diff().dt.total_seconds() / 60
                df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())

                # IA
                model = IsolationForest(contamination=contamination, random_state=42)
                df['IA_Status'] = model.fit_predict(df[['gap_mins']])
                
                df_clean = df[df['IA_Status'] == 1].copy()
                df_noise = df[df['IA_Status'] == -1]

                # KPIs
                media = df_clean['gap_mins'].mean()
                capacidad = ((h_turno*60 - m_descanso)/media) * eficiencia

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Cycle Time", f"{media:.2f} min")
                k2.metric("Capacidad", f"{int(capacidad)} uds")
                k3.metric("Piezas OK", len(df_clean))
                k4.metric("Ruido", len(df_noise), delta_color="inverse")

                st.markdown("---")

                # GRÁFICAS
                if col_u:
                    st.subheader("🏆 Ranking Productividad")
                    stats = df_clean.groupby(col_u)['gap_mins'].agg(['count', 'mean']).reset_index()
                    stats.columns = ['Operario', 'Piezas', 'Velocidad']
                    stats = stats.sort_values('Piezas', ascending=False)

                    c1, c2 = st.columns([1,1])
                    with c1:
                        st.dataframe(stats.style.background_gradient(subset=['Piezas'], cmap='Greens'), use_container_width=True)
                    with c2:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=stats['Operario'], y=stats['Piezas'], name='Piezas', marker_color='#2ecc71', yaxis='y'))
                        fig.add_trace(go.Scatter(x=stats['Operario'], y=stats['Velocidad'], name='Velocidad', marker_color='#e74c3c', yaxis='y2'))
                        
                        # Layout corregido
                        fig.update_layout(
                            title="Volumen vs Velocidad",
                            yaxis=dict(title=dict(text="Piezas", font=dict(color="#2ecc71"))),
                            yaxis2=dict(title=dict(text="Min/Pieza", font=dict(color="#e74c3c")), overlaying='y', side='right')
                        )
                        st.plotly_chart(fig, use_container_width=True)

                st.subheader("Mapa IA")
                fig = px.scatter(df, x=col_f, y='gap_mins', color=df['IA_Status'].astype(str),
                               color_discrete_map={'1': '#2ecc71', '-1': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error cálculo: {e}")

        else:
            st.error("FATAL: Ni siquiera BeautifulSoup pudo leer este archivo.")
