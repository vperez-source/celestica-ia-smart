import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup

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
    except:
        return None

# --- GESTOR DE CARGA ---
@st.cache_data(ttl=3600)
def load_data(file):
    # 1. BeautifulSoup (XML 2003)
    try:
        file.seek(0)
        head = file.read(500).decode('latin-1', errors='ignore')
        file.seek(0)
        if "<?xml" in head or "Workbook" in head:
            return leer_xml_a_la_fuerza(file)
    except: pass

    # 2. Calamine
    try:
        file.seek(0)
        return pd.read_excel(file, engine='calamine', header=None)
    except: pass

    # 3. HTML
    try:
        file.seek(0)
        dfs = pd.read_html(file, header=None)
        if len(dfs) > 0: return dfs[0]
    except: pass

    # 4. Texto
    try:
        file.seek(0)
        return pd.read_csv(file, sep='\t', encoding='latin-1', header=None)
    except: return None

# --- CAZADOR DE CABECERAS (VERSIÓN ESTRICTA) ---
def encontrar_cabecera_y_normalizar(df):
    if df is None: return None, None, None, None
    df = df.astype(str)
    
    k_fecha = ['date', 'time', 'fecha', 'hora', 'timestamp']
    k_estacion = ['station', 'operation', 'work', 'estacion', 'maquina', 'productid']
    k_usuario = ['user', 'operator', 'name', 'usuario', 'created by', 'computername']

    start_row = -1
    
    # Buscamos en las primeras 50 filas
    for i in range(min(50, len(df))):
        fila = df.iloc[i].str.lower().tolist()
        
        # VERIFICACIÓN ESTRICTA:
        # La fila debe tener (FECHA) Y ADEMÁS (ESTACION o USUARIO)
        has_date = any(k in str(v) for v in fila for k in k_fecha)
        has_station = any(k in str(v) for v in fila for k in k_estacion)
        has_user = any(k in str(v) for v in fila for k in k_usuario)
        
        # Solo aceptamos si tiene Fecha Y algo más (para evitar filas de instrucciones)
        if has_date and (has_station or has_user):
            start_row = i
            break
            
    if start_row == -1: return None, None, None, None

    # Cortar y asignar cabecera
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
uploaded_file = st.file_uploader("Sube el archivo XML/Excel", type=["xlsx", "xls", "txt", "xml"])

if uploaded_file:
    with st.spinner("⏳ Procesando..."):
        df_raw = load_data(uploaded_file)

        if df_raw is not None:
            # 1. NORMALIZAR
            df, col_f, col_s, col_u = encontrar_cabecera_y_normalizar(df_raw)

            if not col_f or not col_s:
                st.error("❌ No encontré la fila de cabecera correcta (con Fecha y Estación).")
                st.write("Primeras filas leídas:", df_raw.head(10))
                st.stop()
            
            st.success(f"✅ Mapeo Correcto: Fecha='{col_f}' | Estación='{col_s}' | Usuario='{col_u}'")

            # 2. PROCESAR
            try:
                df[col_f] = pd.to_datetime(df[col_f], errors='coerce')
                df.loc[df[col_f].dt.year < 100, col_f] += pd.offsets.DateOffset(years=2000)
                df = df.dropna(subset=[col_f]).sort_values(col_f)

                if df.empty:
                    st.error("No hay fechas válidas.")
                    st.stop()

                # Gaps
                df['gap_mins'] = df.groupby(col_s)[col_f].diff().dt.total_seconds() / 60
                
                # --- PROTECCIÓN ANTI-INFINITO ---
                # Si el gap es 0 (mismo segundo), lo ponemos a un mínimo seguro (ej. 1 segundo)
                df['gap_mins'] = df['gap_mins'].replace(0, 0.016) 
                df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())

                # IA
                model = IsolationForest(contamination=contamination, random_state=42)
                df['IA_Status'] = model.fit_predict(df[['gap_mins']])
                
                df_clean = df[df['IA_Status'] == 1].copy()
                df_noise = df[df['IA_Status'] == -1]

                # KPIs (CON PROTECCIÓN DE DIVISIÓN POR CERO)
                media = df_clean['gap_mins'].mean()
                
                if media > 0.01:
                    capacidad = ((h_turno*60 - m_descanso)/media) * eficiencia
                else:
                    capacidad = 0 # Evitamos error de infinito

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
            st.error("FATAL: No se pudo leer el archivo.")
