import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import xml.etree.ElementTree as ET # Importante para leer tu archivo

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

# --- FUNCIÓN ESPECIAL: LECTOR XML 2003 (La llave maestra) ---
def leer_xml_2003(file):
    """Lee archivos 'XML Spreadsheet 2003' que fallan con otros métodos."""
    try:
        tree = ET.parse(file)
        root = tree.getroot()
        # Definimos el 'namespace' que usa Microsoft
        ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        
        datos = []
        # Buscamos todas las filas
        for row in root.findall('.//ss:Row', ns):
            fila_datos = []
            # Buscamos todas las celdas de esa fila
            for cell in row.findall('.//ss:Cell', ns):
                data_tag = cell.find('ss:Data', ns)
                if data_tag is not None:
                    fila_datos.append(data_tag.text)
                else:
                    fila_datos.append("") # Celda vacía
            if fila_datos:
                datos.append(fila_datos)
        
        return pd.DataFrame(datos)
    except Exception as e:
        return None

# --- GESTOR DE CARGA ---
@st.cache_data(ttl=3600)
def load_data(file):
    # 1. INTENTO: XML 2003 (Tu caso específico)
    file.seek(0)
    # Leemos la cabecera para ver si es XML
    try:
        head = file.read(100).decode('latin-1', errors='ignore')
        file.seek(0)
        if "<?xml" in head and "Workbook" in head:
            return leer_xml_2003(file)
    except:
        pass

    # 2. INTENTO: Calamine (Excel corrupto normal)
    try:
        return pd.read_excel(file, engine='calamine', header=None)
    except:
        pass

    # 3. INTENTO: HTML (Descargas web)
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

# --- CAZADOR DE CABECERAS ---
def encontrar_cabecera_y_normalizar(df):
    if df is None: return None, None, None, None
    df = df.astype(str)
    
    keywords_fecha = ['date', 'time', 'fecha', 'hora', 'timestamp']
    keywords_estacion = ['station', 'operation', 'work', 'estacion', 'maquina', 'productid'] # Añadido ProductID por tu snippet
    
    start_row = -1
    
    # Buscamos la fila de cabecera
    for i in range(min(20, len(df))):
        fila = df.iloc[i].str.lower().tolist()
        # Buscamos coincidencias flexibles
        has_date = any(k in str(val) for val in fila for k in keywords_fecha)
        # A veces la estación no está clara, pero si hay fecha y usuario/producto, nos vale
        has_id = any(k in str(val) for val in fila for k in keywords_estacion)
        
        if has_date: # Si encontramos fecha, asumimos que es la buena
            start_row = i
            break
    
    if start_row == -1: return None, None, None, None

    # Cortar y asignar cabecera
    df.columns = df.iloc[start_row]
    df = df[start_row + 1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()
    
    # Mapeo
    mapa_f = keywords_fecha + ['in datetime']
    mapa_s = keywords_estacion
    mapa_u = ['user', 'operator', 'name', 'usuario', 'created by']

    col_f, col_s, col_u = None, None, None

    for col in df.columns:
        c_low = col.lower()
        if not col_f and any(k in c_low for k in mapa_f): col_f = col
        if not col_s and any(k in c_low for k in mapa_s): col_s = col
        if not col_u and any(k in c_low for k in mapa_u): col_u = col

    return df, col_f, col_s, col_u

# --- APP ---
uploaded_file = st.file_uploader("Sube el archivo XML/Excel", type=["xlsx", "xls", "txt", "xml"])

if uploaded_file:
    with st.spinner("⏳ Decodificando XML Spreadsheet 2003..."):
        df_raw = load_data(uploaded_file)

        if df_raw is not None:
            # 1. NORMALIZAR
            df, col_f, col_s, col_u = encontrar_cabecera_y_normalizar(df_raw)

            if not col_f:
                st.error("❌ Leí el XML, pero no encontré la columna de FECHA.")
                st.write("Columnas detectadas:", list(df.columns) if df is not None else "Ninguna")
                st.write("Datos crudos:", df_raw.head())
                st.stop()
            
            # Si no encuentra estación, usamos ProductID o lo que haya
            if not col_s:
                st.warning("⚠️ No encontré columna 'Station'. Usando la primera columna como agrupación.")
                col_s = df.columns[0]

            st.success(f"✅ Mapeo: Fecha='{col_f}' | Estación='{col_s}'")

            # 2. PROCESAR
            try:
                # Limpieza de fechas robusta
                df[col_f] = pd.to_datetime(df[col_f], errors='coerce')
                df.loc[df[col_f].dt.year < 100, col_f] += pd.offsets.DateOffset(years=2000)
                df = df.dropna(subset=[col_f]).sort_values(col_f)

                if df.empty:
                    st.error("No quedan fechas válidas tras la limpieza.")
                    st.stop()

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
                        fig.update_layout(yaxis2=dict(overlaying='y', side='right'), title="Volumen vs Velocidad")
                        st.plotly_chart(fig, use_container_width=True)

                st.subheader("Mapa IA")
                fig = px.scatter(df, x=col_f, y='gap_mins', color=df['IA_Status'].astype(str),
                               color_discrete_map={'1': '#2ecc71', '-1': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error cálculo: {e}")

        else:
            st.error("Error fatal: No se pudo decodificar el archivo.")
