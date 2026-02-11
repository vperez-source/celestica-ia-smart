import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica AI Auto", layout="wide", page_icon="🛡️")
st.title("🛡️ Celestica IA: Advance Tracking Analyzer")

with st.sidebar:
    st.header("⚙️ Configuración")
    contamination = st.slider("Sensibilidad IA (% Ruido)", 1, 25, 5) / 100
    st.divider()
    h_turno = st.number_input("Horas Turno", value=8)
    m_descanso = st.number_input("Minutos Descanso", value=45)
    eficiencia = st.slider("Eficiencia %", 50, 100, 75) / 100

# --- FUNCIÓN DE LECTURA TODOTERRENO ---
@st.cache_data(ttl=3600)
def load_data(file):
    """Prueba todos los motores posibles para abrir el archivo."""
    
    # 1. INTENTO: Motor Calamine (El mejor para Excel corrupto)
    try:
        return pd.read_excel(file, engine='calamine')
    except:
        pass

    # 2. INTENTO: HTML (Para archivos .xls descargados de webs)
    try:
        file.seek(0)
        # Busca tablas dentro del código web del archivo
        dfs = pd.read_html(file)
        if len(dfs) > 0:
            return dfs[0]
    except:
        pass

    # 3. INTENTO: Excel Antiguo (xlrd)
    try:
        file.seek(0)
        return pd.read_excel(file, engine='xlrd')
    except:
        pass

    # 4. INTENTO: Texto/CSV (Separado por tabulaciones)
    try:
        file.seek(0)
        return pd.read_csv(file, sep='\t', encoding='latin-1')
    except:
        pass
        
    return None

# --- CEREBRO DE AUTO-MAPEO ---
def normalizar_columnas(df):
    """Detecta automáticamente Fecha, Estación y Usuario."""
    df.columns = df.columns.astype(str).str.strip()
    
    mapa = {
        'FECHA': ['In DateTime', 'Date', 'Time', 'Fecha', 'Hora', 'Timestamp'],
        'ESTACION': ['Station', 'Operation', 'Work Center', 'Estacion', 'Maquina', 'Process'],
        'USUARIO': ['User', 'Operator', 'Name', 'Usuario', 'Empleado', 'Worker']
    }
    
    col_fecha, col_estacion, col_usuario = None, None, None

    for posible in mapa['FECHA']:
        match = next((c for c in df.columns if posible.lower() in c.lower()), None)
        if match: col_fecha = match; break
            
    for posible in mapa['ESTACION']:
        match = next((c for c in df.columns if posible.lower() in c.lower()), None)
        if match: col_estacion = match; break

    for posible in mapa['USUARIO']:
        match = next((c for c in df.columns if posible.lower() in c.lower()), None)
        if match: col_usuario = match; break
            
    return df, col_fecha, col_estacion, col_usuario

# --- INTERFAZ PRINCIPAL ---
uploaded_file = st.file_uploader("Sube tu archivo (Excel, XLS Web, Texto)", type=["xlsx", "xls", "txt"])

if uploaded_file:
    with st.spinner("⏳ Analizando estructura del archivo..."):
        df_raw = load_data(uploaded_file)

        if df_raw is not None:
            # 1. AUTO-DETECTAR COLUMNAS
            df, col_f, col_s, col_u = normalizar_columnas(df_raw)

            if not col_f or not col_s:
                st.error("❌ Archivo leído, pero no encontré las columnas de Fecha o Estación.")
                st.write("Columnas que veo:", list(df.columns))
                st.write("Primeras filas para depurar:", df.head())
                st.stop()
                
            st.caption(f"✅ Lectura OK | Mapeo: Fecha='{col_f}' | Estación='{col_s}' | Usuario='{col_u}'")

            # 2. PROCESAMIENTO
            try:
                df[col_f] = pd.to_datetime(df[col_f], errors='coerce')
                df.loc[df[col_f].dt.year < 100, col_f] += pd.offsets.DateOffset(years=2000)
                df = df.dropna(subset=[col_f]).sort_values(col_f)

                if df.empty:
                    st.error("⚠️ El archivo no tiene fechas válidas.")
                    st.stop()

                # Cálculo de Gaps
                df['gap_mins'] = df.groupby(col_s)[col_f].diff().dt.total_seconds() / 60
                df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())

                # 3. IA (ISOLATION FOREST)
                model = IsolationForest(contamination=contamination, random_state=42)
                df['IA_Status'] = model.fit_predict(df[['gap_mins']])
                
                df_clean = df[df['IA_Status'] == 1].copy()
                df_noise = df[df['IA_Status'] == -1]

                # 4. RESULTADOS (KPIs)
                media = df_clean['gap_mins'].mean()
                capacidad = ((h_turno*60 - m_descanso)/media) * eficiencia

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("⏱️ Cycle Time Global", f"{media:.2f} min")
                k2.metric("📦 Capacidad Turno", f"{int(capacidad)} uds")
                k3.metric("✅ Piezas OK", len(df_clean))
                k4.metric("🗑️ Ruido", len(df_noise), delta_color="inverse")

                st.markdown("---")

                # 5. RANKING AUTOMÁTICO + GRÁFICA COMBINADA
                if col_u:
                    st.subheader("🏆 Ranking: Productividad vs Velocidad")
                    user_stats = df_clean.groupby(col_u)['gap_mins'].agg(['count', 'mean', 'std']).reset_index()
                    user_stats.columns = ['Operario', 'Piezas', 'Velocidad (min)', 'Estabilidad']
                    user_stats = user_stats.sort_values('Piezas', ascending=False)

                    c1, c2 = st.columns([1, 1])
                    with c1:
                        # Tabla
                        st.dataframe(
                            user_stats.style.background_gradient(subset=['Piezas'], cmap='Greens')
                                          .background_gradient(subset=['Velocidad (min)'], cmap='Reds'),
                            use_container_width=True
                        )
                        st.caption("*Estabilidad baja = Ritmo constante.")

                    with c2:
                        # GRÁFICA COMBINADA (Corregida)
                        fig_combo = go.Figure()

                        # Barras (Eje Izquierdo)
                        fig_combo.add_trace(go.Bar(
                            x=user_stats['Operario'],
                            y=user_stats['Piezas'],
                            name='Piezas Realizadas',
                            marker_color='#2ecc71',
                            yaxis='y'
                        ))

                        # Línea (Eje Derecho)
                        fig_combo.add_trace(go.Scatter(
                            x=user_stats['Operario'],
                            y=user_stats['Velocidad (min)'],
                            name='Tiempo Ciclo Medio',
                            marker_color='#e74c3c',
                            yaxis='y2',
                            mode='lines+markers',
                            line=dict(width=3)
                        ))

                        fig_combo.update_layout(
                            title="Volumen (Barras) vs Velocidad (Línea Roja)",
                            hovermode="x unified",
                            yaxis=dict(
                                title=dict(text="Cantidad de Piezas", font=dict(color="#2ecc71")),
                                tickfont=dict(color="#2ecc71")
                            ),
                            yaxis2=dict(
                                title=dict(text="Minutos por Pieza", font=dict(color="#e74c3c")),
                                tickfont=dict(color="#e74c3c"),
                                overlaying='y',
                                side='right'
                            ),
                            legend=dict(x=0.01, y=1.1, orientation='h')
                        )
                        st.plotly_chart(fig_combo, use_container_width=True)

                # 6. GRÁFICO FINAL
                st.subheader("🔍 Mapa de Anomalías (IA)")
                fig_scatter = px.scatter(df, x=col_f, y='gap_mins', color=df['IA_Status'].astype(str),
                                       color_discrete_map={'1': '#2ecc71', '-1': '#e74c3c'},
                                       title="Detección de Ráfagas y Paros (Rojo)")
                st.plotly_chart(fig_scatter, use_container_width=True)

            except Exception as e:
                st.error(f"Error en el cálculo: {e}")
        else:
            st.error("Error grave: No se pudo leer el formato del archivo. Prueba a abrirlo en Excel y guardarlo como CSV o Libro de Excel (.xlsx).")
