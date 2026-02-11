import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Celestica AI Tracker", layout="wide", page_icon="🛡️")
st.title("🛡️ Celestica IA: Advance Tracking Analyzer")

# --- CONFIGURACIÓN ---
with st.sidebar:
    st.header("⚙️ Configuración")
    contamination = st.slider("Sensibilidad IA (% Ruido)", 1, 25, 5) / 100
    st.divider()
    h_turno = st.number_input("Horas Turno", value=8)
    m_descanso = st.number_input("Minutos Descanso", value=45)
    eficiencia = st.slider("Eficiencia %", 50, 100, 75) / 100

def cargar_archivo_robusto(uploaded_file):
    status = st.status("🔍 Leyendo archivo...", expanded=True)
    df = None
    try:
        status.write("🛠️ Motor Calamine (Blindado)...")
        df = pd.read_excel(uploaded_file, engine='calamine')
        status.update(label="✅ Archivo leído con éxito!", state="complete", expanded=False)
        return df
    except Exception as e:
        status.write(f"⚠️ Calamine falló: {e}")
    
    try:
        status.write("🛠️ Motor OpenPyXL...")
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        status.update(label="✅ Leído con OpenPyXL", state="complete", expanded=False)
        return df
    except:
        pass

    try:
        status.write("🛠️ Buscando tablas HTML...")
        uploaded_file.seek(0)
        dfs = pd.read_html(uploaded_file)
        if len(dfs) > 0:
            status.update(label="✅ Tabla HTML encontrada", state="complete", expanded=False)
            return dfs[0]
    except:
        pass

    status.update(label="❌ Error Fatal: No se pudo leer.", state="error")
    return None

uploaded_file = st.file_uploader("Arrastra tu archivo Excel", type=["xlsx", "xls"])

if uploaded_file:
    df = cargar_archivo_robusto(uploaded_file)

    if df is not None:
        # 1. Limpieza de nombres de columnas
        df.columns = df.columns.astype(str).str.strip()
        
        # --- SECCIÓN DE MAPEO DE COLUMNAS (AQUÍ ESTÁ LA SOLUCIÓN) ---
        st.info("👇 Confirma que las columnas seleccionadas sean las correctas:")
        col1, col2, col3 = st.columns(3)
        
        # A. Detectar Fecha (In DateTime)
        candidatos_fecha = [c for c in df.columns if "Date" in c or "Time" in c or "Fecha" in c]
        default_fecha = candidatos_fecha[0] if candidatos_fecha else df.columns[0]
        col_target = col1.selectbox("Columna de FECHA (Hora entrada):", df.columns, index=list(df.columns).index(default_fecha) if default_fecha in df.columns else 0)

        # B. Detectar Estación (Station)
        candidatos_station = [c for c in df.columns if "Station" in c or "Operation" in c or "Estacion" in c or "Work" in c]
        default_station = candidatos_station[0] if candidatos_station else df.columns[0]
        col_station = col2.selectbox("Columna de ESTACIÓN/MÁQUINA:", df.columns, index=list(df.columns).index(default_station) if default_station in df.columns else 0)

        # C. Detectar Usuario (User)
        candidatos_user = [c for c in df.columns if "User" in c or "Operator" in c or "Name" in c or "Usuario" in c]
        index_user = list(df.columns).index(candidatos_user[0]) if candidatos_user and candidatos_user[0] in df.columns else 0
        col_user = col3.selectbox("Columna de OPERARIO (Opcional):", df.columns, index=index_user)

        # Botón para confirmar y procesar
        if st.button("🚀 PROCESAR CON ESTAS COLUMNAS"):
            try:
                # --- PROCESAMIENTO ---
                df[col_target] = pd.to_datetime(df[col_target], errors='coerce')
                df.loc[df[col_target].dt.year < 100, col_target] += pd.offsets.DateOffset(years=2000)
                df = df.dropna(subset=[col_target]).sort_values(col_target)

                # Usamos la columna seleccionada por ti (col_station)
                df['gap_mins'] = df.groupby(col_station)[col_target].diff().dt.total_seconds() / 60
                df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())

                # IA
                model = IsolationForest(contamination=contamination, random_state=42)
                df['IA_Status'] = model.fit_predict(df[['gap_mins']])
                
                df_clean = df[df['IA_Status'] == 1].copy()
                df_noise = df[df['IA_Status'] == -1]

                # DASHBOARD
                media = df_clean['gap_mins'].mean()
                capacidad = ((h_turno*60 - m_descanso)/media) * eficiencia

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Cycle Time IA", f"{media:.2f} min")
                k2.metric("Capacidad Turno", f"{int(capacidad)} uds")
                k3.metric("Datos Válidos", len(df_clean))
                k4.metric("Ruido Eliminado", len(df_noise), delta_color="inverse")

                st.markdown("---")

                # RANKING OPERARIOS (Usando la columna seleccionada col_user)
                if col_user in df.columns:
                    st.subheader(f"🏆 Ranking por {col_user}")
                    user_stats = df_clean.groupby(col_user)['gap_mins'].agg(['count', 'mean', 'std']).reset_index()
                    user_stats.columns = ['Operario', 'Piezas', 'Velocidad (min)', 'Estabilidad']
                    user_stats = user_stats.sort_values('Piezas', ascending=False)

                    c_table, c_chart = st.columns([1, 1])
                    with c_table:
                        st.dataframe(user_stats.style.background_gradient(subset=['Piezas'], cmap='Greens'), use_container_width=True)
                    with c_chart:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=user_stats['Operario'], y=user_stats['Piezas'], marker_color='#2ecc71'))
                        st.plotly_chart(fig, use_container_width=True)

                # GRÁFICO FINAL
                st.subheader("🔍 Mapa de Anomalías")
                fig_scatter = px.scatter(df, x=col_target, y='gap_mins', color=df['IA_Status'].astype(str),
                                       color_discrete_map={'1': '#2ecc71', '-1': '#e74c3c'})
                st.plotly_chart(fig_scatter, use_container_width=True)

            except Exception as e:
                st.error(f"Error durante el cálculo: {e}")
