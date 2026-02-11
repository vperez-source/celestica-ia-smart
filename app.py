import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Celestica AI Tracker", layout="wide", page_icon="🛡️")

st.title("🛡️ Celestica IA: Advance Tracking Analyzer")
st.markdown("---")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    contamination = st.slider("Sensibilidad IA (% Ruido)", 1, 25, 5) / 100
    st.divider()
    h_turno = st.number_input("Horas Turno", value=8)
    m_descanso = st.number_input("Minutos Descanso", value=45)
    eficiencia = st.slider("Eficiencia %", 50, 100, 75) / 100

# --- FUNCIÓN DE CARGA INTELIGENTE CON FEEDBACK ---
def cargar_archivo_robusto(uploaded_file):
    """Intenta leer el archivo con múltiples motores y da feedback."""
    status = st.status("🔍 Analizando estructura del archivo...", expanded=True)
    df = None
    
    # INTENTO 1: Motor Calamine (El "Tanque")
    # Este es el único que puede saltarse el error 'styles.fills.Fill'
    try:
        status.write("🛠️ Intentando lectura con motor blindado (Calamine)...")
        df = pd.read_excel(uploaded_file, engine='calamine')
        status.update(label="✅ Archivo leído correctamente con Calamine!", state="complete", expanded=False)
        return df
    except Exception as e:
        status.write(f"⚠️ Calamine falló: {e}")
    
    # INTENTO 2: OpenPyXL (Estándar)
    # Probablemente fallará con tu archivo, pero hay que intentarlo
    try:
        status.write("🛠️ Intentando lectura estándar (OpenPyXL)...")
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        status.update(label="✅ Archivo leído con motor estándar.", state="complete", expanded=False)
        return df
    except Exception as e:
        status.write(f"⚠️ Estándar falló (Probable error de estilos): {e}")

    # INTENTO 3: HTML/XML
    try:
        status.write("🛠️ Buscando tablas HTML ocultas...")
        uploaded_file.seek(0)
        dfs = pd.read_html(uploaded_file)
        if len(dfs) > 0:
            status.update(label="✅ Tabla HTML detectada y extraída.", state="complete", expanded=False)
            return dfs[0]
    except:
        pass

    status.update(label="❌ Error Fatal: No se pudo leer el archivo.", state="error")
    return None

# --- INTERFAZ PRINCIPAL ---
uploaded_file = st.file_uploader("Arrastra el archivo 'Advance_tracking...'", type=["xlsx", "xls"])

if uploaded_file:
    # Llamamos a la función de carga
    df = cargar_archivo_robusto(uploaded_file)

    if df is not None:
        try:
            # 1. LIMPIEZA DE CABECERAS
            df.columns = df.columns.astype(str).str.strip()
            
            # Buscador de cabecera 'In DateTime'
            col_target = 'In DateTime'
            if col_target not in df.columns:
                # Buscar en las primeras filas
                found = False
                for i in range(min(10, len(df))):
                    row = df.iloc[i].astype(str).values
                    if any(col_target in s for s in row):
                        df.columns = df.iloc[i]
                        df = df[i+1:].reset_index(drop=True)
                        found = True
                        break
                if not found:
                    st.error(f"⚠️ No encuentro la columna '{col_target}'.")
                    st.stop()

            # 2. PROCESAMIENTO
            df[col_target] = pd.to_datetime(df[col_target], errors='coerce')
            df.loc[df[col_target].dt.year < 100, col_target] += pd.offsets.DateOffset(years=2000)
            df = df.dropna(subset=[col_target]).sort_values(col_target)

            if df.empty:
                st.error("El archivo no contiene fechas válidas.")
                st.stop()

            # 3. MACHINE LEARNING
            df['gap_mins'] = df.groupby('Station')[col_target].diff().dt.total_seconds() / 60
            df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())

            model = IsolationForest(contamination=contamination, random_state=42)
            df['IA_Status'] = model.fit_predict(df[['gap_mins']])
            
            df_clean = df[df['IA_Status'] == 1].copy()
            df_noise = df[df['IA_Status'] == -1]

            # 4. DASHBOARD
            media = df_clean['gap_mins'].mean()
            capacidad = ((h_turno*60 - m_descanso)/media) * eficiencia

            st.success(f"✅ Análisis completado: {len(df_clean)} registros válidos.")

            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Cycle Time IA", f"{media:.2f} min")
            k2.metric("Capacidad Turno", f"{int(capacidad)} uds")
            k3.metric("Datos Válidos", len(df_clean))
            k4.metric("Ruido Eliminado", len(df_noise), delta_color="inverse")

            st.markdown("---")

            # RANKING DE OPERARIOS (Si existe columna User)
            if 'User' in df.columns:
                st.subheader("🏆 Ranking de Operarios")
                user_stats = df_clean.groupby('User')['gap_mins'].agg(['count', 'mean', 'std']).reset_index()
                user_stats.columns = ['Operario', 'Piezas', 'Velocidad (min)', 'Estabilidad']
                user_stats = user_stats.sort_values('Piezas', ascending=False)

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.dataframe(
                        user_stats.style.background_gradient(subset=['Piezas'], cmap='Greens')
                                  .background_gradient(subset=['Velocidad (min)'], cmap='Reds'),
                        use_container_width=True
                    )
                with c2:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=user_stats['Operario'], y=user_stats['Piezas'], name='Piezas', marker_color='#2ecc71'))
                    fig.add_trace(go.Scatter(x=user_stats['Operario'], y=user_stats['Velocidad (min)'], name='Velocidad', yaxis='y2', line=dict(color='red')))
                    fig.update_layout(yaxis2=dict(overlaying='y', side='right'), title="Productividad vs Velocidad")
                    st.plotly_chart(fig, use_container_width=True)

            # GRÁFICO IA
            st.subheader("🔍 Mapa de Anomalías")
            fig_scatter = px.scatter(df, x=col_target, y='gap_mins', color=df['IA_Status'].astype(str),
                                   color_discrete_map={'1': '#2ecc71', '-1': '#e74c3c'},
                                   title="Verde = OK | Rojo = Anomalía detectada")
            st.plotly_chart(fig_scatter, use_container_width=True)

        except Exception as e:
            st.error(f"Error procesando datos: {e}")

    else:
        st.error("No se pudo cargar el archivo. Intenta guardarlo como .txt o .csv")
