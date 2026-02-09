import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import io

st.set_page_config(page_title="Celestica IA", layout="wide")

st.title("🛡️ Celestica IA: Smart-Trace Analyzer")
st.markdown("---")

# 1. BARRA LATERAL
with st.sidebar:
    st.header("⚙️ Configuración")
    h_turno = st.number_input("Horas Turno", value=8)
    m_descanso = st.number_input("Minutos Descanso", value=45)
    eficiencia = st.slider("Eficiencia %", 50, 100, 75) / 100

# 2. CARGA DE ARCHIVO
uploaded_file = st.file_uploader("1️⃣ Sube tu archivo (.xls / .xlsx)", type=["xlsx", "xls"])

# 3. LÓGICA CON BOTÓN
if uploaded_file:
    st.info(f"📂 Archivo cargado: {uploaded_file.name} ({uploaded_file.size} bytes)")
    
    if st.button("🚀 2️⃣ PULSA AQUÍ PARA CALCULAR"):
        with st.spinner('⏳ La IA está analizando el archivo... por favor espera...'):
            try:
                content = uploaded_file.read()
                df = None
                status_text = st.empty() # Espacio para mensajes de estado

                # --- INTENTO 1: Excel Moderno ---
                status_text.text("🔍 Intentando leer como Excel moderno (.xlsx)...")
                try:
                    df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
                    st.success("✅ Leído con motor OpenPyXL")
                except:
                    # --- INTENTO 2: Excel Antiguo ---
                    status_text.text("🔍 Falló moderno. Intentando Excel antiguo (.xls)...")
                    try:
                        df = pd.read_excel(io.BytesIO(content), engine='xlrd')
                        st.success("✅ Leído con motor XLRD")
                    except:
                        # --- INTENTO 3: HTML/XML ---
                        status_text.text("🔍 Falló antiguo. Buscando tablas HTML/XML ocultas...")
                        try:
                            # Importante: header=0 busca titulos en la primera fila, header=None coge todo
                            tablas = pd.read_html(io.BytesIO(content), header=0)
                            if len(tablas) > 0:
                                df = tablas[0]
                                st.success(f"✅ Tabla HTML encontrada con {len(df)} filas.")
                        except Exception as e:
                            st.error(f"❌ Fallaron todos los métodos de lectura. Error técnico: {e}")

                # --- VALIDACIÓN DE DATOS ---
                if df is not None:
                    # Limpieza preliminar
                    df.columns = df.columns.astype(str).str.strip()
                    status_text.text("🧹 Limpiando nombres de columnas...")
                    
                    # Búsqueda de columna 'In DateTime'
                    col_target = 'In DateTime'
                    
                    # Si no existe, miramos si está en la fila 1 (común en reportes sucios)
                    if col_target not in df.columns:
                        st.warning("⚠️ Cabecera no detectada en fila 0. Buscando en fila 1...")
                        df.columns = df.iloc[0].astype(str).str.strip()
                        df = df[1:].reset_index(drop=True)

                    # Si sigue sin existir, mostramos qué columnas ve la IA y PARAMOS
                    if col_target not in df.columns:
                        st.error(f"⛔ ERROR CRÍTICO: No encuentro la columna '{col_target}'.")
                        st.write("👀 Esto es lo que la IA está leyendo (primeras 5 filas):")
                        st.dataframe(df.head())
                        st.write("📋 Nombres de columnas detectados:", list(df.columns))
                        st.stop()

                    # --- PROCESAMIENTO ---
                    status_text.text("🧠 Ejecutando Machine Learning (Isolation Forest)...")
                    
                    # Conversión Fechas
                    df[col_target] = pd.to_datetime(df[col_target], errors='coerce')
                    # Fix año 2025 (bug del año 1900)
                    df.loc[df[col_target].dt.year < 100, col_target] += pd.offsets.DateOffset(years=2000)
                    df = df.dropna(subset=[col_target]).sort_values(col_target)

                    if len(df) == 0:
                        st.error("❌ El archivo tiene datos, pero ninguna fecha válida.")
                        st.stop()

                    # Cálculo Gaps
                    df['gap_mins'] = df.groupby('Station')[col_target].diff().dt.total_seconds() / 60
                    df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())

                    # IA
                    model = IsolationForest(contamination=0.05, random_state=42)
                    df['IA_Status'] = model.fit_predict(df[['gap_mins']])

                    # Métricas
                    df_clean = df[df['IA_Status'] == 1]
                    q1, q3 = df_clean['gap_mins'].quantile([0.25, 0.75])
                    df_final = df_clean[(df_clean['gap_mins'] >= q1) & (df_clean['gap_mins'] <= q3)]
                    
                    media = df_final['gap_mins'].mean()
                    capacidad = ((h_turno*60 - m_descanso)/media) * eficiencia

                    # --- RESULTADOS ---
                    status_text.empty() # Borrar mensajes de carga
                    st.balloons()
                    
                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric("⏱️ Cycle Time Real", f"{media:.2f} min")
                    kpi2.metric("📦 Capacidad Turno", f"{int(capacidad)} uds")
                    kpi3.metric("📉 Salud de Datos", f"{(len(df_final)/len(df)*100):.1f}%")

                    st.subheader("Gráfico de Dispersión (IA)")
                    fig = px.scatter(df, x=col_target, y='gap_mins', color=df['IA_Status'].astype(str),
                                     color_discrete_map={'1':'#00cc96', '-1':'#ef553b'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.download_button("📥 Descargar Reporte CSV", df_final.to_csv().encode('utf-8'), "celestica_ia.csv")

                else:
                    st.error("El DataFrame sigue vacío después de intentar leerlo.")

            except Exception as e:
                st.error(f"💥 Error inesperado durante el cálculo: {e}")
