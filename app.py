import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import io

# Configuración de la página
st.set_page_config(page_title="Celestica IA Smart-Trace", layout="wide")

st.title("🛡️ Celestica IA: Smart-Trace Analyzer")
st.markdown("---")

# Barra lateral para parámetros de turno
st.sidebar.header("⚙️ Configuración de Turno")
h_turno = st.sidebar.number_input("Horas de Turno", value=8)
m_descanso = st.sidebar.number_input("Minutos de Descanso", value=45)
eficiencia = st.sidebar.slider("Eficiencia Objetivo %", 50, 100, 75) / 100

uploaded_file = st.file_uploader("Arrastra tu reporte de trazabilidad (.xls o .xlsx)", type=["xlsx", "xls"])

if uploaded_file:
    try:
        content = uploaded_file.read()
        df = None

        # --- MOTOR DE LECTURA MULTI-FORMATO ---
        try:
            # Intento 1: Excel Moderno
            df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
        except:
            try:
                # Intento 2: Excel Antiguo real
                df = pd.read_excel(io.BytesIO(content), engine='xlrd')
            except:
                try:
                    # Intento 3: HTML/XML disfrazado (Caso común en Celestica)
                    tablas = pd.read_html(io.BytesIO(content))
                    if len(tablas) > 0:
                        df = tablas[0]
                        # Si los nombres de columnas están en la primera fila de datos
                        if 'In DateTime' not in df.columns:
                            df.columns = df.iloc[0]
                            df = df[1:].reset_index(drop=True)
                except:
                    try:
                        # Intento 4: XML Directo
                        df = pd.read_xml(io.BytesIO(content))
                    except:
                        st.error("❌ Formato no reconocido. Prueba a abrir el archivo en Excel y guardarlo como .xlsx")
                        st.stop()

        if df is not None:
            # Limpieza de nombres de columnas
            df.columns = df.columns.astype(str).str.strip()
            
            # Validar columna crítica
            col_fecha = 'In DateTime'
            if col_fecha not in df.columns:
                st.error(f"No se encontró la columna '{col_fecha}'. Columnas detectadas: {list(df.columns)}")
                st.stop()

            # --- PROCESAMIENTO E IA ---
            # Conversión de fechas blindada
            df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
            df.loc[df[col_fecha].dt.year < 100, col_fecha] += pd.offsets.DateOffset(years=2000)
            df = df.dropna(subset=[col_fecha]).sort_values(col_fecha)

            # Cálculo de Cycle Time (Gap entre piezas)
            df['gap_mins'] = df.groupby('Station')[col_fecha].diff().dt.total_seconds() / 60
            df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())

            # Machine Learning: Detección de Anomalías (Isolation Forest)
            model = IsolationForest(contamination=0.05, random_state=42)
            df['IA_Status'] = model.fit_predict(df[['gap_mins']])
            
            # Filtro de Calidad (Datos normales + Limpieza de Cuartiles)
            df_normal = df[df['IA_Status'] == 1].copy()
            q1, q3 = df_normal['gap_mins'].quantile([0.25, 0.75])
            df_clean = df_normal[(df_normal['gap_mins'] >= q1) & (df_normal['gap_mins'] <= q3)]
            
            media_ct = df_clean['gap_mins'].mean()
            min_netos = (h_turno * 60) - m_descanso
            capacidad = (min_netos / media_ct) * eficiencia

            # --- DASHBOARD VISUAL ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cycle Time IA", f"{media_ct:.2f} min")
            c2.metric("Capacidad Real", f"{int(capacidad)} uds")
            c3.metric("Salud del Dato", f"{(len(df_clean)/len(df)*100):.1f}%")
            c4.metric("Registros IA", len(df_clean))

            # Gráfico de Dispersión
            st.subheader("📈 Análisis de Flujo (Verde = Válido | Rojo = Anomalía/Ruido)")
            fig_scatter = px.scatter(df, x=col_fecha, y='gap_mins', 
                                   color=df['IA_Status'].astype(str),
                                   color_discrete_map={'1': '#2ecc71', '-1': '#e74c3c'},
                                   title="Detección de ráfagas y paros mediante ML")
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Análisis por Operador
            if 'User' in df.columns:
                st.subheader("👤 Rendimiento por Operador (Solo datos limpios)")
                user_stats = df_clean.groupby('User')['gap_mins'].agg(['mean', 'count']).reset_index()
                user_stats.columns = ['Usuario', 'Cycle Time Medio (min)', 'Piezas Procesadas']
                
                fig_user = px.bar(user_stats, x='Usuario', y='Cycle Time Medio (min)', 
                                 text='Piezas Procesadas', color='Cycle Time Medio (min)',
                                 color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig_user, use_container_width=True)

            # Exportación
            st.download_button("📥 Descargar Reporte Limpio (CSV)", 
                             df_clean.to_csv(index=False).encode('utf-8'), 
                             "reporte_ia_celestica.csv", "text/csv")

    except Exception as e:
        st.error(f"Error inesperado: {e}")
