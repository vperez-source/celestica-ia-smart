import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import io

st.set_page_config(page_title="Celestica IA Analyzer", layout="wide")
st.title("🛡️ Celestica IA: Smart-Trace Analyzer")

# Barra lateral
st.sidebar.header("⚙️ Configuración")
h_turno = st.sidebar.number_input("Horas de Turno", value=8)
m_descanso = st.sidebar.number_input("Minutos de Descanso", value=45)
eficiencia = st.sidebar.slider("Eficiencia %", 50, 100, 75) / 100

uploaded_file = st.file_uploader("Sube tu archivo (.xls o .xlsx)", type=["xlsx", "xls"])

if uploaded_file:
    try:
        content = uploaded_file.read()
        df = None

        # Intentos de lectura
        try: df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
        except:
            try: df = pd.read_excel(io.BytesIO(content), engine='xlrd')
            except:
                try:
                    tablas = pd.read_html(io.BytesIO(content))
                    if len(tablas) > 0: df = tablas[0]
                except:
                    try: df = pd.read_xml(io.BytesIO(content))
                    except: pass

        if df is not None:
            # LIMPIEZA INICIAL: Quitar filas y columnas vacías
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            df.columns = df.columns.astype(str).str.strip()

            # --- MODO DIAGNÓSTICO: Si no encuentra la columna, muestra qué encontró ---
            col_fecha = 'In DateTime'
            if col_fecha not in df.columns:
                # Si la cabecera está en la primera fila real de datos
                df.columns = df.iloc[0].astype(str).str.strip()
                df = df[1:].reset_index(drop=True)

            if col_fecha not in df.columns:
                st.error(f"❌ No encuentro la columna '{col_fecha}'")
                st.write("Columnas detectadas actualmente:", list(df.columns))
                st.write("Previsualización de tus datos (Primeras 5 filas):")
                st.dataframe(df.head()) # Esto nos dirá qué está leyendo la IA
                st.stop()

            # --- PROCESAMIENTO ---
            df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
            df.loc[df[col_fecha].dt.year < 100, col_fecha] += pd.offsets.DateOffset(years=2000)
            df = df.dropna(subset=[col_fecha]).sort_values(col_fecha)

            if len(df) == 0:
                st.warning("⚠️ Se han convertido las fechas pero no quedan registros válidos. Revisa el formato de fecha.")
                st.stop()

            # IA y Cálculos
            df['gap_mins'] = df.groupby('Station')[col_fecha].diff().dt.total_seconds() / 60
            df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())

            model = IsolationForest(contamination=0.05, random_state=42)
            df['IA_Status'] = model.fit_predict(df[['gap_mins']])
            
            df_clean = df[df['IA_Status'] == 1].copy()
            q1, q3 = df_clean['gap_mins'].quantile([0.25, 0.75])
            df_clean = df_clean[(df_clean['gap_mins'] >= q1) & (df_clean['gap_mins'] <= q3)]
            
            if not df_clean.empty:
                media_ct = df_clean['gap_mins'].mean()
                capacidad = ((h_turno * 60 - m_descanso) / media_ct) * eficiencia

                # Mostrar KPIs
                c1, c2, c3 = st.columns(3)
                c1.metric("Cycle Time IA", f"{media_ct:.2f} min")
                c2.metric("Capacidad Real", f"{int(capacidad)} uds")
                c3.metric("Salud del Dato", f"{(len(df_clean)/len(df)*100):.1f}%")

                st.plotly_chart(px.scatter(df, x=col_fecha, y='gap_mins', color=df['IA_Status'].astype(str), color_discrete_map={'1': '#2ecc71', '-1': '#e74c3c'}), use_container_width=True)
            else:
                st.error("La IA ha filtrado todos los datos. El archivo podría tener demasiada variabilidad.")

    except Exception as e:
        st.error(f"Error inesperado: {e}")
