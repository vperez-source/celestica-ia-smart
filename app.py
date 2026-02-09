import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import io

st.set_page_config(page_title="IA Celestica", layout="wide")

st.title("🛡️ Celestica IA: Smart-Trace Analyzer")
st.info("Sube tu archivo .xls o .xlsx generado por el sistema de trazabilidad.")

# Aceptamos ambos formatos
uploaded_file = st.file_uploader("Arrastra tu archivo aquí", type=["xlsx", "xls"])

if uploaded_file:
    try:
        # --- MOTOR DE LECTURA ROBUSTO ---
        # Leemos el contenido para detectar qué hay dentro realmente
        content = uploaded_file.read()
        
        try:
            # Intento 1: Excel Moderno (.xlsx)
            df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
        except:
            try:
                # Intento 2: Excel Antiguo Real (.xls)
                df = pd.read_excel(io.BytesIO(content), engine='xlrd')
            except:
                try:
                    # Intento 3: XML/HTML disfrazado de .xls (Muy común en reportes industriales)
                    df = pd.read_html(io.BytesIO(content))[0]
                except Exception as e:
                    st.error(f"No se pudo leer el formato interno del archivo. Error: {e}")
                    st.stop()

        # --- PROCESAMIENTO DE DATOS ---
        # Aseguramos que los nombres de columnas no tengan espacios raros
        df.columns = df.columns.astype(str).str.strip()

        # Limpieza de fechas blindada
        df['In DateTime'] = pd.to_datetime(df['In DateTime'], errors='coerce')
        mask_short_year = df['In DateTime'].dt.year < 100
        df.loc[mask_short_year, 'In DateTime'] += pd.offsets.DateOffset(years=2000)
        df = df.dropna(subset=['In DateTime']).sort_values('In DateTime')

        # Cálculo de Cycle Time (Gaps)
        df['gap_mins'] = df.groupby('Station')['In DateTime'].diff().dt.total_seconds() / 60
        df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())

        # MACHINE LEARNING: Isolation Forest
        model = IsolationForest(contamination=0.05, random_state=42)
        df['IA_Status'] = model.fit_predict(df[['gap_mins']])
        
        df_normal = df[df['IA_Status'] == 1]
        q1, q3 = df_normal['gap_mins'].quantile([0.25, 0.75])
        df_clean = df_normal[(df_normal['gap_mins'] >= q1) & (df_normal['gap_mins'] <= q3)]
        media_ct = df_clean['gap_mins'].mean()

        # DASHBOARD
        c1, c2, c3 = st.columns(3)
        c1.metric("Cycle Time Real", f"{media_ct:.2f} min")
        c2.metric("Salud del Dato (IA)", f"{(len(df_clean)/len(df)*100):.1f}%")
        c3.metric("Capacidad Real (8h)", f"{int((415/media_ct)*0.75)} uds")

        st.subheader("📊 Mapa de Producción (Verde: OK | Rojo: Ruido)")
        fig = px.scatter(df, x='In DateTime', y='gap_mins', color=df['IA_Status'].astype(str),
                         color_discrete_map={'1': '#2ecc71', '-1': '#e74c3c'})
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("📥 Descargar Reporte Limpio", df_clean.to_csv(index=False).encode('utf-8'), "reporte_ia.csv")
        
    except Exception as e:
        st.error(f"Error crítico en el procesamiento: {e}")
