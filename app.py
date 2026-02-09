import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="IA Celestica", layout="wide")

st.title("🛡️ Celestica IA: Smart-Trace Analyzer")
st.markdown("---")

uploaded_file = st.file_uploader("Arrastra tu Excel (formato .xls o .xlsx)", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, engine='xlrd' if uploaded_file.name.endswith('.xls') else 'openpyxl')
        
        # Limpieza de fechas (Tratamiento para el error 2025)
        df['In DateTime'] = pd.to_datetime(df['In DateTime'], errors='coerce')
        df.loc[df['In DateTime'].dt.year < 100, 'In DateTime'] += pd.offsets.DateOffset(years=2000)
        df = df.dropna(subset=['In DateTime']).sort_values('In DateTime')

        # Cálculo de Cycle Time (Gaps)
        df['gap_mins'] = df.groupby('Station')['In DateTime'].diff().dt.total_seconds() / 60
        df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())

        # MACHINE LEARNING: Isolation Forest (Detección de Anomalías)
        # El modelo aprende qué ritmos son normales y cuáles son "ruido"
        model = IsolationForest(contamination=0.05, random_state=42)
        df['IA_Status'] = model.fit_predict(df[['gap_mins']])
        
        # Filtro de Calidad (Q1-Q3 de los datos normales)
        df_normal = df[df['IA_Status'] == 1]
        q1, q3 = df_normal['gap_mins'].quantile([0.25, 0.75])
        df_clean = df_normal[(df_normal['gap_mins'] >= q1) & (df_normal['gap_mins'] <= q3)]
        media_ct = df_clean['gap_mins'].mean()

        # KPIs Visuales
        c1, c2, c3 = st.columns(3)
        c1.metric("Cycle Time Real", f"{media_ct:.2f} min")
        c2.metric("Salud del Dato (IA)", f"{(len(df_clean)/len(df)*100):.1f}%")
        c3.metric("Capacidad Real (8h)", f"{int((415/media_ct)*0.75)} uds")

        # Gráfico interactivo
        st.subheader("📊 Mapa de Producción (Rojo = Ruido/Paros | Verde = Flujo Real)")
        fig = px.scatter(df, x='In DateTime', y='gap_mins', color=df['IA_Status'].astype(str),
                         color_discrete_map={'1': '#2ecc71', '-1': '#e74c3c'},
                         labels={'gap_mins': 'Minutos/Pieza', 'IA_Status': 'Estado IA'})
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("📥 Descargar Reporte Limpio", df_clean.to_csv(index=False).encode('utf-8'), "reporte_ia.csv")
        
    except Exception as e:
        st.error(f"Error procesando archivo: {e}")
