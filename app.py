import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import io

st.set_page_config(page_title="IA Celestica", layout="wide")
st.title("🛡️ Celestica IA: Smart-Trace Analyzer")

uploaded_file = st.file_uploader("Arrastra tu archivo (.xls o .xlsx)", type=["xlsx", "xls"])

if uploaded_file:
    try:
        content = uploaded_file.read()
        df = None

        # --- MOTOR DE BÚSQUEDA DE TABLAS ---
        # Intento 1: Excel Moderno
        try:
            df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
        except:
            # Intento 2: Excel Antiguo / XML
            try:
                df = pd.read_excel(io.BytesIO(content), engine='xlrd')
            except:
                try:
                    # Intento 3: Buscar tablas en HTML/XML
                    tablas = pd.read_html(io.BytesIO(content))
if len(tablas) > 0:
    df = tablas[0]
    # Si la primera fila parece ser el nombre de las columnas (pasa mucho en reportes de trazabilidad)
    if 'In DateTime' not in df.columns:
        df.columns = df.iloc[0] # Toma la primera fila como cabecera
        df = df[1:] # Borra la primera fila para que no esté duplicada
                except:
                    # Intento 4: XML Directo (Caso Celestica/SAP)
                    try:
                        df = pd.read_xml(io.BytesIO(content))
                    except Exception as e:
                        st.error(f"Error de formato: La IA no reconoce la estructura del archivo. Prueba a guardarlo como 'Libro de Excel' (.xlsx) y vuelve a subirlo.")
                        st.stop()

        if df is not None:
            # Limpiar nombres de columnas (quitar espacios y basura)
            df.columns = df.columns.astype(str).str.strip()
            
            # Verificar si existe la columna de fecha
            col_fecha = 'In DateTime'
            if col_fecha not in df.columns:
                st.warning(f"No encontré la columna '{col_fecha}'. Columnas detectadas: {list(df.columns)}")
                st.stop()

            # --- PROCESAMIENTO ---
            df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
            df.loc[df[col_fecha].dt.year < 100, col_fecha] += pd.offsets.DateOffset(years=2000)
            df = df.dropna(subset=[col_fecha]).sort_values(col_fecha)

            df['gap_mins'] = df.groupby('Station')[col_fecha].diff().dt.total_seconds() / 60
            df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())

            model = IsolationForest(contamination=0.05, random_state=42)
            df['IA_Status'] = model.fit_predict(df[['gap_mins']])
            
            df_clean = df[df['IA_Status'] == 1]
            q1, q3 = df_clean['gap_mins'].quantile([0.25, 0.75])
            df_clean = df_clean[(df_clean['gap_mins'] >= q1) & (df_clean['gap_mins'] <= q3)]
            media_ct = df_clean['gap_mins'].mean()

            # DASHBOARD
            c1, c2, c3 = st.columns(3)
            c1.metric("Cycle Time Real", f"{media_ct:.2f} min")
            c2.metric("Salud del Dato", f"{(len(df_clean)/len(df)*100):.1f}%")
            c3.metric("Capacidad Real (8h)", f"{int((415/media_ct)*0.75)} uds")

            st.plotly_chart(px.scatter(df, x=col_fecha, y='gap_mins', color=df['IA_Status'].astype(str), color_discrete_map={'1': '#2ecc71', '-1': '#e74c3c'}), use_container_width=True)
            st.download_button("Descargar Reporte Limpio", df_clean.to_csv(index=False).encode('utf-8'), "reporte_ia.csv")

    except Exception as e:
        st.error(f"Error inesperado: {e}")
