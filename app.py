import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import io

st.set_page_config(page_title="Celestica IA", layout="wide")
st.title("🛡️ Celestica IA: Smart-Trace Analyzer")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    h_turno = st.number_input("Horas Turno", value=8)
    m_descanso = st.number_input("Minutos Descanso", value=45)
    eficiencia = st.slider("Eficiencia %", 50, 100, 75) / 100

# --- ACEPTAMOS TXT AHORA ---
uploaded_file = st.file_uploader("Sube tu archivo (.txt, .xls, .xlsx)", type=["txt", "xlsx", "xls"])

if uploaded_file:
    if st.button("🚀 ANALIZAR DATOS"):
        try:
            df = None
            st.info(f"Procesando: {uploaded_file.name}")

            # --- MOTOR DE LECTURA ---
            # CASO 1: Archivo de Texto (.txt) -> LA OPCIÓN MÁS SEGURA
            if uploaded_file.name.endswith('.txt'):
                try:
                    # Probamos primero con Tabulaciones (lo estándar en industria)
                    df = pd.read_csv(uploaded_file, sep='\t', encoding='latin-1')
                except:
                    # Si falla, probamos punto y coma
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')

            # CASO 2: Excel (.xlsx / .xls)
            else:
                content = uploaded_file.read()
                try: df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
                except: 
                    try: df = pd.read_excel(io.BytesIO(content), engine='xlrd')
                    except: df = pd.read_html(io.BytesIO(content))[0]

            # --- SI NO HAY DATOS, PARAMOS ---
            if df is None:
                st.error("❌ No se pudo leer el archivo. Asegúrate de guardarlo como 'Texto delimitado por tabulaciones (.txt)'.")
                st.stop()

            # --- LIMPIEZA DE CABECERAS ---
            # Buscamos dónde empieza realmente la tabla
            df.columns = df.columns.astype(str).str.strip()
            col_target = 'In DateTime'

            # Si la columna no está, buscamos en las primeras 10 filas
            if col_target not in df.columns:
                found = False
                for i in range(min(10, len(df))):
                    row = df.iloc[i].astype(str).values
                    # Buscamos si alguna celda contiene "In DateTime"
                    if any(col_target in s for s in row):
                        df.columns = df.iloc[i] # Esta fila es la cabecera
                        df = df[i+1:].reset_index(drop=True) # Borramos lo de arriba
                        found = True
                        break
                
                if not found:
                    st.error(f"⚠️ No encuentro la columna '{col_target}'.")
                    st.write("👀 Así lee la IA tu archivo (primeras 5 filas):")
                    st.dataframe(df.head())
                    st.stop()

            # --- MACHINE LEARNING ---
            df[col_target] = pd.to_datetime(df[col_target], errors='coerce')
            # Arreglo fecha 2025
            df.loc[df[col_target].dt.year < 100, col_target] += pd.offsets.DateOffset(years=2000)
            df = df.dropna(subset=[col_target]).sort_values(col_target)

            if df.empty:
                st.error("No hay fechas válidas.")
                st.stop()

            # Cálculo de Gaps
            df['gap_mins'] = df.groupby('Station')[col_target].diff().dt.total_seconds() / 60
            df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())

            # IA: Isolation Forest
            model = IsolationForest(contamination=0.05, random_state=42)
            df['IA_Status'] = model.fit_predict(df[['gap_mins']])

            # Filtrado Final
            df_clean = df[df['IA_Status'] == 1]
            q1, q3 = df_clean['gap_mins'].quantile([0.25, 0.75])
            df_final = df_clean[(df_clean['gap_mins'] >= q1) & (df_clean['gap_mins'] <= q3)]

            # --- RESULTADOS ---
            media = df_final['gap_mins'].mean()
            capacidad = ((h_turno*60 - m_descanso)/media) * eficiencia

            st.success("✅ Análisis Exitoso")
            k1, k2, k3 = st.columns(3)
            k1.metric("⏱️ Cycle Time", f"{media:.2f} min")
            k2.metric("📦 Capacidad", f"{int(capacidad)} uds")
            k3.metric("📉 Calidad Dato", f"{(len(df_final)/len(df)*100):.1f}%")

            # Gráfica
            fig = px.scatter(df, x=col_target, y='gap_mins', 
                           color=df['IA_Status'].astype(str),
                           color_discrete_map={'1':'#2ecc71', '-1':'#e74c3c'},
                           title="Análisis de Ritmo (Verde=OK, Rojo=Ruido)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Exportar
            st.download_button("📥 Descargar CSV Limpio", df_final.to_csv(index=False).encode('utf-8'), "reporte_ia.csv")

        except Exception as e:
            st.error(f"Error: {e}")
