import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import io

st.set_page_config(page_title="Celestica IA", layout="wide")
st.title("🛡️ Celestica IA: Smart-Trace Analyzer")

# --- CONFIGURACIÓN ---
with st.sidebar:
    st.header("⚙️ Parámetros")
    h_turno = st.number_input("Horas Turno", value=8)
    m_descanso = st.number_input("Minutos Descanso", value=45)
    eficiencia = st.slider("Eficiencia %", 50, 100, 75) / 100

uploaded_file = st.file_uploader("Arrastra tu archivo (.xls / .xlsx)", type=["xlsx", "xls"])

if uploaded_file:
    # Botón para iniciar (evita recargas constantes)
    if st.button("🚀 ANALIZAR ARCHIVO"):
        st.info(f"📂 Procesando: {uploaded_file.name}")
        
        try:
            content = uploaded_file.getvalue()
            df = None
            debug_msg = st.empty()

            # --- MOTOR DE LECTURA (CASCADA) ---
            
            # 1. Intento: Excel Moderno
            try:
                df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
            except:
                # 2. Intento: Excel Antiguo (XLRD)
                try:
                    df = pd.read_excel(io.BytesIO(content), engine='xlrd')
                except:
                    # 3. Intento: HTML (XML disfrazado)
                    try:
                        dfs = pd.read_html(io.BytesIO(content))
                        if len(dfs) > 0: df = dfs[0]
                    except:
                        # 4. INTENTO NUEVO: TEXTO/CSV (El "Abrelatas")
                        # Probamos separador por tabulacion (\t) o punto y coma
                        debug_msg.text("⚠️ Excel falló. Intentando leer como Texto Plano/CSV...")
                        try:
                            # sep='\t' es tabulación (muy común en reportes .xls falsos)
                            df = pd.read_csv(io.BytesIO(content), sep='\t', encoding='latin-1')
                        except:
                            try:
                                # Si falla, probamos separador automático
                                df = pd.read_csv(io.BytesIO(content), sep=None, engine='python', encoding='latin-1')
                            except Exception as e:
                                st.error(f"❌ IMPOSIBLE LEER EL ARCHIVO. Error final: {e}")
                                st.stop()

            # --- VERIFICACIÓN DE DATOS ---
            if df is not None:
                # Limpiar nombres de columnas (quitar espacios invisibles)
                df.columns = df.columns.astype(str).str.strip()
                
                # --- BUSCADOR DE CABECERAS INTELIGENTE ---
                # A veces la cabecera está en la fila 3 o 4. Buscamos dónde está "In DateTime"
                col_target = 'In DateTime'
                
                if col_target not in df.columns:
                    # Buscamos en las primeras 10 filas si aparece el texto "In DateTime"
                    found = False
                    for i in range(10):
                        row_values = df.iloc[i].astype(str).values
                        if any(col_target in s for s in row_values):
                            df.columns = df.iloc[i] # Promocionamos esta fila a cabecera
                            df = df[i+1:] # Borramos lo de arriba
                            df = df.reset_index(drop=True)
                            found = True
                            break
                    
                    if not found:
                        st.error(f"⚠️ No encuentro la columna '{col_target}'.")
                        st.write("👀 Mira las primeras filas de tu archivo tal como las leo:")
                        st.dataframe(df.head())
                        st.write("Columnas detectadas:", list(df.columns))
                        st.stop()

                # --- PROCESAMIENTO IA ---
                df[col_target] = pd.to_datetime(df[col_target], errors='coerce')
                # Fix año 2025/1900
                df.loc[df[col_target].dt.year < 100, col_target] += pd.offsets.DateOffset(years=2000)
                df = df.dropna(subset=[col_target]).sort_values(col_target)

                if df.empty:
                    st.error("El archivo se leyó, pero no quedan fechas válidas tras limpiar.")
                    st.stop()

                # ML Logic
                df['gap_mins'] = df.groupby('Station')[col_target].diff().dt.total_seconds() / 60
                df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())
                
                model = IsolationForest(contamination=0.05, random_state=42)
                df['IA_Status'] = model.fit_predict(df[['gap_mins']])
                
                # Filtro Quartiles
                df_clean = df[df['IA_Status'] == 1]
                q1 = df_clean['gap_mins'].quantile(0.25)
                q3 = df_clean['gap_mins'].quantile(0.75)
                df_final = df_clean[(df_clean['gap_mins'] >= q1) & (df_clean['gap_mins'] <= q3)]

                # KPIs
                media = df_final['gap_mins'].mean()
                capacidad = ((h_turno*60 - m_descanso)/media) * eficiencia

                # --- DASHBOARD ---
                st.success("✅ Análisis Completado")
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ Cycle Time", f"{media:.2f} min")
                c2.metric("📦 Capacidad (75%)", f"{int(capacidad)} uds")
                c3.metric("📉 Calidad Dato", f"{(len(df_final)/len(df)*100):.1f}%")

                fig = px.scatter(df, x=col_target, y='gap_mins', 
                               color=df['IA_Status'].astype(str),
                               color_discrete_map={'1':'#2ecc71', '-1':'#e74c3c'},
                               title="Detección de Anomalías (Rojo)")
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("Error desconocido: El DataFrame es None.")

        except Exception as e:
            st.error(f"Error Crítico: {e}")
