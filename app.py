import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Batch Master", layout="wide", page_icon="🏭")
st.title("🏭 Celestica IA: Análisis de Procesos por Lotes (Batch)")

with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.info("ℹ️ Ajuste de Lotes:")
    umbral_pausa = st.number_input(
        "Minutos máximos entre piezas (Lote):", 
        min_value=1, value=30, 
        help="Si pasan más de X minutos, se considera parada (descanso, avería) y no cuenta."
    )
    
    contamination = st.slider("Sensibilidad Anomalías", 1, 25, 5) / 100
    st.divider()
    eficiencia = st.slider("Eficiencia Objetivo %", 50, 100, 85) / 100

# --- LECTORES ---
def leer_xml_a_la_fuerza(file):
    try:
        content = file.getvalue().decode('latin-1', errors='ignore')
        soup = BeautifulSoup(content, 'xml')
        datos = []
        rows = soup.find_all(['Row', 'ss:Row', 'row'])
        for row in rows:
            fila_datos = []
            cells = row.find_all(['Cell', 'ss:Cell', 'cell'])
            for cell in cells:
                data_tag = cell.find(['Data', 'ss:Data', 'data'])
                if data_tag: fila_datos.append(data_tag.get_text(strip=True))
                else: fila_datos.append("")
            if any(fila_datos): datos.append(fila_datos)
        return pd.DataFrame(datos)
    except: return None

@st.cache_data(ttl=3600)
def load_data(file):
    try:
        file.seek(0)
        head = file.read(500).decode('latin-1', errors='ignore')
        file.seek(0)
        if "<?xml" in head or "Workbook" in head: return leer_xml_a_la_fuerza(file)
    except: pass
    try: file.seek(0); return pd.read_excel(file, engine='calamine', header=None)
    except: pass
    try: file.seek(0); dfs = pd.read_html(file, header=None); return dfs[0] if len(dfs)>0 else None
    except: pass
    try: file.seek(0); return pd.read_csv(file, sep='\t', encoding='latin-1', header=None)
    except: return None

# --- BÚSQUEDA DE CABECERAS ---
def encontrar_cabecera(df):
    if df is None: return None, -1
    df = df.astype(str)
    k_fecha = ['date', 'time', 'fecha', 'hora']
    
    for i in range(min(50, len(df))):
        fila = df.iloc[i].str.lower().tolist()
        if any(k in str(v) for v in fila for k in k_fecha):
            return i
    return -1

# --- APP PRINCIPAL ---
uploaded_file = st.file_uploader("Sube el archivo", type=["xlsx", "xls", "txt", "xml"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        # 1. Detectar cabecera
        idx_cabecera = encontrar_cabecera(df_raw)
        
        if idx_cabecera != -1:
            df = df_raw.copy()
            df.columns = df.iloc[idx_cabecera]
            df = df[idx_cabecera + 1:].reset_index(drop=True)
            df.columns = df.columns.astype(str).str.strip()
            
            # --- ZONA DE CONTROL MANUAL (AQUÍ ESTÁ LA SOLUCIÓN) ---
            st.markdown("### 🔧 Verifica las Columnas")
            c1, c2, c3 = st.columns(3)
            
            # Auto-selección inteligente
            all_cols = list(df.columns)
            
            # Fecha
            def_fecha = next((c for c in all_cols if any(x in c.lower() for x in ['date', 'time', 'fecha'])), all_cols[0])
            col_f = c1.selectbox("📅 Columna FECHA", all_cols, index=all_cols.index(def_fecha))
            
            # Estación (Aquí es donde fallaba, ahora puedes cambiarlo)
            # Quitamos 'productid' de la lista prioritaria para que no se confunda
            def_est = next((c for c in all_cols if any(x in c.lower() for x in ['station', 'operation', 'work', 'maquina'])), all_cols[0])
            col_s = c2.selectbox("🏭 Columna ESTACIÓN (Agrupación)", all_cols, index=all_cols.index(def_est), help="Si eliges ProductID dará 0. Elige la máquina o una columna con valor constante.")
            
            # Usuario
            def_user = next((c for c in all_cols if any(x in c.lower() for x in ['user', 'operator', 'name'])), None)
            idx_u = all_cols.index(def_user) if def_user else 0
            col_u = c3.selectbox("👤 Columna OPERARIO (Opcional)", [None] + all_cols, index=idx_u + 1 if def_user else 0)

            if st.button("🚀 CALCULAR AHORA", type="primary"):
                # 2. PROCESAMIENTO
                try:
                    df[col_f] = pd.to_datetime(df[col_f], errors='coerce')
                    df.loc[df[col_f].dt.year < 100, col_f] += pd.offsets.DateOffset(years=2000)
                    df = df.dropna(subset=[col_f]).sort_values(col_f)

                    # CÁLCULO DE GAPS (POR LOTES)
                    # Agrupamos por la columna que tú elegiste (col_s)
                    df['gap_mins'] = df.groupby(col_s)[col_f].diff().dt.total_seconds() / 60
                    
                    # FILTRO DE LOTES
                    # Solo sumamos tiempos menores al umbral (ej. < 30 min)
                    df_prod = df[df['gap_mins'] < umbral_pausa].copy()
                    
                    tiempo_total = df_prod['gap_mins'].sum()
                    piezas_totales = len(df)
                    
                    if piezas_totales > 0:
                        ct_real = tiempo_total / piezas_totales
                    else:
                        ct_real = 0
                        
                    capacidad = (480 / ct_real * eficiencia) if ct_real > 0 else 0

                    # RESULTADOS
                    st.success(f"✅ Cálculo Realizado. Tiempo Activo Total: {int(tiempo_total)} min")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("⏱️ Cycle Time Real", f"{ct_real:.2f} min/ud")
                    m2.metric("📦 Capacidad (8h)", f"{int(capacidad)} uds")
                    m3.metric("📊 Datos", piezas_totales)

                    st.divider()
                    
                    # GRÁFICAS
                    col_g1, col_g2 = st.columns(2)
                    
                    with col_g1:
                        # Histograma
                        fig = px.histogram(df_prod, x='gap_mins', nbins=50, title="Distribución de Tiempos de Lote")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with col_g2:
                        # Ranking Operarios
                        if col_u:
                            df['gap_prod_user'] = df['gap_mins'].where(df['gap_mins'] < umbral_pausa, 0)
                            stats = df.groupby(col_u).agg(
                                Piezas=('gap_mins', 'count'),
                                Tiempo=('gap_prod_user', 'sum')
                            ).reset_index()
                            stats['CT'] = stats['Tiempo'] / stats['Piezas']
                            stats = stats.sort_values('Piezas', ascending=False)
                            
                            fig_bar = px.bar(stats.head(10), x=col_u, y='Piezas', color='CT', 
                                           title="Top Operarios (Color = Velocidad)",
                                           color_continuous_scale='RdYlGn_r') # Rojo = Lento, Verde = Rápido
                            st.plotly_chart(fig_bar, use_container_width=True)

                except Exception as e:
                    st.error(f"Error en el cálculo: {e}")
                    st.warning("Consejo: Revisa que la Columna FECHA sea realmente una fecha.")

        else:
            st.error("No encontré la cabecera automáticamente. Revisa el archivo.")
    else:
        st.error("No se pudo leer el archivo.")
