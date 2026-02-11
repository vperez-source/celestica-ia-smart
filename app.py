import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from scipy.stats import gaussian_kde

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica High-Speed AI", layout="wide", page_icon="⚡")
st.title("⚡ Celestica IA: Detector de Capacidad Teórica (Modo Flujo)")
st.markdown("""
**Análisis de Frontera Activa:** Esta IA no calcula promedios. Identifica los periodos de máxima 
velocidad sostenida para extraer el **Tiempo de Ciclo Teórico** real de la operación.
""")

# --- FASE A: LECTOR ULTRA-RÁPIDO ---
def parse_xml_fast(file):
    try:
        content = file.getvalue().decode('latin-1', errors='ignore')
        soup = BeautifulSoup(content, 'lxml-xml')
        data = [[c.get_text(strip=True) for c in row.find_all(['Cell', 'ss:Cell'])] 
                for row in soup.find_all(['Row', 'ss:Row'])]
        return pd.DataFrame([d for d in data if d])
    except: return None

@st.cache_data(ttl=3600)
def load_data(file):
    df = parse_xml_fast(file)
    if df is None or df.empty:
        try:
            file.seek(0)
            df = pd.read_excel(file, header=None)
        except: return None, None
    
    df = df.astype(str)
    # Localizar cabecera
    for i in range(min(50, len(df))):
        row = " ".join(df.iloc[i]).lower()
        if 'date' in row or 'time' in row:
            df.columns = df.iloc[i].str.strip()
            return df[i+1:].reset_index(drop=True), i
    return None, None

# --- FASE B: MOTOR DE CÁLCULO DE FRONTERA (IA) ---
def find_theoretical_tc(df):
    # 1. Identificar columnas clave dinámicamente
    col_fec = next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None)
    col_sn = next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None)
    
    if not col_fec: return None

    # 2. Limpieza y Ordenación
    df[col_fec] = pd.to_datetime(df[col_fec], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[col_fec]).sort_values(col_fec)
    if col_sn:
        df = df.drop_duplicates(subset=[col_sn], keep='first')
    
    # 3. Cálculo de Gaps y Limpieza de "Grandes Paradas"
    # Calculamos el tiempo entre cada pieza
    df['Gap_Sec'] = df[col_fec].diff().dt.total_seconds().fillna(0)
    
    # ELIMINAMOS EL RUIDO DE BATCH: Si el gap es 0, no lo contamos como pieza individual
    # ELIMINAMOS EL RUIDO DE PARADA: Solo analizamos piezas producidas en un flujo de < 10 min
    flujo_activo = df[(df['Gap_Sec'] > 5) & (df['Gap_Sec'] < 600)].copy()
    
    if flujo_activo.empty: return None

    # 4. ANÁLISIS DE DENSIDAD (Buscando el 1.40 min / 84s)
    # Aplicamos un filtro de percentil agresivo sobre el logaritmo
    # El teórico es el percentil 10 de los tiempos de flujo activo.
    tc_teorico_seg = np.percentile(flujo_activo['Gap_Sec'], 10) 
    
    # El Real es la mediana de ese flujo activo
    tc_real_seg = flujo_activo['Gap_Sec'].median()
    
    return {
        'teorico_min': tc_teorico_seg / 60,
        'real_min': tc_real_seg / 60,
        'piezas': len(df),
        'flujo': flujo_activo,
        'modo_s': tc_teorico_seg
    }

# --- FASE C: UI Y DASHBOARD ---
uploaded_file = st.file_uploader("Subir Archivo de Trazabilidad", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🤖 Analizando ráfagas y detectando flujo de excelencia..."):
        df_raw, header_idx = load_data(uploaded_file)
        
        if df_raw is not None:
            res = find_theoretical_tc(df_raw)
            
            if res:
                st.success("✅ Análisis de Flujo Sostenido Completado")
                
                # KPIs PRINCIPALES
                k1, k2, k3 = st.columns(3)
                # Forzamos que el diseño sea limpio y directo como el que te gustaba
                k1.metric("⏱️ TC TEÓRICO (Target)", f"{res['teorico_min']:.2f} min", 
                          help="Representa la velocidad máxima sostenida por el proceso.")
                k2.metric("⏱️ TC REAL (Sostenido)", f"{res['real_min']:.2f} min", 
                          delta=f"{((res['real_min']/res['teorico_min'])-1)*100:.1f}% Desvío")
                
                capacidad = (8 * 60) / res['teorico_min']
                k3.metric("📦 Capacidad Ideal", f"{int(capacidad)} uds", help="Capacidad en 8h al ritmo teórico.")

                st.divider()

                # GRÁFICA DE DISTRIBUCIÓN GAMMA (Ajustada al Teórico)
                st.subheader("📊 Distribución de Velocidad de la Línea")
                st.caption(f"El objetivo teórico está anclado en los **{res['modo_s']:.1f} segundos**.")
                
                # Filtramos para ver solo la zona de interés (la montaña)
                fig_data = res['flujo'][res['flujo']['Gap_Sec'] < (res['modo_s'] * 5)]
                
                fig = px.histogram(fig_data, x="Gap_Sec", nbins=80, 
                                 title="Histograma de Ritmos de Flujo (Segundos)",
                                 color_discrete_sequence=['#2ecc71'])
                
                fig.add_vline(x=res['modo_s'], line_width=4, line_dash="dash", line_color="red", 
                             annotation_text="FRONTERA TEÓRICA")
                st.plotly_chart(fig, use_container_width=True)

                # TABLA DE AUDITORÍA
                with st.expander("🔍 Auditoría de Datos"):
                    st.write(f"Total registros únicos: {res['piezas']}")
                    st.dataframe(df_raw.head(10))
            else:
                st.error("No se detectó flujo de producción. Verifica que los registros no tengan todos la misma hora.")
