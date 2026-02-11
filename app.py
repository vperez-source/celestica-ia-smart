import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Process Reconstructor", layout="wide", page_icon="🏭")
st.title("🏭 Celestica IA: Reconstructor de Flujo Real")

# --- 1. LECTOR DE ALTA PRECISIÓN ---
@st.cache_data(ttl=3600)
def load_data_celestica(file):
    try:
        # Intentamos leer CSV primero (más estable para TXT/CSV)
        file.seek(0)
        df = pd.read_csv(file, sep=None, engine='python', encoding='latin-1')
    except:
        try:
            file.seek(0) # Re-intento como Excel/XML
            content = file.getvalue().decode('latin-1', errors='ignore')
            soup = BeautifulSoup(content, 'lxml-xml')
            data = [[c.get_text(strip=True) for c in row.find_all(['Cell', 'ss:Cell'])] 
                    for row in soup.find_all(['Row', 'ss:Row'])]
            df = pd.DataFrame([d for d in data if d])
            # Ajustar cabeceras si es XML
            for i in range(len(df)):
                if any(x in " ".join(df.iloc[i]).lower() for x in ['date', 'time', 'sn']):
                    df.columns = df.iloc[i]; df = df[i+1:]; break
        except: return None
    
    # Limpieza de nombres de columnas
    df.columns = [str(c).strip() for c in df.columns]
    return df

# --- 2. CEREBRO: REDISTRIBUCIÓN DE CARGA (DE-BATCHING) ---
def analyze_pulse_reconstruction(df):
    # Identificar columnas críticas
    c_time = next((c for c in df.columns if 'out' in c.lower() or 'date' in c.lower()), None)
    c_sn = next((c for c in df.columns if 'serial' in c.lower() or 'sn' in c.lower()), None)
    c_prod = next((c for c in df.columns if 'part' in c.lower() or 'model' in c.lower()), "Producto")
    c_oper = next((c for c in df.columns if 'oper' in c.lower() or 'station' in c.lower()), "Operación")

    if not c_time: return None

    # Ordenar y Limpiar
    df[c_time] = pd.to_datetime(df[c_time], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_time]).sort_values(c_time)
    if c_sn in df.columns: df = df.drop_duplicates(subset=[c_sn])

    # --- LÓGICA DE REPARTO DE LOTE ---
    # 1. Agrupamos registros que ocurren casi al mismo tiempo (ruido de servidor < 5s)
    df['Time_Diff'] = df[c_time].diff().dt.total_seconds().fillna(0)
    df['New_Batch'] = df['Time_Diff'] > 5
    df['Batch_ID'] = df['New_Batch'].cumsum()

    # 2. Calculamos el tiempo real por lote
    batch_summary = df.groupby('Batch_ID').agg(
        Final_Time=(c_time, 'max'),
        Units=('Batch_ID', 'count')
    )
    batch_summary['Gap_To_Prev'] = batch_summary['Final_Time'].diff().dt.total_seconds().fillna(0)
    
    # 3. REPARTO: El tiempo de cada unidad es el silencio previo / unidades del lote
    batch_summary['TC_Unitario'] = batch_summary['Gap_To_Prev'] / batch_summary['Units']
    
    # 4. FILTRO DE VALOR REAL (Tu pasillo de 15%)
    # Ignoramos paradas > 20 min y ráfagas residuales
    valid_flow = batch_summary[(batch_summary['TC_Unitario'] > 20) & (batch_summary['TC_Unitario'] < 1200)]
    
    if valid_flow.empty: return None

    return {
        'prod': df[c_prod].mode()[0] if c_prod in df.columns else "N/A",
        'oper': df[c_oper].mode()[0] if c_oper in df.columns else "N/A",
        'tc_real_min': valid_flow['TC_Unitario'].median() / 60,
        'tc_teo_min': valid_flow['TC_Unitario'].quantile(0.20) / 60, # El 20% más rápido del flujo real
        'df_plot': valid_flow,
        'total_units': len(df)
    }

# --- 3. UI ---
uploaded_file = st.file_uploader("Sube tu archivo (XLS, TXT, CSV)", type=["xls", "xml", "xlsx", "csv", "txt"])

if uploaded_file:
    with st.spinner("🤖 Reconstruyendo el latido de la línea..."):
        df_raw = load_data_celestica(uploaded_file)
        if df_raw is not None:
            res = analyze_pulse_reconstruction(df_raw)
            if res:
                st.success(f"📋 **Operación:** {res['oper']} | **Producto:** {res['prod']}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO", f"{res['tc_teo_min']:.2f} min", help="Basado en el mejor 20% de rendimiento sostenido.")
                c2.metric("⏱️ TC REAL (Mediana)", f"{res['tc_real_min']:.2f} min")
                
                h_turno = st.sidebar.number_input("Horas Turno", value=8.0)
                capacidad = (h_turno * 60) / res['tc_teo_min']
                c3.metric("📦 Capacidad Turno", f"{int(capacidad)} uds")

                st.divider()
                st.subheader("📊 Distribución del Tiempo de Ciclo (De-batched)")
                st.markdown("Esta gráfica muestra el tiempo por pieza **recalculado**, eliminando el ruido del servidor.")
                
                fig = px.histogram(res['df_plot'], x="TC_Unitario", nbins=50, 
                                 title="Histograma de Ritmo Real (Segundos)",
                                 labels={'TC_Unitario': 'Segundos / Unidad'},
                                 color_discrete_sequence=['#2ecc71'])
                fig.add_vline(x=res['tc_teo_min']*60, line_dash="dash", line_color="red", line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo extraer el flujo. Comprueba que el archivo tenga segundos en la hora.")
