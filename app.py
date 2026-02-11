import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Process Intelligence", layout="wide", page_icon="⚙️")
st.title("⚙️ Celestica IA: Reconstructor de Ciclo Real")

# --- 1. LECTOR ULTRA-ROBUSTO ---
@st.cache_data(ttl=3600)
def load_data_final(file):
    try:
        file.seek(0)
        # Leemos el CSV con codificación latina para evitar errores de tildes
        df = pd.read_csv(file, sep=None, engine='python', encoding='latin-1')
        
        # Limpiar nombres de columnas (quitar espacios invisibles)
        df.columns = [str(c).strip() for c in df.columns]
        
        # Mapeo específico según tu archivo real
        cols = {
            'Fecha': 'Out DateTime' if 'Out DateTime' in df.columns else None,
            'Producto': 'Part Number' if 'Part Number' in df.columns else 'Model',
            'Operacion': 'Operation' if 'Operation' in df.columns else 'Operation'
        }
        
        if not cols['Fecha']:
            # Si no las encuentra, busca por parecido
            cols['Fecha'] = next((c for c in df.columns if 'date' in c.lower()), None)
            
        return df, cols
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None, None

# --- 2. CEREBRO: FILTRO DE BANDA 80/15/5 ---
def analyze_real_flow(df, cols):
    c_fec = cols['Fecha']
    
    # 1. Convertir fecha (Forzamos formato para evitar errores de idioma)
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce')
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)

    # 2. Calcular Gaps (Tiempo entre registros)
    # Aquí es donde ocurre el 80% de 0-1s que mencionas
    df['Gap'] = df[c_fec].diff().dt.total_seconds().fillna(0)
    
    # 3. FILTRO PASA-BANDA (Buscamos tu 15%)
    # Ignoramos el 80% de ruido (ceros y ráfagas < 30s)
    # Ignoramos el 5% de paradas (> 1500s)
    pasillo_real = df[(df['Gap'] >= 30) & (df['Gap'] <= 1500)]['Gap']
    
    if pasillo_real.empty:
        # Si el filtro es muy duro, bajamos el listón para no dar error
        pasillo_real = df[df['Gap'] > 5]['Gap']

    # 4. RESULTADOS
    # TC Real: La mediana del pasillo (lo que me pediste)
    tc_real_seg = pasillo_real.median()
    # TC Teórico: El percentil 20 del pasillo (el mejor ritmo humano)
    tc_teorico_seg = pasillo_real.quantile(0.20)
    
    return {
        'teo': tc_teorico_seg / 60,
        'real': tc_real_seg / 60,
        't_seg': tc_teorico_seg,
        'prod': df[cols['Producto']].iloc[0] if cols['Producto'] in df.columns else "N/A",
        'oper': df[cols['Operacion']].iloc[0] if cols['Operacion'] in df.columns else "N/A",
        'datos_plot': pasillo_real
    }

# --- 3. UI ---
uploaded_file = st.file_uploader("Sube el archivo Advance_tracking", type=["csv", "txt", "xlsx"])

if uploaded_file:
    with st.spinner("🤖 Analizando el 15% de datos de valor añadido..."):
        df_raw, cols_map = load_data_final(uploaded_file)
        
        if df_raw is not None and cols_map['Fecha']:
            res = analyze_real_flow(df_raw, cols_map)
            
            if res:
                # HEADER CON TU PRODUCTO Y OPERACIÓN
                st.success(f"📌 **Operación:** {res['oper']} | **Producto:** {res['prod']}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO", f"{res['teo']:.2f} min", 
                          help=f"Ritmo de excelencia detectado: {res['t_seg']:.1f}s")
                c2.metric("⏱️ TC REAL (Mediana)", f"{res['real']:.2f} min")
                
                capacidad = (8 * 60) / res['teo']
                c3.metric("📦 Capacidad (8h)", f"{int(capacidad)} uds")

                st.divider()

                # GRÁFICA DEL "SEGUNDO PICO"
                st.subheader("📊 Distribución del Flujo Humano (Excluyendo Ruido)")
                st.markdown("Esta gráfica ignora el 80% de ceros del servidor para que puedas ver tu montaña de 120s.")
                
                fig = px.histogram(res['datos_plot'], x=res['datos_plot'], nbins=50, 
                                 title="Ritmo de Producción Real (Segundos)",
                                 labels={'x': 'Segundos por Pieza'},
                                 color_discrete_sequence=['#2ecc71'])
                fig.add_vline(x=res['t_seg'], line_dash="dash", line_color="red", line_width=4)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo procesar el flujo. Intenta guardar el archivo como CSV antes de subirlo.")
