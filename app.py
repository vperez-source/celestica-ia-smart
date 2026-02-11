import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Engineering Tool", layout="wide", page_icon="⚙️")
st.title("⚙️ Celestica IA: Criterio de Ingeniería de Métodos")
st.markdown("""
**Análisis de Ritmo Humano:** Esta versión aplica filtros de lógica física para ignorar volcados de servidor 
y encontrar el tiempo de ciclo que realmente sucede en el puesto de trabajo.
""")

# --- 1. LECTOR DE DATOS (XML/XLS) ---
def parse_excel_legacy(file):
    try:
        content = file.getvalue().decode('latin-1', errors='ignore')
        soup = BeautifulSoup(content, 'lxml-xml')
        data = [[cell.get_text(strip=True) for cell in row.find_all(['Cell', 'ss:Cell'])] 
                for row in soup.find_all(['Row', 'ss:Row'])]
        return pd.DataFrame([d for d in data if d])
    except: return None

@st.cache_data(ttl=3600)
def load_data(file):
    df = parse_excel_legacy(file)
    if df is None or df.empty:
        try:
            file.seek(0)
            df = pd.read_excel(file, header=None)
        except: return None, None
    
    df = df.astype(str)
    for i in range(min(60, len(df))):
        row = " ".join(df.iloc[i]).lower()
        if 'date' in row or 'time' in row:
            df.columns = df.iloc[i].str.strip()
            return df[i+1:].reset_index(drop=True)
    return None

# --- 2. CEREBRO: FILTRO DE COHERENCIA FÍSICA ---
def analyze_with_manufacturing_logic(df):
    # Identificar Fecha y Serial
    c_fec = next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None)
    c_sn = next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None)
    
    if not c_fec: return None

    # Limpieza
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn: df = df.drop_duplicates(subset=[c_sn], keep='first')

    # --- CÁLCULO DE RITMO REALISTA ---
    # Medimos el tiempo entre eventos
    df['Gap'] = df[c_fec].diff().dt.total_seconds().fillna(0)
    
    # CRITERIO IA: 
    # 1. Ignoramos Gaps < 40s (Considerados ráfagas de sistema o batching)
    # 2. Ignoramos Gaps > 600s (Considerados paradas de línea o descansos)
    ritmos_humanos = df[(df['Gap'] >= 40) & (df['Gap'] <= 600)]['Gap']

    if len(ritmos_humanos) < 5:
        # Si no hay ritmos entre 40s y 10min, el archivo solo tiene ráfagas.
        # Aplicamos lógica de "Tiempo de bloque / Piezas del bloque"
        df['Bloque'] = (df['Gap'] > 600).cumsum()
        bloques = df.groupby('Bloque').agg(
            Duracion=(c_fec, lambda x: (x.max() - x.min()).total_seconds()),
            Piezas=('Gap', 'count')
        )
        # Filtramos bloques con producción real
        bloques = bloques[(bloques['Duracion'] > 0) & (bloques['Piezas'] > 1)]
        if not bloques.empty:
            tc_estimado = (bloques['Duracion'].sum() / bloques['Piezas'].sum())
            return {'teo': tc_estimado/60, 'real': tc_estimado/60, 'metodo': 'Cálculo por Bloques Activos', 'data': df}
        return None

    # Si hay ritmos humanos, buscamos el "Punto Dulce" (Mediana de la zona estable)
    tc_teorico_seg = np.percentile(ritmos_humanos, 20) # El mejor ritmo sostenido
    tc_real_seg = ritmos_humanos.median() # El ritmo promedio de trabajo

    return {
        'teo': tc_teorico_seg / 60,
        'real': tc_real_seg / 60,
        'metodo': 'Análisis de Flujo Humano',
        'data': df,
        'ritmos': ritmos_humanos
    }

# --- 3. INTERFAZ Y RESULTADOS ---
uploaded_file = st.file_uploader("Sube el archivo de 15.4MB / 1.9MB", type=["xls", "xml", "xlsx"])

if uploaded_file:
    with st.spinner("🕵️ Aplicando criterio de ingeniería..."):
        df_raw = load_data(uploaded_file)
        if df_raw is not None:
            res = analyze_manufacturing_logic(df_raw) if 'analyze_manufacturing_logic' in locals() else analyze_with_manufacturing_logic(df_raw)
            
            if res:
                st.success(f"✅ Método aplicado: {res['metodo']}")
                
                # KPIs Limpios
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO (Objetivo)", f"{res['teo']:.2f} min", help="Ritmo de máxima eficiencia detectado.")
                c2.metric("⏱️ TC REAL (Sostenido)", f"{res['real']:.2f} min")
                
                capacidad = (8 * 60) / res['teo']
                c3.metric("📦 Capacidad Turno", f"{int(capacidad)} uds")

                st.divider()

                # GRÁFICA DE CRITERIO
                st.subheader("📊 Distribución de Tiempos de Ciclo")
                if 'ritmos' in res:
                    fig = px.histogram(res['ritmos'], x="Gap", nbins=50, 
                                     title="Frecuencia de Tiempos (Solo zona humana: 40s - 600s)",
                                     labels={'Gap': 'Segundos por Pieza'},
                                     color_discrete_sequence=['#2ecc71'])
                    fig.add_vline(x=res['teo']*60, line_dash="dash", line_color="red", annotation_text="Teórico")
                    st.plotly_chart(fig, use_container_width=True)
                
                # DIAGNÓSTICO PARA TI
                with st.expander("🔍 Por qué estos números? (Auditoría IA)"):
                    total_p = len(df_raw)
                    st.write(f"1. Registros totales: {total_p}")
                    if 'ritmos' in res:
                        st.write(f"2. Piezas en flujo real detectadas: {len(res['ritmos'])}")
                        st.write(f"3. Ruido (Ráfagas o Paradas) eliminado: {total_p - len(res['ritmos'])} registros")
                    st.info("La IA ha decidido ignorar cualquier dato fuera del rango 40s-600s para evitar los errores de red de Spectrum.")

            else:
                st.error("Los datos están demasiado corruptos. El servidor Spectrum grabó todas las piezas con el mismo tiempo o con huecos de horas.")
