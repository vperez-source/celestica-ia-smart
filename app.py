import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Truth Hunter", layout="wide", page_icon="🎯")
st.title("🎯 Celestica IA: Buscador de Tiempos Reales")

# --- 1. LECTOR ROBUSTO ---
@st.cache_data(ttl=3600)
def read_and_clean(file):
    fname = file.name.lower()
    df = None
    try:
        if fname.endswith(('.xml', '.xls')):
            content = file.getvalue().decode('latin-1', errors='ignore')
            soup = BeautifulSoup(content, 'lxml-xml')
            data = [[c.get_text(strip=True) for c in row.find_all(['Cell', 'ss:Cell'])] 
                    for row in soup.find_all(['Row', 'ss:Row'])]
            df = pd.DataFrame([d for d in data if d])
        else:
            file.seek(0)
            df = pd.read_csv(file, sep=None, engine='python', header=None)
        
        # Localizar cabecera
        df = df.astype(str)
        for i in range(min(100, len(df))):
            row = " ".join(df.iloc[i]).lower()
            if any(x in row for x in ['date', 'time', 'fecha', 'sn', 'serial']):
                df.columns = df.iloc[i].str.strip()
                df = df[i+1:].reset_index(drop=True)
                break
        return df
    except: return None

# --- 2. CEREBRO DE ANÁLISIS ---
def run_diagnostic(df):
    # Buscar columnas clave por contenido, no solo por nombre
    c_fec = next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None)
    c_sn = next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None)
    c_prod = next((c for c in df.columns if any(x in c.lower() for x in ['product', 'item', 'part'])), "Producto")
    c_oper = next((c for c in df.columns if any(x in c.lower() for x in ['station', 'oper', 'step', 'workcenter'])), "Operación")

    if not c_fec: return None

    # Limpiar y Ordenar
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    if c_sn in df.columns: df = df.drop_duplicates(subset=[c_sn], keep='first')

    # --- LÓGICA DE RECONSTRUCCIÓN DE TIEMPO ---
    # Calculamos el tiempo entre ráfagas
    # Si entran 10 piezas en el mismo segundo, el gap es con respecto a la ráfaga anterior
    lotes = df.groupby(c_fec).size().reset_index(name='piezas_en_lote')
    lotes['gap_seg'] = lotes[c_fec].diff().dt.total_seconds().fillna(0)
    
    # REPARTO DE CARGA: El tiempo de cada pieza es Gap / Piezas
    # Solo miramos piezas donde el resultado esté entre 30s y 600s (Tu 15% de valor real)
    lotes['tc_estimado'] = lotes['gap_seg'] / lotes['piezas_en_lote']
    
    flujo_real = lotes[(lotes['tc_estimado'] >= 30) & (lotes['tc_estimado'] <= 900)].copy()

    # Si no hay flujo, tomamos el promedio de los huecos más frecuentes
    if flujo_real.empty:
        tc_manual = lotes[lotes['gap_seg'] > 0]['gap_seg'].median() / 2 # Estimación agresiva
    else:
        tc_manual = flujo_real['tc_estimado'].median()

    return {
        'prod': df[c_prod].mode()[0] if c_prod in df.columns else "N/A",
        'oper': df[c_oper].mode()[0] if c_oper in df.columns else "N/A",
        'tc_real': tc_manual / 60,
        'tc_teo': (tc_manual * 0.85) / 60, # El teórico es un 15% mejor que el real observado
        'df_lotes': lotes,
        'df_clean': flujo_real
    }

# --- 3. UI ---
uploaded_file = st.file_uploader("Sube el archivo para que la IA lo analice", type=["xls", "xml", "xlsx", "csv", "txt"])

if uploaded_file:
    df_raw = read_and_clean(uploaded_file)
    if df_raw is not None:
        res = run_diagnostic(df_raw)
        if res:
            st.success(f"📦 Producto: {res['prod']} | ⚙️ Operación: {res['oper']}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("⏱️ TC TEÓRICO", f"{res['tc_teo']:.2f} min")
            c2.metric("⏱️ TC REAL", f"{res['tc_real']:.2f} min")
            
            h_turno = st.sidebar.number_input("Horas Turno", value=8.0)
            capacidad = (h_turno * 60) / res['tc_teo']
            c3.metric("📦 Capacidad", f"{int(capacidad)} uds")

            st.divider()
            
            # GRÁFICA DE DIAGNÓSTICO
            st.subheader("📊 Mapa de Calor de Producción")
            st.markdown("Cada punto es un registro. Los grupos de puntos indican **Lotes de Producción**.")
            fig = px.scatter(res['df_lotes'], x=res['df_lotes'].columns[0], y='tc_estimado', 
                            title="Tiempos Detectados (Incluyendo ruido)",
                            labels={'tc_estimado': 'Segundos por pieza'})
            # Limitar vista para ver el pasillo de 120s
            fig.update_yaxes(range=[0, 600]) 
            st.plotly_chart(fig, use_container_width=True)
