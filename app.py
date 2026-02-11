import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica Process Intelligence", layout="wide", page_icon="⚙️")
st.title("⚙️ Celestica IA: Reconstructor de Flujo (v29.0)")

# --- 1. MOTOR DE CARGA MULTIFORMATO (Recuperado y Mejorado) ---
@st.cache_data(ttl=3600)
def load_data_universal(file):
    fname = file.name.lower()
    df = None
    try:
        # CASO A: Archivos XLS / XML (Legacy Spectrum)
        if fname.endswith(('.xml', '.xls')):
            content = file.getvalue().decode('latin-1', errors='ignore')
            if "<?xml" in content or "Workbook" in content:
                soup = BeautifulSoup(content, 'lxml-xml')
                data = [[c.get_text(strip=True) for c in row.find_all(['Cell', 'ss:Cell'])] 
                        for row in soup.find_all(['Row', 'ss:Row'])]
                df = pd.DataFrame([d for d in data if d])
            else:
                file.seek(0)
                df = pd.read_excel(file, header=None)
        # CASO B: TXT / CSV
        else:
            file.seek(0)
            df = pd.read_csv(file, sep=None, engine='python', encoding='latin-1', header=None)
    except Exception as e:
        st.error(f"Error de lectura: {e}")
        return None, {}

    if df is None or df.empty: return None, {}

    # Buscador de cabeceras flexible
    df = df.astype(str)
    header_idx = 0
    for i in range(min(100, len(df))):
        row_str = " ".join(df.iloc[i]).lower()
        if any(x in row_str for x in ['date', 'time', 'fecha', 'sn', 'serial']):
            header_idx = i; break
    
    df.columns = df.iloc[header_idx].str.strip()
    df = df[header_idx + 1:].reset_index(drop=True)

    cols = {
        'Fecha': next((c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'fecha'])), None),
        'SN': next((c for c in df.columns if any(x in c.lower() for x in ['serial', 'sn', 'unitid'])), None),
        'Producto': next((c for c in df.columns if any(x in c.lower() for x in ['product', 'part', 'model'])), "Producto"),
        'Operacion': next((c for c in df.columns if any(x in c.lower() for x in ['station', 'oper', 'step'])), "Operación")
    }
    return df, cols

# --- 2. CEREBRO: DETECTOR DE SEGUNDO PICO (Lógica 80/15/5) ---
def analyze_reconstruction(df, cols):
    c_fec = cols['Fecha']
    
    # TRATAMIENTO DE FECHA ESPECIAL: Jan 16,25 01:04:28
    # Intentamos parsear con formatos comunes de Celestica
    df[c_fec] = pd.to_datetime(df[c_fec], errors='coerce', infer_datetime_format=True)
    
    # Si falla, intentamos una limpieza manual para el formato "Jan 16,25"
    if df[c_fec].isna().all():
        try:
            # Reemplazamos la coma por un espacio y corregimos el año '25
            temp_date = df[c_fec].str.replace(',', ' 20', regex=False)
            df[c_fec] = pd.to_datetime(temp_date, errors='coerce')
        except: pass

    df = df.dropna(subset=[c_fec]).sort_values(c_fec)
    
    if df.empty:
        return {"error": "No se pudo interpretar el formato de fecha. Revisa si es 'Jan 16,25'"}

    # Identidad (Inmune a IndexError)
    prod_name = df[cols['Producto']].iloc[0] if not df.empty and cols['Producto'] in df.columns else "N/A"
    oper_name = df[cols['Operacion']].iloc[0] if not df.empty and cols['Operacion'] in df.columns else "N/A"

    # Lógica de Gaps (Batching)
    df['Gap'] = df[c_fec].diff().dt.total_seconds().fillna(0)
    
    # APLICACIÓN DEL CRITERIO 80/15/5
    # Ordenamos los tiempos para encontrar los cortes
    tiempos = df[df['Gap'] > 0]['Gap'].sort_values().values
    if len(tiempos) < 10:
        # Fallback si hay pocos datos
        tc_med = df['Gap'].median()
        return {'teo': tc_med/60, 'real': tc_med/60, 'prod': prod_name, 'oper': oper_name, 'error_logic': True}

    # El "Pasillo de Producción": Saltamos el 80% (ruido) y cortamos el 5% final (paradas)
    p80 = np.percentile(tiempos, 80)
    p95 = np.percentile(tiempos, 95)
    
    # Filtramos los datos que pertenecen al 15% real
    pasillo = tiempos[(tiempos >= p80) & (tiempos <= p95)]
    
    if len(pasillo) == 0:
        tc_teo = p80
    else:
        tc_teo = pasillo[0] # El inicio del pasillo es el Teórico
        
    tc_real = np.median(pasillo) if len(pasillo) > 0 else p80

    return {
        'teo': tc_teo / 60,
        'real': tc_real / 60,
        't_seg': tc_teo,
        'prod': prod_name,
        'oper': oper_name,
        'datos_plot': pasillo if len(pasillo) > 0 else tiempos,
        'p80': p80,
        'p95': p95
    }

# --- 3. UI ---
uploaded_file = st.file_uploader("Sube el archivo (XLS, TXT, CSV)", type=["xls", "xml", "xlsx", "csv", "txt"])

if uploaded_file:
    with st.spinner("🕵️ Reconstruyendo el 15% de flujo real..."):
        df_raw, cols_map = load_data_universal(uploaded_file)
        
        if df_raw is not None and cols_map.get('Fecha'):
            res = analyze_reconstruction(df_raw, cols_map)
            
            if "error" in res:
                st.error(res["error"])
            else:
                st.success(f"📌 **Operación:** {res['oper']} | **Producto:** {res['prod']}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱️ TC TEÓRICO", f"{res['teo']:.2f} min", 
                          help=f"Frontera detectada tras el 80% de ruido: {res['t_seg']:.1f}s")
                c2.metric("⏱️ TC REAL (Mediana)", f"{res['real']:.2f} min")
                
                capacidad = (8 * 60) / res['teo'] if res['teo'] > 0 else 0
                c3.metric("📦 Capacidad (8h)", f"{int(capacidad)} uds")

                st.divider()
                st.subheader("📊 Pasillo de Producción Real (15% del total)")
                st.write(f"La IA ha detectado que tu ritmo real está entre **{res.get('p80', 0):.1f}s** y **{res.get('p95', 0):.1f}s**.")
                
                fig = px.histogram(res['datos_plot'], nbins=30, title="Distribución de Tiempos de Valor Añadido",
                                 color_discrete_sequence=['#2ecc71'])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No se detectó la estructura del archivo.")
