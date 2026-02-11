import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica AI Auto-Adaptive", layout="wide", page_icon="🧠")
st.title("🧠 Celestica IA: Análisis Auto-Adaptativo")
st.markdown("""
**Modo Inteligente:** La IA detecta automáticamente los límites de los lotes y separa los descansos 
usando estadística avanzada (IQR), sin necesidad de introducir filtros manuales.
""")

# --- LECTORES ---
def leer_xml_robusto(file):
    try:
        content = file.getvalue().decode('latin-1', errors='ignore')
        soup = BeautifulSoup(content, 'xml')
        datos = []
        rows = soup.find_all(['Row', 'ss:Row', 'row'])
        for row in rows:
            fila = [cell.get_text(strip=True) for cell in row.find_all(['Cell', 'ss:Cell', 'cell'])]
            if any(fila): datos.append(fila)
        return pd.DataFrame(datos)
    except: return None

@st.cache_data(ttl=3600)
def load_data(file):
    try: 
        file.seek(0)
        if "<?xml" in file.read(500).decode('latin-1', errors='ignore'): 
            file.seek(0); return leer_xml_robusto(file)
    except: pass
    try: file.seek(0); return pd.read_excel(file, engine='calamine', header=None)
    except: pass
    try: file.seek(0); return pd.read_csv(file, sep='\t', encoding='latin-1', header=None)
    except: return None

# --- MAPEO SEGURO ---
def mapear_columnas(df):
    if df is None: return None, {}
    df = df.astype(str)
    start = -1
    for i in range(min(50, len(df))):
        row = df.iloc[i].str.lower().tolist()
        if any('date' in str(v) for v in row) and any('station' in str(v) for v in row):
            start = i; break
    if start == -1: return None, {}
    df.columns = df.iloc[start]
    df = df[start+1:].reset_index(drop=True)
    df.columns = df.columns.astype(str).str.strip()
    cols = {'Fecha': None, 'Producto': None, 'Familia': None, 'Usuario': None}
    for c in df.columns:
        cl = c.lower()
        if not cols['Fecha'] and ('date' in cl or 'time' in cl): cols['Fecha'] = c
        if not cols['Producto'] and ('product' in cl or 'item' in cl): cols['Producto'] = c
        if not cols['Familia'] and ('family' in cl): cols['Familia'] = c
        if not cols['Usuario'] and ('user' in cl or 'operator' in cl): cols['Usuario'] = c
    # Rellenar faltantes
    if not cols['Fecha']: return None, {}
    for k, v in cols.items():
        if v is None:
            df[f'Col_{k}'] = "General"
            cols[k] = f'Col_{k}'
    return df, cols

# --- CEREBRO: DETECCIÓN ESTADÍSTICA DE OUTLIERS (IQR) ---
def procesar_ia_adaptativa(df, col_fec):
    # 1. Limpieza inicial
    df[col_fec] = pd.to_datetime(df[col_fec], errors='coerce')
    df = df.dropna(subset=[col_fec]).sort_values(col_fec)
    
    # 2. Calcular Gaps (Diferencia de tiempo)
    df['Gap_Min'] = df[col_fec].diff().dt.total_seconds().fillna(0) / 60
    
    # 3. FILTRO ESTADÍSTICO (IQR)
    # Buscamos qué gaps son "anormalmente largos" (descansos)
    # Solo miramos gaps > 0 para la estadística
    gaps_positivos = df[df['Gap_Min'] > 0]['Gap_Min']
    if not gaps_positivos.empty:
        Q1 = gaps_positivos.quantile(0.25)
        Q3 = gaps_positivos.quantile(0.75)
        IQR = Q3 - Q1
        # El límite superior para no ser outlier suele ser Q3 + 1.5*IQR
        # Para procesos de fabricación, somos más laxos: Q3 + 3*IQR
        limite_superior = Q3 + (3 * IQR)
        # Aseguramos un mínimo razonable (ej. si todo es muy rápido, no cortar a los 2 min)
        limite_superior = max(limite_superior, 20) 
    else:
        limite_superior = 30

    # 4. Clasificar
    # Si Gap > limite_superior -> Marcamos como 'Parada/Outlier'
    df['Es_Parada'] = df['Gap_Min'] > limite_superior
    
    # Tiempo productivo: Si es parada, no sumamos ese tiempo al ciclo
    df['Tiempo_Productivo'] = df['Gap_Min']
    df.loc[df['Es_Parada'], 'Tiempo_Productivo'] = 0
    
    # 5. Crear Bloques de Trabajo
    df['Bloque_ID'] = df['Es_Parada'].cumsum()
    
    return df, limite_superior

# --- INTERFAZ ---
uploaded_file = st.file_uploader("Sube el archivo", type=["xlsx", "xls", "xml", "txt"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
    if df_raw is not None:
        df_clean, cols = mapear_columnas(df_raw)
        if cols:
            # PROCESAMIENTO IA
            df_final, umbral_detectado = procesar_ia_adaptativa(df_clean, cols['Fecha'])
            
            # KPIs
            total_piezas = len(df_final)
            tiempo_total = df_final['Tiempo_Productivo'].sum()
            ct_medio = tiempo_total / total_piezas if total_piezas > 0 else 0
            
            st.success(f"✅ Análisis completado. La IA ha detectado paradas a partir de {umbral_detectado:.1f} minutos.")

            k1, k2, k3 = st.columns(3)
            k1.metric("⏱️ Cycle Time Estimado", f"{ct_medio:.2f} min/ud")
            k2.metric("📊 Datos Procesados", total_piezas)
            k3.metric("🚫 Paradas Detectadas", df_final['Es_Parada'].sum())

            st.divider()

            # --- ANÁLISIS PRODUCTO ---
            st.subheader("🔬 Desglose por Familia y Producto")
            c_prod, c_fam = cols['Producto'], cols['Familia']
            
            resumen = df_final.groupby([c_fam, c_prod]).agg(
                Piezas=('Tiempo_Productivo', 'count'),
                Tiempo_Total=('Tiempo_Productivo', 'sum')
            ).reset_index()
            resumen['CT_Real'] = resumen['Tiempo_Total'] / resumen['Piezas']
            
            st.dataframe(resumen.sort_values('Piezas', ascending=False), use_container_width=True)

            # --- VISUALIZACIÓN ---
            st.subheader("📈 Distribución de Tiempos (Detección de Outliers)")
            fig = px.scatter(df_final, x=cols['Fecha'], y='Gap_Min', color='Es_Parada',
                           color_discrete_map={True: 'red', False: 'green'},
                           title="Puntos Verdes = Producción | Puntos Rojos = Paradas Ignoradas")
            st.plotly_chart(fig, use_container_width=True)

        else: st.error("No se detectó cabecera válida.")
    else: st.error("Error al leer el archivo.")
