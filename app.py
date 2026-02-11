import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import xml.etree.ElementTree as ET # Importante para leer tu archivo

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Celestica AI Ultimate", layout="wide", page_icon="🛡️")
st.title("🛡️ Celestica IA: Advance Tracking Analyzer")

with st.sidebar:
    st.header("⚙️ Configuración")
    contamination = st.slider("Sensibilidad IA (% Ruido)", 1, 25, 5) / 100
    st.divider()
    h_turno = st.number_input("Horas Turno", value=8)
    m_descanso = st.number_input("Minutos Descanso", value=45)
    eficiencia = st.slider("Eficiencia %", 50, 100, 75) / 100

# --- FUNCIÓN ESPECIAL: LECTOR XML 2003 (La llave maestra) ---
def leer_xml_2003(file):
    """Lee archivos 'XML Spreadsheet 2003' que fallan con otros métodos."""
    try:
        tree = ET.parse(file)
        root = tree.getroot()
        # Definimos el 'namespace' que usa Microsoft
        ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        
        datos = []
        # Buscamos todas las filas
        for row in root.findall('.//ss:Row', ns):
            fila_datos = []
            # Buscamos todas las celdas de esa fila
            for cell in row.findall('.//ss:Cell', ns):
                data_tag = cell.find('ss:Data', ns)
                if data_tag is not None:
                    fila_datos.append(data_tag.text)
                else:
                    fila_datos.append("") # Celda vacía
            if fila_datos:
                datos.append(fila_datos)
        
        return pd.DataFrame(datos)
    except Exception as e:
        return None

# --- GESTOR DE CARGA ---
@st.cache_data(ttl=3600)
def load_data(file):
    # 1. INTENTO: XML 2003 (Tu caso específico)
    file.seek(0)
    # Leemos la cabecera para ver si es XML
    try:
        head = file.read(100).decode('latin-1', errors='ignore')
        file.seek(0)
        if "<?xml" in head and "Workbook" in head:
            return leer_xml_2003(file)
    except:
        pass

    # 2. INTENTO: Calamine (Excel corrupto normal)
    try:
        return pd.read_excel(file, engine='calamine', header=None)
    except:
        pass

    # 3. INTENTO: HTML (Descargas web)
    try:
        file.seek(0)
        dfs = pd.read_html(file, header=None)
        if len(dfs)
