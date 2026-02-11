import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Celestica Diagnóstico", layout="wide")
st.title("🕵️‍♂️ Modo Diagnóstico Forense")
st.warning("Esta pantalla es solo para averiguar qué formato tiene tu archivo rebelde.")

uploaded_file = st.file_uploader("Sube el archivo maldito (.xls)", type=["xlsx", "xls", "txt", "xml"])

if uploaded_file:
    st.subheader("1. 🔍 Inspección de 'Tripas' (Magic Bytes)")
    
    # Leemos los primeros 1000 caracteres del archivo tal cual (en crudo)
    try:
        uploaded_file.seek(0)
        # Intentamos leer como texto
        content_text = uploaded_file.read(1000).decode('latin-1', errors='ignore')
        st.code(content_text, language='html')
        
        st.subheader("2. 🧠 Análisis de la IA sobre el formato")
        
        if "MIME-Version" in content_text:
            st.error("🚨 ¡DETECTADO! Este archivo es un 'MHTML Web Archive'.")
            st.info("💡 Solución: Python no puede leer esto directo. Abrelo en Excel, dale a 'Guardar como' y elige 'Libro de Excel (.xlsx)'.")
            
        elif "<html" in content_text.lower() or "<!DOCTYPE html" in content_text:
            st.success("✅ Es un archivo HTML. Deberíamos poder leerlo con 'pd.read_html'.")
            try:
                uploaded_file.seek(0)
                dfs = pd.read_html(uploaded_file.getvalue())
                st.write(f"🎉 ¡Éxito! He encontrado {len(dfs)} tablas dentro.")
                st.dataframe(dfs[0].head())
            except Exception as e:
                st.error(f"Parece HTML pero falló al extraer tabla: {e}")

        elif "<?xml" in content_text:
            st.success("✅ Es un archivo XML puro (XML Spreadsheet).")
            try:
                uploaded_file.seek(0)
                df = pd.read_xml(uploaded_file.getvalue())
                st.dataframe(df.head())
            except:
                st.warning("Fallo lectura XML directa. Intentando como texto...")

        elif "PK" in content_text[0:2]:
            st.success("✅ Es un Excel Real (.xlsx comprimido).")
            
        else:
            st.warning("⚠️ Formato desconocido (Posiblemente Texto plano o Binario antiguo).")
            st.text("Intentando leer como CSV separado por tabulaciones...")
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep='\t', encoding='latin-1', on_bad_lines='skip')
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Tampoco funcionó como CSV: {e}")

    except Exception as e:
        st.error(f"Error fatal leyendo el archivo: {e}")
