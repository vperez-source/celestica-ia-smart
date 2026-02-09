import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Celestica IA", layout="wide")

# --- CABECERA ---
st.title("🛡️ Celestica IA: Dashboard de Operaciones")
st.markdown("---")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("🧠 Inteligencia Artificial")
    contamination = st.slider("Sensibilidad IA (Ruido %)", 1, 20, 5, help="Más alto = Borra más datos.") / 100
    
    st.divider()
    st.header("⚙️ Turno")
    h_turno = st.number_input("Horas Turno", value=8)
    m_descanso = st.number_input("Minutos Descanso", value=45)
    eficiencia = st.slider("Eficiencia Objetivo %", 50, 100, 75) / 100

# --- CARGA ---
uploaded_file = st.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        
        # Limpieza básica
        df.columns = df.columns.astype(str).str.strip()
        col_target = 'In DateTime'
        
        if col_target in df.columns:
            # Procesar Fechas
            df[col_target] = pd.to_datetime(df[col_target], errors='coerce')
            df.loc[df[col_target].dt.year < 100, col_target] += pd.offsets.DateOffset(years=2000)
            df = df.dropna(subset=[col_target]).sort_values(col_target)
            
            # Calcular Tiempos
            df['gap_mins'] = df.groupby('Station')[col_target].diff().dt.total_seconds() / 60
            df['gap_mins'] = df['gap_mins'].fillna(df['gap_mins'].median())
            
            # --- CEREBRO IA ---
            model = IsolationForest(contamination=contamination, random_state=42)
            df['IA_Status'] = model.fit_predict(df[['gap_mins']])
            
            # Datos Limpios vs Ruido
            df_clean = df[df['IA_Status'] == 1].copy()
            df_noise = df[df['IA_Status'] == -1]

            # --- MÉTRICAS GLOBALES ---
            media_global = df_clean['gap_mins'].mean()
            capacidad = ((h_turno*60 - m_descanso)/media_global) * eficiencia
            
            # Visualización KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("⏱️ Cycle Time Global", f"{media_global:.2f} min", delta="Promedio Planta")
            k2.metric("📦 Capacidad", f"{int(capacidad)} uds")
            k3.metric("✅ Piezas Válidas", len(df_clean))
            k4.metric("🗑️ Ruido Eliminado", len(df_noise), delta_color="inverse")

            st.markdown("---")

            # --- NUEVA SECCIÓN: ANÁLISIS POR USUARIO ---
            if 'User' in df.columns:
                st.subheader("🏆 Ranking de Operarios")
                
                # Agrupamos por usuario y calculamos estadísticas
                user_stats = df_clean.groupby('User')['gap_mins'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
                
                # Renombramos columnas para que se vea bonito
                user_stats.columns = ['Operario', 'Piezas', 'CT Medio (min)', 'Estabilidad (Std)', 'Min', 'Max']
                
                # Redondeamos decimales
                user_stats = user_stats.round(2)
                
                # Ordenamos por quién ha hecho más piezas (Productividad)
                user_stats = user_stats.sort_values('Piezas', ascending=False).reset_index(drop=True)

                # Mostramos la tabla interactiva
                c_table, c_chart = st.columns([1, 1])
                
                with c_table:
                    st.dataframe(
                        user_stats.style.background_gradient(subset=['Piezas'], cmap='Greens')
                                  .background_gradient(subset=['CT Medio (min)'], cmap='Reds'),
                        use_container_width=True
                    )
                    st.caption("*Estabilidad: Cuanto más bajo el número, más constante es el ritmo del operario.")

                with c_chart:
                    # Gráfico Combinado: Barras (Piezas) y Línea (Tiempo)
                    fig_combo = go.Figure()
                    
                    # Barras de producción
                    fig_combo.add_trace(go.Bar(
                        x=user_stats['Operario'], 
                        y=user_stats['Piezas'],
                        name='Piezas Realizadas',
                        marker_color='#2ecc71'
                    ))
                    
                    # Línea de Tiempo de Ciclo
                    fig_combo.add_trace(go.Scatter(
                        x=user_stats['Operario'], 
                        y=user_stats['CT Medio (min)'],
                        name='Cycle Time (min)',
                        yaxis='y2',
                        line=dict(color='#e74c3c', width=3)
                    ))

                    fig_combo.update_layout(
                        title="Productividad vs Velocidad",
                        yaxis=dict(title="Cantidad de Piezas"),
                        yaxis2=dict(title="Minutos por Pieza", overlaying='y', side='right'),
                        legend=dict(x=0.1, y=1.1, orientation='h')
                    )
                    st.plotly_chart(fig_combo, use_container_width=True)

            else:
                st.warning("No encontré la columna 'User' en tu Excel para hacer el ranking.")

            # --- GRÁFICO DE DISPERSIÓN (El mapa de puntos) ---
            st.markdown("---")
            st.subheader("🔍 Detalle de Puntos (IA)")
            fig_scatter = px.scatter(df, x=col_target, y='gap_mins', 
                                   color=df['IA_Status'].astype(str),
                                   color_discrete_map={'1':'#2ecc71', '-1':'#e74c3c'},
                                   title="Mapa de Producción (Verde=Válido | Rojo=Ruido)")
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Botón Descarga
            st.download_button("📥 Descargar Datos de Tabla", user_stats.to_csv(index=False).encode('utf-8'), "ranking_operarios.csv")

        else:
            st.error("Falta la columna 'In DateTime'.")

    except Exception as e:
        st.error(f"Error: {e}")
