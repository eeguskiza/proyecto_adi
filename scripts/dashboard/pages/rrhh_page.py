import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def page_rrhh(filtered: dict, prod: pd.DataFrame) -> None:
    st.subheader("Gestión de Recursos Humanos")
    
    rrhh = filtered.get("rrhh", pd.DataFrame())
    
    if rrhh.empty:
        st.info("Sin datos de RRHH en el rango seleccionado.")
        return

    if pd.api.types.is_string_dtype(rrhh["año_mes"]):
        rrhh["año_mes"] = pd.to_datetime(rrhh["año_mes"])

    # Calculos de totales
    total_teoricas = rrhh["horas_teoricas"].sum()
    total_netas = rrhh["horas_netas"].sum()
    
    total_absentismo = (
        rrhh["horas_enfermedad"].sum() + 
        rrhh["horas_accidente"].sum() + 
        rrhh["horas_permiso"].sum()
    )
    
    base_calculo = rrhh["horas_ajustadas"].sum() if "horas_ajustadas" in rrhh.columns else total_teoricas
    tasa_absentismo = (total_absentismo / base_calculo) if base_calculo > 0 else 0
    tasa_disponibilidad = (total_netas / total_teoricas) if total_teoricas > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Horas Disponibles (Netas)", f"{total_netas:,.0f} h")
    k2.metric("Horas Perdidas (Absentismo)", f"{total_absentismo:,.0f} h", delta=f"-{total_absentismo:,.0f} h", delta_color="inverse")
    k3.metric("Tasa Absentismo Global", f"{tasa_absentismo:.1%}", help="Total Absentismo / Horas Planificadas")
    k4.metric("Disponibilidad Total", f"{tasa_disponibilidad:.1%}", help="Horas Netas / Horas Teóricas")

    st.markdown("---")

    # Analisis de Capacidad
    st.markdown("### Análisis de Capacidad: De Teóricas a Netas")
    
    sum_data = rrhh.sum(numeric_only=True)
    
    val_teoricas = sum_data.get("horas_teoricas", 0)
    val_tco = sum_data.get("reduccion_tco", 0)
    val_enf = sum_data.get("horas_enfermedad", 0)
    val_acc = sum_data.get("horas_accidente", 0)
    val_perm = sum_data.get("horas_permiso", 0)
    val_netas = val_teoricas - val_tco - val_enf - val_acc - val_perm
    
    fig_waterfall = go.Figure(go.Waterfall(
        name="Horas",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
        x=["Teóricas", "TCO (Ajuste)", "Enfermedad", "Accidentes", "Permisos", "Disponibles (Netas)"],
        textposition="auto",
        text=[
            f"{val_teoricas:,.0f}",
            f"-{val_tco:,.0f}",
            f"-{val_enf:,.0f}",
            f"-{val_acc:,.0f}",
            f"-{val_perm:,.0f}",
            f"{val_netas:,.0f}"
        ],
        y=[
            val_teoricas,
            -val_tco,
            -val_enf,
            -val_acc,
            -val_perm,
            val_netas
        ],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#ef4444"}},
        increasing={"marker": {"color": "#22c55e"}},
        totals={"marker": {"color": "#10b981"}}
    ))

    fig_waterfall.update_layout(
        title="Cascada de Disponibilidad (Teóricas - TCO - Absentismo = Netas)",
        showlegend=False,
        height=450,
        yaxis_title="Horas Totales",
        yaxis=dict(range=[val_netas * 0.9, val_teoricas * 1.05])
    )
    st.plotly_chart(fig_waterfall, width='stretch')

    # Preparacion de datos cruzados
    df_prod_mes = pd.DataFrame()
    if not prod.empty:
        prod_calc = prod.copy()
        prod_calc["ts_ini"] = pd.to_datetime(prod_calc["ts_ini"])
        prod_calc["año_mes"] = prod_calc["ts_ini"].dt.to_period("M").dt.to_timestamp()
        
        df_prod_mes = prod_calc.groupby("año_mes").agg(
            piezas_ok=("piezas_ok", "sum"),
            duracion_min=("duracion_min", "sum")
        ).reset_index()

    rrhh["año_mes"] = pd.to_datetime(rrhh["año_mes"])
    df_merged = rrhh.merge(df_prod_mes, on="año_mes", how="left").fillna(0)
    
    df_merged["productividad"] = df_merged.apply(
        lambda row: row["piezas_ok"] / row["horas_netas"] if row["horas_netas"] > 0 else 0, 
        axis=1
    )

    # Graficos de evolucion
    st.markdown("### Evolución Mensual y Productividad")
    c1, c2 = st.columns(2)
    
    df_absentismo = rrhh.melt(
        id_vars=["año_mes"], 
        value_vars=["horas_enfermedad", "horas_accidente", "horas_permiso"],
        var_name="Tipo", 
        value_name="Horas"
    )
    
    fig_abs = px.bar(
        df_absentismo, 
        x="año_mes", 
        y="Horas", 
        color="Tipo", 
        title="Evolución del Absentismo (Horas perdidas)",
        color_discrete_map={
            "horas_enfermedad": "#f59e0b",
            "horas_accidente": "#ef4444",
            "horas_permiso": "#3b82f6"
        }
    )
    c1.plotly_chart(fig_abs, width='stretch')

    fig_prod = go.Figure()
    fig_prod.add_trace(go.Bar(
        x=df_merged["año_mes"],
        y=df_merged["piezas_ok"],
        name="Piezas OK",
        marker_color="#cbd5e1",
        opacity=0.6
    ))
    fig_prod.add_trace(go.Scatter(
        x=df_merged["año_mes"],
        y=df_merged["productividad"],
        name="Piezas / Hora Neta",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="#22c55e", width=3)
    ))
    fig_prod.update_layout(
        title="Productividad Laboral (Piezas vs Eficiencia)",
        yaxis=dict(title="Volumen Piezas OK"),
        yaxis2=dict(title="Piezas / Hora Neta", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        hovermode="x unified"
    )
    c2.plotly_chart(fig_prod, width='stretch')

    st.markdown("---")

    # Analisis de saturacion
    st.markdown("### Saturación de la Plantilla")

    if not prod.empty:
        df_merged["horas_prod"] = df_merged["duracion_min"] / 60
        df_merged["horas_gap"] = df_merged["horas_netas"] - df_merged["horas_prod"]
        df_merged["pct_uso"] = df_merged.apply(
            lambda x: x["horas_prod"] / x["horas_netas"] if x["horas_netas"] > 0 else 0, axis=1
        )

        c3, c4 = st.columns(2)

        fig_sat = go.Figure()
        fig_sat.add_trace(go.Bar(
            x=df_merged["año_mes"],
            y=df_merged["horas_netas"],
            name="Horas Disponibles (RRHH)",
            marker_color="#94a3b8"
        ))
        fig_sat.add_trace(go.Bar(
            x=df_merged["año_mes"],
            y=df_merged["horas_prod"],
            name="Horas Imputadas (Producción)",
            marker_color="#3b82f6"
        ))
        fig_sat.update_layout(
            title="Comparativa: Disponibilidad vs Uso Real",
            barmode="overlay",
            yaxis_title="Horas",
            legend=dict(orientation="h", y=1.1)
        )
        c3.plotly_chart(fig_sat, width='stretch')

        fig_gap = px.area(
            df_merged, 
            x="año_mes", 
            y="pct_uso", 
            markers=True,
            title="% Saturación (Horas Prod / Horas Netas)"
        )
        fig_gap.layout.yaxis.tickformat = ',.0%'
        fig_gap.add_hline(y=0.85, line_dash="dot", annotation_text="Objetivo (85%)", line_color="green")
        c4.plotly_chart(fig_gap, width='stretch')
        
        total_gap = df_merged["horas_gap"].sum()
        st.caption(f"Diferencia acumulada: {total_gap:,.0f} horas pagadas no imputadas a órdenes de producción en el periodo.")

    with st.expander("Ver datos detallados mensuales"):
        cols_orden = ["año_mes", "horas_teoricas", "horas_netas", "horas_prod", "horas_enfermedad", "productividad", "pct_uso"]
        cols_final = [c for c in cols_orden if c in df_merged.columns]
        
        st.dataframe(
            df_merged[cols_final].style.format({
                "productividad": "{:.2f}", 
                "horas_netas": "{:,.0f}",
                "horas_prod": "{:,.0f}",
                "pct_uso": "{:.1%}"
            }),
            width='stretch'
        )