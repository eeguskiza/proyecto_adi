import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..oee import compute_oee, compute_oee_daily, render_kpi_cards


def page_dashboard(filtered: dict, ciclos: pd.DataFrame, recurso_sel: str) -> None:
    st.subheader("Cuadro de mando general")
    prod_all = filtered["produccion"]
    if prod_all.empty:
        st.info("Sin datos en el rango seleccionado.")
        return

    prod = prod_all if recurso_sel == "(Todos)" else prod_all[prod_all["machine_name"] == recurso_sel]
    if "estado_oee" not in prod.columns:
        prod = prod.copy()
        prod["estado_oee"] = prod["evento"].str.lower().map(
            {"producción": "produccion", "produccion": "produccion", "preparación": "preparacion", "preparacion": "preparacion"}
        )
        prod["estado_oee"] = prod["estado_oee"].fillna("incidencia")

    oee_data = compute_oee(prod, ciclos)
    with st.container():
        render_kpi_cards(oee_data)

    # Información de rendimiento (debug)
    with st.expander("Informacion de calculo de rendimiento"):
        st.markdown("**Metodo de calculo**: Basado en tiempo de ciclo real vs objetivo (percentil 25 historico)")

        col_d1, col_d2, col_d3, col_d4 = st.columns(4)

        uph_real = oee_data.get("uph_real", np.nan)
        uph_teorico = oee_data.get("uph_teorico", np.nan)
        actual_output = oee_data.get("actual_output", 0)
        ideal_output = oee_data.get("ideal_output", 0)
        performance = oee_data.get("performance", np.nan)

        # Calcular tiempos de ciclo equivalentes
        cycle_time_real = (3600 / uph_real) if pd.notna(uph_real) and uph_real > 0 else np.nan
        cycle_time_teorico = (3600 / uph_teorico) if pd.notna(uph_teorico) and uph_teorico > 0 else np.nan

        col_d1.metric("UPH Real", f"{uph_real:.1f}" if pd.notna(uph_real) else "N/A",
                     help=f"Tiempo de ciclo: {cycle_time_real:.1f} seg/pieza" if pd.notna(cycle_time_real) else "N/A")
        col_d2.metric("UPH Objetivo", f"{uph_teorico:.1f}" if pd.notna(uph_teorico) else "N/A",
                     help=f"Tiempo de ciclo objetivo: {cycle_time_teorico:.1f} seg/pieza" if pd.notna(cycle_time_teorico) else "N/A")
        col_d3.metric("Piezas Reales", f"{actual_output:,.0f}")
        col_d4.metric("Piezas Ideales", f"{ideal_output:,.0f}")

        st.markdown("---")

        col_info1, col_info2 = st.columns(2)

        with col_info1:
            if pd.notna(cycle_time_real) and pd.notna(cycle_time_teorico):
                st.markdown(f"""
                **Tiempo de ciclo promedio ponderado**:
                - Real: **{cycle_time_real:.1f} seg/pieza**
                - Objetivo: **{cycle_time_teorico:.1f} seg/pieza**
                - Diferencia: **{(cycle_time_real - cycle_time_teorico):.1f} seg/pieza**
                """)

        with col_info2:
            if pd.notna(performance):
                if performance > 1.0:
                    st.success(f"**Rendimiento: {performance:.1%}**\n\nSuperando objetivo (Excelente)")
                elif performance >= 0.90:
                    st.success(f"**Rendimiento: {performance:.1%}**\n\nExcelente desempeno")
                elif performance >= 0.80:
                    st.info(f"**Rendimiento: {performance:.1%}**\n\nBuen desempeno")
                elif performance >= 0.60:
                    st.warning(f"**Rendimiento: {performance:.1%}**\n\nMejorable")
                else:
                    st.error(f"**Rendimiento: {performance:.1%}**\n\nBajo rendimiento")
            else:
                st.warning("No hay suficientes datos para calcular el rendimiento")

        st.markdown("---")
        st.caption("El objetivo se calcula usando el percentil 25 de los tiempos de ciclo historicos por maquina y referencia. Esto representa un objetivo ambicioso pero alcanzable basado en los mejores desempenos pasados.")

    # Gráfico de evolución temporal del OEE
    st.markdown("---")
    st.markdown("### Evolucion Temporal del OEE")
    daily_oee = compute_oee_daily(prod, ciclos)
    if not daily_oee.empty:
        col_ev1, col_ev2 = st.columns(2)

        # Filtrar NaN y asegurar que fecha sea del tipo correcto
        daily_oee_clean = daily_oee.dropna(subset=["oee"]).copy()

        if not daily_oee_clean.empty:
            # Gráfico de OEE diario
            fig_oee_trend = px.line(
                daily_oee_clean,
                x="fecha",
                y="oee",
                markers=True,
                title="OEE Diario",
                labels={"oee": "OEE", "fecha": "Fecha"}
            )
            fig_oee_trend.add_hline(y=0.75, line=dict(dash="dash", color="green"), annotation_text="Objetivo 75%")
            fig_oee_trend.update_traces(
                line_color="#3b82f6",
                line_width=3,
                marker=dict(size=8, line=dict(width=2, color="white"))
            )
            fig_oee_trend.update_layout(
                yaxis_tickformat=".0%",
                yaxis_range=[0, 1],
                hovermode='x unified'
            )
            col_ev1.plotly_chart(fig_oee_trend, use_container_width=True)

            # Gráfico de componentes del OEE
            daily_melted = daily_oee_clean.melt(
                id_vars=["fecha"],
                value_vars=["availability", "performance", "quality"],
                var_name="componente",
                value_name="valor"
            )
            daily_melted["componente"] = daily_melted["componente"].map({
                "availability": "Disponibilidad",
                "performance": "Rendimiento",
                "quality": "Calidad"
            })
            # Filtrar NaN en los valores
            daily_melted = daily_melted.dropna(subset=["valor"])

            fig_components = px.line(
                daily_melted,
                x="fecha",
                y="valor",
                color="componente",
                markers=True,
                title="Componentes del OEE (Diario)",
                labels={"valor": "Valor", "fecha": "Fecha", "componente": "Componente"}
            )
            fig_components.update_traces(line_width=2, marker=dict(size=6))
            fig_components.update_layout(
                yaxis_tickformat=".0%",
                yaxis_range=[0, 1],
                hovermode='x unified'
            )
            col_ev2.plotly_chart(fig_components, use_container_width=True)
        else:
            col_ev1.warning("No hay datos válidos de OEE para graficar")
            col_ev2.warning("No hay datos válidos de componentes OEE para graficar")

    st.markdown("---")
    st.markdown("### Analisis Detallado")

    # Gráfico Waterfall de pérdidas OEE
    col_w1, col_w2, col_w3 = st.columns([2, 1, 1])

    avail = oee_data["availability"] if pd.notna(oee_data["availability"]) else 0
    perf = oee_data["performance"] if pd.notna(oee_data["performance"]) else 0
    qual = oee_data["quality"] if pd.notna(oee_data["quality"]) else 0
    oee_final = oee_data["oee"] if pd.notna(oee_data["oee"]) else 0

    # Pérdidas en puntos porcentuales
    perdida_disp = 1 - avail
    perdida_perf = avail * (1 - perf)
    perdida_qual = avail * perf * (1 - qual)

    fig_waterfall = go.Figure(go.Waterfall(
        name="OEE",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Inicial", "Pérdida Disponibilidad", "Pérdida Rendimiento", "Pérdida Calidad", "OEE Final"],
        y=[1, -perdida_disp, -perdida_perf, -perdida_qual, oee_final],
        text=[f"{1:.0%}", f"-{perdida_disp:.1%}", f"-{perdida_perf:.1%}", f"-{perdida_qual:.1%}", f"{oee_final:.1%}"],
        textposition="outside",
        connector={"line": {"color": "#64748b"}},
        decreasing={"marker": {"color": "#ef4444"}},
        increasing={"marker": {"color": "#22c55e"}},
        totals={"marker": {"color": "#3b82f6"}},
    ))
    fig_waterfall.update_layout(
        title="Cascada de Pérdidas del OEE",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.1],
        showlegend=False,
        height=400
    )
    col_w1.plotly_chart(fig_waterfall, use_container_width=True)

    # Distribución del tiempo
    dur = oee_data["durations"]
    disp_df = pd.DataFrame(
        {
            "estado": ["Producción", "Preparación", "Incidencias"],
            "minutos": [dur.get("produccion", 0), dur.get("preparacion", 0), dur.get("incidencia", 0)],
        }
    )
    disp_df["barra"] = "Disponibilidad"

    fig_disp = px.bar(
        disp_df,
        x="minutos",
        y="barra",
        color="estado",
        orientation="h",
        text="minutos",
        title="Distribución del tiempo",
        color_discrete_map={"Producción": "#22c55e", "Preparación": "#fbbf24", "Incidencias": "#ef4444"},
    )
    fig_disp.update_layout(barmode="stack", xaxis_title="Minutos", yaxis_title="", height=400)
    fig_disp.update_traces(texttemplate="%{text:.0f} min", textposition="inside")
    col_w2.plotly_chart(fig_disp, use_container_width=True)

    # Distribución de calidad
    calidad_df = pd.DataFrame(
        {
            "tipo": ["OK", "Scrap"],
            "piezas": [oee_data["total_ok"], oee_data["total_scrap"]],
        }
    )
    fig_calidad = px.pie(calidad_df, values="piezas", names="tipo", title="Distribución de calidad", color="tipo")
    fig_calidad.update_layout(height=400)
    col_w3.plotly_chart(fig_calidad, use_container_width=True)

    prod_perf = prod[prod["estado_oee"] == "produccion"].copy()
    ciclos_ref_machine = ciclos[["ref_id_str", "machine_name", "piezas_hora_teorico"]].drop_duplicates()
    prod_perf = prod_perf.merge(ciclos_ref_machine, on=["ref_id_str", "machine_name"], how="left")

    hist_perf = (
        prod_perf.groupby(["machine_name", "ref_id_str"])
        .agg(piezas=("piezas_ok", "sum"), dur_min=("duracion_min", "sum"))
        .reset_index()
    )
    hist_perf = hist_perf[hist_perf["dur_min"] > 0]
    hist_perf["uph_hist"] = hist_perf["piezas"] / (hist_perf["dur_min"] / 60)
    hist_perf["uph_hist"] = hist_perf["uph_hist"].clip(30, 180)
    prod_perf = prod_perf.merge(hist_perf[["machine_name", "ref_id_str", "uph_hist"]], on=["machine_name", "ref_id_str"], how="left")

    prod_perf["piezas_hora_teorico"] = prod_perf["piezas_hora_teorico"].replace(0, pd.NA)
    prod_perf["piezas_hora_teorico"] = prod_perf["piezas_hora_teorico"].fillna(prod_perf["uph_hist"])
    prod_perf["piezas_hora_teorico"] = prod_perf["piezas_hora_teorico"].fillna(80)
    prod_perf = prod_perf.drop(columns=["uph_hist"])

    prod_perf["ideal_piezas"] = prod_perf["piezas_hora_teorico"] * (prod_perf["duracion_min"] / 60)
    prod_perf["fecha"] = prod_perf["ts_ini"].dt.date

    perf_diaria = (
        prod_perf.groupby("fecha")
        .agg(
            piezas_ok=("piezas_ok", "sum"),
            duracion_min=("duracion_min", "sum"),
            ideal_piezas=("ideal_piezas", "sum"),
        )
        .reset_index()
    )
    perf_diaria = perf_diaria[perf_diaria["duracion_min"] > 0]
    perf_diaria["uph_real"] = perf_diaria["piezas_ok"] / (perf_diaria["duracion_min"] / 60)
    perf_diaria["uph_ideal"] = np.where(
        perf_diaria["duracion_min"] > 0, perf_diaria["ideal_piezas"] / (perf_diaria["duracion_min"] / 60), np.nan
    )
    perf_diaria["uph_real"] = pd.to_numeric(perf_diaria["uph_real"], errors="coerce")
    perf_diaria["uph_ideal"] = pd.to_numeric(perf_diaria["uph_ideal"], errors="coerce")

    prod_inci = prod.copy()
    prod_inci["estado_oee"] = prod_inci["evento"].str.lower().map(
        {"producción": "produccion", "produccion": "produccion", "preparación": "preparacion", "preparacion": "preparacion"}
    )
    prod_inci["estado_oee"] = prod_inci["estado_oee"].fillna("incidencia")
    incidencias = prod_inci[prod_inci["estado_oee"] == "incidencia"]
    fig_inci = None
    if not incidencias.empty:
        top_inci = (
            incidencias.groupby("tipo_incidencia")["duracion_min"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        # Calcular el porcentaje acumulado para Pareto
        top_inci["pct"] = (top_inci["duracion_min"] / top_inci["duracion_min"].sum()) * 100
        top_inci["pct_acum"] = top_inci["pct"].cumsum()

        # Crear gráfico de Pareto con barras y línea
        fig_inci = go.Figure()

        # Barras de tiempo
        fig_inci.add_trace(go.Bar(
            x=top_inci["tipo_incidencia"],
            y=top_inci["duracion_min"],
            name="Tiempo perdido",
            marker_color="#ef4444",
            yaxis="y",
            text=top_inci["duracion_min"].apply(lambda x: f"{x:.0f} min"),
            textposition="outside"
        ))

        # Línea de porcentaje acumulado
        fig_inci.add_trace(go.Scatter(
            x=top_inci["tipo_incidencia"],
            y=top_inci["pct_acum"],
            name="% Acumulado",
            mode="lines+markers",
            marker=dict(color="#fbbf24", size=8),
            line=dict(color="#fbbf24", width=3),
            yaxis="y2"
        ))

        # Línea del 80% para identificar el principio de Pareto
        fig_inci.add_hline(
            y=80, line=dict(dash="dash", color="#22c55e"),
            annotation_text="80%", yref="y2"
        )

        fig_inci.update_layout(
            title="Análisis de Pareto - Incidencias (80/20)",
            xaxis_title="Tipo de Incidencia",
            yaxis=dict(title="Minutos perdidos", side="left"),
            yaxis2=dict(
                title="% Acumulado",
                overlaying="y",
                side="right",
                range=[0, 105],
                tickformat=".0f",
                ticksuffix="%"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450
        )

    st.markdown("---")
    st.markdown("### Heatmap de Disponibilidad por Maquina")

    # Calcular disponibilidad por máquina y día
    if recurso_sel == "(Todos)":
        prod_heat = prod.copy()
        prod_heat["fecha"] = prod_heat["ts_ini"].dt.date

        # Calcular disponibilidad por máquina y día
        disp_by_machine_day = []
        for machine in prod_heat["machine_name"].unique():
            if pd.notna(machine):
                for fecha in prod_heat["fecha"].unique():
                    prod_subset = prod_heat[(prod_heat["machine_name"] == machine) & (prod_heat["fecha"] == fecha)]
                    if not prod_subset.empty:
                        dur_prod = prod_subset[prod_subset["estado_oee"] == "produccion"]["duracion_min"].sum()
                        dur_total = prod_subset["duracion_min"].sum()
                        disponibilidad = dur_prod / dur_total if dur_total > 0 else 0
                        disp_by_machine_day.append({
                            "machine": machine,
                            "fecha": fecha,
                            "disponibilidad": disponibilidad
                        })

        if disp_by_machine_day:
            disp_df = pd.DataFrame(disp_by_machine_day)

            # Crear pivot para el heatmap
            pivot_disp = disp_df.pivot(index="machine", columns="fecha", values="disponibilidad")

            # Crear heatmap
            fig_heatmap = px.imshow(
                pivot_disp.values,
                labels=dict(x="Fecha", y="Máquina", color="Disponibilidad"),
                x=[str(d) for d in pivot_disp.columns],
                y=pivot_disp.index.tolist(),
                color_continuous_scale=["#ef4444", "#fbbf24", "#22c55e"],
                aspect="auto",
                title="Disponibilidad por Máquina y Día"
            )
            fig_heatmap.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Máquina",
                coloraxis_colorbar=dict(
                    title="Disponibilidad",
                    tickformat=".0%"
                ),
                height=400
            )
            fig_heatmap.update_traces(
                hovertemplate="Máquina: %{y}<br>Fecha: %{x}<br>Disponibilidad: %{z:.1%}<extra></extra>"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("El heatmap de disponibilidad solo se muestra cuando se seleccionan todas las máquinas.")

    st.markdown("---")
    st.markdown("### Analisis por Turno")

    # Calcular OEE por turno
    if "turno" in prod.columns:
        turnos_oee = []
        for turno in ["mañana", "tarde", "noche"]:
            prod_turno = prod[prod["turno"] == turno]
            if not prod_turno.empty:
                oee_turno = compute_oee(prod_turno, ciclos)
                turnos_oee.append({
                    "turno": turno.capitalize(),
                    "oee": oee_turno["oee"],
                    "disponibilidad": oee_turno["availability"],
                    "rendimiento": oee_turno["performance"],
                    "calidad": oee_turno["quality"],
                    "piezas": oee_turno["total_piezas"],
                })

        if turnos_oee:
            turnos_df = pd.DataFrame(turnos_oee)
            col_t1, col_t2 = st.columns(2)

            # Gráfico de barras comparando OEE por turno
            fig_turno_oee = px.bar(
                turnos_df,
                x="turno",
                y="oee",
                title="OEE por Turno",
                labels={"oee": "OEE", "turno": "Turno"},
                color="oee",
                color_continuous_scale=["#ef4444", "#fbbf24", "#22c55e"],
                text="oee"
            )
            fig_turno_oee.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_turno_oee.update_layout(yaxis_tickformat=".0%", showlegend=False, height=350)
            fig_turno_oee.add_hline(y=0.75, line=dict(dash="dash", color="green"), annotation_text="Objetivo")
            col_t1.plotly_chart(fig_turno_oee, use_container_width=True)

            # Gráfico de componentes por turno
            turnos_melted = turnos_df.melt(
                id_vars=["turno"],
                value_vars=["disponibilidad", "rendimiento", "calidad"],
                var_name="componente",
                value_name="valor"
            )
            turnos_melted["componente"] = turnos_melted["componente"].str.capitalize()

            fig_turno_comp = px.bar(
                turnos_melted,
                x="turno",
                y="valor",
                color="componente",
                barmode="group",
                title="Componentes del OEE por Turno",
                labels={"valor": "Valor", "turno": "Turno", "componente": "Componente"}
            )
            fig_turno_comp.update_layout(yaxis_tickformat=".0%", height=350)
            col_t2.plotly_chart(fig_turno_comp, use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns((2, 1))
    if not perf_diaria.empty:
        c3.plotly_chart(
            px.line(
                perf_diaria,
                x="fecha",
                y=["uph_real", "uph_ideal"],
                markers=True,
                labels={"value": "Piezas/hora", "variable": "Serie"},
                title="Rendimiento real vs. ideal",
            ),
            use_container_width=True,
            config={'displayModeBar': False}
        )
    if fig_inci:
        c4.plotly_chart(fig_inci, use_container_width=True)
