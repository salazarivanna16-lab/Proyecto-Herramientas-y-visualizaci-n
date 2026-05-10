import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURACION DE PAGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Inteligencia de Marketing | ADM-3083",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# PALETA DE COLORES CORPORATIVA
# ---------------------------------------------------------------------------
COLOR = {
    "azul_oscuro": "#1C2B4A",
    "rojo":        "#B03A2E",
    "dorado":      "#C9A84C",
    "verde":       "#1A7A6E",
    "gris":        "#6C7A89",
    "fondo":       "#F5F6FA",
    "blanco":      "#FFFFFF",
    "cat": ["#1C2B4A", "#B03A2E", "#C9A84C", "#1A7A6E",
            "#6C7A89", "#5B2C6F", "#1F618D", "#784212"]
}

# ---------------------------------------------------------------------------
# ESTILOS CSS GLOBALES
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"] {{
        background-color: {COLOR['fondo']};
    }}
    .block-container {{
        padding-top: 1.8rem;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {COLOR['azul_oscuro']} !important;
        font-family: Georgia, serif;
    }}
    p, span, label, div {{
        color: {COLOR['azul_oscuro']} !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {COLOR['azul_oscuro']};
    }}
    [data-testid="stSidebar"] * {{
        color: {COLOR['blanco']} !important;
    }}
    [data-testid="stMetric"] {{
        background-color: {COLOR['blanco']};
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
        border-top: 4px solid {COLOR['azul_oscuro']};
    }}
    [data-testid="stMetricLabel"] p {{
        font-size: 0.82rem;
        font-weight: 600;
        color: {COLOR['gris']} !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {COLOR['azul_oscuro']} !important;
    }}
    [data-testid="stAlert"] {{
        background-color: #EEF2FB !important;
        border-radius: 6px;
    }}
    [data-testid="stAlert"] * {{
        color: {COLOR['azul_oscuro']} !important;
    }}
    .encabezado-seccion {{
        background-color: {COLOR['azul_oscuro']};
        color: {COLOR['blanco']} !important;
        padding: 8px 18px;
        border-radius: 5px;
        font-size: 0.92rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 24px 0 14px 0;
    }}
    .encabezado-seccion * {{
        color: {COLOR['blanco']} !important;
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# FUNCION DE ESTILO UNIFORME PARA GRAFICOS
# ---------------------------------------------------------------------------
def estilo_grafico(fig, leyenda_abajo=False, alto=430, cartesiano=True):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLOR["blanco"],
        plot_bgcolor="#F8F9FA",
        height=alto,
        font=dict(family="Arial", size=12, color=COLOR["azul_oscuro"]),
        title=dict(
            font=dict(size=14, color=COLOR["azul_oscuro"], family="Georgia, serif"),
            x=0.02,
            xanchor="left"
        ),
        legend=dict(
            font=dict(size=11, color=COLOR["azul_oscuro"]),
            orientation="h" if leyenda_abajo else "v",
            yanchor="top",
            y=-0.22 if leyenda_abajo else 1,
            xanchor="left",
            x=0 if leyenda_abajo else 1.02
        ),
        margin=dict(t=70, b=110 if leyenda_abajo else 60, l=70, r=40)
    )
    if cartesiano:
        fig.update_xaxes(
            showgrid=True, gridcolor="#E5E8ED",
            linecolor="#C0C7D0", zeroline=False,
            tickfont=dict(color=COLOR["azul_oscuro"], size=11),
            title_font=dict(color=COLOR["azul_oscuro"], size=12),
            automargin=True
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="#E5E8ED",
            linecolor="#C0C7D0", zeroline=False,
            tickfont=dict(color=COLOR["azul_oscuro"], size=11),
            title_font=dict(color=COLOR["azul_oscuro"], size=12),
            automargin=True
        )
    return fig


# ---------------------------------------------------------------------------
# CARGA Y TRANSFORMACION DE DATOS
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("marketing_campaign.csv", encoding="utf-8-sig")

    df = df.dropna(subset=["Income"])
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    df["Edad"] = 2024 - df["Year_Birth"]

    df["Estado_Civil"] = df["Marital_Status"].replace({
        "Alone": "Soltero", "Absurd": "Soltero", "YOLO": "Soltero",
        "Single": "Soltero", "Together": "Casado", "Married": "Casado",
        "Divorced": "Divorciado", "Widow": "Viudo"
    })

    df["Total_Hijos"] = df["Kidhome"] + df["Teenhome"]
    df["Perfil_Familiar"] = (
        df["Total_Hijos"]
        .map({0: "Sin hijos", 1: "1 hijo", 2: "2 hijos"})
        .fillna("3 o mas hijos")
    )

    df["Gasto_Total"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] +
        df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )
    df["Gasto_Premium"] = df["MntWines"] + df["MntGoldProds"]

    df["Campanas_Aceptadas"] = (
        df[["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3",
            "AcceptedCmp4","AcceptedCmp5"]].sum(axis=1)
    )

    df["Segmento_Ingreso"] = pd.qcut(
        df["Income"], q=3,
        labels=["Ingreso bajo", "Ingreso medio", "Ingreso alto"]
    )

    def canal_preferido(row):
        opciones = {
            "Tienda fisica": row["NumStorePurchases"],
            "Web":           row["NumWebPurchases"],
            "Catalogo":      row["NumCatalogPurchases"]
        }
        return max(opciones, key=opciones.get)

    df["Canal_Preferido"] = df.apply(canal_preferido, axis=1)

    return df


df = cargar_datos()


# ---------------------------------------------------------------------------
# BARRA LATERAL — FILTROS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filtros de Analisis")
    st.markdown("---")

    edu_opts = sorted(df["Education"].unique())
    edu_sel = st.multiselect("Nivel educativo", edu_opts, default=edu_opts)

    civil_opts = sorted(df["Estado_Civil"].unique())
    civil_sel = st.multiselect("Estado civil", civil_opts, default=civil_opts)

    inc_min = int(df["Income"].min())
    inc_max = int(df["Income"].max())
    rango_ingreso = st.slider(
        "Rango de ingreso anual ($)",
        inc_min, inc_max, (inc_min, inc_max), step=1000, format="$%d"
    )

    hijos_opts = sorted(df["Total_Hijos"].unique())
    hijos_sel = st.multiselect("Hijos en el hogar", hijos_opts, default=hijos_opts)

    resp_opciones = {"Todos": None, "Respondio a la campana": 1, "No respondio": 0}
    resp_label = st.selectbox("Respuesta ultima campana", list(resp_opciones.keys()))

    st.markdown("---")
    st.markdown("**Proyecto Final — ADM-3083**")
    st.markdown("Herramientas y Visualizacion")


# ---------------------------------------------------------------------------
# APLICAR FILTROS
# ---------------------------------------------------------------------------
dff = df[
    df["Education"].isin(edu_sel) &
    df["Estado_Civil"].isin(civil_sel) &
    df["Income"].between(rango_ingreso[0], rango_ingreso[1]) &
    df["Total_Hijos"].isin(hijos_sel)
]

if resp_opciones[resp_label] is not None:
    dff = dff[dff["Response"] == resp_opciones[resp_label]]

if dff.empty:
    st.warning("No existen registros para la combinacion de filtros seleccionada.")
    st.stop()


# ---------------------------------------------------------------------------
# ENCABEZADO PRINCIPAL
# ---------------------------------------------------------------------------
st.markdown(f"""
<h1 style='margin-bottom:2px; font-family:Georgia,serif;'>
    Inteligencia de Marketing y Segmentacion de Clientes
</h1>
<p style='color:{COLOR["gris"]} !important; font-size:0.97rem; margin-bottom:18px;'>
    Analisis de campanas publicitarias — Identificacion del perfil de cliente de alto valor<br>
    Muestra activa: {len(dff):,} registros de {len(df):,} clientes totales
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# INDICADORES CLAVE (KPIs)
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric("Clientes analizados", f"{len(dff):,}")
with k2:
    st.metric("Ingreso promedio", f"${dff['Income'].mean():,.0f}")
with k3:
    st.metric("Gasto premium promedio", f"${dff['Gasto_Premium'].mean():,.0f}")
with k4:
    tasa = dff["Response"].mean() * 100
    st.metric("Tasa de respuesta", f"{tasa:.1f}%")
with k5:
    canal_top = dff["Canal_Preferido"].value_counts().idxmax()
    st.metric("Canal dominante", canal_top)

st.markdown("---")


# ===========================================================================
# SECCION 1 — PERFIL DEL CLIENTE DE ALTO VALOR
# ===========================================================================
st.markdown(
    "<div class='encabezado-seccion'>Seccion 1 — Perfil del cliente de alto valor</div>",
    unsafe_allow_html=True
)

st.markdown(
    "El primer bloque identifica las caracteristicas demograficas que definen "
    "al cliente con mayor potencial de gasto en productos premium."
)

col1, col2 = st.columns(2)

with col1:
    p99 = dff["Income"].quantile(0.99)
    scatter_df = dff[dff["Income"] <= p99].copy()

    fig1 = px.scatter(
        scatter_df,
        x="Income", y="MntWines",
        color="Perfil_Familiar",
        size="MntGoldProds",
        size_max=12,
        opacity=0.80,
        color_discrete_sequence=COLOR["cat"],
        title="Los hogares sin hijos concentran el mayor gasto en vinos a ingresos altos",
        labels={
            "Income": "Ingreso anual ($)",
            "MntWines": "Gasto en vinos ($)",
            "Perfil_Familiar": "Composicion familiar",
            "MntGoldProds": "Gasto Gold ($)"
        },
        hover_data=["Edad", "Education"]
    )
    fig1.update_traces(marker=dict(line=dict(width=0.6, color="white")))
    fig1 = estilo_grafico(fig1, leyenda_abajo=True, alto=460)
    fig1.update_xaxes(tickprefix="$", tickformat=",")
    fig1.update_yaxes(tickprefix="$", tickformat=",")
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Se excluye el percentil 99 de ingresos para evitar distorsion visual.")
    st.info(
        "**Observacion:** A medida que el ingreso aumenta, el gasto en vinos crece de "
        "forma marcada en los segmentos sin hijos. El tamano del punto refleja el gasto "
        "adicional en productos Gold, que sigue el mismo patron."
    )

with col2:
    fig2 = px.box(
        dff,
        x="Education", y="Gasto_Premium",
        color="Education",
        color_discrete_sequence=COLOR["cat"],
        title="Los clientes con posgrado lideran el gasto en productos de alto valor",
        labels={
            "Education": "Nivel educativo",
            "Gasto_Premium": "Gasto premium — Vinos + Gold ($)"
        },
        category_orders={"Education": ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]}
    )
    fig2.update_traces(showlegend=False, marker=dict(opacity=0.7), line=dict(width=2))
    fig2 = estilo_grafico(fig2, alto=460)
    fig2.update_yaxes(tickprefix="$", tickformat=",")
    st.plotly_chart(fig2, use_container_width=True)
    st.info(
        "**Observacion:** PhD registra la mediana y el promedio mas altos en gasto premium. "
        "Master muestra un desempeno comparable. Ambos grupos deben ser el foco principal "
        "de las proximas campanas de productos de lujo."
    )


# ===========================================================================
# SECCION 2 — COMPOSICION FAMILIAR Y COMPORTAMIENTO DE COMPRA
# ===========================================================================
st.markdown(
    "<div class='encabezado-seccion'>Seccion 2 — Composicion familiar y comportamiento de compra</div>",
    unsafe_allow_html=True
)

st.markdown(
    "La presencia de hijos en el hogar es la variable que mejor explica "
    "la disponibilidad de gasto en productos de lujo."
)

col3, col4 = st.columns(2)

with col3:
    categorias = ["MntWines","MntMeatProducts","MntGoldProds",
                  "MntFishProducts","MntSweetProducts","MntFruits"]
    etiquetas  = ["Vinos","Carnes","Gold / Lujo","Pescados","Dulces","Frutas"]

    gasto_familia = dff.groupby("Perfil_Familiar")[categorias].mean().reset_index()
    gasto_melted  = gasto_familia.melt(
        id_vars="Perfil_Familiar", var_name="Categoria", value_name="Gasto promedio"
    )
    gasto_melted["Categoria"] = gasto_melted["Categoria"].map(
        dict(zip(categorias, etiquetas))
    )

    fig3 = px.bar(
        gasto_melted,
        x="Categoria", y="Gasto promedio",
        color="Perfil_Familiar",
        barmode="group",
        color_discrete_sequence=COLOR["cat"],
        title="Sin hijos en el hogar: el doble de gasto en vinos y productos exclusivos",
        labels={"Perfil_Familiar": "Composicion familiar"}
    )
    fig3.update_traces(marker_line_color="white", marker_line_width=0.6)
    fig3 = estilo_grafico(fig3, leyenda_abajo=True, alto=460)
    fig3.update_yaxes(tickprefix="$", tickformat=",")
    st.plotly_chart(fig3, use_container_width=True)
    st.info(
        "**Observacion:** La brecha de gasto entre hogares sin hijos y con dos o mas hijos "
        "es especialmente pronunciada en vinos y productos Gold. Carnes tambien muestra "
        "una diferencia relevante a favor del segmento sin hijos."
    )

with col4:
    sunburst_df = (
        dff.groupby(["Estado_Civil", "Perfil_Familiar"])["Gasto_Premium"]
        .mean().reset_index()
    )

    fig4 = px.sunburst(
        sunburst_df,
        path=["Estado_Civil", "Perfil_Familiar"],
        values="Gasto_Premium",
        color="Gasto_Premium",
        color_continuous_scale=["#D5D8DC", "#85C1E9", "#1C2B4A"],
        title="En todos los estados civiles, la ausencia de hijos maximiza el gasto de lujo",
        labels={
            "Gasto_Premium": "Gasto promedio ($)",
            "Estado_Civil":  "Estado civil",
            "Perfil_Familiar": "Composicion familiar"
        }
    )
    fig4.update_traces(
        textinfo="label",
        insidetextorientation="radial",
        insidetextfont=dict(size=11, color="white"),
        hovertemplate=(
            "<b>%{label}</b><br>Gasto promedio: $%{value:,.0f}"
            "<br>Participacion: %{percentParent:.1%}<extra></extra>"
        )
    )
    fig4 = estilo_grafico(fig4, alto=500, cartesiano=False)
    fig4.update_layout(
        uniformtext=dict(minsize=9, mode="hide"),
        margin=dict(t=75, b=35, l=15, r=110),
        coloraxis_colorbar=dict(
            title=dict(text="Gasto ($)", font=dict(color=COLOR["azul_oscuro"])),
            tickfont=dict(color=COLOR["azul_oscuro"])
        )
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.info(
        "**Observacion:** El estado civil introduce matices, pero la composicion familiar "
        "es la variable mas consistente. Los solteros y divorciados sin hijos presentan "
        "los mayores promedios de gasto premium en la mayoria de los segmentos."
    )


# ===========================================================================
# SECCION 3 — PREFERENCIA DE CANAL POR SEGMENTO DE INGRESO
# ===========================================================================
st.markdown(
    "<div class='encabezado-seccion'>Seccion 3 — Canal de compra preferido por segmento de ingreso</div>",
    unsafe_allow_html=True
)

st.markdown(
    "Identificar el canal correcto es tan importante como identificar al cliente. "
    "Esta seccion determina donde se concentran las compras segun ingreso y educacion."
)

col5, col6 = st.columns(2)

with col5:
    canal_df = (
        dff.groupby("Segmento_Ingreso", observed=True)
        [["NumWebPurchases","NumStorePurchases","NumCatalogPurchases"]]
        .mean().reset_index()
    )
    canal_melted = canal_df.melt(
        id_vars="Segmento_Ingreso", var_name="Canal", value_name="Compras promedio"
    )
    canal_melted["Canal"] = canal_melted["Canal"].map({
        "NumWebPurchases":     "Web",
        "NumStorePurchases":   "Tienda fisica",
        "NumCatalogPurchases": "Catalogo"
    })

    fig5 = px.bar(
        canal_melted,
        x="Segmento_Ingreso", y="Compras promedio",
        color="Canal", barmode="group",
        color_discrete_sequence=[COLOR["azul_oscuro"], COLOR["rojo"], COLOR["dorado"]],
        title="El ingreso alto se concentra en tienda fisica y catalogo, no en la web",
        labels={
            "Segmento_Ingreso": "Segmento de ingreso",
            "Compras promedio":  "Numero promedio de compras"
        }
    )
    fig5.update_traces(marker_line_color="white", marker_line_width=0.6)
    fig5 = estilo_grafico(fig5, leyenda_abajo=True, alto=460)
    st.plotly_chart(fig5, use_container_width=True)
    st.info(
        "**Observacion:** La tienda fisica domina como canal de compra en todos los "
        "segmentos de ingreso. El catalogo cobra mayor importancia en el segmento alto, "
        "mientras que la web mantiene una participacion secundaria pero consistente."
    )

with col6:
    heat_df = dff.groupby("Education")[
        ["NumWebPurchases","NumStorePurchases","NumCatalogPurchases","NumDealsPurchases"]
    ].mean()
    heat_df = heat_df.reindex(["Basic","2n Cycle","Graduation","Master","PhD"])
    heat_df.columns = ["Web","Tienda fisica","Catalogo","Con descuento"]

    z = heat_df.values
    z_max = np.nanmax(z)

    fig6 = go.Figure(data=go.Heatmap(
        z=z,
        x=heat_df.columns.tolist(),
        y=heat_df.index.tolist(),
        colorscale=[[0,"#EBF5FB"],[0.5,"#2E86C1"],[1,"#1C2B4A"]],
        showscale=True,
        colorbar=dict(
            title=dict(text="Compras<br>promedio", font=dict(color=COLOR["azul_oscuro"])),
            tickfont=dict(color=COLOR["azul_oscuro"])
        ),
        hovertemplate=(
            "<b>%{y}</b><br>Canal: %{x}<br>"
            "Compras promedio: %{z:.1f}<extra></extra>"
        )
    ))

    anotaciones = []
    for i, edu in enumerate(heat_df.index):
        for j, canal in enumerate(heat_df.columns):
            val = z[i][j]
            color_texto = "white" if val >= z_max * 0.55 else COLOR["azul_oscuro"]
            anotaciones.append(dict(
                x=canal, y=edu, text=f"{val:.1f}",
                showarrow=False,
                font=dict(color=color_texto, size=12, family="Arial")
            ))

    fig6.update_layout(
        title="PhD lidera en catalogo ademas de tienda; educacion basica concentra descuentos",
        xaxis_title="Canal de compra",
        yaxis_title="Nivel educativo",
        annotations=anotaciones
    )
    fig6 = estilo_grafico(fig6, alto=460)
    st.plotly_chart(fig6, use_container_width=True)
    st.info(
        "**Observacion:** La tienda fisica registra los valores mas altos en todos los "
        "niveles educativos. Los clientes con PhD destacan adicionalmente en catalogo y "
        "web, lo que sugiere un perfil de comprador mas diversificado y activo."
    )


# ===========================================================================
# SECCION 4 — EFECTIVIDAD DE LAS CAMPANAS
# ===========================================================================
st.markdown(
    "<div class='encabezado-seccion'>Seccion 4 — Efectividad comparativa de las cinco campanas</div>",
    unsafe_allow_html=True
)

st.markdown(
    "El historial de conversion permite identificar cuales estrategias generaron "
    "mayor respuesta y que perfil de cliente respondio con mayor frecuencia."
)

col7, col8 = st.columns([1.3, 1])

with col7:
    camp_cols   = ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3",
                   "AcceptedCmp4","AcceptedCmp5","Response"]
    camp_labels = ["Campana 1","Campana 2","Campana 3",
                   "Campana 4","Campana 5","Campana actual"]
    tasas = [dff[c].mean() * 100 for c in camp_cols]

    fig7 = go.Figure(go.Funnel(
        y=camp_labels,
        x=tasas,
        texttemplate="%{value:.1f}%",
        marker=dict(color=COLOR["cat"][:6]),
        connector=dict(line=dict(color="#BFC9CA", width=1.5))
    ))
    fig7.update_layout(
        title="La campana actual supera a todas las anteriores en tasa de conversion"
    )
    fig7 = estilo_grafico(fig7, alto=460)
    st.plotly_chart(fig7, use_container_width=True)
    st.info(
        "**Observacion:** La campana actual registra la mayor tasa de aceptacion del "
        "periodo. Entre las anteriores, las campanas 3, 4 y 5 mostraron tasas superiores "
        "a las campanas 1 y 2, lo que indica una mejora progresiva en la segmentacion."
    )

with col8:
    respondio    = dff[dff["Response"] == 1]
    no_respondio = dff[dff["Response"] == 0]

    variables = ["MntWines","MntGoldProds","Income",
                 "NumWebPurchases","NumStorePurchases","MntMeatProducts"]
    etiq_radar = ["Gasto vinos","Gasto Gold","Ingreso",
                  "Compras web","Compras tienda","Gasto carnes"]

    max_vals = [dff[v].max() for v in variables]

    def normalizar(grupo, var, maximo):
        if grupo.empty or maximo == 0:
            return 0
        return grupo[var].mean() / maximo * 10

    r1 = [normalizar(respondio,    v, m) for v, m in zip(variables, max_vals)]
    r2 = [normalizar(no_respondio, v, m) for v, m in zip(variables, max_vals)]

    fig8 = go.Figure()
    fig8.add_trace(go.Scatterpolar(
        r=r1 + [r1[0]], theta=etiq_radar + [etiq_radar[0]],
        fill="toself", name="Respondio",
        line=dict(color=COLOR["verde"], width=2),
        fillcolor="rgba(26, 122, 110, 0.20)"
    ))
    fig8.add_trace(go.Scatterpolar(
        r=r2 + [r2[0]], theta=etiq_radar + [etiq_radar[0]],
        fill="toself", name="No respondio",
        line=dict(color=COLOR["rojo"], width=2),
        fillcolor="rgba(176, 58, 46, 0.20)"
    ))
    fig8.update_layout(
        title="Quien responde tiene mayor ingreso y concentra su gasto en vinos y carnes",
        polar=dict(
            bgcolor="#F8F9FA",
            radialaxis=dict(
                visible=True, range=[0, 10],
                gridcolor="#D5D8DC", linecolor="#AEB6BF",
                tickfont=dict(color=COLOR["azul_oscuro"], size=10)
            ),
            angularaxis=dict(
                gridcolor="#D5D8DC", linecolor="#AEB6BF",
                tickfont=dict(color=COLOR["azul_oscuro"], size=11)
            )
        )
    )
    fig8 = estilo_grafico(fig8, leyenda_abajo=True, alto=460, cartesiano=False)
    st.plotly_chart(fig8, use_container_width=True)
    st.info(
        "**Observacion:** El cliente que acepta la campana presenta un perfil de mayor "
        "valor economico. Sus promedios de ingreso, gasto en vinos y carnes superan "
        "notablemente a los de quienes no responden."
    )


# ===========================================================================
# CONCLUSIONES ESTRATEGICAS
# ===========================================================================
st.markdown("---")
st.markdown(
    "<div class='encabezado-seccion'>Conclusiones estrategicas para la planificacion del proximo ano</div>",
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    **Perfil del cliente objetivo**

    El segmento de mayor valor corresponde a adultos de 50 anos en adelante,
    con nivel educativo de posgrado (PhD o Master), ingresos en el tercio
    superior y sin hijos en el hogar. Su gasto se concentra en vinos,
    productos Gold y carnes, categorias de alto margen para la empresa.
    """)

with c2:
    st.markdown("""
    **Canal de contacto recomendado**

    La tienda fisica es el canal con mayor volumen de compras en todos los
    segmentos. El catalogo adquiere relevancia adicional en el segmento de
    ingreso alto y educacion PhD. La web funciona como canal de apoyo pero
    no lidera en ningun segmento prioritario. La estrategia optima combina
    atencion presencial con catalogo personalizado.
    """)

with c3:
    st.markdown("""
    **Recomendaciones de presupuesto**

    Se recomienda concentrar la inversion en hogares sin hijos con ingresos
    altos y formacion academica avanzada. Los hogares con dos o mas hijos
    muestran un retorno significativamente menor en productos premium.
    La campana actual debe tomarse como modelo de referencia dado que supero
    en conversion a todas las campanas previas del periodo analizado.
    """)

st.markdown("---")
st.caption(
    "Dashboard desarrollado con Python, Streamlit y Plotly Express  |  "
    "Proyecto Final ADM-3083 — Herramientas y Visualizacion  |  "
    "Fuente: Marketing Campaign Dataset, Kaggle (Rodsaldanha)"
)
