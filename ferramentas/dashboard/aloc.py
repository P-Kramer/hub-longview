from datetime import date
import io
import requests
import streamlit as st
import pandas as pd
from utils import BASE_URL_API, CARTEIRAS

# ===== Configs =====
PIZZA_LIMIAR_OUTROS = 0.02  # classes com <2% vão para "Outros"

INSTRUMENT_TYPE_MAP = {
    1: "stock",
    2: "etf",
    3: "fund",
    7: "currency",
    9: "future",
    10: "option",
    11: "swap",
    12: "cfd",
}

CURRENCY_ID_MAP = {
    1: "BRL",
    2: "USD",
    3: "EUR",
    4: "GBP",
}

# =========================
# Helpers
# =========================
def _to_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        return float(str(x).replace(",", ""))
    except Exception:
        return default


def _fmt_brl(v: float) -> str:
    try:
        return f"{float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "0,00"


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _contains_any(series: pd.Series, words: list[str]) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    for w in words:
        mask = mask | series.str.contains(w, na=False)
    return mask


# =========================
# Pizza
# =========================
def _consolidar_outros(agg: pd.DataFrame, limiar: float = PIZZA_LIMIAR_OUTROS) -> pd.DataFrame:
    if agg.empty:
        return agg

    if "pct" not in agg.columns:
        total = float(agg["asset_value"].sum())
        agg = agg.assign(pct=(agg["asset_value"] / total) if total else 0.0)

    grandes = agg[agg["pct"] >= limiar].copy()
    pequenos = agg[agg["pct"] < limiar].copy()

    if pequenos.empty:
        return grandes.sort_values("asset_value", ascending=False).reset_index(drop=True)

    total_all = float(agg["asset_value"].sum())
    outros_val = float(pequenos["asset_value"].sum())

    outros = pd.DataFrame({
        "book_name": ["Outros"],
        "asset_value": [outros_val],
        "pct": [(outros_val / total_all) if total_all else 0.0],
    })

    res = pd.concat([grandes, outros], ignore_index=True)
    return res.sort_values("asset_value", ascending=False).reset_index(drop=True)


def _color_map_for_pie(agg: pd.DataFrame):
    palette = [
        "#1F77B4","#2CA02C","#FF7F0E","#9467BD","#17BECF",
        "#D62728","#8C564B","#BCBD22","#E377C2","#7F7F7F",
        "#0B4F6C","#14532D","#B45309","#4C1D95","#0F766E",
    ]
    tmp = agg.copy().sort_values("asset_value", ascending=False).reset_index(drop=True)
    color_by_label, i = {}, 0
    for _, row in tmp.iterrows():
        lab = str(row["book_name"])
        if lab.strip().lower() == "outros":
            color_by_label[lab] = "#9CA3AF"
        else:
            color_by_label[lab] = palette[i % len(palette)]
            i += 1
    return [color_by_label[str(l)] for l in agg["book_name"].tolist()]


def _fig_pizza(agg: pd.DataFrame):
    import plotly.graph_objects as go
    labels = agg["book_name"].tolist()
    values = agg["asset_value"].tolist()
    colors = _color_map_for_pie(agg)

    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            sort=False,
            direction="clockwise",
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            texttemplate="<b>%{percent}</b>",
            textinfo="percent",
            textposition="inside",
            textfont=dict(size=14, family="Arial, DejaVu Sans, sans-serif", color="white"),
            hovertemplate="<b>%{label}</b><br>Financeiro: R$ %{value:,.2f}<br>% Alocado: %{percent}<extra></extra>",
            showlegend=True
        )]
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=12)),
        height=600,
    )
    return fig


# =========================
# Métricas (API-only)
# =========================
def _calcular_metricas_gestao(overview: dict, df_positions: pd.DataFrame) -> dict:
    df = df_positions.copy()

    # PL e enquadramento oficial
    PL = _to_float(overview.get("net_asset_value", 0))
    equity_exposure = _to_float(overview.get("equity_exposure", 0))
    enquadramento_rv = (equity_exposure / PL * 100) if PL else 0.0

    # valor com sinal (obrigatório para líquido/hedge)
    value_col = _pick_first_existing(df, ["exposure_value", "last_exposure_value"])
    if value_col is None:
        # fallback: vai rodar, mas líquido/hedge fica menos confiável
        value_col = _pick_first_existing(df, ["financial_value", "market_value", "asset_value"])

    if value_col is None:
        return {"_erro": "Sem coluna de valor (exposure_value/asset_value/etc).", "PL": PL}

    df["val"] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)

    # moeda
    currency_id_col = _pick_first_existing(df, ["original_currency_id", "currency_id", "instrument_currency_id"])
    if currency_id_col is not None:
        df["currency"] = df[currency_id_col].map(CURRENCY_ID_MAP).fillna(df[currency_id_col].astype(str))
    else:
        df["currency"] = "BRL"

    # instrument_type
    if "instrument_type" in df.columns:
        df["type_l"] = df["instrument_type"].map(INSTRUMENT_TYPE_MAP).fillna(df["instrument_type"].astype(str)).astype(str).str.lower()
    else:
        df["type_l"] = "unknown"

    is_deriv = df["type_l"].isin(["future", "option", "swap", "cfd", "currency"])

    # textos
    df["book_l"] = df.get("book_name", "").astype(str).str.lower()
    df["name_l"] = df.get("instrument_name", "").astype(str).str.lower()
    df["cc_l"] = df.get("contract_code", "").astype(str).str.lower()

    # =========================
    # Buckets por book_name (API)
    # =========================
    # Equity BR: book indica RV Brasil e NÃO é derivativo
    br_mask = (
        _contains_any(df["book_l"], ["renda variável brasil", "renda variavel brasil", "renda variavel brasil >>", "renda variável brasil >>", "fundos >> fundos ações", "fundos >> fundos acoes"])
        & (~is_deriv)
    )

    # Equity Global: book indica offshore/global e NÃO é derivativo
    gl_mask = (
        _contains_any(df["book_l"], ["renda variável offshore", "renda variavel offshore", "offshore", "renda variavel global", "renda variável global"])
        & (~is_deriv)
    )

    br = df[br_mask]
    gl = df[gl_mask]

    bruto_br = float(br["val"].abs().sum())
    liquido_br = float(br["val"].sum())
    hedge_br = bruto_br - abs(liquido_br)

    bruto_gl = float(gl["val"].abs().sum())
    liquido_gl = float(gl["val"].sum())
    hedge_gl = bruto_gl - abs(liquido_gl)

    # =========================
    # USD (ativos vs hedge)
    # =========================
    # Hedge USD: derivativos/currency que parecem DOL/WDO/NDF ou book de moedas/câmbio
    usd_hedge_mask = (
        is_deriv
        & (
            _contains_any(df["cc_l"], ["dol", "wdo", "ndf"])
            | _contains_any(df["name_l"], ["ndf", "dol", "wdo", "dólar", "dolar"])
            | _contains_any(df["book_l"], ["moedas", "câmbio", "cambio", "fx", "hedge"])
        )
    )

    # Ativos USD: moeda USD (não-hedge)
    usd_assets_mask = (df["currency"] == "USD") & (~usd_hedge_mask)

    usd_assets = df[usd_assets_mask]
    usd_hedge = df[usd_hedge_mask]

    bruto_usd = float(usd_assets["val"].abs().sum()) + float(usd_hedge["val"].abs().sum())
    liquido_usd = float(usd_assets["val"].sum()) + float(usd_hedge["val"].sum())

    return {
        "PL": PL,
        "Exposição Bruta Ações Brasil": bruto_br,
        "Exposição Líquida Ações Brasil": liquido_br,
        "Hedges/Alavancagem Ações Brasil": hedge_br,
        "Exposição Bruta Ações Globais": bruto_gl,
        "Exposição Líquida Ações Globais": liquido_gl,
        "Hedges/Alavancagem Ações Globais": hedge_gl,
        "Exposição Bruta USD": bruto_usd,
        "Exposição Líquida USD": liquido_usd,
        "Exposição Bruta RV (overview)": equity_exposure,
        "Enquadramento Renda Variável (%)": enquadramento_rv,
        "_value_col": value_col,
        "_currency_col": currency_id_col,
        "_df_debug": df,
    }


# =========================
# Tela principal
# =========================
def tela_alocacao():
    st.header("Alocação por Classe de Ativo")

    if "headers" not in st.session_state or not st.session_state.headers:
        st.warning("Faça login para consultar os dados.")
        return

    # ===== Filtros =====
    c1, c2, c3 = st.columns([1.1, 2, 0.8])
    with c1:
        data_base = st.date_input("Data-base", value=date.today())
    with c2:
        nome_carteira = st.selectbox("Carteira", options=sorted(CARTEIRAS.values()), index=0)
    with c3:
        consultar = st.button("Consultar", use_container_width=True)

    try:
        carteira_id = next(k for k, v in CARTEIRAS.items() if v == nome_carteira)
    except StopIteration:
        st.error("Carteira inválida.")
        return

    if not consultar:
        st.info("Selecione os filtros e clique em Consultar.")
        return

    payload = {
        "start_date": str(data_base),
        "end_date": str(data_base),
        "instrument_position_aggregation": 3,
        "portfolio_ids": [carteira_id],
    }

    # ===== API =====
    try:
        with st.spinner("Buscando posições..."):
            resp = requests.post(
                f"{BASE_URL_API.rstrip('/')}/portfolio_position/positions/get",
                json=payload,
                headers=st.session_state.headers,
                timeout=60,
            )
            resp.raise_for_status()
            resultado = resp.json()

        objetos = resultado.get("objects", {})
        inst_positions = []
        overviews = []

        iter_values = objetos.values() if isinstance(objetos, dict) else (objetos if isinstance(objetos, list) else [])
        for obj in iter_values:
            if not isinstance(obj, dict):
                continue
            overviews.append(obj)
            pos = obj.get("instrument_positions")
            if isinstance(pos, list):
                inst_positions.extend(pos)
            elif isinstance(pos, dict):
                inst_positions.append(pos)

        if not inst_positions:
            st.info("Nenhuma posição encontrada.")
            return

        df = pd.json_normalize(inst_positions)

    except Exception as e:
        st.error(f"Erro ao buscar posições: {e}")
        return

    overview = overviews[0] if overviews else {}

    # ===== métricas =====
    metricas = _calcular_metricas_gestao(overview, df)
    if metricas.get("_erro"):
        st.error(metricas["_erro"])
        return

    st.subheader("Métricas de Gestão")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("PL", f"R$ {_fmt_brl(metricas['PL'])}")
        st.metric("Bruto Ações BR", f"R$ {_fmt_brl(metricas['Exposição Bruta Ações Brasil'])}")
        st.metric("Líquido Ações BR", f"R$ {_fmt_brl(metricas['Exposição Líquida Ações Brasil'])}")
        st.metric("Hedges/Alav. BR", f"R$ {_fmt_brl(metricas['Hedges/Alavancagem Ações Brasil'])}")

    with c2:
        st.metric("Bruto Ações Globais", f"R$ {_fmt_brl(metricas['Exposição Bruta Ações Globais'])}")
        st.metric("Líquido Ações Globais", f"R$ {_fmt_brl(metricas['Exposição Líquida Ações Globais'])}")
        st.metric("Hedges/Alav. Globais", f"R$ {_fmt_brl(metricas['Hedges/Alavancagem Ações Globais'])}")

    with c3:
        st.metric("USD Bruto", f"R$ {_fmt_brl(metricas['Exposição Bruta USD'])}")
        st.metric("USD Líquido", f"R$ {_fmt_brl(metricas['Exposição Líquida USD'])}")
        st.metric("Enquadramento RV (oficial)", f"{metricas['Enquadramento Renda Variável (%)']:.1f}%")

    with st.expander("Debug (sanity + colunas usadas)"):
        st.write("Coluna valor usada:", metricas["_value_col"])
        st.write("Coluna moeda usada:", metricas["_currency_col"])
        st.write("Equity exposure (overview): R$ " + _fmt_brl(metricas["Exposição Bruta RV (overview)"]))
        st.write("BR+Global calculado: R$ " + _fmt_brl(metricas["Exposição Bruta Ações Brasil"] + metricas["Exposição Bruta Ações Globais"]))

    # ===== Pizza (asset_value) =====
    if "asset_value" not in df.columns:
        st.warning("Sem 'asset_value' nas posições; pizza de alocação não disponível.")
        return

    df["asset_value"] = pd.to_numeric(df["asset_value"], errors="coerce").fillna(0.0)
    agg = df.groupby("book_name", as_index=False)["asset_value"].sum()
    total = float(agg["asset_value"].sum())
    agg["pct"] = (agg["asset_value"] / total) if total else 0.0

    st.subheader("Alocação por Classe")
    tabela_resumo = (
        agg.assign(
            Financeiro=lambda x: x["asset_value"].map(_fmt_brl),
            **{"% Alocado": lambda x: (x["pct"] * 100).round(2).map(lambda v: str(v).replace(".", ","))}
        )
        .rename(columns={"book_name": "Classe"})
        .drop(columns=["asset_value", "pct"])
    )
    st.dataframe(tabela_resumo, use_container_width=True, hide_index=True)

    agg_pizza = _consolidar_outros(agg, limiar=PIZZA_LIMIAR_OUTROS)
    st.plotly_chart(_fig_pizza(agg_pizza), use_container_width=True)



