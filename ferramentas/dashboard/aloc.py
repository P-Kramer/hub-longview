from datetime import date
import io
import requests
import streamlit as st
import pandas as pd
from utils import BASE_URL_API, CARTEIRAS
from ferramentas.dashboard.metricas_dash import (
    ativos_renda_var,
    exposicao_bruta_rv_brasil,
    hedge_indice,
    exposicao_bruta_rv_global,
    hedge_sp500,
    hedge_dol,
)

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
        return f"{float(v):,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
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


def _pct_from_series_sum(s: pd.Series) -> float:
    """
    Soma percentuais respeitando sinal.
    Se vier em fração (0-1), converte para 0-100.
    Se vier em % já (ex 12.3), mantém.
    """
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mx = float(s.abs().max()) if len(s) else 0.0
    # Heurística robusta:
    # - se máximo <= 1.5 -> fração
    # - senão -> já é %
    return float(s.sum() * 100.0) if mx <= 1.5 else float(s.sum())


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
# Métricas (GABARITO + Hedge DOL em %Exposição)
# =========================
def _calcular_metricas_gestao(overview: dict, df_positions: pd.DataFrame) -> dict:
    df = df_positions.copy()

    PL = _to_float(overview.get("net_asset_value", 0))

    # =========================
    # Colunas básicas
    # =========================
    value_col = _pick_first_existing(df, ["exposure_value", "last_exposure_value"])
    if value_col is None:
        return {"_erro": "Sem exposure_value/last_exposure_value. Preciso do valor com sinal.", "PL": PL}

    df["val"] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)

    if "book_name" not in df.columns:
        df["book_name"] = ""
    df["book_l"] = df["book_name"].astype(str).str.strip().str.lower()

    name_col = _pick_first_existing(df, ["instrument_name", "name", "instrument", "description", "ticker_description"])
    if name_col is None:
        df["name_l"] = ""
    else:
        df["name_l"] = df[name_col].astype(str).str.strip().str.lower()

    # =========================
    # Normalização das regras
    # =========================
    ativos_rv_set = {str(x).strip().lower() for x in (ativos_renda_var or [])}

    book_rv_br_fundos = str(exposicao_bruta_rv_brasil).strip().lower()
    books_hedge_indice_br = [str(x).strip().lower() for x in (hedge_indice or [])]

    books_rv_global = [str(x).strip().lower() for x in (exposicao_bruta_rv_global or [])]
    book_hedge_sp500 = str(hedge_sp500).strip().lower()

    book_hedge_dol = str(hedge_dol).strip().lower()

    # =========================
    # 1) ENQUADRAMENTO RV (%)  -> por NOMES (lista)
    # =========================
    # regra: soma ABS das posições cujos nomes estejam na lista
    # (se quiser com sinal, troco para .sum() em vez de .abs().sum())
    mask_enq_rv = df["name_l"].isin(ativos_rv_set)
    equity_exposure_lista = float(df.loc[mask_enq_rv, "val"].abs().sum())
    enquadramento_rv = (equity_exposure_lista / PL * 100) if PL else 0.0

    # =========================
    # 2) RV BRASIL -> por BOOK (SEM filtro por nome)
    # =========================
    rv_br_book = df["book_l"].eq(book_rv_br_fundos)
    idx_br = df["book_l"].isin(books_hedge_indice_br)

    exp_bruta_rv_br = float(df.loc[rv_br_book, "val"].abs().sum())
    hedge_indice_br = float(df.loc[idx_br, "val"].sum())
    exp_liq_rv_br = exp_bruta_rv_br + hedge_indice_br

    # =========================
    # 3) RV GLOBAL -> por BOOK
    # =========================
    rv_gl_book = df["book_l"].isin(books_rv_global)
    idx_gl = df["book_l"].eq(book_hedge_sp500)

    exp_bruta_rv_gl = float(df.loc[rv_gl_book, "val"].abs().sum())
    hedge_indice_gl = float(df.loc[idx_gl, "val"].sum())
    exp_liq_rv_gl = exp_bruta_rv_gl + hedge_indice_gl

    # =========================
    # 4) HEDGE DÓLAR -> por BOOK (e % por PL com chave compatível)
    # =========================
    hedge_dol_mask = df["book_l"].eq(book_hedge_dol)
    hedge_dol_val = float(df.loc[hedge_dol_mask, "val"].sum())
    hedge_dol_pct_pl = (hedge_dol_val / PL * 100) if PL else 0.0

    # pct API (opcional)
    hedge_dol_pct_api = None
    if "pct_asset_value" in df.columns:
        hedge_dol_pct_api = _pct_from_series_sum(df.loc[hedge_dol_mask, "pct_asset_value"])

    # Moedas >> Outros (se você quiser manter)
    outros = _contains_any(df["book_l"], ["moedas >> outros"])
    moedas_outros = float(df.loc[outros, "val"].sum())

    net_dolar = exp_bruta_rv_gl + hedge_dol_val + moedas_outros
    net_dolar_pct = (net_dolar / PL * 100) if PL else 0.0

    def pct_pl(x: float) -> float:
        return (x / PL * 100) if PL else 0.0

    return {
        "PL": PL,

        # Enquadramento por nomes
        "Enquadramento RV (%)": enquadramento_rv,
        "_enq_rv_val_lista": equity_exposure_lista,  # debug: total em R$ da lista

        # RV BR por book
        "Exp. Bruta RV Brasil": exp_bruta_rv_br,
        "HEDGE ÍNDICE BR": hedge_indice_br,
        "Exp. Líquida RV Brasil": exp_liq_rv_br,

        # RV Global por book
        "Exp. Bruta RV Global": exp_bruta_rv_gl,
        "HEDGE ÍNDICE Global": hedge_indice_gl,
        "Exp. Líquida RV Global": exp_liq_rv_gl,

        # Hedge Dólar (compatível com UI)
        "HEDGE DOL": hedge_dol_val,
        "HEDGE DOL %": hedge_dol_pct_pl,
        "HEDGE DOL % (API pct_asset_value)": hedge_dol_pct_api,

        # Net dólar
        "Net Dólar": net_dolar,
        "Net Dólar %": net_dolar_pct,

        # percentuais por PL
        "Exp. Bruta RV Brasil %": pct_pl(exp_bruta_rv_br),
        "HEDGE ÍNDICE BR %": pct_pl(hedge_indice_br),
        "Exp. Líquida RV Brasil %": pct_pl(exp_liq_rv_br),

        "Exp. Bruta RV Global %": pct_pl(exp_bruta_rv_gl),
        "HEDGE ÍNDICE Global %": pct_pl(hedge_indice_gl),
        "Exp. Líquida RV Global %": pct_pl(exp_liq_rv_gl),

        "_value_col": value_col,
        "_name_col": name_col,
    }

def _normalize_pct_asset_value(df: pd.DataFrame) -> pd.Series:
    pct = pd.to_numeric(df.get("pct_asset_value", 0), errors="coerce").fillna(0.0)
    # se vier como fração (0-1), converte para %
    if pct.abs().max() <= 1.5:
        pct = pct * 100.0
    return pct


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

    metricas = _calcular_metricas_gestao(overview, df)
    if metricas.get("_erro"):
        st.error(metricas["_erro"])
        return

    st.subheader("Métricas de Gestão")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("PL", f"R$ {_fmt_brl(metricas['PL'])}")
        st.metric("Enquadramento RV", f"{metricas['Enquadramento RV (%)']:.1f}%")

    with c2:
        st.metric("Exp. Bruta RV Brasil", f"R$ {_fmt_brl(metricas['Exp. Bruta RV Brasil'])}  ({metricas['Exp. Bruta RV Brasil %']:.1f}%)")
        st.metric("HEDGE ÍNDICE BR", f"R$ {_fmt_brl(metricas['HEDGE ÍNDICE BR'])}  ({metricas['HEDGE ÍNDICE BR %']:.1f}%)")
        st.metric("Exp. Líquida RV Brasil", f"R$ {_fmt_brl(metricas['Exp. Líquida RV Brasil'])}  ({metricas['Exp. Líquida RV Brasil %']:.1f}%)")

    with c3:
        st.metric("Exp. Bruta RV Global", f"R$ {_fmt_brl(metricas['Exp. Bruta RV Global'])}  ({metricas['Exp. Bruta RV Global %']:.1f}%)")
        st.metric("HEDGE ÍNDICE Global", f"R$ {_fmt_brl(metricas['HEDGE ÍNDICE Global'])}  ({metricas['HEDGE ÍNDICE Global %']:.1f}%)")
        st.metric("Exp. Líquida RV Global", f"R$ {_fmt_brl(metricas['Exp. Líquida RV Global'])}  ({metricas['Exp. Líquida RV Global %']:.1f}%)")

    st.markdown("---")
    c4, c5 = st.columns(2)
    with c4:
        st.metric("HEDGE DOL", f"R$ {_fmt_brl(metricas['HEDGE DOL'])}  ({metricas['HEDGE DOL %']:.1f}%)")
    with c5:
        st.metric("Net Dólar", f"R$ {_fmt_brl(metricas['Net Dólar'])}  ({metricas['Net Dólar %']:.1f}%)")

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
 

    agg_pizza = _consolidar_outros(agg, limiar=PIZZA_LIMIAR_OUTROS)
    st.plotly_chart(_fig_pizza(agg_pizza), use_container_width=True)
