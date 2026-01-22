# ferramentas/overview/router.py
from __future__ import annotations

from datetime import date
from typing import Dict, Any, List, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px

import numpy as np
import plotly.graph_objects as go
from utils import BASE_URL_API, CARTEIRAS  # CARTEIRAS: {id: nome}


# =========================
# API
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_positions_all_portfolios(data_base: date, portfolio_ids: List[str], headers: Dict[str, str]) -> pd.DataFrame:
    """
    Busca positions/get para várias carteiras no mesmo request.
    Retorna DF LONG: portfolio_id | portfolio_name | book_name | instrument_name | asset_value
    """
    payload = {
        "start_date": str(data_base),
        "end_date": str(data_base),
        "instrument_position_aggregation": 3,
        "portfolio_ids": portfolio_ids,
    }

    resp = st.session_state.get("_overview_last_resp")  # só pra debug, opcional
    r = None
    try:
        r = __import__("requests").post(
            f"{BASE_URL_API.rstrip('/')}/portfolio_position/positions/get",
            json=payload,
            headers=headers,
            timeout=60,
        )
        r.raise_for_status()
        res = r.json()
    except Exception as e:
        # mostra um erro útil
        msg = f"Falha ao buscar positions/get: {type(e).__name__}: {e}"
        if r is not None:
            msg += f" | HTTP {getattr(r, 'status_code', '')}"
        raise RuntimeError(msg)

    objetos = res.get("objects", {})
    rows: List[Dict[str, Any]] = []

    # objetos pode vir dict {portfolio_id: [overviews...]} ou lista
    iter_values = objetos.values() if isinstance(objetos, dict) else (objetos if isinstance(objetos, list) else [])
    for obj in iter_values:
        # cada obj geralmente é um overview de uma carteira
        if not isinstance(obj, dict):
            continue

        pid = obj.get("portfolio_id") or obj.get("portfolio") or obj.get("id")
        pname = obj.get("name")

        positions = obj.get("instrument_positions")
        if not positions:
            continue

        if isinstance(positions, dict):
            positions = [positions]

        if not isinstance(positions, list):
            continue

        for p in positions:
            if not isinstance(p, dict):
                continue

            rows.append({
                "portfolio_id": str(pid) if pid is not None else None,
                "portfolio_name": str(pname) if pname else None,
                "book_name": p.get("book_name") or "Sem Classe",
                "instrument_name": p.get("instrument_name") or "Sem Nome",
                "asset_value": p.get("asset_value"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["portfolio_id"] = df["portfolio_id"].astype(str)
    df["portfolio_name"] = df["portfolio_name"].fillna(df["portfolio_id"].map(lambda x: CARTEIRAS.get(x, x)))
    df["book_name"] = df["book_name"].fillna("Sem Classe").astype(str)
    df["instrument_name"] = df["instrument_name"].fillna("Sem Nome").astype(str)
    df["asset_value"] = pd.to_numeric(df["asset_value"], errors="coerce").fillna(0.0)

    return df


def _compute_pl_by_portfolio(df_long: pd.DataFrame) -> pd.Series:
    return df_long.groupby("portfolio_name")["asset_value"].sum()


def _allocation_by_book_pct(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna DF: portfolio_name | book_name | value | pct
    """
    grp = df_long.groupby(["portfolio_name", "book_name"], as_index=False)["asset_value"].sum()
    pl = grp.groupby("portfolio_name")["asset_value"].transform("sum")
    grp["pct"] = grp["asset_value"] / pl.replace(0, pd.NA)
    grp["pct"] = grp["pct"].fillna(0.0)
    grp = grp.rename(columns={"asset_value": "value"})
    return grp


def _heatmap_df(alloc: pd.DataFrame, top_books: int = 25) -> pd.DataFrame:
    """
    Pivota para portfolio x book.
    Reduz dimensão pegando top books por soma total.
    """
    book_tot = alloc.groupby("book_name")["value"].sum().sort_values(ascending=False)
    keep = book_tot.head(int(top_books)).index.tolist()
    sub = alloc[alloc["book_name"].isin(keep)].copy()

    piv = sub.pivot_table(index="portfolio_name", columns="book_name", values="pct", aggfunc="sum").fillna(0.0)
    # ordena colunas por relevância
    piv = piv[keep]
    return piv

import re

def _suggest_instruments(df_long: pd.DataFrame, query: str, limit: int = 20) -> list[str]:
    q = (query or "").strip().lower()
    if len(q) < 2:
        return []

    names = df_long["instrument_name"].dropna().astype(str)

    # contém (case-insensitive)
    mask = names.str.lower().str.contains(re.escape(q), na=False)
    hits = names[mask]

    if hits.empty:
        return []

    # ordena: começa com o texto primeiro, depois contém
    ql = q
    uniq = hits.drop_duplicates().tolist()
    uniq.sort(key=lambda s: (0 if str(s).lower().startswith(ql) else 1, len(str(s))))
    return uniq[:limit]


def _search_asset(df_long: pd.DataFrame, query: str, exact: bool = False) -> pd.DataFrame:
    if not query:
        return pd.DataFrame()

    tmp = df_long.copy()
    inst = tmp["instrument_name"].astype(str)
    inst_l = inst.str.lower()
    q = query.strip().lower()

    if exact:
        hit = tmp[inst_l == q].copy()
    else:
        import re
        hit = tmp[inst_l.str.contains(re.escape(q), na=False)].copy()

    if hit.empty:
        return pd.DataFrame()

    g = hit.groupby(
        ["portfolio_name", "instrument_name"],
        as_index=False
    )["asset_value"].sum()

    pl = tmp.groupby("portfolio_name")["asset_value"].sum().rename("pl")
    g = g.merge(pl, on="portfolio_name", how="left")

    g["pct_pl"] = (g["asset_value"] / g["pl"].replace(0, pd.NA)).fillna(0.0)

    return g.sort_values("asset_value", ascending=False).drop(columns=["pl"])



# =========================
# UI
# =========================
def render(ctx) -> None:
    st.markdown("### Overview")
    st.caption("Carteiras × Classes (book) + Busca de Ativo")
    st.divider()

    if "headers" not in st.session_state or not st.session_state.headers:
        st.warning("Faça login para consultar os dados.")
        return

    c1, c2, c3 = st.columns([1.2, 1.6, 1.2])
    with c1:
        data_base = st.date_input("Data-base", value=date.today(), key="ov_data_base")
    with c2:
        top_books = st.number_input("Top classes (books) no heatmap", min_value=5, max_value=80, value=25, step=5)
    with c3:
        only_selected = st.checkbox("Selecionar carteiras", value=False)

    all_portfolios = [(str(pid), name) for pid, name in CARTEIRAS.items()]
    selected_ids = [pid for pid, _ in all_portfolios]

    if only_selected:
        selected_names = st.multiselect(
            "Carteiras",
            options=[name for _, name in all_portfolios],
            default=[name for _, name in all_portfolios[:8]],
            key="ov_sel_names",
        )
        selected_ids = [pid for pid, name in all_portfolios if name in selected_names]

    if not selected_ids:
        st.info("Selecione ao menos 1 carteira.")
        return

    with st.spinner("Carregando posições de todas as carteiras..."):
        df_long = _fetch_positions_all_portfolios(data_base, selected_ids, st.session_state.headers)

    if df_long.empty:
        st.info("Sem posições retornadas para a data/carteiras selecionadas.")
        return

    # KPIs rápidos
    pl = _compute_pl_by_portfolio(df_long)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Carteiras", f"{pl.shape[0]}")
    with k2:
        st.metric("PL total", f"R$ {pl.sum():,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
    with k3:
        st.metric("Posições (linhas)", f"{df_long.shape[0]}")

    st.divider()

    st.markdown("#### Heatmap — % alocado por classe (book)")

    alloc = _allocation_by_book_pct(df_long)
    piv = _heatmap_df(alloc, top_books=int(top_books))

    # matriz em %
    z = (piv.values * 100)

    x = piv.columns.tolist()
    y = piv.index.tolist()

    # ---- LIMIAR (ajuste aqui)
    LIMIAR_BRANCO = 20.0  # %; acima disso texto branco, abaixo texto preto

    # ---- máscara
    mask_white = z >= LIMIAR_BRANCO
    mask_black = ~mask_white

    # ---- texto formatado (e esconde zeros)
    txt = np.where(z > 0, np.round(z, 1).astype(str) + "%", "0%")


    # ---- duas camadas de texto (uma branca e outra preta)
    txt_white = np.where(mask_white, txt, "")
    txt_black = np.where(mask_black, txt, "")

    # ---- base heatmap (cores)
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Reds",
            zmin=0,
            zmax=float(np.nanmax(z) if np.isfinite(z).any() else 100),
            colorbar=dict(title="% do PL", ticksuffix="%"),
            hovertemplate="Carteira=%{y}<br>Classe=%{x}<br>% do PL=%{z:.2f}%<extra></extra>",
        )
    )

    # ---- overlay texto preto (células claras)
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],  # transparente
            showscale=False,
            text=txt_black,
            texttemplate="%{text}",
            textfont=dict(color="black", size=11),
            hoverinfo="skip",
        )
    )

    # ---- overlay texto branco (células escuras)
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],  # transparente
            showscale=False,
            text=txt_white,
            texttemplate="%{text}",
            textfont=dict(color="white", size=11),
            hoverinfo="skip",
        )
    )

    # ---- eixos (books em cima)
    fig.update_xaxes(side="top", tickangle=-45, tickfont=dict(size=11), title="Classe (book)")
    fig.update_yaxes(tickfont=dict(size=11), title="Carteira")

    fig.update_layout(
        height=max(420, 42 * len(y)),
        margin=dict(l=10, r=10, t=90, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)



    # tabela de apoio (top linhas)
    with st.expander("Tabela (% por classe)"):
        show = piv.copy()
        st.dataframe((show * 100).round(2), use_container_width=True)

    st.divider()

    # =========================
    # BUSCA POR ATIVO
    # =========================
    st.markdown("#### Buscar ativo — onde está e quanto tem")

    # lista única de ativos (uma vez)
    ativos_unicos = sorted(
        df_long["instrument_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # SELECTBOX com autocomplete NATIVO
    ativo_sel = st.selectbox(
        "Digite para buscar o ativo",
        options=[""] + ativos_unicos,
        index=0,
        key="ov_ativo_sel",
    )

    # só busca quando houver seleção
    if ativo_sel:
        res = _search_asset(df_long, ativo_sel, exact=True)

        if res.empty:
            st.info("Nenhuma carteira contém esse ativo.")
        else:
            out = res.copy()

            out["asset_value"] = out["asset_value"].map(
                lambda v: f"{v:,.2f}"
                .replace(",", "X")
                .replace(".", ",")
                .replace("X", ".")
            )

            out["pct_pl"] = (out["pct_pl"] * 100).round(2).map(
                lambda v: f"{v:.2f}%".replace(".", ",")
            )

            out = out.rename(columns={
                "portfolio_name": "Carteira",
                "instrument_name": "Ativo",
                "asset_value": "Valor (R$)",
                "pct_pl": "% do PL",
            })

            st.dataframe(out, use_container_width=True, hide_index=True)

            st.caption(
                f"Carteiras com o ativo: {res['portfolio_name'].nunique()} | "
                f"Total alocado: R$ {res['asset_value'].sum():,.2f}"
                .replace(",", "X")
                .replace(".", ",")
                .replace("X", ".")
            )
