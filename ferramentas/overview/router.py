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
CARTEIRAS_OCULTAS = {
    "307",   # exemplo: carteira interna, espelho, teste
    "308",
    "1361"
}



def aplicar_regras_consolidacao_pl(df_long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    UMA função com regras hardcoded:
      - exclusão mútua (espelhos)
      - include (pai inclui filha)
      - investimento cruzado parcial (A investe X% em B)

    Retorna:
      (df_long_ajustado, report)

    IMPORTANTE:
      - Tudo por portfolio_id (string)
      - Investimento cruzado aqui é um ajuste de PL por carteira (aproximação),
        NÃO é look-through por instrumento.
    """

    if df_long.empty:
        return df_long, {"drops": [], "pl_adjustments": {}}

    df = df_long.copy()
    df["portfolio_id"] = df["portfolio_id"].astype(str)

    # =========================================================
    # REGRAS (edite aqui dentro, como você quer)
    # =========================================================

    # 1) Grupos que NÃO podem coexistir: mantém o primeiro da lista
    EXCLUSAO_MUTUA = [
        ["1212", "307"],              # exemplo: espelho -> fica 101, drop 102
        ["1211", "308"],
        ["1478","1361"]       # exemplo: variações da mesma carteira
    ]

    # 3) Investimento cruzado parcial:
    #    ("A", "B", 0.70) => A investe 70% do seu PL em B.
    #    Se A e B estiverem selecionadas, reduz o PL de A em 70% (aproximação).
    INVESTE_EM = [
        ("401", "402", 0.70),
        # ("A", "B", 0.25),
    ]

    # =========================================================
    # 0) conjuntos e PL base
    # =========================================================
    presentes = set(df["portfolio_id"].unique())
    pl_base = df.groupby("portfolio_id")["asset_value"].sum().to_dict()

    drops = set()

    # =========================================================
    # 1) Exclusão mútua
    # =========================================================
    for grupo in EXCLUSAO_MUTUA:
        grupo = [str(x) for x in grupo]
        presentes_grupo = [pid for pid in grupo if pid in presentes and pid not in drops]
        if len(presentes_grupo) > 1:
            # mantém o primeiro na ordem definida
            drops.update(presentes_grupo[1:])


    # aplica drops
    if drops:
        df = df[~df["portfolio_id"].isin(drops)].copy()

    # recalcula PL após drops (porque muda universo)
    pl = df.groupby("portfolio_id")["asset_value"].sum().to_dict()

    # =========================================================
    # 3) Ajuste por investimento cruzado parcial (netting)
    # =========================================================
    # regra: se A e B estiverem presentes (e não dropados), reduz PL(A) por pct*PL(A)
    # e distribui esse ajuste proporcionalmente nas linhas de A (para não quebrar heatmap)
    ajustes_pl = {}  # pid -> fator multiplicativo final

    # começa com fator 1.0
    for pid in pl.keys():
        ajustes_pl[pid] = 1.0

    for a, b, pct in INVESTE_EM:
        a, b = str(a), str(b)
        pct = float(pct)

        if a in pl and b in pl:
            # reduz A por pct do próprio A
            # fator vira (1 - pct). Se tiver múltiplas relações, multiplica fatores.
            ajustes_pl[a] *= (1.0 - pct)

    # aplica fatores nas linhas de cada carteira
    # (proporcional: multiplicar asset_value por fator do portfolio)
    df["asset_value"] = df.apply(
        lambda r: float(r["asset_value"]) * float(ajustes_pl.get(str(r["portfolio_id"]), 1.0)),
        axis=1,
    )

    report = {
        "drops": sorted(drops),
        "pl_adjustments_factor": {k: round(v, 6) for k, v in ajustes_pl.items() if abs(v - 1.0) > 1e-12},
        "pl_base": {k: float(pl_base.get(k, 0.0)) for k in sorted(pl_base.keys())},
        "pl_final": {k: float(df[df["portfolio_id"] == k]["asset_value"].sum()) for k in sorted(df["portfolio_id"].unique())},
    }

    return df, report

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


def _allocation_by_book_pct(df_long: pd.DataFrame, modo: str = "Subclasse") -> pd.DataFrame:
    """
    modo:
      - "Subclasse" -> book_name completo (como vem)
      - "Classe"    -> pega antes do '>>'
    Retorna DF: portfolio_name | book_key | value | pct
    """
    tmp = df_long.copy()

    if modo == "Classe":
        tmp["book_key"] = (
            tmp["book_name"]
            .astype(str)
            .str.split(">>", n=1)
            .str[0]
            .str.strip()
            .replace("", "Sem Classe")
        )
    else:
        tmp["book_key"] = tmp["book_name"].astype(str).str.strip().replace("", "Sem Classe")

    grp = tmp.groupby(["portfolio_name", "book_key"], as_index=False)["asset_value"].sum()
    pl = grp.groupby("portfolio_name")["asset_value"].transform("sum")

    grp["pct"] = grp["asset_value"] / pl.replace(0, pd.NA)
    grp["pct"] = grp["pct"].fillna(0.0)

    grp = grp.rename(columns={"asset_value": "value"})
    return grp



def _heatmap_df(alloc: pd.DataFrame, top_books: int = 25) -> pd.DataFrame:
    """
    Pivota para portfolio x book_key.
    Reduz dimensão pegando top por soma total.
    """
    book_tot = alloc.groupby("book_key")["value"].sum().sort_values(ascending=False)
    keep = book_tot.head(int(top_books)).index.tolist()
    sub = alloc[alloc["book_key"].isin(keep)].copy()

    piv = sub.pivot_table(index="portfolio_name", columns="book_key", values="pct", aggfunc="sum").fillna(0.0)
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

import io

def _excel_bytes_formatado(
    sheets: list[tuple[str, pd.DataFrame, bool, dict[str, str]]]
) -> bytes:
    """
    sheets: lista de tuplas (nome_aba, df, index, col_formats)
      - col_formats: dict col_name -> tipo ("text"|"brl"|"pct0_1"|"pct0_2"|"num2")
        pct0_1 = percentual em 0..1 (ex: 0.187) -> 18,7%
        pct0_2 = percentual em 0..100 (ex: 18.7) -> 18,7%
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book

        fmt_title = wb.add_format({"bold": True, "font_size": 12})
        fmt_header = wb.add_format({
            "bold": True, "font_color": "white", "bg_color": "#111827",
            "align": "center", "valign": "vcenter"
        })
        fmt_text = wb.add_format({"align": "left", "valign": "vcenter"})
        fmt_num2 = wb.add_format({"num_format": "#,##0.00", "valign": "vcenter"})
        fmt_brl  = wb.add_format({"num_format": u'R$ #,##0.00', "valign": "vcenter"})
        fmt_pct01 = wb.add_format({"num_format": "0.00%", "valign": "vcenter"})     # 0..1
        fmt_pct02 = wb.add_format({"num_format": '0.00"%"', "valign": "vcenter"})  # 0..100 (número 18.7)

        def _autowidth(ws, df, start_col=0, max_w=55, min_w=10):
            for i, col in enumerate(df.columns):
                vals = [str(col)]
                if not df.empty:
                    vals += df[col].head(200).astype(str).tolist()
                w = min(max_w, max(min_w, max(len(v) for v in vals) + 2))
                ws.set_column(start_col + i, start_col + i, w)

        for sheet_name, df, index, col_formats in sheets:
            df.to_excel(writer, sheet_name=sheet_name, index=index)
            ws = writer.sheets[sheet_name]
            ws.hide_gridlines(2)
            ws.set_default_row(18)

            # header
            header_row = 0
            for c, colname in enumerate(df.columns if not index else ["index"] + list(df.columns)):
                ws.write(header_row, c, colname, fmt_header)

            # freeze + filter
            ws.freeze_panes(1, 0)
            last_row = len(df) if len(df) > 0 else 1
            last_col = (len(df.columns) - 1) + (1 if index else 0)
            ws.autofilter(0, 0, last_row, last_col)

            # formatos por coluna (respeita index)
            offset = 1 if index else 0
            for col, typ in (col_formats or {}).items():
                if col not in df.columns:
                    continue
                j = df.columns.get_loc(col) + offset
                if typ == "text":
                    ws.set_column(j, j, None, fmt_text)
                elif typ == "brl":
                    ws.set_column(j, j, None, fmt_brl)
                elif typ == "num2":
                    ws.set_column(j, j, None, fmt_num2)
                elif typ == "pct0_1":
                    ws.set_column(j, j, None, fmt_pct01)
                elif typ == "pct0_2":
                    ws.set_column(j, j, None, fmt_pct02)

            # largura
            _autowidth(ws, df, start_col=(1 if index else 0))

    return buf.getvalue()


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

    c1, c2, c3, c4 = st.columns([1.2, 1.3, 1.3, 1.2])
    with c1:
        data_base = st.date_input("Data-base", value=date.today(), key="ov_data_base")
    with c2:
        top_books = st.number_input("Top classes (books) no heatmap", min_value=5, max_value=35, value=15, step=5)
    with c3:
        modo_heatmap = st.selectbox("Heatmap por", options=["Subclasse", "Classe"], index=0, key="ov_heatmap_modo")
    with c4:
        only_selected = st.checkbox("Selecionar carteiras", value=False)


    # -------------------------
    # Carteiras escondidas / bloqueadas na UI
    # -------------------------

    # monta lista de carteiras já filtrada
    all_portfolios = [
        (str(pid), name)
        for pid, name in CARTEIRAS.items()
        if str(pid) not in CARTEIRAS_OCULTAS
    ]

    # ids default só das visíveis
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




    st.divider()

    st.markdown(f"#### Heatmap — % alocado por {('subclasse' if modo_heatmap=='Subclasse' else 'classe')} (book)")


    alloc = _allocation_by_book_pct(df_long, modo=modo_heatmap)
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
            hovertemplate="Carteira=%{y}<br>Book=%{x}<br>% do PL=%{z:.2f}%<extra></extra>",
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
    fig.update_xaxes(side="top", tickangle=-45, tickfont=dict(size=11), title=f"Book ({modo_heatmap})")
    fig.update_yaxes(tickfont=dict(size=11), title="Carteira")

    fig.update_layout(
        height=max(420, 42 * len(y)),
        margin=dict(l=10, r=10, t=90, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    prefix = "subclasse" if modo_heatmap == "Subclasse" else "classe"

    with st.expander(f"Tabela (% por {prefix})"):
        show = piv.copy()                 # piv é 0..1
        show_pct = (show * 100).round(2)  # só pra visualizar

        st.dataframe(show_pct, use_container_width=True)

        # Excel: uma aba numérica (0..1) + uma aba em 0..100 (pra leitura)
        df_num = show.copy()  # 0..1
        df_100 = (show * 100.0)  # 0..100

        
        xls = _excel_bytes_formatado([
            (f"heatmap_{prefix}_pct_01", df_num, True, {c: "pct0_1" for c in df_num.columns}),
            (f"heatmap_{prefix}_pct_100", df_100, True, {c: "pct0_2" for c in df_100.columns}),
        ])


        st.download_button(
            f"Baixar tabela (% por {prefix}) — Excel",
            data=xls,
            file_name=f"overview_heatmap_{prefix}_{str(data_base)}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="ov_dl_heatmap_xlsx",
        )



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
    st.divider()
    # SELECTBOX com autocomplete NATIVO
    st.subheader ("Busca por Ativo")
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
