# simul.py
from __future__ import annotations

import io
import re
from datetime import date
from typing import List, Dict, Any, Tuple

import requests
import streamlit as st
import pandas as pd
import numpy as np
from utils import BASE_URL_API, CARTEIRAS
from .metricas_dash import caixa as caixa_ativos
from .aloc import _calcular_metricas_gestao

# ======================================================
# AJUSTE ESTE IMPORT PARA O SEU ARQUIVO REAL DE MÉTRICAS
# ======================================================


PIZZA_LIMIAR_OUTROS = 0.02  # classes <2% viram "Outros"


# ======================================================
# UI / CSS
# ======================================================
CSS = """
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }

.card { border:1px solid rgba(0,0,0,0.08); border-radius:12px; padding:14px; background:white; }
.card-muted { border:1px dashed rgba(0,0,0,0.10); border-radius:12px; padding:12px; background:rgba(0,0,0,0.02); }
.h-label { font-weight:600; font-size:0.95rem; margin:0 0 6px 0; }
.help { color:#6b7280; font-size:0.85rem; margin-top:-2px; margin-bottom:8px; }
.hr { height:1px; background:rgba(0,0,0,0.06); margin:10px 0 14px 0; }
.small { color:#6b7280; font-size:0.85rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ======================================================
# HELPERS
# ======================================================
def _format_ptbr_num(v: float) -> str:
    try:
        return f"{float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "0,00"


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _is_deriv_row(df: pd.DataFrame, row_idx: int) -> bool:
    """
    Decide se uma linha é derivativo (opção/futuro).
    Usa instrument_type/instrument_type_name quando existir.
    Senão usa instrument_type_id (se você configurar).
    Senão fallback por heurística de exposure vs asset.
    """
    if df is None or df.empty or row_idx not in df.index:
        return False

    # 1) por texto
    col_txt = _pick_col(df, ["instrument_type", "instrument_type_name", "instrument_type_desc"])
    print(df)
    if col_txt:
        s = str(df.at[row_idx, col_txt] or "").strip().lower()
        
        if re.search(r"\b(5|4)\b", s):
            return True

    # 2) por id (ajuste quando souber)
    if "instrument_type_id" in df.columns:
        it = pd.to_numeric(df.at[row_idx, "instrument_type_id"], errors="coerce")
        if pd.notna(it):
            it = int(it)
            OPCOES_IDS: set[int] = set()
            FUTUROS_IDS: set[int] = set()
            if (OPCOES_IDS or FUTUROS_IDS) and it in (OPCOES_IDS | FUTUROS_IDS):
                return True

    # 3) fallback por característica: asset ~ 0 e exposure grande
    valcol = _pick_col(df, ["exposure_value", "last_exposure_value"])
    if not valcol:
        return False

    av = float(pd.to_numeric(df.at[row_idx, "asset_value"], errors="coerce") or 0.0)
    ex = float(pd.to_numeric(df.at[row_idx, valcol], errors="coerce") or 0.0)
    eps = 1e-6
    return (abs(av) < eps and abs(ex) > eps)


def _user_error(msg: str):
    st.error(msg)


def _guide_cash_mask_message():
    st.info(
        "Não foi possível identificar a posição de CAIXA para fazer a contrapartida do rebalanceamento.\n\n"
        "Como resolver:\n"
        "1) Verifique qual é o book_name / instrument_name do caixa na sua base.\n"
        "2) Edite `_guess_cash_mask()` e inclua palavras-chave corretas.\n"
        "3) Recarregue a base e tente novamente."
    )


def _parse_ptbr_currency(txt: str) -> float:
    """
    Converte '1.234.567,89' ou '-1.000.000,00' em float.
    """
    if not txt:
        return 0.0
    txt = txt.strip()
    txt = re.sub(r"[^\d,\-\.]", "", txt)
    txt = txt.replace(".", "").replace(",", ".")
    try:
        return float(txt)
    except ValueError:
        return 0.0


def _fmt_ptbr_currency(v: float) -> str:
    """
    Formata float para padrão PT-BR com milhar.
    """
    try:
        return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "0,00"


def _guess_cash_mask(df: pd.DataFrame, caixa_ativos: list[str]) -> pd.Series:
    book = df.get("book_name", "").astype(str).str.lower().str.strip()
    inst = df.get("instrument_name", "").astype(str).str.lower().str.strip()

    mask = pd.Series(False, index=df.index)

    termos = [str(t).strip().lower() for t in (caixa_ativos or []) if str(t).strip()]
    for t in termos:
        if len(t) <= 3:  # usd, brl
            # palavra inteira
            mask |= inst.str.contains(rf"\b{re.escape(t)}\b", regex=True, na=False)
        else:
            # contém
            mask |= inst.str.contains(re.escape(t), regex=True, na=False)

    keywords = [
        "caixa", "cash", "liquidez", "conta corrente", "saldo",
        "disponível", "disponivel", "cash management"
    ]
    for k in keywords:
        mask |= book.str.contains(k, na=False) | inst.str.contains(k, na=False)

    return mask




def _recalc_pct_asset_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mantém como você tinha: pct_asset_value baseado no TOTAL LÍQUIDO de asset_value.
    Usado nas suas métricas (hedge dol %). Não mexo.
    """
    df = df.copy()
    df["asset_value"] = pd.to_numeric(df.get("asset_value", 0), errors="coerce").fillna(0.0)
    total = float(df["asset_value"].sum())
    df["pct_asset_value"] = (df["asset_value"] / total * 100.0) if total else 0.0
    return df


def _get_cash_available(df: pd.DataFrame) -> float:
    """
    Caixa disponível (líquido) em R$.
    Se seu caixa estiver distribuído em múltiplas linhas, soma tudo.
    """
    cash_mask = _guess_cash_mask(df, caixa_ativos)
    if not cash_mask.any():
        raise RuntimeError("CASH_NOT_FOUND")
    return float(pd.to_numeric(df.loc[cash_mask, "asset_value"], errors="coerce").fillna(0.0).sum())


# ======================================================
# PIZZA (cores estáveis por classe)
# ======================================================
def _consolidar_outros(agg: pd.DataFrame, limiar: float = PIZZA_LIMIAR_OUTROS) -> pd.DataFrame:
    if agg.empty:
        return agg
    if "pct" not in agg.columns:
        total = float(agg["asset_value"].sum())
        agg = agg.assign(pct=agg["asset_value"] / total if total else 0.0)

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


def _color_map_by_labels(all_labels: list[str]) -> dict[str, str]:
    palette = [
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
        "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
        "#0B4F6C", "#B45309", "#14532D", "#4C1D95", "#0F766E",
        "#9A3412", "#1D4ED8", "#BE123C", "#15803D", "#A21CAF",
    ]
    uniq = sorted({str(x) for x in all_labels if x is not None})
    color_by = {}
    i = 0
    for lab in uniq:
        if lab.strip().lower() == "outros":
            color_by[lab] = "#9CA3AF"
        else:
            color_by[lab] = palette[i % len(palette)]
            i += 1
    return color_by


def _fig_pizza(agg: pd.DataFrame, *, color_by: dict[str, str], height: int = 600):
    import plotly.graph_objects as go

    labels = agg["book_name"].astype(str).tolist()
    values = agg["asset_value"].tolist()
    colors = [color_by.get(l, "#9CA3AF") for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        sort=False,
        direction="clockwise",
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        texttemplate="<b>%{percent}</b>",
        textinfo="percent",
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>Financeiro: R$ %{value:,.2f}<br>Alocação: %{percent}<extra></extra>",
        showlegend=True,
    )])

    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(t=8, b=150, l=8, r=8),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            itemwidth=40,
        ),
        uniformtext_minsize=12,
        uniformtext_mode="show",
    )
    return fig


# ======================================================
# API
# ======================================================
def _carregar_posicoes(data_base: date, carteira_id: str, headers: Dict[str, str]) -> Tuple[pd.DataFrame, dict]:
    payload = {
        "start_date": str(data_base),
        "end_date": str(data_base),
        "instrument_position_aggregation": 3,
        "portfolio_ids": [carteira_id],
    }

    resp = requests.post(
        f"{BASE_URL_API.rstrip('/')}/portfolio_position/positions/get",
        json=payload, headers=headers, timeout=60,
    )
    resp.raise_for_status()
    res = resp.json()

    objetos = res.get("objects", {})
    inst_positions: List[Dict[str, Any]] = []
    overviews: List[dict] = []

    iter_values = objetos.values() if isinstance(objetos, dict) else (objetos if isinstance(objetos, list) else [])
    for obj in iter_values:
        if isinstance(obj, dict):
            overviews.append(obj)
            pos = obj.get("instrument_positions")
        else:
            pos = None

        if not pos:
            continue
        if isinstance(pos, list):
            inst_positions.extend(pos)
        elif isinstance(pos, dict):
            inst_positions.append(pos)

    if not inst_positions:
        return pd.DataFrame(), {}

    df = pd.json_normalize(inst_positions)
    df["asset_value"] = pd.to_numeric(df.get("asset_value", 0), errors="coerce").fillna(0.0)
    df["book_name"] = df.get("book_name").fillna("Sem Classe")
    df["instrument_name"] = df.get("instrument_name").fillna("Sem Nome")

    return df, (overviews[0] if overviews else {})

def _debug_top_exposures(df, label):
    valcol = _pick_col(df, ["exposure_value", "last_exposure_value"])
    if not valcol:
        st.write(f"DEBUG | {label} sem exposure_value")
        return
    tmp = df.copy()
    tmp[valcol] = pd.to_numeric(tmp[valcol], errors="coerce").fillna(0.0)
    tmp["asset_value"] = pd.to_numeric(tmp.get("asset_value", 0), errors="coerce").fillna(0.0)

    show = tmp.sort_values(valcol, key=lambda s: s.abs(), ascending=False).head(15)
    st.write(f"DEBUG TOP EXPOSURES — {label}")
    st.dataframe(show[["book_name","instrument_name","asset_value",valcol]].reset_index(drop=True))


# ======================================================
# ENGINE – REBALANCE + VALIDAÇÃO DE VENDA + CAIXA
# ======================================================
def _aplicar_movimentos_rebalance(df_base: pd.DataFrame, movimentos: list[dict]) -> pd.DataFrame:
    """
    REGRAS (DO JEITO QUE VOCÊ PEDIU):
    - NÃO-derivativo:
        * delta é ΔFINANCEIRO (R$)
        * altera asset_value
        * contrapartida SEMPRE no caixa
        * trava venda > posição e compra > caixa
        * exposure_value acompanha (exposure += delta) para ativo “normal”
    - Derivativo (futuro/opção):
        * delta é ΔEXPOSIÇÃO (R$)
        * altera SOMENTE exposure_value (ou last_exposure_value)
        * NÃO mexe em asset_value
        * NÃO mexe em caixa
        * NÃO tem trava (se exp = -100 e digita -100, vira -200)
    """

    if df_base is None or df_base.empty:
        raise ValueError("Base vazia. Carregue a carteira antes de simular.")

    df = df_base.copy()

    # normaliza chaves
    df["_iname"] = df["instrument_name"].astype(str).str.strip()
    df["_bname"] = df["book_name"].astype(str).str.strip()

    # coluna de exposure
    value_col = _pick_col(df, ["exposure_value", "last_exposure_value"])
    if value_col is None:
        raise ValueError(
            "A base não contém exposure_value / last_exposure_value. "
            "Sem isso não dá para recalcular métricas oficiais."
        )

    # garante numérico
    df["asset_value"] = pd.to_numeric(df.get("asset_value", 0), errors="coerce").fillna(0.0)
    df[value_col] = pd.to_numeric(df.get(value_col, 0), errors="coerce").fillna(0.0)

    # identifica caixa (pega o MAIOR, não o primeiro)
    cash_mask = _guess_cash_mask(df, caixa_ativos)
    if not cash_mask.any():
        raise RuntimeError("CASH_NOT_FOUND")

    cash_candidates = df.loc[cash_mask].copy()
    cash_candidates["asset_value"] = pd.to_numeric(cash_candidates["asset_value"], errors="coerce").fillna(0.0)
    cash_idx = cash_candidates["asset_value"].abs().idxmax()

    eps = 1e-6

    # limpa e ordena movimentos (vendas primeiro)
    movs = []
    for mv in movimentos:
        nome = str(mv.get("instrument_name") or "").strip()
        classe = str(mv.get("book_name") or "").strip()
        delta = float(mv.get("delta") or 0.0)

        if not nome or not classe:
            raise ValueError("Movimento inválido: faltou instrument_name ou book_name.")

        if abs(delta) < eps:
            continue

        movs.append({"instrument_name": nome, "book_name": classe, "delta": delta})

    movs.sort(key=lambda x: 0 if x["delta"] < 0 else 1)

    # aplica movimentos
    for mv in movs:
        nome = mv["instrument_name"]
        classe = mv["book_name"]
        delta = float(mv["delta"])

        filtro = (df["_iname"] == nome) & (df["_bname"] == classe)
        
        # trava ambiguidade (OBRIGATÓRIO)
        if filtro.sum() != 1:
            raise ValueError(
                f"Movimento ambíguo: '{nome}' em '{classe}' bateu em {int(filtro.sum())} linhas."
            )

        row_idx = df.index[filtro][0]
        is_deriv = _is_deriv_row(df, row_idx)

        # ==========================
        # DERIVATIVO: mexe SÓ exposure
        # ==========================
        if is_deriv:
            df.loc[filtro, value_col] += delta
            continue  # sem trava, sem caixa, sem asset_value

        # ==========================
        # NÃO-DERIVATIVO: fluxo normal
        # ==========================

        # VENDA: trava por posição
        if delta < 0:
            pos_atual = float(df.loc[filtro, "asset_value"].sum())
            if abs(delta) - pos_atual > eps:
                raise ValueError(
                    f"Venda inválida: tentou vender R$ {_format_ptbr_num(abs(delta))}, "
                    f"mas a posição atual é R$ {_format_ptbr_num(pos_atual)}."
                )

        # COMPRA: trava por caixa disponível
        if delta > 0:
            caixa_disp = float(df.loc[cash_mask, "asset_value"].sum())
            if delta - caixa_disp > eps:
                raise ValueError(
                    f"Compra inválida: caixa insuficiente. "
                    f"Solicitado: R$ {_format_ptbr_num(delta)} | "
                    f"Disponível: R$ {_format_ptbr_num(caixa_disp)}."
                )

        # ===== APLICA FINANCEIRO (com contrapartida no caixa) =====
        df.loc[filtro, "asset_value"] += delta
        df.at[cash_idx, "asset_value"] -= delta

        # ===== EXPOSURE acompanha para ativo “normal” (não-caixa) =====
        # Para ativo normal, exposure ~ asset. Então seguimos com exposure += delta.
        df.loc[filtro, value_col] += delta

    # NÃO remova derivativos: asset_value pode ser 0, mas exposure é relevante
    exp_col = _pick_col(df, ["exposure_value", "last_exposure_value"])
    if exp_col:
        df = df[(df["asset_value"].abs() > 1e-6) | (df[exp_col].abs() > 1e-6)].reset_index(drop=True)
    else:
        df = df[df["asset_value"].abs() > 1e-6].reset_index(drop=True)

    # limpa auxiliares
    df = df.drop(columns=[c for c in ["_iname", "_bname"] if c in df.columns], errors="ignore")

    return _recalc_pct_asset_value(df)




# ======================================================
# AGG / EXCEL
# ======================================================
def _agg_por_classe(df: pd.DataFrame, include_cash: bool = True) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["book_name", "asset_value", "pct"])

    tmp = df.copy()
    tmp["asset_value"] = pd.to_numeric(tmp.get("asset_value", 0), errors="coerce").fillna(0.0)

    if not include_cash:
        cash_mask = _guess_cash_mask(tmp, caixa_ativos)
        tmp = tmp.loc[~cash_mask].copy()

    tmp["asset_value_abs"] = tmp["asset_value"].abs()
    agg = tmp.groupby("book_name", as_index=False)["asset_value_abs"].sum()
    agg = agg.rename(columns={"asset_value_abs": "asset_value"})

    total = float(agg["asset_value"].sum())
    agg["pct"] = (agg["asset_value"] / total) if total else 0.0
    return agg.sort_values("asset_value", ascending=False).reset_index(drop=True)




def _excel_simulacao(
    agg_antes: pd.DataFrame, agg_depois: pd.DataFrame,
    df_antes: pd.DataFrame, df_depois: pd.DataFrame,
    metricas_antes: dict, metricas_depois: dict
) -> bytes:
    # ---------- prepara dataframes ----------
    delta = pd.merge(
        agg_antes, agg_depois,
        on="book_name", how="outer", suffixes=("_bef", "_aft")
    ).fillna(0)

    delta["Δ Financeiro (R$)"] = delta["asset_value_aft"] - delta["asset_value_bef"]
    delta["Δ p.p."] = (delta["pct_aft"] - delta["pct_bef"]) * 100

    keys = [
        "Enquadramento RV (%)",
        "Exp. Bruta RV Brasil", "HEDGE ÍNDICE BR", "Exp. Líquida RV Brasil",
        "Exp. Bruta RV Global", "HEDGE ÍNDICE Global", "Exp. Líquida RV Global",
        "HEDGE DOL", "Net Dólar",
        "Exp. Bruta RV Brasil %", "HEDGE ÍNDICE BR %", "Exp. Líquida RV Brasil %",
        "Exp. Bruta RV Global %", "HEDGE ÍNDICE Global %", "Exp. Líquida RV Global %",
        "HEDGE DOL %", "Net Dólar %"
    ]
    met_rows = []
    for k in keys:
        a = float(metricas_antes.get(k, 0) or 0)
        b = float(metricas_depois.get(k, 0) or 0)
        met_rows.append({"Métrica": k, "Antes": a, "Depois": b, "Delta": b - a})
    met_df = pd.DataFrame(met_rows)

    df_res_antes = agg_antes.rename(columns={"book_name": "Classe", "asset_value": "Financeiro", "pct": "%_Alocado"}).copy()
    df_res_depois = agg_depois.rename(columns={"book_name": "Classe", "asset_value": "Financeiro", "pct": "%_Alocado"}).copy()
    df_delta_classes = delta.rename(columns={"book_name": "Classe"}).copy()

    def _safe_to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    _safe_to_numeric(df_res_antes, ["Financeiro", "%_Alocado"])
    _safe_to_numeric(df_res_depois, ["Financeiro", "%_Alocado"])
    _safe_to_numeric(df_delta_classes, ["asset_value_bef", "asset_value_aft", "pct_bef", "pct_aft", "Δ Financeiro (R$)", "Δ p.p."])
    _safe_to_numeric(met_df, ["Antes", "Depois", "Delta"])

    # ---------- writer ----------
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book

        # formatos
        fmt_header = wb.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#111827",
            "align": "center",
            "valign": "vcenter",
        })
        fmt_section = wb.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#1F2937",
            "align": "left",
            "valign": "vcenter",
        })
        fmt_text = wb.add_format({"align": "left", "valign": "vcenter"})
        fmt_brl = wb.add_format({"num_format": u'R$ #,##0.00', "valign": "vcenter"})
        fmt_pct = wb.add_format({"num_format": "0.00%", "valign": "vcenter"})
        fmt_pp = wb.add_format({"num_format": '0.00" p.p."', "valign": "vcenter"})
        fmt_neg = wb.add_format({"font_color": "#B91C1C"})
        fmt_pos = wb.add_format({"font_color": "#047857"})

        def _autowidth(ws, df, start_col=0, max_w=52, min_w=10):
            for i, col in enumerate(df.columns):
                s = [str(col)]
                if not df.empty:
                    s.extend(df[col].head(200).astype(str).tolist())
                w = max(len(x) for x in s)
                w = max(min_w, min(max_w, w + 2))
                ws.set_column(start_col + i, start_col + i, w)

        def _write_df(ws, df, start_row: int, start_col: int, *, header=True) -> int:
            """Escreve df em ws a partir de (start_row, start_col). Retorna última linha escrita."""
            if header:
                for c, name in enumerate(df.columns):
                    ws.write(start_row, start_col + c, name, fmt_header)
                data_row0 = start_row + 1
            else:
                data_row0 = start_row

            # dados
            for r in range(len(df)):
                for c, colname in enumerate(df.columns):
                    v = df.iloc[r, c]
                    if pd.isna(v):
                        ws.write_blank(data_row0 + r, start_col + c, None)
                        continue
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        # padrão numérico genérico; vamos sobrepor formatos por coluna depois
                        ws.write_number(data_row0 + r, start_col + c, float(v))
                    else:
                        ws.write(data_row0 + r, start_col + c, str(v), fmt_text)

            last_row = data_row0 + max(0, len(df) - 1)
            return last_row

        def _apply_col_formats(ws, df, start_col: int, col_formats: dict[str, Any]):
            for col_name, f in col_formats.items():
                if col_name in df.columns:
                    idx = df.columns.get_loc(col_name)
                    ws.set_column(start_col + idx, start_col + idx, None, f)

        def _conditional_sign(ws, first_row, last_row, col_idx_abs):
            ws.conditional_format(first_row, col_idx_abs, last_row, col_idx_abs,
                                  {"type": "cell", "criteria": ">", "value": 0, "format": fmt_pos})
            ws.conditional_format(first_row, col_idx_abs, last_row, col_idx_abs,
                                  {"type": "cell", "criteria": "<", "value": 0, "format": fmt_neg})

        # ===============================
        # ABA ÚNICA: RESUMO (ANTES + DEPOIS)
        # ===============================
        sheet_name = "Resumo"
        df_empty = pd.DataFrame({"_": []})  # só pra criar a sheet
        df_empty.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        ws.hide_gridlines(2)
        ws.set_default_row(18)

        row = 0
        col_left = 0
        col_right = len(df_res_antes.columns) + 2  # gap

        # Título seção ANTES
        ws.merge_range(row, col_left, row, col_left + len(df_res_antes.columns) - 1, "ANTES", fmt_section)
        row += 1
        last_left = _write_df(ws, df_res_antes, row, col_left, header=True)
        _apply_col_formats(ws, df_res_antes, col_left, {"Classe": fmt_text, "Financeiro": fmt_brl, "%_Alocado": fmt_pct})
        _autowidth(ws, df_res_antes, start_col=col_left)

        # Título seção DEPOIS (mesma altura do antes)
        ws.merge_range((row - 1), col_right, (row - 1), col_right + len(df_res_depois.columns) - 1, "DEPOIS", fmt_section)
        last_right = _write_df(ws, df_res_depois, row, col_right, header=True)
        _apply_col_formats(ws, df_res_depois, col_right, {"Classe": fmt_text, "Financeiro": fmt_brl, "%_Alocado": fmt_pct})
        _autowidth(ws, df_res_depois, start_col=col_right)

        # freeze no header (linha do df)
        ws.freeze_panes(row + 1, 0)

        # autofilter para os dois blocos
        ws.autofilter(row, col_left, last_left, col_left + len(df_res_antes.columns) - 1)
        ws.autofilter(row, col_right, last_right, col_right + len(df_res_depois.columns) - 1)

        # ===============================
        # ABA: DELTA CLASSES
        # ===============================
        df_delta_classes.to_excel(writer, index=False, sheet_name="Delta_Classes")
        ws2 = writer.sheets["Delta_Classes"]
        ws2.hide_gridlines(2)
        ws2.set_default_row(18)

        # header
        for c, name in enumerate(df_delta_classes.columns):
            ws2.write(0, c, name, fmt_header)
        ws2.freeze_panes(1, 0)
        ws2.autofilter(0, 0, max(0, len(df_delta_classes)), max(0, len(df_delta_classes.columns) - 1))
        _autowidth(ws2, df_delta_classes, start_col=0)

        _apply_col_formats(ws2, df_delta_classes, 0, {
            "Classe": fmt_text,
            "asset_value_bef": fmt_brl,
            "asset_value_aft": fmt_brl,
            "pct_bef": fmt_pct,
            "pct_aft": fmt_pct,
            "Δ Financeiro (R$)": fmt_brl,
            "Δ p.p.": fmt_pp,
        })

        # sinal no delta
        if "Δ Financeiro (R$)" in df_delta_classes.columns:
            col = df_delta_classes.columns.get_loc("Δ Financeiro (R$)")
            _conditional_sign(ws2, 1, len(df_delta_classes), col)
        if "Δ p.p." in df_delta_classes.columns:
            col = df_delta_classes.columns.get_loc("Δ p.p.")
            _conditional_sign(ws2, 1, len(df_delta_classes), col)

        # ===============================
        # ABA: DELTA METRICAS
        # ===============================
        met_df.to_excel(writer, index=False, sheet_name="Delta_Metricas")
        ws3 = writer.sheets["Delta_Metricas"]
        ws3.hide_gridlines(2)
        ws3.set_default_row(18)

        for c, name in enumerate(met_df.columns):
            ws3.write(0, c, name, fmt_header)
        ws3.freeze_panes(1, 0)
        ws3.autofilter(0, 0, max(0, len(met_df)), max(0, len(met_df.columns) - 1))
        ws3.set_column(0, 0, 44, fmt_text)

        # reescreve números por linha com formato correto (%, ou R$)
        for r in range(1, len(met_df) + 1):
            met_name = str(met_df.iloc[r - 1]["Métrica"])
            is_percent = ("%" in met_name)
            f = fmt_pct if is_percent else fmt_brl
            ws3.write_number(r, 1, float(met_df.iloc[r - 1]["Antes"] or 0), f)
            ws3.write_number(r, 2, float(met_df.iloc[r - 1]["Depois"] or 0), f)
            ws3.write_number(r, 3, float(met_df.iloc[r - 1]["Delta"] or 0), f)

        # cor no delta
        if "Delta" in met_df.columns:
            col = met_df.columns.get_loc("Delta")
            _conditional_sign(ws3, 1, len(met_df), col)

    return buf.getvalue()



# ======================================================
# STATE
# ======================================================
def _init_state():
    if "sim_exist_rows" not in st.session_state:
        st.session_state.sim_exist_rows = {}
    if "sim_base_df" not in st.session_state:
        st.session_state.sim_base_df = None
    if "sim_overview" not in st.session_state:
        st.session_state.sim_overview = {}
    if "sim_loaded_key" not in st.session_state:
        st.session_state.sim_loaded_key = None
    if "sim_loaded" not in st.session_state:
        st.session_state.sim_loaded = False



# ======================================================
# UI: MÉTRICAS COMPARATIVAS (resumo)
# ======================================================
def _render_metricas_resumo(m_before: dict, m_after: dict):
    eps = 1e-9

    def g(m, k):
        if k not in m:
            raise KeyError(f"Métrica ausente: '{k}'")
        return float(m[k] or 0.0)

    def fmt_brl(v: float) -> str:
        return f"{v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def fmt_pct(v: float) -> str:
        # v já vem em % (ex: 18.7). você quer 18,7%
        return f"{v:.1f}".replace(".", ",")

    def delta_color_num(d: float) -> str:
        # "normal": + verde, - vermelho
        # "off": sem cor (pra zero)
        if abs(d) <= eps:
            return "off"
        return "normal"
    def caption_delta_pp(d_pct: float, *, eps: float = 1e-6):
        if abs(d_pct) <= eps:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            return

        color = "#0cf862" if d_pct > 0 else "#ff0000"  # verde | vermelho
        txt = f"{d_pct:+.2f} p.p.".replace(".", ",")

        st.markdown(
            f"<div style='color:{color}; font-size:0.85rem; margin-top:-6px'>{txt}</div>",
            unsafe_allow_html=True,
        )

    def metric_val_pct(title: str, val_bef: float, val_aft: float, pct_bef: float, pct_aft: float):
        d_val = val_aft - val_bef
        d_pct = pct_aft - pct_bef

        st.metric(
            title,
            f"R$ {fmt_brl(val_aft)} ({fmt_pct(pct_aft)}%)",
            delta=d_val,  # NUMÉRICO -> cor correta
            delta_color=delta_color_num(d_val),
        )

        caption_delta_pp(d_pct)
    # ================= LINHA 1 =================
    c1, c2, c3 = st.columns(3)

    with c1:
        pl_a, pl_b = g(m_before, "PL"), g(m_after, "PL")
        rv_a, rv_b = g(m_before, "Enquadramento RV (%)"), g(m_after, "Enquadramento RV (%)")

        # PL não tem % ao lado (se quiser, me diga qual % é)
        d_pl = pl_b - pl_a
        st.metric("PL", f"R$ {fmt_brl(pl_b)}", delta=d_pl, delta_color=delta_color_num(d_pl))

        d_rv = rv_b - rv_a
        st.metric("Enquadramento RV", f"{fmt_pct(rv_b)}%", delta=d_rv, delta_color=delta_color_num(d_rv))

    with c2:
        a, b = g(m_before, "Exp. Bruta RV Brasil"), g(m_after, "Exp. Bruta RV Brasil")
        ap, bp = g(m_before, "Exp. Bruta RV Brasil %"), g(m_after, "Exp. Bruta RV Brasil %")

        h_a, h_b = g(m_before, "HEDGE ÍNDICE BR"), g(m_after, "HEDGE ÍNDICE BR")
        hp_a, hp_b = g(m_before, "HEDGE ÍNDICE BR %"), g(m_after, "HEDGE ÍNDICE BR %")

        l_a, l_b = g(m_before, "Exp. Líquida RV Brasil"), g(m_after, "Exp. Líquida RV Brasil")
        lp_a, lp_b = g(m_before, "Exp. Líquida RV Brasil %"), g(m_after, "Exp. Líquida RV Brasil %")

        metric_val_pct("Exp. Bruta RV Brasil", a, b, ap, bp)
        metric_val_pct("HEDGE ÍNDICE BR", h_a, h_b, hp_a, hp_b)
        metric_val_pct("Exp. Líquida RV Brasil", l_a, l_b, lp_a, lp_b)

    with c3:
        a, b = g(m_before, "Exp. Bruta RV Global"), g(m_after, "Exp. Bruta RV Global")
        ap, bp = g(m_before, "Exp. Bruta RV Global %"), g(m_after, "Exp. Bruta RV Global %")

        h_a, h_b = g(m_before, "HEDGE ÍNDICE Global"), g(m_after, "HEDGE ÍNDICE Global")
        hp_a, hp_b = g(m_before, "HEDGE ÍNDICE Global %"), g(m_after, "HEDGE ÍNDICE Global %")

        l_a, l_b = g(m_before, "Exp. Líquida RV Global"), g(m_after, "Exp. Líquida RV Global")
        lp_a, lp_b = g(m_before, "Exp. Líquida RV Global %"), g(m_after, "Exp. Líquida RV Global %")

        metric_val_pct("Exp. Bruta RV Global", a, b, ap, bp)
        metric_val_pct("HEDGE ÍNDICE Global", h_a, h_b, hp_a, hp_b)
        metric_val_pct("Exp. Líquida RV Global", l_a, l_b, lp_a, lp_b)

    st.markdown("---")

    # ================= LINHA 2 =================
    c4, c5 = st.columns(2)

    with c4:
        a, b = g(m_before, "HEDGE DOL"), g(m_after, "HEDGE DOL")
        ap, bp = g(m_before, "HEDGE DOL %"), g(m_after, "HEDGE DOL %")
        metric_val_pct("HEDGE Dólar", a, b, ap, bp)

    with c5:
        a, b = g(m_before, "Net Dólar"), g(m_after, "Net Dólar")
        ap, bp = g(m_before, "Net Dólar %"), g(m_after, "Net Dólar %")
        metric_val_pct("Net Dólar", a, b, ap, bp)




def _df_sem_caixa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cash_mask = _guess_cash_mask(df, caixa_ativos)
    return df.loc[~cash_mask].copy()

def _book_name_caixa_preferido(df: pd.DataFrame) -> str:
    """Escolhe um book_name de caixa existente (o maior) para manter consistência."""
    cash_mask = _guess_cash_mask(df, caixa_ativos)
    if not cash_mask.any():
        return "Caixa"
    tmp = df.loc[cash_mask].copy()
    tmp["asset_value"] = pd.to_numeric(tmp.get("asset_value", 0), errors="coerce").fillna(0.0)
    row = tmp.iloc[tmp["asset_value"].abs().values.argmax()]
    return str(row.get("book_name") or "Caixa")

def _aplicar_deposito_saque(df: pd.DataFrame, deposito: float, saque: float) -> pd.DataFrame:
    """
    Depósito: cria ativo novo 'Depósito' em classe Caixa (não mexe nos outros).
    Saque: reduz caixa existente (maior linha de caixa).
    Reusa _guess_cash_mask e _get_cash_available.
    """
    df = df.copy()
    df["asset_value"] = pd.to_numeric(df.get("asset_value", 0), errors="coerce").fillna(0.0)

    eps = 1e-6
    deposito = float(deposito or 0.0)
    saque = float(saque or 0.0)

    if deposito < -eps or saque < -eps:
        raise ValueError("Depósito e Saque devem ser valores positivos (o sinal é dado pelo campo).")

    # --- SAQUE (valida e debita do caixa) ---
    if saque > eps:
        caixa_disp = _get_cash_available(df)  # REUSA
        if saque - caixa_disp > eps:
            raise ValueError(
                f"Saque inválido: caixa insuficiente. "
                f"Solicitado: R$ {_format_ptbr_num(saque)} | Disponível: R$ {_format_ptbr_num(caixa_disp)}."
            )

        cash_mask = _guess_cash_mask(df, caixa_ativos)
        cash_candidates = df.loc[cash_mask].copy()
        cash_candidates["asset_value"] = pd.to_numeric(cash_candidates["asset_value"], errors="coerce").fillna(0.0)
        cash_idx = cash_candidates["asset_value"].abs().idxmax()

        df.at[cash_idx, "asset_value"] -= saque

    # --- DEPÓSITO (cria ativo novo em caixa) ---
    if deposito > eps:
        book_caixa = _book_name_caixa_preferido(df)
        nova = {}

        # mantém colunas que você já usa no app
        nova["instrument_name"] = "Caixa - Depósito"
        nova["book_name"] = "Caixa"
        nova["asset_value"] = deposito

        # se existir exposure, define exposure = deposito (ativo “normal”)
        value_col = _pick_col(df, ["exposure_value", "last_exposure_value"])
        if value_col:
            nova[value_col] = deposito

        # preserva campos mínimos pra não quebrar métricas
        if "instrument_type_id" in df.columns:
            nova["instrument_type_id"] = 0  # “outros”/desconhecido (não entra no equity mask)

        # completa colunas faltantes com NaN
        for c in df.columns:
            if c not in nova:
                nova[c] = np.nan

        df = pd.concat([df, pd.DataFrame([nova])], ignore_index=True)

    return _recalc_pct_asset_value(df)  # REUSA

# ======================================================
# PAGE
# ======================================================
def tela_simulacao() -> None:
    st.markdown("### Simulação de Carteira")
    st.caption(
    "Ativos: ΔR$ (compra > 0, venda < 0) com travas de posição e caixa. "
    "Derivativos: Δ ajusta somente a EXPOSIÇÃO (sem travas e sem contrapartida no caixa)."
)

    if "headers" not in st.session_state or not st.session_state.headers:
        st.warning("Faça login para consultar os dados.")
        return

    _init_state()

    # --------- filtros ----------
    with st.container():
        c_f1, c_f2, c_f3 = st.columns([1.1, 2, 1])
        with c_f1:
            data_base = st.date_input("Data-base", value=date.today(), key="sim_data_base")
        with c_f2:
            nome_carteira = st.selectbox("Carteira", sorted(CARTEIRAS.values()), index=0, key="sim_carteira")
        with c_f3:
            carregar = st.button("Carregar Base", use_container_width=True)

    try:
        carteira_id = next(k for k, v in CARTEIRAS.items() if v == nome_carteira)
    except StopIteration:
        _user_error("Carteira inválida. Selecione uma carteira válida.")
        return

    load_key = f"{carteira_id}|{str(data_base)}"
    load_key = f"{carteira_id}|{str(data_base)}"

# Se mudou data/carteira, invalida o carregamento e não mostra nada até clicar
    if st.session_state.sim_loaded_key != load_key:
        st.session_state.sim_loaded = False
        st.session_state.sim_base_df = None
        st.session_state.sim_overview = {}
        st.session_state.sim_exist_rows = {}

    if carregar:
        try:
            with st.spinner("Buscando posições..."):
                df_antes, overview = _carregar_posicoes(data_base, carteira_id, st.session_state.headers)
        except Exception:
            _user_error("Falha ao buscar posições na API. Verifique login/autorização e tente novamente.")
            return

        if df_antes.empty:
            st.info("Nenhuma posição encontrada para essa data/carteira.")
            return

        st.session_state.sim_base_df = df_antes
        st.session_state.sim_overview = overview
        st.session_state.sim_loaded_key = load_key
        st.session_state.sim_exist_rows = {}
        st.session_state.sim_loaded = True

    # GATE: nada abaixo aparece sem clique
    if not st.session_state.sim_loaded:
        st.info("Selecione data e carteira e clique em **Carregar Base** para iniciar a simulação.")
        return


    df_antes = st.session_state.sim_base_df
    overview = st.session_state.sim_overview
    

    ativos_unicos = sorted(df_antes["instrument_name"].dropna().unique().tolist())
    value_col = _pick_col(df_antes, ["exposure_value", "last_exposure_value"])

    # --------- callbacks ----------
    def _ensure_row(nome: str):
        if nome not in st.session_state.sim_exist_rows:
            classes_ativo = df_antes.loc[df_antes["instrument_name"] == nome, "book_name"].dropna().unique().tolist()
            default_class = classes_ativo[0] if classes_ativo else "Sem Classe"
            st.session_state.sim_exist_rows[nome] = {"classe": default_class, "delta": 0.0}

    def _on_change_delta(nome: str, pos_atual: float, is_deriv: bool):
        _ensure_row(nome)

        key_txt = f"sim_delta_txt_{nome}"
        raw = str(st.session_state.get(key_txt, "0,00"))
        delta = _parse_ptbr_currency(raw)

        # trava venda > posição (SÓ não-derivativo)
        if (not is_deriv) and delta < 0 and abs(delta) > pos_atual:
            delta = -float(pos_atual)


        st.session_state.sim_exist_rows[nome]["delta"] = float(delta)
        st.session_state[key_txt] = _fmt_ptbr_currency(float(delta))

    colL, colR = st.columns([1.15, 1])

    # ================= CONTROLES =================
    with colL:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown('<div class="h-label">Fluxo externo </div>', unsafe_allow_html=True)


        dep_txt = st.text_input("Depósito (R$)", value="0,00", key="sim_dep_txt")
        saq_txt = st.text_input("Saque (R$)", value="0,00", key="sim_saq_txt")

        deposito = _parse_ptbr_currency(dep_txt)
        saque = _parse_ptbr_currency(saq_txt)




        st.markdown('<div class="h-label">Ativos</div>', unsafe_allow_html=True)
        sel = st.multiselect(
            "Escolha os ativos",
            options=ativos_unicos,
            default=list(st.session_state.sim_exist_rows.keys()),
            key="sim_sel_exist",
            label_visibility="collapsed",
        )
        st.markdown('<div class="help">Selecione ativos e defina ΔR$ (venda = negativo). Venda ≤ posição atual. Compra ≤ caixa disponível.</div>', unsafe_allow_html=True)

        for nome in list(st.session_state.sim_exist_rows.keys()):
            if nome not in sel:
                del st.session_state.sim_exist_rows[nome]
                k = f"sim_delta_txt_{nome}"
                if k in st.session_state:
                    del st.session_state[k]

        for nome in sel:
            _ensure_row(nome)

        for nome in sel:
            dado = st.session_state.sim_exist_rows[nome]
            classes_ativo = df_antes.loc[df_antes["instrument_name"] == nome, "book_name"].dropna().unique().tolist()

            st.markdown('<div class="card-muted">', unsafe_allow_html=True)
            cA, cB = st.columns([2, 1])

            with cA:
                idx = classes_ativo.index(dado["classe"]) if dado["classe"] in classes_ativo else 0
                classe = st.selectbox(
                    f"Classe — {nome}",
                    options=classes_ativo if classes_ativo else ["Sem Classe"],
                    index=idx if classes_ativo else 0,
                    key=f"sim_cls_exist_{nome}",
                )
                dado["classe"] = classe

                filtro = (df_antes["instrument_name"].astype(str) == str(nome)) & (df_antes["book_name"].astype(str) == str(classe))
                pos_atual = float(pd.to_numeric(df_antes.loc[filtro, "asset_value"], errors="coerce").fillna(0.0).sum())
                is_deriv = False
                if filtro.sum() == 1:
                    row_idx = df_antes.index[filtro][0]
                    is_deriv = _is_deriv_row(df_antes, row_idx)


                if value_col:
                    pos_expo = float(pd.to_numeric(df_antes.loc[filtro, value_col], errors="coerce").fillna(0.0).sum())
                    pl_before = float(pd.to_numeric(df_antes.get("asset_value", 0), errors="coerce").fillna(0.0).sum())
                    st.caption(f"Exposição: R$ {_format_ptbr_num(pos_expo)} ({pos_expo/pl_before * 100:.1f} %)")
                else:
                    st.caption(f"Posição atual: R$ {_format_ptbr_num(pos_atual)}")

            with cB:
                key_txt = f"sim_delta_txt_{nome}"
                if key_txt not in st.session_state:
                    st.session_state[key_txt] = _fmt_ptbr_currency(float(dado.get("delta", 0.0)))

                st.text_input(
                    "Δ R$" if not is_deriv else "Δ Exposição",
                    key=key_txt,
                    on_change=_on_change_delta,
                    args=(nome, pos_atual, is_deriv),
                )

                if not is_deriv:
                    st.caption(f"Venda máx.: R$ {_format_ptbr_num(pos_atual)}")
                else:
                    st.caption("Derivativo: soma Δ direto na exposição (sem trava e sem caixa).")


            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown('<div class="small">', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    movimentos = [
        {"instrument_name": nome, "book_name": info["classe"], "delta": float(info.get("delta", 0.0))}
        for nome, info in st.session_state.sim_exist_rows.items()
        if abs(float(info.get("delta", 0.0))) > 1e-6
    ]

    df_depois = None
    m_before = None
    m_after = None
    agg_antes = None
    agg_depois = None

    # ================= GRÁFICOS (coluna direita) =================
    with colR:
        try:
            # 1) rebalance (compras/vendas com contrapartida no caixa)
            df_depois = _aplicar_movimentos_rebalance(df_antes, movimentos) if movimentos else df_antes.copy()

            # 2) fluxo externo (depósito/saque) — TEM que vir ANTES do st.metric e das métricas
            df_depois = _aplicar_deposito_saque(
                df_depois,
                deposito=deposito,
                saque=saque,
            )

        except RuntimeError as re_err:
            if str(re_err) == "CASH_NOT_FOUND":
                _guide_cash_mask_message()
                return
            _user_error("Erro inesperado na simulação. Recarregue a base e tente novamente.")
            return
        except ValueError as ve:
            _user_error(str(ve))
            return
        except Exception as e:
            st.error(f"Falha ao aplicar movimentos/fluxo: {type(e).__name__}: {e}")
            st.exception(e)
            return

        # 3) Agora sim: caixa após simulação (já com depósito/saque)
        try:
            caixa_depois = _get_cash_available(df_depois)
            st.metric("Caixa (líquido) após simulação", f"R$ {_format_ptbr_num(caixa_depois)}")
        except RuntimeError as re_err:
            if str(re_err) == "CASH_NOT_FOUND":
                _guide_cash_mask_message()
                return
        except Exception:
            pass

        # ======================================================
        # MÉTRICAS — before/after com overview coerente
        # ======================================================
        try:
            overview_before = dict(overview or {})
            overview_after  = dict(overview or {})

            pl_before = float(pd.to_numeric(df_antes.get("asset_value", 0), errors="coerce").fillna(0.0).sum())
            pl_after  = float(pd.to_numeric(df_depois.get("asset_value", 0), errors="coerce").fillna(0.0).sum())
            overview_before["net_asset_value"] = pl_before
            overview_after["net_asset_value"]  = pl_after

            if "instrument_type_id" in df_antes.columns:
                it_bef = pd.to_numeric(df_antes["instrument_type_id"], errors="coerce").fillna(-1)
                it_aft = pd.to_numeric(df_depois["instrument_type_id"], errors="coerce").fillna(-1)

                eq_mask_bef = it_bef.isin([1, 2])  # 1=stock, 2=etf (ajuste se for diferente)
                eq_mask_aft = it_aft.isin([1, 2])

                eq_bef = float(pd.to_numeric(df_antes.loc[eq_mask_bef, "asset_value"], errors="coerce").fillna(0.0).sum())
                eq_aft = float(pd.to_numeric(df_depois.loc[eq_mask_aft, "asset_value"], errors="coerce").fillna(0.0).sum())

                overview_before["equity_exposure"] = eq_bef
                overview_after["equity_exposure"]  = eq_aft

            df_antes_m  = _df_sem_caixa(df_antes)
            df_depois_m = _df_sem_caixa(df_depois)

            m_before = _calcular_metricas_gestao(overview_before, df_antes_m)
            m_after  = _calcular_metricas_gestao(overview_after, df_depois_m)

            if m_before.get("_erro"):
                raise ValueError(m_before["_erro"])
            if m_after.get("_erro"):
                raise ValueError(m_after["_erro"])

        except Exception as e:
            _user_error(f"Falha ao recalcular métricas: {e}")
            return

        # ======================================================
        # GRÁFICOS
        # ======================================================
        agg_antes  = _agg_por_classe(df_antes, include_cash=True)
        agg_depois = _agg_por_classe(df_depois, include_cash=True)


        aggA = _consolidar_outros(agg_antes)
        aggB = _consolidar_outros(agg_depois)
        all_labels = aggA["book_name"].astype(str).tolist() + aggB["book_name"].astype(str).tolist()
        color_by = _color_map_by_labels(all_labels)

        st.markdown('<div class="card" style="margin-top:10px;">', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            st.caption("Distribuição — antes")
            st.plotly_chart(_fig_pizza(aggA, color_by=color_by), use_container_width=True, key="pizza_antes")
        with d2:
            st.caption("Distribuição — depois")
            st.plotly_chart(_fig_pizza(aggB, color_by=color_by), use_container_width=True, key="pizza_depois")
        st.markdown('</div>', unsafe_allow_html=True)

    # ======================= FULL WIDTH =======================
    st.markdown("---")
    st.subheader("Métricas — antes vs depois")
    _render_metricas_resumo(m_before, m_after)

    st.markdown("---")
    st.subheader("Delta por Classe")

    delta_df = pd.merge(agg_antes, agg_depois, on="book_name", how="outer", suffixes=("_bef", "_aft")).fillna(0)
    delta_df["Δ Financeiro (R$)"] = delta_df["asset_value_aft"] - delta_df["asset_value_bef"]
    delta_df["Δ p.p."] = (delta_df["pct_aft"] - delta_df["pct_bef"]) * 100

    view = delta_df[["book_name", "asset_value_bef", "asset_value_aft", "Δ Financeiro (R$)", "pct_bef", "pct_aft", "Δ p.p."]].copy()
    view["asset_value_bef"] = view["asset_value_bef"].map(_format_ptbr_num)
    view["asset_value_aft"] = view["asset_value_aft"].map(_format_ptbr_num)
    view["Δ Financeiro (R$)"] = view["Δ Financeiro (R$)"].map(_format_ptbr_num)
    view["pct_bef"] = view["pct_bef"].map(lambda v: f"{v*100:.2f}".replace(".", ","))
    view["pct_aft"] = view["pct_aft"].map(lambda v: f"{v*100:.2f}".replace(".", ","))
    view["Δ p.p."] = view["Δ p.p."].map(lambda v: f"{v:+.2f}".replace(".", ","))

    st.dataframe(view.rename(columns={
        "book_name": "Classe",
        "asset_value_bef": "Antes (R$)",
        "asset_value_aft": "Depois (R$)",
        "pct_bef": "Antes (%)",
        "pct_aft": "Depois (%)",
    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Exportação")

    try:
        xls = _excel_simulacao(agg_antes, agg_depois, df_antes, df_depois, m_before, m_after)
        st.download_button(
            "Baixar Excel (estado atual)",
            data=xls,
            file_name=f"{nome_carteira}_simulacao_{str(data_base)}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="sim_btn_dl",
        )
    except Exception:
        _user_error(
            "Falha ao gerar Excel.\n"
            "Como resolver: tente novamente; se persistir, reduza o volume (menos linhas) e reteste."
        )
