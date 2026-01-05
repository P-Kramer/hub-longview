# simul.py
from __future__ import annotations

import io
from datetime import date
from typing import List, Dict, Any

import requests
import streamlit as st
import pandas as pd

from utils import BASE_URL_API, CARTEIRAS

PIZZA_LIMIAR_OUTROS = 0.02  # classes <2% viram "Outros"


# --------------------------
# Helpers visuais e gerais
# --------------------------
CSS = """
<style>
/* Reset leve */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }

/* Cards minimalistas */
.card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 12px;
  padding: 14px 14px;
  background: white;
}
.card-muted {
  border: 1px dashed rgba(0,0,0,0.10);
  border-radius: 12px;
  padding: 12px 12px;
  background: rgba(0,0,0,0.02);
}

/* Títulos menores e limpos */
.h-label {
  font-weight: 600; font-size: 0.95rem; margin: 0 0 6px 0;
}
.help {
  color: #6b7280; font-size: 0.85rem; margin-top: -4px; margin-bottom: 8px;
}

/* Separador curto */
.hr { height: 1px; background: rgba(0,0,0,0.06); margin: 10px 0 14px 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def _format_ptbr_num(v: float) -> str:
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _consolidar_outros(agg: pd.DataFrame, limiar: float = PIZZA_LIMIAR_OUTROS) -> pd.DataFrame:
    if "pct" not in agg.columns:
        total = agg["asset_value"].sum()
        agg = agg.assign(pct=agg["asset_value"] / total if total else 0.0)
    grandes = agg[agg["pct"] >= limiar].copy()
    pequenos = agg[agg["pct"] < limiar].copy()
    if pequenos.empty:
        return grandes.sort_values("asset_value", ascending=False).reset_index(drop=True)
    outros = pd.DataFrame({
        "book_name": ["Outros"],
        "asset_value": [pequenos["asset_value"].sum()],
        "pct": [pequenos["asset_value"].sum() / agg["asset_value"].sum() if agg["asset_value"].sum() else 0.0],
    })
    res = pd.concat([grandes, outros], ignore_index=True)
    return res.sort_values("asset_value", ascending=False).reset_index(drop=True)


def _fig_pizza(agg: pd.DataFrame, *, height: int = 600):
    import plotly.graph_objects as go

    labels = agg["book_name"].tolist()
    values = agg["asset_value"].tolist()

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        sort=False,
        direction="clockwise",
        texttemplate="<b>%{percent}</b>",
        textinfo="percent",
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>Financeiro: R$ %{value:,.2f}<br>% Alocado: %{percent}<extra></extra>",
        showlegend=True,
    )])

    fig.update_layout(
        template="plotly_white",
        # altura maior e espaço pra legenda
        height=height,
        margin=dict(t=8, b=150, l=8, r=8),
        # legenda em linha, centralizada, com fonte menor
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,             # empurra a legenda para baixo dentro do canvas
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            itemwidth=40,
        ),
        uniformtext_minsize=12,
        uniformtext_mode="show",
    )
    return fig


# --------------------------
# API – Posições
# --------------------------
def _carregar_posicoes(data_base: date, carteira_id: str, headers: Dict[str, str]) -> pd.DataFrame:
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
    resultado = resp.json()

    objetos = resultado.get("objects", {})
    inst_positions_acumulado: List[Dict[str, Any]] = []
    iter_values = objetos.values() if isinstance(objetos, dict) else (objetos if isinstance(objetos, list) else [])
    for obj in iter_values:
        pos = obj.get("instrument_positions") if isinstance(obj, dict) else None
        if not pos:
            continue
        if isinstance(pos, list):
            inst_positions_acumulado.extend(pos)
        elif isinstance(pos, dict):
            inst_positions_acumulado.append(pos)

    if not inst_positions_acumulado:
        return pd.DataFrame()

    df = pd.json_normalize(inst_positions_acumulado)
    df["asset_value"] = pd.to_numeric(df.get("asset_value", 0), errors="coerce").fillna(0.0)
    df["book_name"] = df.get("book_name").fillna("Sem Classe")
    df["instrument_name"] = df.get("instrument_name").fillna("Sem Nome")
    return df


# --------------------------
# Engine da simulação
# --------------------------
def _aplicar_movimentos(df_base: pd.DataFrame, movimentos: List[Dict[str, Any]]) -> pd.DataFrame:
    if df_base is None or df_base.empty:
        return pd.DataFrame(columns=["instrument_name", "book_name", "asset_value"])

    df = df_base.copy()
    for col in ("instrument_name", "book_name", "asset_value"):
        if col not in df.columns:
            df[col] = "" if col != "asset_value" else 0.0
    df["asset_value"] = pd.to_numeric(df["asset_value"], errors="coerce").fillna(0.0)

    records = df[["instrument_name", "book_name"]].reset_index(drop=True).to_dict("records")
    idx = {(r.get("instrument_name"), r.get("book_name")): i for i, r in enumerate(records)}

    for mv in movimentos:
        if not isinstance(mv, dict):
            raise ValueError(f"Movimento inválido: {mv!r}")

        nome = mv.get("instrument_name")
        classe = mv.get("book_name")
        delta = mv.get("delta", 0)

        try:
            delta = float(delta or 0)
        except Exception:
            raise ValueError(f"Delta inválido para '{nome}': {delta!r}")

        if not nome:
            raise ValueError(f"Movimento sem 'instrument_name': {mv!r}")
        if delta == 0:
            continue

        key_candidates = [(nome, classe)] if classe else []
        if not key_candidates:
            key_candidates.extend([(nome, bn) for bn in df["book_name"].unique()])

        row_index = None
        for k in key_candidates:
            if k in idx:
                row_index = idx[k]
                break

        if row_index is not None:
            df.loc[row_index, "asset_value"] = float(df.loc[row_index, "asset_value"]) + delta
        else:
            if not classe:
                raise ValueError(f"Ativo '{nome}' não existe na base e 'book_name' não foi informado.")
            new_row = {"instrument_name": nome, "book_name": classe, "asset_value": delta}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            idx[(nome, classe)] = len(df) - 1

    df["asset_value"] = df["asset_value"].fillna(0.0)
    df = df[df["asset_value"] != 0].reset_index(drop=True)
    return df


def _agg_por_classe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["book_name", "asset_value", "pct"])
    agg = df.groupby("book_name", as_index=False)["asset_value"].sum()
    total = float(agg["asset_value"].sum())
    agg["pct"] = agg["asset_value"] / total if total else 0.0
    return agg.sort_values("asset_value", ascending=False).reset_index(drop=True)


def _excel_simulacao(agg_antes: pd.DataFrame, agg_depois: pd.DataFrame,
                     df_antes: pd.DataFrame, df_depois: pd.DataFrame) -> bytes:
    delta = pd.merge(agg_antes, agg_depois, on="book_name", how="outer", suffixes=("_bef", "_aft")).fillna(0)
    delta["delta_value"] = delta["asset_value_aft"] - delta["asset_value_bef"]
    delta["delta_pct_pt"] = (delta["pct_aft"] - delta["pct_bef"]) * 100

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        agg_antes.rename(columns={"book_name": "Classe", "asset_value": "Financeiro", "pct": "%_Alocado"}).to_excel(
            writer, index=False, sheet_name="Resumo_antes"
        )
        agg_depois.rename(columns={"book_name": "Classe", "asset_value": "Financeiro", "pct": "%_Alocado"}).to_excel(
            writer, index=False, sheet_name="Resumo_depois"
        )
        delta.rename(columns={"book_name": "Classe"}).to_excel(writer, index=False, sheet_name="Delta")
        df_antes.to_excel(writer, index=False, sheet_name="Raw_antes")
        df_depois.to_excel(writer, index=False, sheet_name="Raw_depois")

        wb = writer.book
        pct_fmt = wb.add_format({"num_format": "0.00%"})
        money_fmt = wb.add_format({"num_format": "#,##0.00"})
        bold = wb.add_format({"bold": True})

        for sh in ["Resumo_antes", "Resumo_depois"]:
            ws = writer.sheets[sh]
            ws.set_row(0, None, bold)
            ws.set_column(0, 0, 28)
            ws.set_column(1, 1, 18, money_fmt)
            ws.set_column(2, 2, 12, pct_fmt)

        ws = writer.sheets["Delta"]
        ws.set_row(0, None, bold)
        ws.set_column(0, 0, 28)
        ws.set_column(1, 1, 18, money_fmt)
        ws.set_column(2, 2, 12, pct_fmt)
        ws.set_column(3, 3, 18, money_fmt)
        ws.set_column(4, 4, 12, pct_fmt)
        ws.set_column(5, 5, 18, money_fmt)
        ws.set_column(6, 6, 14)

    return buf.getvalue()


# --------------------------
# Estado
# --------------------------
def _init_sim_state():
    if "sim_movs_novos" not in st.session_state:
        st.session_state.sim_movs_novos = []  # lista de dicts (apenas NOVOS)
    if "sim_exist_rows" not in st.session_state:
        st.session_state.sim_exist_rows = {}  # nome -> {"classe": str, "delta": float}
    if "sim_base_df" not in st.session_state:
        st.session_state.sim_base_df = None
    if "sim_loaded_key" not in st.session_state:
        st.session_state.sim_loaded_key = None
    # campos de novo ativo
    if "sim_novo_nome" not in st.session_state:
        st.session_state.sim_novo_nome = ""
    if "sim_novo_classe" not in st.session_state:
        st.session_state.sim_novo_classe = ""
    if "sim_novo_valor" not in st.session_state:
        st.session_state.sim_novo_valor = 0.0


def _stage_move_novo(nome: str, classe: str, delta: float):
    if not nome or not classe or delta == 0:
        return
    st.session_state.sim_movs_novos.append({
        "instrument_name": nome,
        "book_name": classe,
        "delta": float(delta),
    })


# --------------------------
# Tela 2 – Simulação (UI clean)
# --------------------------
def tela_simulacao() -> None:
    st.markdown("### ⚙️ Simulação de Carteira — ao vivo")
    st.caption("Selecione ativos, ajuste ΔR$ e veja o impacto instantâneo por classe. Δ>0 compra, Δ<0 venda.")

    if "headers" not in st.session_state or not st.session_state.headers:
        st.warning("Faça login para consultar os dados.")
        return

    _init_sim_state()

    # ----- Filtros em card -----
    with st.container():
        c_f1, c_f2, c_f3, c_f4 = st.columns([1.1, 2, 1, 1])
        with c_f1:
            st.markdown('<div class="h-label">Data-base</div>', unsafe_allow_html=True)
            data_base = st.date_input("", value=date.today(), key="sim_data_base", label_visibility="collapsed")
        with c_f2:
            st.markdown('<div class="h-label">Carteira</div>', unsafe_allow_html=True)
            nome_carteira = st.selectbox("", sorted(CARTEIRAS.values()), index=0, key="sim_carteira", label_visibility="collapsed")
        with c_f3:
            st.markdown('<div class="h-label">&nbsp;</div>', unsafe_allow_html=True)
            carregar = st.button("🔄 Carregar Base", key="sim_btn_carregar", use_container_width=True)
        with c_f4:
            st.markdown('<div class="h-label">&nbsp;</div>', unsafe_allow_html=True)
            reset = st.button("🧹 Limpar simulação", key="sim_btn_reset", use_container_width=True)

    try:
        carteira_id = next(k for k, v in CARTEIRAS.items() if v == nome_carteira)
    except StopIteration:
        st.error("Carteira inválida.")
        return

    load_key = f"{carteira_id}|{str(data_base)}"
    if reset:
        st.session_state.sim_exist_rows = {}
        st.session_state.sim_movs_novos = []

    if carregar or st.session_state.sim_base_df is None or st.session_state.sim_loaded_key != load_key:
        with st.spinner("Buscando posições..."):
            df_antes = _carregar_posicoes(data_base, carteira_id, st.session_state.headers)
        if df_antes.empty:
            st.info("Nenhuma posição encontrada.")
            return
        st.session_state.sim_base_df = df_antes
        st.session_state.sim_loaded_key = load_key
        st.session_state.sim_exist_rows = {}
        st.session_state.sim_movs_novos = []
        st.session_state.sim_novo_nome = ""
        st.session_state.sim_novo_classe = ""
        st.session_state.sim_novo_valor = 0.0

    df_antes = st.session_state.sim_base_df
    ativos_unicos = sorted(df_antes["instrument_name"].unique().tolist())

    # ----- Layout em 2 colunas: Controles | Resultados -----
    colL, colR = st.columns([1.1, 1])

    # ===================== Coluna Esquerda – Controles =====================
    with colL:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        tabs = st.tabs(["🧰 Existentes", "➕ Novo"])

        # -------- Tab: Existentes --------
        with tabs[0]:
            st.markdown('<div class="h-label">Ativos</div>', unsafe_allow_html=True)
            sel = st.multiselect(
                "Escolha os ativos",
                options=ativos_unicos,
                default=list(st.session_state.sim_exist_rows.keys()),
                key="sim_sel_exist",
                label_visibility="collapsed"
            )
            st.markdown('<div class="help">Selecione para adicionar linhas abaixo. Ajuste a Classe e ΔR$.</div>', unsafe_allow_html=True)

            # remove quem saiu
            for nome in list(st.session_state.sim_exist_rows.keys()):
                if nome not in sel:
                    del st.session_state.sim_exist_rows[nome]

            # cria entradas novas
            for nome in sel:
                if nome not in st.session_state.sim_exist_rows:
                    classes_ativo = df_antes.loc[df_antes["instrument_name"] == nome, "book_name"].unique().tolist()
                    default_class = classes_ativo[0] if classes_ativo else "Sem Classe"
                    st.session_state.sim_exist_rows[nome] = {"classe": default_class, "delta": 0.0}

            # linhas
            for nome in sel:
                classes_ativo = df_antes.loc[df_antes["instrument_name"] == nome, "book_name"].unique().tolist()
                dado = st.session_state.sim_exist_rows[nome]
                st.markdown('<div class="card-muted">', unsafe_allow_html=True)
                cA, cB = st.columns([2, 1])
                with cA:
                    idx = classes_ativo.index(dado["classe"]) if dado["classe"] in classes_ativo else 0
                    classe = st.selectbox(
                        f"Classe — {nome}",
                        options=classes_ativo,
                        index=idx,
                        key=f"sim_cls_exist_{nome}",
                    )
                    st.session_state.sim_exist_rows[nome]["classe"] = classe
                with cB:
                    delta = st.number_input(
                        "Δ R$",
                        value=float(dado["delta"]),
                        step=1000.0,
                        key=f"sim_delta_exist_{nome}",
                    )
                    st.session_state.sim_exist_rows[nome]["delta"] = float(delta)
                st.markdown('</div>', unsafe_allow_html=True)

        # -------- Tab: Novo --------
        with tabs[1]:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.session_state.sim_novo_nome = st.text_input("Nome do novo ativo", value=st.session_state.sim_novo_nome, key="sim_input_novo_nome")
            with c2:
                st.session_state.sim_novo_classe = st.text_input("Classe (book_name)", value=st.session_state.sim_novo_classe, key="sim_input_novo_classe")

            c3, c4 = st.columns([1, 1])
            with c3:
                st.session_state.sim_novo_valor = st.number_input("Valor (R$)", value=float(st.session_state.sim_novo_valor), step=1000.0, key="sim_input_novo_valor")
            with c4:
                if st.button("Adicionar NOVO", key="sim_btn_add_novo", use_container_width=True):
                    if not st.session_state.sim_novo_nome or not st.session_state.sim_novo_classe or float(st.session_state.sim_novo_valor) == 0:
                        st.error("Preencha nome, classe e valor ≠ 0.")
                    else:
                        _stage_move_novo(st.session_state.sim_novo_nome, st.session_state.sim_novo_classe, float(st.session_state.sim_novo_valor))
                        st.success(f"Novo ativo: {st.session_state.sim_novo_nome} / {st.session_state.sim_novo_classe} / Δ R$ {_format_ptbr_num(st.session_state.sim_novo_valor)}")
                        st.session_state.sim_novo_nome = ""
                        st.session_state.sim_novo_classe = ""
                        st.session_state.sim_novo_valor = 0.0
                        st.experimental_rerun()

            if st.session_state.sim_movs_novos:
                st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
                st.caption("Novos adicionados")
                st.table(pd.DataFrame(st.session_state.sim_movs_novos))

        st.markdown('</div>', unsafe_allow_html=True)

    # ===================== Coluna Direita – Resultados =====================
    # Monta movimentos
    mov_existentes = [
        {"instrument_name": nome, "book_name": info["classe"], "delta": float(info["delta"])}
        for nome, info in st.session_state.sim_exist_rows.items()
        if float(info["delta"]) != 0.0
    ]
    mov_novos = list(st.session_state.sim_movs_novos)
    movimentos = mov_existentes + mov_novos

    try:
        df_depois = _aplicar_movimentos(df_antes, movimentos) if movimentos else df_antes.copy()
    except Exception as e:
        st.error(f"Erro aplicando movimentos: {e}")
        return

    agg_antes = _agg_por_classe(df_antes)
    agg_depois = _agg_por_classe(df_depois)

    total_antes = float(agg_antes["asset_value"].sum()) if not agg_antes.empty else 0.0
    total_depois  = float(agg_depois["asset_value"].sum()) if not agg_depois.empty else 0.0
    cash_delta   = total_depois - total_antes

    with colR:

        # Cards: Donuts lado a lado
        st.markdown('<div class="card" style="margin-top:10px;">', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            st.caption("Distribuição — antes")
            st.plotly_chart(_fig_pizza(_consolidar_outros(agg_antes)), use_container_width=True,key="antes")
        with d2:
            st.caption("Distribuição — depois")
            st.plotly_chart(_fig_pizza(_consolidar_outros(agg_depois)), use_container_width=True,key="depois")
        st.markdown('</div>', unsafe_allow_html=True)


        delta_df = pd.merge(agg_antes, agg_depois, on="book_name", how="outer", suffixes=("_bef", "_aft")).fillna(0)
        delta_df["Δ Financeiro (R$)"] = delta_df["asset_value_aft"] - delta_df["asset_value_bef"]
        delta_df["Δ p.p."] = (delta_df["pct_aft"] - delta_df["pct_bef"]) * 100

        view_delta = delta_df[["book_name", "asset_value_bef", "asset_value_aft",
                               "Δ Financeiro (R$)", "pct_bef", "pct_aft", "Δ p.p."]].copy()
        view_delta["asset_value_bef"] = view_delta["asset_value_bef"].map(_format_ptbr_num)
        view_delta["asset_value_aft"] = view_delta["asset_value_aft"].map(_format_ptbr_num)
        view_delta["pct_bef"] = view_delta["pct_bef"].map(lambda v: f"{v*100:.2f}".replace(".", ","))
        view_delta["pct_aft"] = view_delta["pct_aft"].map(lambda v: f"{v*100:.2f}".replace(".", ","))
        view_delta["Δ p.p."] = view_delta["Δ p.p."].map(lambda v: f"{v:.2f}".replace(".", ","))


        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        try:
            xls = _excel_simulacao(agg_antes, agg_depois, df_antes, df_depois)
            st.download_button(
                "⬇️ Baixar Excel (estado atual)",
                key="sim_btn_dl",
                data=xls,
                file_name=f"{nome_carteira}_simulacao_{str(data_base)}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Falha ao gerar Excel: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
