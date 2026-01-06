import math
from datetime import date
from io import BytesIO

import pandas as pd
import requests
import streamlit as st

from utils import CARTEIRAS
from .rename_risco import rename

from .turso_http import init_favoritos_schema, load_favoritos, toggle_favorito

from .functions import (
    buscar_dados_liquidez,
    buscar_dados_resgates,
    get_portfolio_name,
    clear_data_if_portfolios_changed,
    format_excel_sheet_risco,
)

TELA_ID = "risco"
ABAS = ["Liquidez", "Resgates", "ADTV"]


# -----------------------------
# Estado mínimo
# -----------------------------
def _init_state():
    # seleções por aba
    if "selecoes_colunas_risco" not in st.session_state or not isinstance(st.session_state.selecoes_colunas_risco, dict):
        st.session_state.selecoes_colunas_risco = {aba: [] for aba in ABAS}
    else:
        for aba in ABAS:
            st.session_state.selecoes_colunas_risco.setdefault(aba, [])

    # carregamento
    if "risco_dados_carregados" not in st.session_state:
        st.session_state.risco_dados_carregados = False

    # dfs
    if "df_liquidez" not in st.session_state or not isinstance(st.session_state.df_liquidez, pd.DataFrame):
        st.session_state.df_liquidez = pd.DataFrame()
    if "df_resgates" not in st.session_state or not isinstance(st.session_state.df_resgates, pd.DataFrame):
        st.session_state.df_resgates = pd.DataFrame()
    if "df_adtv" not in st.session_state or not isinstance(st.session_state.df_adtv, pd.DataFrame):
        st.session_state.df_adtv = pd.DataFrame()

    # colunas
    if "colunas_liquidez" not in st.session_state:
        st.session_state.colunas_liquidez = []
    if "colunas_resgates" not in st.session_state:
        st.session_state.colunas_resgates = []
    if "colunas_adtv" not in st.session_state:
        st.session_state.colunas_adtv = []

    # portfolios
    if "selected_portfolios" not in st.session_state:
        st.session_state.selected_portfolios = []
    if "last_selected_portfolios" not in st.session_state:
        st.session_state.last_selected_portfolios = []


def _init_favoritos_db():
    init_favoritos_schema()

    if "favoritos_db" not in st.session_state or not isinstance(st.session_state.favoritos_db, dict):
        st.session_state.favoritos_db = {}

    if TELA_ID not in st.session_state.favoritos_db:
        st.session_state.favoritos_db[TELA_ID] = load_favoritos(TELA_ID)

    for a in ABAS:
        st.session_state.favoritos_db[TELA_ID].setdefault(a, [])


def _colunas_disponiveis():
    return {
        "Liquidez": [c for c in st.session_state.colunas_liquidez if "repetido" not in str(c).lower()],
        "Resgates": [c for c in st.session_state.colunas_resgates if "repetido" not in str(c).lower()],
        "ADTV": [c for c in st.session_state.colunas_adtv if "repetido" not in str(c).lower()],
    }


# -----------------------------
# Tela
# -----------------------------
def mostrar_risco(ctx=None):
    _init_state()
    _init_favoritos_db()

    st.title("Risco")
    st.subheader("Buscar posições")

    data_inicio, data_fim = st.date_input(
        "Escolha o intervalo de datas",
        [date.today(), date.today()],
        key="risco_intervalo_datas",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        portfolio_names = st.multiselect(
            "Selecione as carteiras",
            options=list(CARTEIRAS.values()),
            default=[],
            format_func=lambda x: x,
            key="risco_portfolio_multiselect",
        )

    selected_ids = [k for k, v in CARTEIRAS.items() if v in portfolio_names]
    st.session_state.selected_portfolios = selected_ids

    # limpa dados se mudou carteira (não mexe em favoritos)
    clear_data_if_portfolios_changed()

    with col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

        if st.button("Buscar dados", key="risco_btn_buscar") and selected_ids:
            try:
                # Liquidez e Resgates
                df_liq = buscar_dados_liquidez(data_inicio, data_fim, selected_ids)
                df_res = buscar_dados_resgates(data_inicio, data_fim, selected_ids)

                # ADTV vem das observations do retorno de liquidez (como você fazia)
                adtv_rows = []
                # df_liq pode ser dict: {"main": df, "observations": [...]}
                obs_list = df_liq.get("observations", []) if isinstance(df_liq, dict) else []
                for obs in obs_list:
                    if isinstance(obs, list):
                        adtv_rows.extend(obs)

                df_adtv = pd.DataFrame(adtv_rows) if adtv_rows else pd.DataFrame()

                # main de liquidez
                df_liquidez = df_liq.get("main", pd.DataFrame()) if isinstance(df_liq, dict) else pd.DataFrame()
                df_resgates = df_res if isinstance(df_res, pd.DataFrame) else pd.DataFrame()

                # renomeia
                df_liquidez = rename(df_liquidez)
                df_resgates = rename(df_resgates)
                df_adtv = rename(df_adtv)

                # salva
                st.session_state.df_liquidez = df_liquidez
                st.session_state.df_resgates = df_resgates
                st.session_state.df_adtv = df_adtv

                st.session_state.colunas_liquidez = sorted(df_liquidez.columns) if not df_liquidez.empty else []
                st.session_state.colunas_resgates = sorted(df_resgates.columns) if not df_resgates.empty else []
                st.session_state.colunas_adtv = sorted(df_adtv.columns) if not df_adtv.empty else []

                st.session_state.risco_dados_carregados = True
                st.success("Dados carregados com sucesso.")

            except Exception as e:
                st.error(f"Erro ao buscar dados: {e}")

    # -----------------------------
    # Configuração após carregar
    # -----------------------------
    if not st.session_state.risco_dados_carregados:
        return

    aba = st.radio("Escolha a aba para configurar:", ABAS, key="risco_aba_config")

    colunas_disponiveis = _colunas_disponiveis()
    colunas_aba = sorted(colunas_disponiveis.get(aba, []))

    prefixo_checkbox = {"Liquidez": "liq_", "Resgates": "res_", "ADTV": "adtv_"}
    prefixo = prefixo_checkbox[aba]

    # Ações rápidas
    st.markdown("### Ações rápidas")
    ac1, ac2, ac3 = st.columns([1, 1, 1])

    with ac1:
        if st.button("✅ Selecionar todas", key=f"risco_select_all_{aba}"):
            st.session_state.selecoes_colunas_risco[aba] = colunas_aba.copy()
            for c in colunas_aba:
                st.session_state[f"{prefixo}{c}"] = True
            st.rerun()

    with ac2:
        if st.button("❌ Limpar seleção", key=f"risco_clear_all_{aba}"):
            st.session_state.selecoes_colunas_risco[aba] = []
            for c in colunas_aba:
                st.session_state[f"{prefixo}{c}"] = False
            st.rerun()

    with ac3:
        if st.button("⭐ Favoritos", key=f"risco_apply_fav_{aba}"):
            favs = st.session_state.favoritos_db[TELA_ID].get(aba, [])
            favs_ok = [c for c in favs if c in colunas_aba]
            st.session_state.selecoes_colunas_risco[aba] = favs_ok
            for c in colunas_aba:
                st.session_state[f"{prefixo}{c}"] = c in favs_ok
            st.rerun()

    # Export
    if st.button("Exportar para Excel", key="risco_export_excel"):
        try:
            portfolio_names_for_file = "_".join(
                [get_portfolio_name(pid).replace(" ", "_") for pid in st.session_state.selected_portfolios]
            )
            output = BytesIO()

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                for pid in st.session_state.selected_portfolios:
                    nome = get_portfolio_name(pid)

                    df_liq = st.session_state.df_liquidez.copy()
                    df_res = st.session_state.df_resgates.copy()
                    df_adtv = st.session_state.df_adtv.copy() if isinstance(st.session_state.df_adtv, pd.DataFrame) else pd.DataFrame()

                    # filtra por carteira
                    if not df_liq.empty and "ID Carteira" in df_liq.columns:
                        df_liq = df_liq[df_liq["ID Carteira"] == pid]
                    if not df_res.empty and "ID Carteira" in df_res.columns:
                        df_res = df_res[df_res["ID Carteira"] == pid]
                    if not df_adtv.empty and "ID Carteira" in df_adtv.columns:
                        df_adtv = df_adtv[df_adtv["ID Carteira"] == pid]

                    # aplica seleção de colunas
                    if not df_liq.empty:
                        cols = [c for c in st.session_state.selecoes_colunas_risco["Liquidez"] if c in df_liq.columns]
                        if cols:
                            df_liq = df_liq[cols]
                    if not df_res.empty:
                        cols = [c for c in st.session_state.selecoes_colunas_risco["Resgates"] if c in df_res.columns]
                        if cols:
                            df_res = df_res[cols]
                    if not df_adtv.empty:
                        cols = [c for c in st.session_state.selecoes_colunas_risco["ADTV"] if c in df_adtv.columns]
                        if cols:
                            df_adtv = df_adtv[cols]

                    format_excel_sheet_risco(writer, df_liq, df_res, df_adtv, nome)

            st.download_button(
                label="📥 Baixar Excel",
                data=output.getvalue(),
                file_name=f"risco_{portfolio_names_for_file}_{data_inicio}_a_{data_fim}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Erro ao exportar: {e}")

    # -----------------------------
    # Grid de colunas + estrela persistente
    # -----------------------------
    st.markdown("---")
    st.markdown("### Colunas disponíveis")

    if not colunas_aba:
        st.info("Nenhuma coluna disponível nesta aba.")
        return

    num_colunas = 3
    n_linhas = math.ceil(len(colunas_aba) / num_colunas)

    colunas_selecionadas_aba = []

    for linha in range(n_linhas):
        cols = st.columns(num_colunas, gap="large")
        fatia = colunas_aba[linha * num_colunas:(linha + 1) * num_colunas]

        for i, col in enumerate(fatia):
            with cols[i]:
                key_cb = f"{prefixo}{col}"
                row_cols = st.columns([0.85, 0.15])

                with row_cols[0]:
                    checked = st.checkbox(
                        col,
                        value=(col in st.session_state.selecoes_colunas_risco[aba]),
                        key=key_cb,
                    )

                with row_cols[1]:
                    is_fav = col in st.session_state.favoritos_db[TELA_ID].get(aba, [])
                    estrela = "⭐" if is_fav else "☆"

                    if st.button(estrela, key=f"fav_{TELA_ID}_{aba}_{col}"):
                        novo = toggle_favorito(TELA_ID, aba, col)

                        # atualiza cache local
                        st.session_state.favoritos_db[TELA_ID].setdefault(aba, [])
                        if novo and col not in st.session_state.favoritos_db[TELA_ID][aba]:
                            st.session_state.favoritos_db[TELA_ID][aba].append(col)
                        if (not novo) and col in st.session_state.favoritos_db[TELA_ID][aba]:
                            st.session_state.favoritos_db[TELA_ID][aba].remove(col)

                        st.rerun()

                if checked:
                    colunas_selecionadas_aba.append(col)

    st.session_state.selecoes_colunas_risco[aba] = colunas_selecionadas_aba
