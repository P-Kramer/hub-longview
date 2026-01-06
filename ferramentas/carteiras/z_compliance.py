import math
from datetime import date
from io import BytesIO

import pandas as pd
import requests
import streamlit as st

from utils import CARTEIRAS
from .rename_compliance import rename, mapa_compliance

from .turso_http import init_favoritos_schema, load_favoritos, toggle_favorito

from .functions import (
    aplicar_estilo_percentual,
    get_portfolio_name,
    clear_data_if_portfolios_changed,
    reset_column_selection,
    formatar_percentuais_df,
    BASE_URL_API,
    format_excel_sheet_compliance,
)

TELA_ID = "compliance"
ABA_ID = "Compliance"


def _init_state():
    if "df_compliance" not in st.session_state or not isinstance(st.session_state.df_compliance, pd.DataFrame):
        st.session_state.df_compliance = pd.DataFrame()

    if "colunas_compliance" not in st.session_state:
        st.session_state.colunas_compliance = []

    if "selecoes_colunas_compliance" not in st.session_state or not isinstance(st.session_state.selecoes_colunas_compliance, list):
        st.session_state.selecoes_colunas_compliance = []

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

    st.session_state.favoritos_db[TELA_ID].setdefault(ABA_ID, [])


def mostrar_compliance(ctx=None):
    _init_state()
    _init_favoritos_db()

    st.title("Compliance")
    st.subheader("Buscar posições")

    data_inicio, data_fim = st.date_input(
        "Escolha o intervalo de datas",
        [date.today(), date.today()],
        key="compliance_intervalo_datas",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        portfolio_names = st.multiselect(
            "Selecione as carteiras",
            options=list(CARTEIRAS.values()),
            default=[],
            format_func=lambda x: x,
            key="compliance_portfolio_multiselect",
        )

    selected_ids = [k for k, v in CARTEIRAS.items() if v in portfolio_names]
    st.session_state.selected_portfolios = selected_ids

    clear_data_if_portfolios_changed()

    with col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

        if st.button("Buscar dados", key="compliance_btn_buscar"):
            if not selected_ids:
                st.warning("Selecione pelo menos uma carteira!")
            else:
                payload = {
                    "start_date": str(data_inicio),
                    "end_date": str(data_fim),
                    "portfolio_ids": selected_ids,
                }

                try:
                    r = requests.post(
                        f"{BASE_URL_API}/compliance/compliancestatus",
                        json=payload,
                        headers=st.session_state.headers,
                        timeout=60,
                    )
                    r.raise_for_status()
                    resultado = r.json()
                    dados = resultado.get("objects", {})

                    registros = []
                    for item in dados.values():
                        if isinstance(item, list):
                            registros.extend(item)
                        else:
                            registros.append(item)

                    df = pd.json_normalize(registros)
                    df = rename(df)

                    if "Status de Compliance" in df.columns:
                        df["Status de Compliance"] = df["Status de Compliance"].replace({
                            1: "ENQUADRADO",
                            2: "ALERTA",
                            3: "DESENQUADRADO",
                        })

                    # drop repetidos
                    cols_to_drop = [c for c in df.columns if "repetido" in str(c).lower()]
                    if cols_to_drop:
                        df = df.drop(columns=cols_to_drop, errors="ignore")

                    df = df.dropna(axis=1, how="all")

                    st.session_state.df_compliance = df
                    st.session_state.colunas_compliance = sorted(df.columns)

                    if df.empty:
                        st.warning("Nenhum dado encontrado para os filtros informados.")
                    else:
                        nomes = ", ".join([get_portfolio_name(pid) for pid in selected_ids])
                        st.success(f"Dados recebidos com sucesso para: {nomes}")

                        # Se sua formatação mexe em df no session_state, ok.
                        reset_column_selection()
                        formatar_percentuais_df()

                except Exception as e:
                    if str(e) in ["'Status de Compliance'", '"Status de Compliance"']:
                        st.info("Não há dados de Status de Compliance para estas datas.")
                    else:
                        st.error(f"Erro ao buscar dados: {e}")

    # -----------------------------------
    # UI de seleção
    # -----------------------------------
    if st.session_state.df_compliance.empty:
        return

    st.dataframe(aplicar_estilo_percentual(st.session_state.df_compliance), use_container_width=True)

    st.markdown("## Selecionar colunas para exportar")

    # Só as colunas do mapa_compliance
    colunas_disponiveis = [
        col for col in sorted(st.session_state.colunas_compliance)
        if col in set(mapa_compliance.values())
    ]
    prefixo = "cmp_"

    # Ações rápidas
    st.markdown("### Ações rápidas")
    ac1, ac2, ac3 = st.columns([1, 1, 1])

    with ac1:
        if st.button("✅ Selecionar todas", key="compliance_select_all"):
            st.session_state.selecoes_colunas_compliance = colunas_disponiveis.copy()
            for c in colunas_disponiveis:
                st.session_state[f"{prefixo}{c}"] = True
            st.rerun()

    with ac2:
        if st.button("❌ Limpar seleção", key="compliance_clear_all"):
            st.session_state.selecoes_colunas_compliance = []
            for c in colunas_disponiveis:
                st.session_state[f"{prefixo}{c}"] = False
            st.rerun()

    with ac3:
        if st.button("⭐ Favoritos", key="compliance_apply_fav"):
            favs = st.session_state.favoritos_db[TELA_ID].get(ABA_ID, [])
            favs_ok = [c for c in favs if c in colunas_disponiveis]
            st.session_state.selecoes_colunas_compliance = favs_ok
            for c in colunas_disponiveis:
                st.session_state[f"{prefixo}{c}"] = c in favs_ok
            st.rerun()

    # Exportação
    if st.button("Exportar para Excel", key="compliance_export_excel"):
        if not st.session_state.selecoes_colunas_compliance:
            st.warning("Selecione pelo menos uma coluna!")
        else:
            try:
                df_filtrado = st.session_state.df_compliance.copy()
                portfolio_names_for_file = "_".join(
                    [get_portfolio_name(pid).replace(" ", "_") for pid in st.session_state.selected_portfolios]
                )

                output = BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    for pid in st.session_state.selected_portfolios:
                        nome = get_portfolio_name(pid)

                        if "ID Carteira" in df_filtrado.columns:
                            df_portfolio = df_filtrado[df_filtrado["ID Carteira"] == pid].copy()
                        else:
                            df_portfolio = df_filtrado.copy()

                        cols = [c for c in st.session_state.selecoes_colunas_compliance if c in df_portfolio.columns]
                        if cols:
                            df_portfolio = df_portfolio[cols]

                        if not df_portfolio.empty:
                            format_excel_sheet_compliance(writer, df_portfolio, nome)

                st.download_button(
                    label="📥 Baixar Excel",
                    data=output.getvalue(),
                    file_name=f"compliance_{portfolio_names_for_file}_{data_inicio}_a_{data_fim}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.error(f"Erro ao exportar: {str(e)}")

    # Grid de colunas + estrela
    st.markdown("---")
    st.markdown("### Colunas disponíveis")

    if not colunas_disponiveis:
        st.info("Nenhuma coluna disponível para Compliance (mapa_compliance).")
        return

    n_cols = 3
    n_linhas = math.ceil(len(colunas_disponiveis) / n_cols)

    colunas_selecionadas = []

    for linha in range(n_linhas):
        st_cols = st.columns(n_cols, gap="large")
        fatia = colunas_disponiveis[linha * n_cols:(linha + 1) * n_cols]

        for i, col in enumerate(fatia):
            with st_cols[i]:
                key_cb = f"{prefixo}{col}"

                row_cols = st.columns([0.8, 0.2])
                with row_cols[0]:
                    checked = st.checkbox(
                        col,
                        value=(col in st.session_state.selecoes_colunas_compliance),
                        key=key_cb,
                    )

                with row_cols[1]:
                    is_fav = col in st.session_state.favoritos_db[TELA_ID].get(ABA_ID, [])
                    estrela = "⭐" if is_fav else "☆"

                    if st.button(estrela, key=f"fav_{TELA_ID}_{ABA_ID}_{col}"):
                        novo = toggle_favorito(TELA_ID, ABA_ID, col)

                        # atualiza cache local
                        st.session_state.favoritos_db[TELA_ID].setdefault(ABA_ID, [])
                        if novo and col not in st.session_state.favoritos_db[TELA_ID][ABA_ID]:
                            st.session_state.favoritos_db[TELA_ID][ABA_ID].append(col)
                        if (not novo) and col in st.session_state.favoritos_db[TELA_ID][ABA_ID]:
                            st.session_state.favoritos_db[TELA_ID][ABA_ID].remove(col)

                        st.rerun()

                if checked:
                    colunas_selecionadas.append(col)

    st.session_state.selecoes_colunas_compliance = colunas_selecionadas
    st.session_state.colunas_selecionadas = colunas_selecionadas
