import math
from datetime import date
from io import BytesIO

import pandas as pd
import requests
import streamlit as st

from utils import CARTEIRAS
from .rename_carteira import mapa_renomeacao_ativos, mapa_renomeacao_cpr

from .turso_http import init_favoritos_schema, load_favoritos, toggle_favorito

TELA_ID = "carteiras"
ABAS = ["Overview Carteira", "Ativos", "CPR"]

from .functions import (
    aplicar_estilo_percentual,
    clear_data_if_portfolios_changed,
    reset_column_selection,
    formatar_percentuais_df,
    format_excel_sheet,
    get_portfolio_name,
    BASE_URL_API,
)

# -----------------------------
# Helpers de estado
# -----------------------------
def _init_state():
    # DataFrames
    if "df" not in st.session_state or not isinstance(st.session_state.df, pd.DataFrame):
        st.session_state.df = pd.DataFrame()

    if "df_ativos" not in st.session_state or not isinstance(st.session_state.df_ativos, pd.DataFrame):
        st.session_state.df_ativos = pd.DataFrame()

    if "df_cpr" not in st.session_state or not isinstance(st.session_state.df_cpr, pd.DataFrame):
        st.session_state.df_cpr = pd.DataFrame()

    # Portfolios
    if "selected_portfolios" not in st.session_state:
        st.session_state.selected_portfolios = []

    if "last_selected_portfolios" not in st.session_state:
        st.session_state.last_selected_portfolios = []

    # Colunas por dataset
    if "colunas_overview" not in st.session_state:
        st.session_state.colunas_overview = []

    if "colunas_ativos" not in st.session_state:
        st.session_state.colunas_ativos = []

    if "colunas_cpr" not in st.session_state:
        st.session_state.colunas_cpr = []

    # Seleções por aba (para export)
    if "selecoes_colunas" not in st.session_state or not isinstance(st.session_state.selecoes_colunas, dict):
        st.session_state.selecoes_colunas = {aba: [] for aba in ABAS}
    else:
        for aba in ABAS:
            st.session_state.selecoes_colunas.setdefault(aba, [])

    if "colunas_selecionadas" not in st.session_state:
        st.session_state.colunas_selecionadas = []

    return ABAS


def _drop_repetidos(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols_to_drop = [c for c in df.columns if "repetido" in str(c).lower()]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")
    return df


def _renomear_overview(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    rename_map = {
        "profitability_start_date": "%Dt Início",
        "profitability_in_day": "%Dia",
        "profitability_in_month": "%Mês",
        "profitability_in_semester": "%Semestre",
        "profitability_in_6_months": "%6 Meses",
        "profitability_in_year": "%Ano",
        "profitability_in_12_months": "%12 Meses",
        "profitability_in_18_months": "%18 Meses",
        "profitability_in_24_months": "%24 Meses",
        "profitability_in_30_months": "%30 Meses",
        "profitability_in_36_months": "%36 Meses",
        "profitability_in_48_months": "%48 Meses",
        "profitability_in_60_months": "%60 Meses",
        "net_asset_value": "PL",
        "portfolio_id": "ID Carteira",
        "overview_type": "Tipo de Overview",
        "date": "Data",
        "name": "Carteira",
        "instrument_positions": "Ativos",
        "financial_transaction_positions": "CPR",
        "last_shares": "Qtd. Cotas D-1",
        "is_opening": "Carteira de Abertura",
        "id": "ID Overview",
        "navps": "Cota Líquida",
        "gross_navps": "Cota bruta",
        "shares": "Qtd. Cotas",
        "fixed_shares": "Qtd. Cotas Fixas",
        "portfolio_average_duration": "Duração Média Carteira",
        "created_on": "Data de Criação",
        "modified_on": "Modificado em",
        "released_on": "Data de Liberação",
        "benchmark_profitability.symbol": "Nome Bench",
        "gross_asset_value": "Valor Bruto",
        "asset_value_for_allocation": "Valor para Alocação",
        "last_net_asset_value": "PL D-1",
        "last_navps": "Cota Líquida D-1",
        "fixed_navps": "Cota Fixa",
        "corp_actions_adjusted_navps": "Cota Líquida Ajustada por Eventos Societários",
        "corp_actions_factor": "Fator de Ajuste por Eventos Societários",
        "equity_exposure": "Exposição em Renda Variável",
        "is_system_generated": "Gerado pelo Sistema",
        "overview_status": "Status do Overview",
        "pct_lent_exposure": "Exposição % Doada",
        "portfolio_average_term": "Prazo Médio da Carteira",
        # benchmarks
        "benchmark_profitability.profitability_in_day": "Bench %Dia",
        "benchmark_profitability.profitability_in_month": "Bench %Mês",
        "benchmark_profitability.profitability_in_year": "Bench %Ano",
        "benchmark_profitability.profitability_in_12_months": "Bench %12 Meses",
        "benchmark_profitability.profitability_start_date": "Bench %Dt Início",
        "benchmark_profitability.profitability_in_semester": "Bench %Semestre",
        "benchmark_profitability.profitability_in_6_months": "Bench %6 Meses",
        "benchmark_profitability.profitability_in_18_months": "Bench %18 Meses",
        "benchmark_profitability.profitability_in_24_months": "Bench %24 Meses",
        "benchmark_profitability.profitability_in_30_months": "Bench %30 Meses",
        "benchmark_profitability.profitability_in_36_months": "Bench %36 Meses",
        "benchmark_profitability.profitability_in_48_months": "Bench %48 Meses",
        "benchmark_profitability.profitability_in_60_months": "Bench %60 Meses",
        # attribution
        "attribution.portfolio_beta.financial_value": "PnL Beta",
        "attribution.portfolio_beta.percentage_value": "PnL % Beta",
        "attribution.total.financial_value": "PnL Total",
        "attribution.total.percentage_value": "PnL % Total",
        "attribution.currency.financial_value": "PnL Moeda",
        "attribution.currency.percentage_value": "PnL % Moeda",
        "attribution_maximums.par_price": "PnL Máximo Preço Par",
        "attribution_maximums.portfolio_beta": "PnL Máximo Beta da Carteira",
        "attribution_maximums.total": "PnL Máximo Total",
        "attribution_maximums.total_hedged": "PnL Máximo Total Hedgeado",
        "attribution_maximums.corp_actions": "PnL Máximo Eventos Societários",
        "attribution_maximums.currency": "PnL Máximo Moeda",
    }

    df = df.rename(columns=rename_map)
    df = _drop_repetidos(df)
    df = df.dropna(axis=1, how="all")
    return df


def _explode_e_traduz_listas(df_overview: pd.DataFrame):
    if df_overview is None or df_overview.empty:
        return pd.DataFrame(), pd.DataFrame(), df_overview

    # ATIVOS
    df_ativos_concat = pd.DataFrame()
    if "Ativos" in df_overview.columns:
        lista_ativos_df = []
        for lst in df_overview["Ativos"]:
            if isinstance(lst, list) and lst:
                tmp = pd.json_normalize(lst).rename(columns=mapa_renomeacao_ativos)
                tmp = tmp.dropna(axis=1, how="all")
                tmp = _drop_repetidos(tmp)
                lista_ativos_df.append(tmp)
            else:
                lista_ativos_df.append(pd.DataFrame())

        df_overview["Ativos"] = [d.to_dict("records") if not d.empty else [] for d in lista_ativos_df]
        nao_vazios = [d for d in lista_ativos_df if not d.empty]
        if nao_vazios:
            df_ativos_concat = pd.concat(nao_vazios, ignore_index=True)
    else:
        df_overview["Ativos"] = [[] for _ in range(len(df_overview))]

    # CPR
    df_cpr_concat = pd.DataFrame()
    if "CPR" in df_overview.columns:
        lista_cpr_df = []
        for lst in df_overview["CPR"]:
            if isinstance(lst, list) and lst:
                tmp = pd.json_normalize(lst).rename(columns=mapa_renomeacao_cpr)
                tmp = tmp.dropna(axis=1, how="all")
                tmp = _drop_repetidos(tmp)
                lista_cpr_df.append(tmp)
            else:
                lista_cpr_df.append(pd.DataFrame())

        df_overview["CPR"] = [d.to_dict("records") if not d.empty else [] for d in lista_cpr_df]
        nao_vazios = [d for d in lista_cpr_df if not d.empty]
        if nao_vazios:
            df_cpr_concat = pd.concat(nao_vazios, ignore_index=True)
    else:
        df_overview["CPR"] = [[] for _ in range(len(df_overview))]

    return df_ativos_concat, df_cpr_concat, df_overview


# -----------------------------
# Tela
# -----------------------------
def mostrar_carteira(ctx=None):
    # ---------- FAVORITOS (Turso) ----------
    init_favoritos_schema()

    if "favoritos_db" not in st.session_state or not isinstance(st.session_state.favoritos_db, dict):
        st.session_state.favoritos_db = {}

    if TELA_ID not in st.session_state.favoritos_db:
        st.session_state.favoritos_db[TELA_ID] = load_favoritos(TELA_ID)

    for a in ABAS:
        st.session_state.favoritos_db[TELA_ID].setdefault(a, [])

    # ---------- STATE GERAL ----------
    abas = _init_state()

    st.title("Carteira")
    st.subheader("Buscar posições")

    data_inicio, data_fim = st.date_input(
        "Escolha o intervalo de datas",
        [date.today(), date.today()],
        key="intervalo_datas_carteira",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        portfolio_names = st.multiselect(
            "Selecione as carteiras",
            options=list(CARTEIRAS.values()),
            default=[],
            format_func=lambda x: x,
            key="portfolio_multiselect",
        )

    selected_ids = [k for k, v in CARTEIRAS.items() if v in portfolio_names]
    st.session_state.selected_portfolios = selected_ids

    with col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

        # garante que a função não exploda por falta de last_selected_portfolios
        if "last_selected_portfolios" not in st.session_state:
            st.session_state.last_selected_portfolios = []
        clear_data_if_portfolios_changed()

        if st.button("Buscar dados", key="btn_buscar_dados_carteira"):
            if not selected_ids:
                st.warning("Selecione pelo menos uma carteira.")
            else:
                payload = {
                    "start_date": str(data_inicio),
                    "end_date": str(data_fim),
                    "instrument_position_aggregation": 3,
                    "portfolio_ids": selected_ids,
                }

                try:
                    if "headers" not in st.session_state or not st.session_state.headers:
                        raise RuntimeError("headers não encontrados em st.session_state. Você precisa autenticar antes.")

                    r = requests.post(
                        f"{BASE_URL_API}/portfolio_position/positions/get",
                        json=payload,
                        headers=st.session_state.headers,
                        timeout=60,
                    )
                    r.raise_for_status()
                    resultado = r.json()
                    objetos = resultado.get("objects", {})

                    registros = []
                    for item in objetos.values():
                        if isinstance(item, list):
                            registros.extend(item)
                        else:
                            registros.append(item)

                    df = pd.json_normalize(registros)
                    df = _renomear_overview(df)

                    df_ativos, df_cpr, df = _explode_e_traduz_listas(df)

                    st.session_state.df = df
                    st.session_state.df_ativos = df_ativos
                    st.session_state.df_cpr = df_cpr

                    st.session_state.colunas_overview = sorted(df.columns) if not df.empty else []
                    st.session_state.colunas_ativos = sorted(df_ativos.columns) if not df_ativos.empty else []
                    st.session_state.colunas_cpr = sorted(df_cpr.columns) if not df_cpr.empty else []

                    if df.empty:
                        st.warning("Nenhum dado encontrado para os filtros informados.")
                    else:
                        nomes = ", ".join([get_portfolio_name(pid) for pid in selected_ids])
                        st.success(f"Dados recebidos com sucesso para: {nomes}")

                        formatar_percentuais_df()

                        # resetar seleções só quando buscar novos dados
                        reset_column_selection()

                except Exception as e:
                    st.error(f"Erro ao buscar dados: {e}")

    # -----------------------------
    # Visualização + seleção + export
    # -----------------------------
    if st.session_state.df.empty:
        return

    st.dataframe(aplicar_estilo_percentual(st.session_state.df), use_container_width=True)

    st.markdown("## Selecionar colunas para exportar")
    aba = st.radio("Escolha a aba para configurar:", abas, key="aba_config_carteira")

    colunas_disponiveis = {
        "Overview Carteira": [c for c in st.session_state.colunas_overview if "repetido" not in c.lower()],
        "Ativos": [c for c in st.session_state.colunas_ativos if "repetido" not in c.lower()],
        "CPR": [c for c in st.session_state.colunas_cpr if "repetido" not in c.lower()],
    }

    prefixo_checkbox = {"Overview Carteira": "ov_", "Ativos": "atv_", "CPR": "cpr_"}
    colunas_aba = sorted(colunas_disponiveis.get(aba, []))
    prefixo = prefixo_checkbox[aba]

    # Ações rápidas
    st.markdown("### Ações rápidas")
    ac1, ac2, ac3 = st.columns([1, 1, 1])

    with ac1:
        if st.button("Selecionar todas", key=f"select_all_{TELA_ID}_{aba}"):
            st.session_state.selecoes_colunas[aba] = colunas_aba
            for c in colunas_aba:
                st.session_state[f"{prefixo}{c}"] = True
            st.rerun()

    with ac2:
        if st.button("Limpar seleção", key=f"clear_all_{TELA_ID}_{aba}"):
            st.session_state.selecoes_colunas[aba] = []
            for c in colunas_aba:
                st.session_state[f"{prefixo}{c}"] = False
            st.rerun()

    with ac3:
        if st.button("Favoritos", key=f"apply_fav_{TELA_ID}_{aba}"):
            favs = st.session_state.favoritos_db[TELA_ID].get(aba, [])
            favs_ok = [c for c in favs if c in colunas_aba]
            st.session_state.selecoes_colunas[aba] = favs_ok
            for c in colunas_aba:
                st.session_state[f"{prefixo}{c}"] = c in favs_ok
            st.rerun()

    # Export
    if st.button("Exportar para Excel", key="btn_export_excel_carteira"):
        if (
            not st.session_state.selecoes_colunas["Overview Carteira"]
            or not st.session_state.selecoes_colunas["Ativos"]
            or not st.session_state.selecoes_colunas["CPR"]
        ):
            st.warning("Selecione pelo menos uma coluna de cada aba.")
        else:
            try:
                df_overview_full = st.session_state.df.copy()

                # seleção do overview
                cols_ov = [c for c in st.session_state.selecoes_colunas["Overview Carteira"] if c in df_overview_full.columns]

                output = BytesIO()
                nomes_arquivo = "_".join([get_portfolio_name(pid).replace(" ", "_") for pid in st.session_state.selected_portfolios])

                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    for pid in st.session_state.selected_portfolios:
                        nome = get_portfolio_name(pid)

                        if "ID Carteira" in st.session_state.df.columns:
                            df_port = st.session_state.df[st.session_state.df["ID Carteira"] == pid].copy()
                        else:
                            df_port = st.session_state.df.copy()

                        if cols_ov:
                            df_port = df_port[[c for c in cols_ov if c in df_port.columns]]

                        if not df_port.empty:
                            format_excel_sheet(writer, df_port, nome)

                st.download_button(
                    label="Baixar Excel",
                    data=output.getvalue(),
                    file_name=f"portfolio_{nomes_arquivo}_{data_inicio}_a_{data_fim}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            except Exception as e:
                st.error(f"Erro ao exportar: {e}")

    st.markdown("---")
    st.markdown("### Colunas disponíveis")

    if not colunas_aba:
        st.info("Nenhuma coluna disponível nesta aba.")
        return

    n_cols = 3
    n_linhas = math.ceil(len(colunas_aba) / n_cols)

    colunas_selecionadas_aba = []

    for linha in range(n_linhas):
        st_cols = st.columns(n_cols, gap="large")
        for i, col in enumerate(colunas_aba[linha * n_cols:(linha + 1) * n_cols]):
            with st_cols[i]:
                key_cb = f"{prefixo}{col}"
                row_cols = st.columns([0.85, 0.15])

                with row_cols[0]:
                    checked = st.checkbox(
                        col,
                        value=(col in st.session_state.selecoes_colunas[aba]),
                        key=key_cb,
                    )

                with row_cols[1]:
                    is_fav = col in st.session_state.favoritos_db[TELA_ID].get(aba, [])
                    estrela = "⭐" if is_fav else "☆"

                    if st.button(estrela, key=f"fav_{TELA_ID}_{aba}_{col}"):
                        novo = toggle_favorito(TELA_ID, aba, col)

                        st.session_state.favoritos_db[TELA_ID].setdefault(aba, [])
                        if novo and col not in st.session_state.favoritos_db[TELA_ID][aba]:
                            st.session_state.favoritos_db[TELA_ID][aba].append(col)
                        if (not novo) and col in st.session_state.favoritos_db[TELA_ID][aba]:
                            st.session_state.favoritos_db[TELA_ID][aba].remove(col)

                        st.rerun()

                if checked:
                    colunas_selecionadas_aba.append(col)

    st.session_state.selecoes_colunas[aba] = colunas_selecionadas_aba

    st.session_state.colunas_selecionadas = (
        st.session_state.selecoes_colunas["Overview Carteira"]
        + st.session_state.selecoes_colunas["Ativos"]
        + st.session_state.selecoes_colunas["CPR"]
    )
