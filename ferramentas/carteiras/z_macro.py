import requests
import streamlit as st
from datetime import date
import pandas as pd
from io import BytesIO
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment, Border, Side, PatternFill, Font
import numpy as np

from .functions import aplicar_estilo_percentual

from utils import BASE_URL_API, CARTEIRAS

# -------- ORDEM DESEJADA DAS COLUNAS (ajuste como quiser) ----------
ordem_colunas_traduzidas = [
    "PL do Fundo", "Valor", "Valor (%)", "Pior Cenário",
    "Med. Retornos Positivos", "Med. Retornos Positivos (%)",
    "Med. Retornos Negativos", "Med. Retornos Negativos (%)", "Pior Macro", "Nome"
]

mapa_rename = {
    "fund_pl": "PL do Fundo",
    "worst_scenario_name": "Pior Cenário",
    "value": "Valor",
    "percentual_value": "Valor (%)",
    "median_positive_returns": "Med. Retornos Positivos",
    "percentual_median_positive_returns": "Med. Retornos Positivos (%)",
    "median_negative_returns": "Med. Retornos Negativos",
    "percentual_median_negative_returns": "Med. Retornos Negativos (%)",
    "worst_macro_name": "Pior Macro",
    "name": "Nome"
}

def garantir_abas_favoritos(dicionario, abas):
    if not isinstance(dicionario, dict):
        dicionario = {}
    for aba in abas:
        if aba not in dicionario or not isinstance(dicionario[aba], list):
            dicionario[aba] = []
    return dicionario

def renomear_df(df, mapa):
    cols = df.columns
    return df.rename(columns={c: mapa.get(c, c) for c in cols})

def sanitize_for_excel(value):
    if isinstance(value, (list, dict)):
        return str(value)
    if pd.isnull(value):
        return ""
    return value

def formatar_bloco(ws, start_row, start_col, nrows, ncols):
    align_center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin_black = Side(border_style="thin", color="000000")
    border = Border(left=thin_black, right=thin_black, top=thin_black, bottom=thin_black)
    for r in range(start_row, start_row + nrows):
        for c in range(start_col, start_col + ncols):
            cell = ws.cell(row=r, column=c)
            cell.alignment = align_center
            cell.border = border
            try:
                if (isinstance(cell.value, (int, float, np.integer, np.floating)) or
                    (isinstance(cell.value, str) and cell.value.replace('.', '', 1).replace('-', '', 1).isdigit())):
                    val = float(cell.value)
                    if val < 0:
                        cell.font = Font(color="FF0000")
            except Exception:
                pass

def pintar_titulo(ws, row, start_col, ncols):
    fill = PatternFill(start_color="B7E1FA", end_color="B7E1FA", fill_type="solid")
    bold = Font(bold=True)
    for c in range(start_col, start_col + ncols):
        cell = ws.cell(row=row, column=c)
        cell.fill = fill
        cell.font = bold

def ajustar_colunas(ws, max_row):
    for col in ws.columns:
        max_length = 0
        for cell in col:
            if cell.row > max_row:
                break
            try:
                cell_value = str(cell.value) if cell.value is not None else ""
                if len(cell_value) > max_length:
                    max_length = len(cell_value)
            except:
                pass
        col_letter = col[0].column_letter
        ws.column_dimensions[col_letter].width = min(max(12, max_length + 2), 50)
    for row in ws.iter_rows(min_row=1, max_row=max_row):
        for cell in row:
            ws.row_dimensions[cell.row].height = 30

def interface_selecao_colunas_macro(dfs, abas, mapa_rename):
    if "aba_macro_selecionada" not in st.session_state:
        st.session_state.aba_macro_selecionada = abas[0]



    # Cria mapeamento col_traduzida -> col_original
    colunas_traduzidas = {}
    for aba in abas:
        df = dfs.get(aba, pd.DataFrame())
        mapping = {mapa_rename.get(c, c): c for c in df.columns} if not df.empty else {}
        colunas_traduzidas[aba] = mapping

    if "selecoes_colunas_macro" not in st.session_state or not isinstance(st.session_state.selecoes_colunas_macro, dict):
        st.session_state.selecoes_colunas_macro = {aba: list(colunas_traduzidas[aba].keys()) for aba in abas}

    st.markdown("## Selecionar colunas para exportar")
    st.radio(
        "Escolha a aba para configurar:",
        options=abas,
        index=abas.index(st.session_state.aba_macro_selecionada),
        key="aba_macro_selecionada"
    )
    aba = st.session_state.aba_macro_selecionada
    col_trad = list(colunas_traduzidas[aba].keys())

    st.markdown("#### Ações rápidas")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        if st.button("✅ Selecionar todas"):
            st.session_state.selecoes_colunas_macro[aba] = col_trad.copy()
    with colB:
        if st.button("❌ Limpar seleção"):
            st.session_state.selecoes_colunas_macro[aba] = []
    with colC:
        if st.button("⭐ Favoritos"):
            st.session_state.selecoes_colunas_macro[aba] = st.session_state.favoritos_colunas_macro[aba].copy()

    st.markdown("---")
    st.markdown("#### Colunas disponíveis")
    ncols = 3
    rows = [col_trad[i:i+ncols] for i in range(0, len(col_trad), ncols)]
    for row in rows:
        cols = st.columns(ncols)
        for idx, col_name in enumerate(row):
            checked = col_name in st.session_state.selecoes_colunas_macro[aba]
            is_fav = col_name in st.session_state.favoritos_colunas_macro[aba]
            with cols[idx]:
                col_check, col_star = cols[idx].columns([6,2])
                with col_check:
                    ch = st.checkbox(col_name, value=checked, key=f"{aba}_{col_name}_check")
                with col_star:
                    fav = st.button("⭐" if is_fav else "☆", key=f"{aba}_{col_name}_fav")

                # Lógica seleção colunas (como antes)
                if ch and col_name not in st.session_state.selecoes_colunas_macro[aba]:
                    st.session_state.selecoes_colunas_macro[aba].append(col_name)
                if not ch and col_name in st.session_state.selecoes_colunas_macro[aba]:
                    st.session_state.selecoes_colunas_macro[aba].remove(col_name)
                # Sincronizar favoritos ao Sheets
                if fav:
                    if not is_fav:
                        st.session_state.favoritos_colunas_macro[aba].append(col_name)
                    else:
                        st.session_state.favoritos_colunas_macro[aba].remove(col_name)


    st.markdown("---")
    return colunas_traduzidas

def exportar_excel_macro_formatado(
    dfs_por_carteira, headers_macro_por_carteira, resultados_macro_por_carteira,
    selecoes_colunas_macro, colunas_traduzidas, mapa_rename=mapa_rename):

    output = BytesIO()
    wb = Workbook()
    wb.remove(wb.active)
    abas = ["Header", "Resultados Macro", "Cenários", "Cenários Excluídos"]

    for carteira in dfs_por_carteira:
        ws = wb.create_sheet(carteira)
        row = 1

        # HEADER
        df_header = headers_macro_por_carteira.get(carteira, pd.DataFrame())
        if not df_header.empty:
            df_header = renomear_df(df_header, mapa_rename)
            cols = [c for c in selecoes_colunas_macro["Header"] if c in df_header.columns]
            df_header = df_header[cols] if cols else df_header
            rows = list(dataframe_to_rows(df_header, index=False, header=True))
            ncols = len(rows[0])
            for r_idx, r in enumerate(rows):
                for c_idx, val in enumerate(r):
                    ws.cell(row=row + r_idx, column=1 + c_idx, value=sanitize_for_excel(val))
            pintar_titulo(ws, row, 1, ncols)
            formatar_bloco(ws, row, 1, len(rows), ncols)
            row += len(rows)
        else:
            ws.cell(row=row, column=1, value="Header vazio")
            row += 1
        row += 2

        # RESULTADOS MACRO - "Nome" sempre na primeira coluna se selecionada
        df_result = resultados_macro_por_carteira.get(carteira, pd.DataFrame())
        if not df_result.empty:
            df_result = renomear_df(df_result, mapa_rename)
            colunas_disponiveis = [c for c in selecoes_colunas_macro["Resultados Macro"] if c in df_result.columns]
            if "Nome" in colunas_disponiveis:
                colunas_disponiveis.remove("Nome")
                colunas_disponiveis.remove("CEN -2")
                colunas_disponiveis.remove("CEN -1")
                colunas_disponiveis.remove("CEN 0")
                colunas_disponiveis.remove("CEN +1")
                colunas_ordenadas = ["Nome"] + ["CEN -2"] + ["CEN -1"] + ["CEN 0"] + ["CEN +1"] +  colunas_disponiveis
            else:
                colunas_ordenadas = colunas_disponiveis
            df_result = df_result[colunas_ordenadas] if colunas_ordenadas else df_result
            rows = list(dataframe_to_rows(df_result, index=False, header=True))
            ncols = len(rows[0])
            for r_idx, r in enumerate(rows):
                for c_idx, val in enumerate(r):
                    ws.cell(row=row + r_idx, column=1 + c_idx, value=sanitize_for_excel(val))
            pintar_titulo(ws, row, 1, ncols)
            formatar_bloco(ws, row, 1, len(rows), ncols)
            row += len(rows)
        else:
            ws.cell(row=row, column=1, value="Resultados Macro vazio")
            row += 1
        row += 2

        # CENÁRIOS
        df_cenarios = dfs_por_carteira[carteira].get("Cenários", pd.DataFrame())
        cen_row = row
        if not df_cenarios.empty:
            df_cenarios = renomear_df(df_cenarios, mapa_rename)
            cols = [c for c in selecoes_colunas_macro["Cenários"] if c in df_cenarios.columns]
            df_cenarios = df_cenarios[cols] if cols else df_cenarios
            rows = list(dataframe_to_rows(df_cenarios, index=False, header=True))
            ncols = len(rows[0])
            for r_idx, r in enumerate(rows):
                for c_idx, val in enumerate(r):
                    ws.cell(row=cen_row + r_idx, column=1 + c_idx, value=sanitize_for_excel(val))
            pintar_titulo(ws, cen_row, 1, ncols)
            formatar_bloco(ws, cen_row, 1, len(rows), ncols)
        else:
            ws.cell(row=cen_row, column=1, value="Cenários vazio")
            ncols = 1
            rows = [[]]

        # CENÁRIOS EXCLUÍDOS
        df_exc = dfs_por_carteira[carteira].get("Cenários Excluídos", pd.DataFrame())
        exc_col = 1 + ncols + 1
        if not df_exc.empty:
            df_exc = renomear_df(df_exc, mapa_rename)
            cols = [c for c in selecoes_colunas_macro["Cenários Excluídos"] if c in df_exc.columns]
            df_exc = df_exc[cols] if cols else df_exc
            exc_rows = list(dataframe_to_rows(df_exc, index=False, header=True))
            exc_ncols = len(exc_rows[0])
            for r_idx, r in enumerate(exc_rows):
                for c_idx, val in enumerate(r):
                    ws.cell(row=cen_row + r_idx, column=exc_col + c_idx, value=sanitize_for_excel(val))
            pintar_titulo(ws, cen_row, exc_col, exc_ncols)
            formatar_bloco(ws, cen_row, exc_col, len(exc_rows), exc_ncols)
            max_row = max(cen_row + len(rows) - 1, cen_row + len(exc_rows) - 1)
        else:
            ws.cell(row=cen_row, column=exc_col, value="Cenários Excluídos vazio")
            max_row = cen_row + (len(rows) if rows != [[]] else 1) - 1

        ajustar_colunas(ws, max_row + 2)

    wb.save(output)
    output.seek(0)
    return output


def mostrar_macro_stress():
    URL_HEADER = f"{BASE_URL_API}/risk/macro_stress/general_info"
    URL_RESULTADOS = f"{BASE_URL_API}/risk/macro_stress/results"
    mapa_rename_cenarios = {
        "scenario_id": "ID do Cenário",
        "scenario_name": "Cenário",
        "percentual_value": "% Valor",
        "value": "Valor",
        "Carteira": "Carteira"
    }
    mapa_rename_excluidos = {
        "desconsidered_observations": "Cenários Excluídos",
        "Carteira": "Carteira"
    }
    cenarios = {16: "Macro LongView", 3: "USD -1%(Perfil Mensal)", 2: "DIXPRE -1%(Perfil Mensal)", 4: "Cupom Cambial -1%(Perfil Mensal)", 7: "IBOV - 10%", 1: "IBOVESPA -1%(Perfil Mensal)", 11: "MACRO STRESS | Fundos", 12: "COVID-19", 15: "GREVE DOS CAMINHONEIROS", 13: "JOESLEY DAY", 6: "Histórico 3M", 14: "SUBPRIME", 8: "Fundos FIA e MM RV", 10: "Alternativos",  5: "MACRO STRESS" }
    cenario_nomes = [v for k, v in cenarios.items()]
    cenario_id_por_nome = {v: k for k, v in cenarios.items()}
    abas = ["Header", "Resultados Macro", "Cenários", "Cenários Excluídos"]

    if "selecoes_colunas_macro" not in st.session_state or not isinstance(st.session_state.selecoes_colunas_macro, dict):
        st.session_state.selecoes_colunas_macro = {aba: [] for aba in abas}
    else:
        for aba_key in abas:
            if aba_key not in st.session_state.selecoes_colunas_macro:
                st.session_state.selecoes_colunas_macro[aba_key] = []

    if "dfs_macro" not in st.session_state:
        st.session_state.dfs_macro = {}

    st.title("Macro Stress")
    st.subheader("Consulta de cenários de estresse macroeconômico")

    data_macro = st.date_input("Data de referência", date.today())
    col1, col2 = st.columns([3, 1])
    with col1:
        portfolio_names = st.multiselect(
            "Selecione as carteiras",
            options=list(CARTEIRAS.values()),
            default=[],
            format_func=lambda x: x
        )
    portfolio_ids = [k for k, v in CARTEIRAS.items() if v in portfolio_names]
    st.session_state.selected_portfolios_macro = portfolio_ids

    with col2:
        cenario_nome = st.selectbox("Selecione o cenário macroeconômico", options=cenario_nomes)
        scenario_id = cenario_id_por_nome[cenario_nome]
        buscar = st.button("Buscar Macro Stress")

    # --- RESET AUTOMÁTICO SE CARTEIRA/CENÁRIO/DATA MUDAR ---
    filtros_atuais = (tuple(portfolio_ids), scenario_id, str(data_macro))
    if "last_macro_filters" not in st.session_state or st.session_state.last_macro_filters != filtros_atuais:
        st.session_state.dfs_macro = {}
        st.session_state.header_macro_por_carteira = {}
        st.session_state.resultados_macro_por_carteira = {}
        st.session_state.last_macro_filters = filtros_atuais

    if buscar:
        if len(portfolio_ids) < 1:
            st.warning("Selecione pelo menos uma carteira!")
        st.session_state.dfs_macro = {}
        st.session_state.header_macro_por_carteira = {}
        st.session_state.resultados_macro_por_carteira = {}

        for pid in portfolio_ids:
            nome_carteira = CARTEIRAS[pid]

            # HEADER
            try:
                payload_header = {
                    "start_date": str(data_macro),
                    "portfolio_ids": [pid],
                    "scenario_id": scenario_id
                }
                r_header = requests.post(
                    URL_HEADER,
                    json=payload_header,
                    headers=st.session_state.headers
                )
                r_header.raise_for_status()
                resultado_header = r_header.json()
                observations = resultado_header.get("observations", [])
                df_header = pd.json_normalize(observations) if observations else pd.DataFrame()
                st.session_state.header_macro_por_carteira[nome_carteira] = df_header
            except Exception as e:
                st.session_state.header_macro_por_carteira[nome_carteira] = pd.DataFrame()
                st.error(f"Erro ao buscar header macro ({nome_carteira}): {str(e)}")

            # RESULTADOS
            try:
                payload_resultados = {
                    "start_date": str(data_macro),
                    "portfolio_ids": [pid],
                    "scenario_id": scenario_id
                }
                r_resultados = requests.post(
                    URL_RESULTADOS,
                    json=payload_resultados,
                    headers=st.session_state.headers
                )
                r_resultados.raise_for_status()
                resultado_resultados = r_resultados.json()
                observations = resultado_resultados.get("observations", [])
                df_resultados = pd.json_normalize(observations) if observations else pd.DataFrame()
                st.session_state.resultados_macro_por_carteira[nome_carteira] = df_resultados
            except Exception as e:
                st.session_state.resultados_macro_por_carteira[nome_carteira] = pd.DataFrame()
                st.error(f"Erro ao buscar resultados macro ({nome_carteira}): {str(e)}")

            # CENÁRIOS
            payload = {
                "start_date": str(data_macro),
                "portfolio_ids": [pid],
                "scenario_id": scenario_id
            }
            try:
                r = requests.post(
                    f"{BASE_URL_API}/risk/macro_stress/full_values",
                    json=payload,
                    headers=st.session_state.headers
                )
                r.raise_for_status()
                resultado = r.json()
                dados = resultado.get("objects", resultado)
                if isinstance(dados, dict):
                    registros = []
                    for item in dados.values():
                        if isinstance(item, list):
                            registros.extend(item)
                        else:
                            registros.append(item)
                    df = pd.json_normalize(registros)
                elif isinstance(dados, list):
                    df = pd.json_normalize(dados)
                else:
                    df = pd.DataFrame([dados])
                df = df.dropna(axis=1, how='all')
                if "scenario_id" in df.columns:
                    df = df[df["scenario_id"] == scenario_id]
                df["Carteira"] = nome_carteira
                df_cenarios = df.drop(columns=["desconsidered_observations"], errors="ignore").rename(columns=mapa_rename_cenarios)
                if "desconsidered_observations" in df.columns:
                    excluidos = df[["desconsidered_observations"]].copy()
                    excluidos["Carteira"] = nome_carteira
                    excluidos = excluidos.rename(columns=mapa_rename_excluidos)
                else:
                    excluidos = pd.DataFrame(columns=[mapa_rename_excluidos["desconsidered_observations"], mapa_rename_excluidos["Carteira"]])
                st.session_state.dfs_macro[nome_carteira] = {
                    "Cenários": df_cenarios,
                    "Cenários Excluídos": excluidos
                }
            except Exception as e:
                st.error(f"Erro ao buscar dados para {nome_carteira}: {str(e)}")

    # ---- INTERFACE DE SELEÇÃO DE COLUNAS POR ABA ----
    colunas_traduzidas = None
    if st.session_state.get("header_macro_por_carteira") and st.session_state.get("resultados_macro_por_carteira"):
        if portfolio_names:
            carteira = portfolio_names[0]
            dfs_colunas = {
                "Header": st.session_state.header_macro_por_carteira.get(carteira, pd.DataFrame()),
                "Resultados Macro": st.session_state.resultados_macro_por_carteira.get(carteira, pd.DataFrame()),
                "Cenários": st.session_state.dfs_macro.get(carteira, {}).get("Cenários", pd.DataFrame()),
                "Cenários Excluídos": st.session_state.dfs_macro.get(carteira, {}).get("Cenários Excluídos", pd.DataFrame()),
            }
            colunas_traduzidas = interface_selecao_colunas_macro(dfs_colunas, abas, mapa_rename)

    # ---- EXPORTAÇÃO ----
    if (st.session_state.get("dfs_macro") and
        st.session_state.get("header_macro_por_carteira") and
        st.session_state.get("resultados_macro_por_carteira") and
        colunas_traduzidas and
        any(len(df) > 0 for df in st.session_state.header_macro_por_carteira.values())):

        if st.button("Exportar para Excel (Selecionado)"):
            output = exportar_excel_macro_formatado(
                st.session_state.dfs_macro,
                st.session_state.header_macro_por_carteira,
                st.session_state.resultados_macro_por_carteira,
                st.session_state.selecoes_colunas_macro,
                colunas_traduzidas,
            )
            if output is not None:
                st.download_button(
                    label="📥 Baixar Excel Selecionado",
                    data=output.getvalue(),
                    file_name=f"macro_selecionado_{date.today()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("Nenhum dado disponível para exportar!")

        if st.button("Exportar para Excel (Tudo)"):
            output = exportar_excel_macro_formatado(
                st.session_state.dfs_macro,
                st.session_state.header_macro_por_carteira,
                st.session_state.resultados_macro_por_carteira,
                {aba: [] for aba in abas},
                colunas_traduzidas,
            )
            if output is not None:
                st.download_button(
                    label="📥 Baixar Excel Tudo",
                    data=output.getvalue(),
                    file_name=f"macro_tudo_{date.today()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("Nenhum dado disponível para exportar!")
