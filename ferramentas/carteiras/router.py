import streamlit as st
import requests
import pandas as pd

# Configuração da página deve vir primeiro
st.set_page_config(page_title="Consulta de Portfólio", layout="centered")

# Imports locais
from utils import  CLIENT_ID, CLIENT_SECRET,BASE_URL_API
from utils import CARTEIRAS
from .functions import (
    format_excel_sheet, get_column_letter, get_portfolio_name,
    get_repeated_columns, get_valid_columns, ir_para, aplicar_estilo_percentual,
    reset_column_selection, select_all, clear_all,
    clear_data_if_portfolios_changed, formatar_percentuais_df
)
from .turso_http import init_favoritos_schema
from .rename_carteira import (
    mapa_renomeacao_ativos, mapa_renomeacao_cpr, rename
)



from .z_carteira import mostrar_carteira
from .z_compliance import mostrar_compliance
from .z_risco import mostrar_risco
from .z_instrucoes import mostrar_instrucoes
from .z_adicionar_carteira import alterar_carteiras
from .z_macro import mostrar_macro_stress

st.session_state.pagina_atual_carteira = "menu"
if "token" not in st.session_state:
    st.session_state.token = None
if "headers" not in st.session_state:
    st.session_state.headers = None
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "colunas_selecionadas" not in st.session_state:
    st.session_state.colunas_selecionadas = []
if "current_columns" not in st.session_state:
    st.session_state.current_columns = []
if "colunas_validas" not in st.session_state:
    st.session_state.colunas_validas = []
if "selected_portfolios" not in st.session_state:
    st.session_state.selected_portfolios = []
if "last_selected_portfolios" not in st.session_state:
    st.session_state.last_selected_portfolios = []
if "colunas_overview" not in st.session_state:
    st.session_state.colunas_overview = []
if "colunas_ativos" not in st.session_state:
    st.session_state.colunas_ativos = []
if "colunas_cpr" not in st.session_state:
    st.session_state.colunas_cpr = []
import gspread
import streamlit as st



def render(ctx):
    init_favoritos_schema()
    if st.session_state.pagina_atual_carteira == "menu":
        op = st.sidebar.radio(
        "Telas do Dashboard",
        ["Tela 1 – Carteira", "Tela 2 – Risco", "Tela 3 – Compliance", "Tela 4 - Macro Stress"],
        index=0,
    )

    if op.startswith("Tela 1"):
        mostrar_carteira()
    elif op.startswith("Tela 2"):
        mostrar_risco()
    elif op.startswith("Tela 3"):
        mostrar_compliance()
    elif op.startswith("Tela 4"):
        mostrar_macro_stress()
            