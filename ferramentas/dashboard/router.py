# apps/compara_relatorios/router.py
import streamlit as st
import pandas as pd
from io import BytesIO
from .aloc import tela_alocacao
from .perform import tela_performance
from .simul import tela_simulacao

from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from PIL import Image

def render(ctx):
    op = st.sidebar.radio(
        "Telas do Dashboard",
        ["Tela 1 – Alocação e Métricas", "Tela 2 – Simulação", "Tela 3 – Performance"],
        index=0,
    )

    if op.startswith("Tela 1"):
        tela_alocacao()
    elif op.startswith("Tela 2"):
        tela_simulacao()
    elif op.startswith("Tela 3"):
        tela_performance()