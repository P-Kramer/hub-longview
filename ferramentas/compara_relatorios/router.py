# apps/compara_relatorios/router.py
import streamlit as st
import pandas as pd
from io import BytesIO
from .main import processar_pdf
from .diferencas import checar_divergencias
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from PIL import Image

def render(ctx):
    st.set_page_config(page_title="Analisador de Ativos", layout="centered")

    # ==== CABEÇALHO ====

    st.markdown("## 🧾 Comparador de Ativos: PDF vs COMDINHEIRO")
    st.markdown(
        """
        Esta ferramenta compara os ativos de um extrato em PDF com os dados do sistema COMDINHEIRO, 
        identificando divergências de valor, quantidade ou identificação.
        """
    )

    # ==== UPLOADS ====
    st.markdown("### 📁 Upload dos Arquivos")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("📄 Extrato em PDF (.pdf)")
        pdf_file = st.file_uploader("", type="pdf", key="pdf")

    with col2:
        st.markdown("📊 Planilha COMDINHEIRO (.xlsx)")
        excel_file = st.file_uploader("", type=["xlsx"], key="excel")
        st.markdown("Colunas Necessárias: 'Carteira', 'Ativo', 'Descrição', 'Quant.', 'Saldo Bruto', 'Classe', 'ticker_cmd_puro'")

    # ==== BOTÃO DE PROCESSAMENTO ====
    st.markdown("---")
    if st.button("🔍 Iniciar Comparação") and pdf_file and excel_file:
        with st.spinner("⏳ Processando arquivos..."):
            try:
                # 1) Extrai dados do PDF
                df_ativos, excel_buffer = processar_pdf(pdf_file.read(), return_excel=True)
                st.success("✅ PDF processado com sucesso!")

                with st.expander("📋 Visualizar dados extraídos do PDF"):
                    st.dataframe(df_ativos, use_container_width=True)

                # 2) Lê Excel COMDINHEIRO
                df_cd = pd.read_excel(excel_file)

                # MOSTRAR O QUE VEIO DO EXCEL
                with st.expander("📊 Visualizar dados lidos do COMDINHEIRO (Excel)"):
                    st.dataframe(df_cd, use_container_width=True)
                    st.caption(f"Linhas: {len(df_cd)} | Colunas: {len(df_cd.columns)}")

                # 3) Compara os dados
                df_diferencas, report_buffer = checar_divergencias(df_ativos, df_cd)

                # 4) Sempre tentar mostrar os PAREADOS a partir do relatório
                pareados_df = None
                outras_abas = {}
                try:
                    report_buffer.seek(0)
                    xls = pd.ExcelFile(report_buffer)
                    # guarda todas as abas
                    sheet_names = xls.sheet_names

                    if "Pareados" in sheet_names:
                        pareados_df = pd.read_excel(xls, sheet_name="Pareados")

                    # opcional: guardar outras
                    for sh in sheet_names:
                        if sh != "Pareados":
                            outras_abas[sh] = pd.read_excel(xls, sheet_name=sh)

                except Exception as err:
                    st.warning("⚠ Não foi possível ler o relatório gerado (aba(s) do Excel). Verifique o checar_divergencias().")
                    pareados_df = None
                    outras_abas = {}

                # 5) Mostrar divergências (se houver)
                if not df_diferencas.empty:
                    st.success("✅ Comparação concluída com sucesso! Divergências encontradas.")
                    with st.expander("🔎 Visualizar divergências encontradas"):
                        st.dataframe(df_diferencas, use_container_width=True)
                else:
                    st.info("✅ Nenhuma divergência encontrada entre os dados.")

            

                # 6.1) Mostrar também as outras abas do relatório (se quiser inspecionar)
                if outras_abas:
                    with st.expander("📁 Outras abas do relatório gerado"):
                        for nome, df_tab in outras_abas.items():
                            st.markdown(f"**Aba: {nome}**")
                            st.dataframe(df_tab, use_container_width=True)

                # 7) Botão de download SEMPRE
                st.download_button(
                    label="📥 Baixar Relatório em Excel",
                    data=report_buffer.getvalue(),
                    file_name="relatorio_consolidado_equity.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            except Exception as e:
                st.error("❌ Ocorreu um erro ao processar os arquivos.")
                st.exception(e)

    # ==== RODAPÉ ====
    st.markdown("---")
    st.caption("Desenvolvido por Pedro Averame • Última atualização: Julho/2025")
