# ferramentas/compara_relatorios/router.py
import streamlit as st
import pandas as pd

from .main import processar_pdf
from .diferencas import checar_divergencias


def render(ctx=None):
    # NÃO use st.set_page_config aqui. Deixe no app.py principal.

    st.markdown("## 🧾 Comparador de Ativos: PDF vs COMDINHEIRO")
    st.markdown(
        """
        Esta ferramenta compara os ativos de um extrato em PDF com os dados do sistema COMDINHEIRO,
        identificando divergências de valor, quantidade ou identificação.
        """
    )

    st.markdown("### 📁 Upload dos Arquivos")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("📄 Extrato em PDF (.pdf)")
        pdf_file = st.file_uploader("", type=["pdf"], key="pdf")

    with col2:
        st.markdown("📊 Planilha COMDINHEIRO (.xlsx)")
        excel_file = st.file_uploader("", type=["xlsx", "xls"], key="excel")
        st.caption("Colunas necessárias: Carteira, Ativo, Descrição, Quant., Saldo Bruto, Classe, ticker_cmd_puro")

    st.divider()
    st.subheader("Critérios de Divergências")
    diff_mv_max = st.number_input("Máxima diferença em MarketValue entre Equities ($)", min_value=0, max_value=10000000000000, step=1)
    diff_pct_max = st.number_input("Máxima diferença percentual de MarketValue entre não Equities (%)", min_value=0.00, max_value=100.00, step=1/100)

    if st.button("🔍 Iniciar Comparação", use_container_width=True) and pdf_file and excel_file:
        with st.spinner("⏳ Processando arquivos..."):
            try:
                # 1) PDF -> df_ativos
                df_ativos, _excel_buffer = processar_pdf(pdf_file.read(), return_excel=True)
                st.success("✅ PDF processado com sucesso!")

                with st.expander("📋 Visualizar dados extraídos do PDF"):
                    st.dataframe(df_ativos, use_container_width=True)

                # 2) Excel -> df_cd (ENGINE EXPLÍCITO)
                name = (excel_file.name or "").lower()

                # Diagnóstico (ajuda a acabar com achismo)
                with st.expander("🔧 Diagnóstico do arquivo Excel"):
                    st.write("Nome:", excel_file.name)
                    st.write("Tipo:", getattr(excel_file, "type", None))
                    try:
                        st.write("Tamanho (bytes):", len(excel_file.getbuffer()))
                    except Exception:
                        pass

                if name.endswith(".xlsx"):
                    df_cd = pd.read_excel(excel_file, engine="openpyxl")
                elif name.endswith(".xls"):
                    # Só funciona se você tiver xlrd instalado e compatível
                    df_cd = pd.read_excel(excel_file, engine="xlrd")
                else:
                    st.error("Arquivo inválido. Envie um .xlsx (recomendado) ou .xls.")
                    st.stop()

                with st.expander("📊 Visualizar dados lidos do COMDINHEIRO (Excel)"):
                    st.dataframe(df_cd, use_container_width=True)
                    st.caption(f"Linhas: {len(df_cd)} | Colunas: {len(df_cd.columns)}")

                # 3) Comparação
                df_diferencas, report_buffer = checar_divergencias(df_ativos, df_cd, diff_pct_max/100, diff_mv_max)

                if not df_diferencas.empty:
                    st.success("✅ Comparação concluída. Divergências encontradas.")
                    with st.expander("🔎 Visualizar divergências"):
                        st.dataframe(df_diferencas, use_container_width=True)
                else:
                    st.info("✅ Nenhuma divergência encontrada entre os dados.")

                # 4) Mostrar abas do relatório gerado (opcional)
                outras_abas = {}
                try:
                    report_buffer.seek(0)
                    xls = pd.ExcelFile(report_buffer, engine="openpyxl")
                    for sh in xls.sheet_names:
                        if sh != "Pareados":
                            outras_abas[sh] = pd.read_excel(xls, sheet_name=sh, engine="openpyxl")
                except Exception:
                    outras_abas = {}

                if outras_abas:
                    with st.expander("📁 Outras abas do relatório gerado"):
                        for nome, df_tab in outras_abas.items():
                            st.markdown(f"**Aba: {nome}**")
                            st.dataframe(df_tab, use_container_width=True)

                # 5) Download
                st.download_button(
                    label="📥 Baixar Relatório em Excel",
                    data=report_buffer.getvalue(),
                    file_name="relatorio_consolidado_equity.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

            except Exception as e:
                st.error("❌ Ocorreu um erro ao processar os arquivos.")
                st.exception(e)

    st.divider()
    st.caption("Desenvolvido por Pedro Averame • Última atualização: Julho/2025")
