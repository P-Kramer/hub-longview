import streamlit as st

def mostrar_instrucoes ():
    # instrucoes.py
    st.title("📘 Instruções de Uso")

    st.markdown("""
    ### Bem-vindo ao Analisador de Portfólio Longview!

    Este aplicativo permite consultar as informações das carteiras a partir da coleta de dados do sistema MARAVI!

    ---

    #### 🗂 Seções disponíveis:
    - **Carteira:** análise geral da carteira, dos ativos e do CPR.
    - **Compliance:** validações e regras específicas da sua operação.
    - **Risco:** análise de liquidez, resgates e ADTV.
   
    ---
                
    #### 🚀 Como utilizar:

    **1. Escolha o intervalo de datas**  
    Use o seletor de data para definir o período desejado.  
    > *Dica: clique duas vezes no dia para selecionar início e fim iguais.*

    **2. Selecione as carteiras**  
    Você pode escolher uma ou mais entre as disponíveis, como:
    - PEPENERO FIM  
    - FILIPINA FIM  
    - PARMIGIANO FIM  
    - HARPYJA FIM  

    **3. Clique em “Buscar dados”**  
    Aguarde o carregamento automático das informações da carteira.
    """)

    st.markdown("---")

    st.subheader("📤 Exportação para Excel")

    st.markdown("""
    **4. Escolha a aba para configurar**

    **5. Selecione as colunas desejadas para exportação**
    - ✅ Use “Selecionar todas” ou marque individualmente.
    - ⭐ Clique em “Favoritos” para recuperar seleções frequentes.

    **6. Clique em “Exportar para Excel”**
    > ⚠️ *Importante: selecione ao menos uma coluna por aba antes de exportar.*
    """)

    st.markdown("---")