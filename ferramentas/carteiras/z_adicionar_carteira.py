def alterar_carteiras ():
    import streamlit as st
    import ast
    import os

    UTILS_PATH = "info_carteiras.py"

    def load_carteiras():
        with open(UTILS_PATH, "r", encoding="utf-8") as f:
            code = f.read()
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if target.id == "CARTEIRAS":
                        value = ast.get_source_segment(code, node.value)
                        return ast.literal_eval(value)
        return {}

    def save_carteiras(novo_dict):
        with open(UTILS_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        start, end = None, None
        for i, line in enumerate(lines):
            if line.strip().startswith("CARTEIRAS"):
                start = i
                for j in range(i, len(lines)):
                    if lines[j].strip().endswith("}"):
                        end = j
                        break
                break

        if start is not None and end is not None:
            dict_str = "CARTEIRAS = {\n"
            for k, v in sorted(novo_dict.items(), key=lambda x: int(x[0])):
                dict_str += f"    {k}: \"{v}\",\n"
            dict_str += "}\n"
            lines = lines[:start] + [dict_str] + lines[end+1:]
            with open(UTILS_PATH, "w", encoding="utf-8") as f:
                f.writelines(lines)

    st.title("Administrador de Carteiras")

    # Carrega o dicionário atual
    if "carteiras_admin" not in st.session_state:
        st.session_state.carteiras_admin = load_carteiras()

    carteiras = st.session_state.carteiras_admin

    st.subheader("Adicionar nova carteira")
    with st.form("add_carteira"):
        novo_id = st.number_input("ID da carteira", min_value=1, step=1, key="add_id")
        novo_nome = st.text_input("Nome da carteira", key="add_nome")
        if st.form_submit_button("Adicionar"):
            if str(int(novo_id)) in carteiras:
                st.warning("ID já existe.")
            elif not novo_nome.strip():
                st.warning("Nome obrigatório.")
            else:
                carteiras[str(int(novo_id))] = novo_nome.strip()
                save_carteiras(carteiras)
                st.success("Carteira adicionada!")
                st.rerun()

    st.markdown("---")
    st.subheader("Editar/remover carteiras existentes")

    for k in sorted(carteiras, key=int):
        col1, col2, col3 = st.columns([5,1,1])
        with col1:
            novo_nome = st.text_input(f"Nome para {k}", value=carteiras[k], key=f"edit_{k}")
        with col2:
            st.markdown("")
            if st.button("Salvar", key=f"salvar_{k}"):
                carteiras[k] = novo_nome.strip()
                save_carteiras(carteiras)
                st.success(f"Carteira {k} editada!")
                st.rerun()
        with col3:
            st.markdown("")
            if st.button("Remover", key=f"remover_{k}"):
                del carteiras[k]
                save_carteiras(carteiras)
                st.success(f"Carteira {k} removida!")
                st.rerun()

    st.markdown("---")
    st.write("Carteiras atuais:", carteiras)