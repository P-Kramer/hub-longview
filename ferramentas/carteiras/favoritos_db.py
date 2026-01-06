import streamlit as st
from libsql_client import create_client

@st.cache_resource
def get_db():
    url = st.secrets["TURSO"]["DATABASE_URL"]
    token = st.secrets["TURSO"]["AUTH_TOKEN"]
    return create_client(url=url, auth_token=token)

def init_schema():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS favoritos_colunas (
            tela    TEXT NOT NULL,
            aba     TEXT NOT NULL,
            coluna  TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (tela, aba, coluna)
        );
    """)

def load_favoritos(tela: str) -> dict:
    """Retorna {aba: [colunas]} para a tela."""
    db = get_db()
    rows = db.execute(
        "SELECT aba, coluna FROM favoritos_colunas WHERE tela=?",
        (tela,),
    ).rows

    favs = {}
    for aba, coluna in rows:
        favs.setdefault(aba, []).append(coluna)
    return favs

def toggle_favorito(tela: str, aba: str, coluna: str) -> bool:
    """True adicionou, False removeu."""
    db = get_db()
    exists = db.execute(
        "SELECT 1 FROM favoritos_colunas WHERE tela=? AND aba=? AND coluna=? LIMIT 1",
        (tela, aba, coluna),
    ).rows

    if exists:
        db.execute(
            "DELETE FROM favoritos_colunas WHERE tela=? AND aba=? AND coluna=?",
            (tela, aba, coluna),
        )
        return False

    db.execute(
        "INSERT OR IGNORE INTO favoritos_colunas (tela, aba, coluna) VALUES (?, ?, ?)",
        (tela, aba, coluna),
    )
    return True

def ensure_favoritos_loaded(tela: str, abas: list[str]):
    init_schema()

    if "favoritos_db" not in st.session_state or not isinstance(st.session_state.favoritos_db, dict):
        st.session_state.favoritos_db = {}

    if tela not in st.session_state.favoritos_db:
        st.session_state.favoritos_db[tela] = load_favoritos(tela)

    for aba in abas:
        st.session_state.favoritos_db[tela].setdefault(aba, [])
def is_favorito(tela: str, aba: str, coluna: str) -> bool:
    return coluna in st.session_state.favoritos_db.get(tela, {}).get(aba, [])

def ui_estrela(tela: str, aba: str, coluna: str, key: str):
    fav = is_favorito(tela, aba, coluna)
    estrela = "⭐" if fav else "☆"

    if st.button(estrela, key=key):
        novo = toggle_favorito(tela, aba, coluna)

        st.session_state.favoritos_db.setdefault(tela, {})
        st.session_state.favoritos_db[tela].setdefault(aba, [])

        if novo and coluna not in st.session_state.favoritos_db[tela][aba]:
            st.session_state.favoritos_db[tela][aba].append(coluna)
        if (not novo) and coluna in st.session_state.favoritos_db[tela][aba]:
            st.session_state.favoritos_db[tela][aba].remove(coluna)

        st.rerun()

def aplicar_favoritos_na_selecao(tela: str, aba: str, colunas_disponiveis: list[str], prefixo: str):
    favs = st.session_state.favoritos_db.get(tela, {}).get(aba, [])
    favs_ok = [c for c in favs if c in colunas_disponiveis]

    st.session_state.selecoes_colunas[aba] = favs_ok
    for c in colunas_disponiveis:
        st.session_state[f"{prefixo}{c}"] = (c in favs_ok)