import requests
import streamlit as st
from datetime import datetime, timedelta

from utils import BASE_URL_API, CLIENT_ID, CLIENT_SECRET

from ferramentas.compara_relatorios.router import render as render_compara_relatorios
from ferramentas.dashboard.router import render as render_dashboard
from ferramentas.carteiras.router import render as render_carteiras





# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Longview Hub",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# STATE INIT
# =========================
if "pagina_atual" not in st.session_state:
    st.session_state.pagina_atual = "login"  # login | hub | app:<id>
if "token" not in st.session_state:
    st.session_state.token = None
if "headers" not in st.session_state:
    st.session_state.headers = {}
if "token_expira_em" not in st.session_state:
    st.session_state.token_expira_em = None

# =========================
# NAV / SESSION
# =========================
def ir_para(pagina: str):
    st.session_state.pagina_atual = pagina

def limpar_sessao():
    st.session_state.token = None
    st.session_state.headers = {}
    st.session_state.token_expira_em = None
    st.query_params.clear()
    ir_para("login")
    st.rerun()

def token_valido() -> bool:
    exp = st.session_state.get("token_expira_em")
    token = st.session_state.get("token")
    if not token or not exp:
        return False
    return datetime.utcnow() < (exp - timedelta(seconds=30))

def exigir_auth():
    if not token_valido():
        limpar_sessao()

# =========================
# AUTH
# =========================
def autenticar(email: str, senha: str):
    url = f"{BASE_URL_API.rstrip('/')}/auth/token"
    client_headers = {
        "CF-Access-Client-Id": CLIENT_ID,
        "CF-Access-Client-Secret": CLIENT_SECRET,
    }
    data = {"username": email.strip(), "password": senha}
    resp = requests.post(url, data=data, headers=client_headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    token = payload.get("access_token")
    expires_in = int(payload.get("expires_in", 3600))
    if not token:
        raise ValueError("Resposta sem access_token.")

    headers = {**client_headers, "Authorization": f"Bearer {token}"}
    expira_em = datetime.utcnow() + timedelta(seconds=expires_in)
    return token, headers, expira_em

# =========================
# API WRAPPER (recomendado)
# =========================
def api_request(method: str, path: str, **kwargs) -> requests.Response:
    exigir_auth()
    url = f"{BASE_URL_API.rstrip('/')}/{path.lstrip('/')}"
    headers = {**st.session_state.headers, **kwargs.pop("headers", {})}

    resp = requests.request(method, url, headers=headers, timeout=30, **kwargs)

    if resp.status_code in (401, 403):
        limpar_sessao()

    resp.raise_for_status()
    return resp

class Ctx:
    def __init__(self):
        self.api = api_request
        self.headers = st.session_state.headers

ctx = Ctx()

# =========================
# APPS (UI nome bonito, ID nome limpo)
# =========================
APPS = [
    {"id": "carteiras",          "nome": "Carteiras",          "icone": "🗂️", "render": render_carteiras},
    {"id": "dashboard",          "nome": "Dashboard",          "icone": "📊", "render": render_dashboard},
    {"id": "compara_relatorios", "nome": "Compara Relatórios", "icone": "🧾", "render": render_compara_relatorios},
    {"id": "transacoes",         "nome": "Transações",         "icone": "🔁", "render": 'render_transacoes'},
]

# =========================
# CSS do HUB (cards HTML)
# =========================
st.markdown(
    """
    <style>
      .hub-wrap { max-width: 1100px; margin: 0 auto; }
      .hub-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
        margin-top: 28px;
      }
      .hub-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 180px;
        border-radius: 18px;
        border: 1px solid rgba(0,0,0,0.12);
        background: white;
        text-decoration: none !important;
        transition: transform 0.08s ease, border-color 0.08s ease;
      }
      .hub-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0,0,0,0.30);
      }
      .hub-icon { font-size: 46px; line-height: 1; margin-bottom: 10px; }
      .hub-title { font-size: 1.05rem; color: rgba(0,0,0,0.78); font-weight: 650; text-align: center; }
      .hub-sub { font-size: 0.9rem; color: rgba(0,0,0,0.55); margin-top: 4px; }
      @media (max-width: 900px) {
        .hub-grid { grid-template-columns: 1fr; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# TELAS
# =========================
def tela_login():
    import streamlit as st
    from libsql_client import create_client

    @st.cache_resource
    def get_db():
        try:
            url = st.secrets["TURSO"]["DATABASE_URL"]
            token = st.secrets["TURSO"]["AUTH_TOKEN"]
        except Exception as e:
            raise RuntimeError(f"Secrets TURSO ausentes/errados: {e}")

        return create_client(url=url, auth_token=token)

    def check_db():
        st.title("DB Check (Turso)")

        url = st.secrets["TURSO"]["DATABASE_URL"]
        token = st.secrets["TURSO"]["AUTH_TOKEN"]

        st.write("DATABASE_URL ok:", bool(url))
        st.write("AUTH_TOKEN ok:", bool(token))

        with libsql_client.create_client_sync(url, auth_token=token) as db:
            r = db.execute("SELECT 1 AS ok;").rows
            st.success(f"Conexão ok. SELECT 1 retornou: {r}")

            db.execute("""
                CREATE TABLE IF NOT EXISTS _db_check (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT (datetime('now')),
                    note TEXT
                );
            """)
            db.execute("INSERT INTO _db_check (note) VALUES (?);", ("hello",))
            rows = db.execute("SELECT id, created_at, note FROM _db_check ORDER BY id DESC LIMIT 5;").rows
            st.success("Escrita/leitura ok. Últimas linhas:")
            st.write(rows)
            db.execute("DELETE FROM _db_check WHERE note = ?;", ("hello",))
            st.info("Cleanup feito.")

    try:
        check_db()
    except Exception as e:
        st.error(f"Falhou: {e}")

def tela_hub():
    exigir_auth()

    st.sidebar.image("longview.png")
    st.sidebar.success("Sessão autenticada")


    st.markdown("## Longview's Hub")
    
    st.caption("Escolha uma aplicação para continuar")
    st.divider()

    # Grid 2x2 SEM link (sem abrir nova aba)
    col1, col2 = st.columns(2, gap="large")
    col3, col4 = st.columns(2, gap="large")
    slots = [col1, col2, col3, col4]

    for slot, app in zip(slots, APPS):
        with slot:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="text-align:center; padding:22px 0;">
                        <div style="font-size:46px">{app['icone']}</div>
                        <div style="margin-top:10px; font-weight:650; font-size:1.05rem;">
                            {app['nome']}
                        </div>
               
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("Abrir", key=f"open_{app['id']}", use_container_width=True):
                    st.query_params.clear()
                    st.query_params["app"] = app["id"]   # mantém seu fluxo atual (?app=...)
                    ir_para(f"app:{app['id']}")
                    st.rerun()


def tela_app(app_id: str):
    exigir_auth()

    st.sidebar.image("longview.png")
    st.sidebar.button(
        "Voltar ao Hub",
        on_click=lambda: (st.query_params.clear(), ir_para("hub")),
        use_container_width=True,
    )


    app = next((a for a in APPS if a["id"] == app_id), None)
    if not app:
        st.error("App não encontrada.")
        st.button("Voltar", on_click=lambda: (st.query_params.clear(), ir_para("hub")))
        return

    # SEM HEADER DO HUB: a ferramenta manda em tudo
    app["render"](ctx)

# =========================
# ROUTER + query param (?app=...)
# =========================
qp_app = st.query_params.get("app")
if qp_app:
    ir_para(f"app:{qp_app}")

if st.session_state.pagina_atual == "login":
    if token_valido():
        ir_para("hub")
        st.rerun()
    tela_login()

elif st.session_state.pagina_atual == "hub":
    tela_hub()

elif st.session_state.pagina_atual.startswith("app:"):
    app_id = st.session_state.pagina_atual.split("app:", 1)[1].strip()
    if not app_id:
        st.query_params.clear()
        ir_para("hub")
        st.rerun()
    tela_app(app_id)

else:
    limpar_sessao()
