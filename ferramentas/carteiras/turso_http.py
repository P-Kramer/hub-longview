import streamlit as st
import requests

def _pipeline_url() -> str:
    base = st.secrets["TURSO"]["DATABASE_URL"].replace("libsql://", "https://").rstrip("/")
    return f"{base}/v2/pipeline"

def turso_exec(sql: str, args=None) -> dict:
    payload = {
        "requests": [
            {"type": "execute", "stmt": {"sql": sql}},
            {"type": "close"},
        ]
    }
    if args is not None:
        payload["requests"][0]["stmt"]["args"] = args

    r = requests.post(
        _pipeline_url(),
        headers={
            "Authorization": f"Bearer {st.secrets['TURSO']['AUTH_TOKEN']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def init_favoritos_schema():
    turso_exec("""
    CREATE TABLE IF NOT EXISTS favoritos_colunas (
      tela   TEXT NOT NULL,
      aba    TEXT NOT NULL,
      coluna TEXT NOT NULL,
      created_at TEXT DEFAULT (datetime('now')),
      PRIMARY KEY (tela, aba, coluna)
    );
    """)

def load_favoritos(tela: str) -> dict:
    out = turso_exec(
        "SELECT aba, coluna FROM favoritos_colunas WHERE tela = ? ORDER BY aba, coluna;",
        [{"type": "text", "value": tela}],
    )
    res = out["results"][0]["response"].get("result")
    rows = res.get("rows", []) if res else []

    favs = {}
    for r in rows:
        aba = r[0]["value"]
        coluna = r[1]["value"]
        favs.setdefault(aba, []).append(coluna)
    return favs

def _exists(tela: str, aba: str, coluna: str) -> bool:
    out = turso_exec(
        "SELECT 1 FROM favoritos_colunas WHERE tela=? AND aba=? AND coluna=? LIMIT 1;",
        [{"type":"text","value":tela},{"type":"text","value":aba},{"type":"text","value":coluna}],
    )
    res = out["results"][0]["response"].get("result")
    return bool(res and res.get("rows"))

def toggle_favorito(tela: str, aba: str, coluna: str) -> bool:
    """True = adicionou, False = removeu"""
    if _exists(tela, aba, coluna):
        turso_exec(
            "DELETE FROM favoritos_colunas WHERE tela=? AND aba=? AND coluna=?;",
            [{"type":"text","value":tela},{"type":"text","value":aba},{"type":"text","value":coluna}],
        )
        return False

    turso_exec(
        "INSERT OR IGNORE INTO favoritos_colunas (tela, aba, coluna) VALUES (?, ?, ?);",
        [{"type":"text","value":tela},{"type":"text","value":aba},{"type":"text","value":coluna}],
    )
    return True
