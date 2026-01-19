# perform.py
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils import BASE_URL_API, CARTEIRAS


# ============================================================
# BENCHMARKS (backend instrument_ids -> label)
# ============================================================
BENCHMARKS: Dict[int, str] = {
    9: "CDI",
    6: "Dolar",
    11: "IPCA",
    265: "Ibovespa",
    1533: "IMAB",
}
BENCHMARK_NAME_TO_ID: Dict[str, int] = {v: k for k, v in BENCHMARKS.items()}


# --------------------------
# Janelas (ordem + labels)
# --------------------------
WINDOW_ORDER = ["1d", "1w", "1m", "3m", "6m", "12m", "18m", "24m"]
WINDOW_LABELS = {
    "1d": "1 dia",
    "1w": "1 semana",
    "1m": "1 mês",
    "3m": "3 meses",
    "6m": "6 meses",
    "12m": "12 meses",
    "18m": "18 meses",
    "24m": "24 meses",
}


# --------------------------
# Utilitários
# --------------------------
def _periods(today: Optional[pd.Timestamp] = None, windows: Optional[List[str]] = None) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """Devolve janelas móveis padronizadas terminando em 'today'. Se windows for passado, filtra."""
    if today is None:
        today = pd.Timestamp.now().normalize()
    else:
        today = pd.Timestamp(today).normalize()

    base = {
        "1d":  (today - pd.Timedelta(days=1), today),
        "1w":  (today - pd.Timedelta(weeks=1), today),
        "1m":  (today - pd.DateOffset(months=1), today),
        "3m":  (today - pd.DateOffset(months=3), today),
        "6m":  (today - pd.DateOffset(months=6), today),
        "12m": (today - pd.DateOffset(months=12), today),
        "18m": (today - pd.DateOffset(months=18), today),
        "24m": (today - pd.DateOffset(months=24), today),
    }

    if not windows:
        return base

    valid = [w for w in windows if w in base]
    return {w: base[w] for w in valid}


def _fmt_pct(x: Any) -> str:
    
    try:
        x = float(x)
    except Exception:
        return ""
    if pd.isna(x):
        return ""

    # Se vier em decimal (0.0188), vira 1.88%
    # Se vier em % já (1.88), deixa como 1.88%
    if abs(x) <= 1.0:
        return f"{x * 100:.2f}%"
    return f"{x:.2f}%"



def _fmt_pp(pp: Any) -> str:
    try:
        pp = float(pp)
    except Exception:
        return ""
    return "" if pd.isna(pp) else f"{pp:+.2f} pp"


def _parse_percentage_value(value) -> float:
    """Converte valores de porcentagem (string ou número) para float decimal."""
    if pd.isna(value):
        return np.nan
    try:
        if isinstance(value, str):
            cleaned = value.replace("%", "").replace(",", ".").strip()
            return float(cleaned)
        if isinstance(value, (int, float, np.number)):
            return float(value)
        return np.nan
    except Exception:
        return np.nan


def _asof_on_or_before(series: pd.Series, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Retorna o último índice <= ts (as-of)."""
    if series.empty:
        return None
    idx = series.index.searchsorted(ts, side="right") - 1
    if idx >= 0:
        return series.index[idx]
    return None

from datetime import date, timedelta
import calendar
from dateutil.relativedelta import relativedelta

def nearest_business_day(d: date) -> date:
    """
    Dia útil mais próximo considerando apenas seg-sex.
    - Sábado -> sexta (volta 1 dia)
    - Domingo -> segunda (avança 1 dia)
    """
    wd = d.weekday()  # Mon=0 ... Sun=6
    if wd == 5:       # Saturday
        return d - timedelta(days=1)
    if wd == 6:       # Sunday
        return d + timedelta(days=1)
    return d

def _shift_months_keep_day_clamped(d: date, months_back: int) -> date:
    """
    Volta 'months_back' meses tentando manter o mesmo dia do mês.
    Se o dia não existir no mês alvo (ex: 31/02), usa o último dia do mês.
    """
    target = d + relativedelta(months=-months_back)
    y, m = target.year, target.month
    last_day = calendar.monthrange(y, m)[1]
    day = min(d.day, last_day)  # mantém o dia original quando possível
    return date(y, m, day)

def anchor_dates(d: date) -> dict:
    """
    Recebe uma data 'd' e devolve 4 datas ajustadas:
    - 3 meses atrás
    - 6 meses atrás
    - 18 meses atrás
    - 24 meses atrás
    Regras:
      1) volta meses mantendo o dia quando possível; se não existir, cai no último dia do mês
      2) ajusta para o dia útil mais próximo (seg-sex)
    """
    raw = {
        "3m":  _shift_months_keep_day_clamped(d, 3),
        "6m":  _shift_months_keep_day_clamped(d, 6),
        "18m": _shift_months_keep_day_clamped(d, 18),
        "24m": _shift_months_keep_day_clamped(d, 24),
    }
    return {k: nearest_business_day(v) for k, v in raw.items()}
def find_dates(d: date) -> List[str]:
    anchors = anchor_dates(d)  # {"3m": date, "6m": date, ...}

    return [
        anchors["3m"].isoformat(),
        anchors["6m"].isoformat(),
        anchors["18m"].isoformat(),
        anchors["24m"].isoformat(),
    ]


# --------------------------
# API (posições)
# --------------------------
def _post_positions(start_date: date, end_date: date, portfolio_ids: List[str], headers: Dict[str, str]) -> pd.DataFrame:
    custom_dates = find_dates(end_date)
    payload = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "instrument_position_aggregation": 3,
        "include_profitabilities": True,
        "portfolio_ids": portfolio_ids
    }
    payload2 = {
        "date": str(end_date),
        "profitability_periods": [2,3,5,6,7],
        "portfolio_ids": portfolio_ids,
        "custom_dates": custom_dates
    }
    
    try:
        r = requests.post(
            f"{BASE_URL_API}/portfolio_position/profitability/portfolio_profitability_view",
            json=payload2,
            headers=headers,
            timeout=60
        )
        
        r.raise_for_status()
        resultado = r.json()
        
        prof = resultado.get("portfolios", {})

        registros: List[dict] = []
        for item in prof.values():
            if isinstance(item, list):
                registros.extend(item)
            else:
                registros.append(item)

        df_valores = pd.json_normalize(registros)
        df_valores = df_valores.filter(like="profitability_by_custom_date.")
        vals = pd.to_numeric(df_valores.iloc[0], errors="coerce").tolist()
        prof_3m, prof_6m, prof_18m, prof_24m = vals
   
        r = requests.post(
            f"{BASE_URL_API}/portfolio_position/positions/get",
            json=payload,
            headers=headers,
            timeout=60
        )
        
        r.raise_for_status()
        resultado = r.json()
        dados = resultado.get("objects", {})
 
        registros: List[dict] = []
        for item in dados.values():
            if isinstance(item, list):
                registros.extend(item)
            else:
                registros.append(item)

        df = pd.json_normalize(registros)

        df.rename(columns={
            "profitability_start_date": "%Dt Início",
            "profitability_in_day": "%Dia",
            "profitability_in_month": "%Mês",
            "profitability_in_semester": "%Semestre",
            "profitability_in_6_months": "%6 Meses",
            "profitability_in_year": "%Ano",
            "profitability_in_12_months": "%12 Meses",
            "profitability_in_18_months": "%18 Meses",
            "profitability_in_24_months": "%24 Meses",
            "profitability_in_30_months": "%30 Meses",
            "profitability_in_36_months": "%36 Meses",
            "profitability_in_48_months": "%48 Meses",
            "profitability_in_60_months": "%60 Meses",
            "net_asset_value": "PL",
            "portfolio_id": "ID Carteira",
            "overview_type": "Tipo de Overview",
            "date": "Data",
            "name": "Carteira",
            "instrument_positions": "Ativos",
            "last_shares": "Qtd. Cotas D-1",
            "is_opening": "Carteira de Abertura",
            "id": "ID Overview",
            "navps": "Cota Líquida",
            "gross_navps": "Cota bruta",
            "shares": "Qtd. Cotas",
            "fixed_shares": "Qtd. Cotas Fixas",
            "portfolio_average_duration": "Duração Média Carteira",
            "created_on": "Data de Criação",
            "benchmark_profitability.profitability_in_day": "Bench %Dia",
            "benchmark_profitability.profitability_in_month": "Bench %Mês",
            "benchmark_profitability.profitability_in_year": "Bench %Ano",
            "benchmark_profitability.profitability_in_12_months": "Bench %12 Meses",
            "benchmark_profitability.profitability_start_date": "Bench %Dt Início",
            "benchmark_profitability.profitability_in_semester": "Bench %Semestre",
            "benchmark_profitability.profitability_in_6_months": "Bench %6 Meses",
            "benchmark_profitability.profitability_in_18_months": "Bench %18 Meses",
            "benchmark_profitability.profitability_in_24_months": "Bench %24 Meses",
            "benchmark_profitability.profitability_in_30_months": "Bench %30 Meses",
            "benchmark_profitability.profitability_in_36_months": "Bench %36 Meses",
            "benchmark_profitability.profitability_in_48_months": "Bench %48 Meses",
            "benchmark_profitability.profitability_in_60_months": "Bench %60 Meses",
            "modified_on": "Modificado em",
            "released_on": "Data de Liberação",
            "benchmark_profitability.symbol": "Nome Bench",
            "gross_asset_value": "Valor Bruto",
            "asset_value_for_allocation": "Valor para Alocação",
            "last_net_asset_value": "PL D-1",
            "last_navps": "Cota Líquida D-1",
            "fixed_navps": "Cota Fixa",
            "financial_transaction_positions": "CPR",
        }, inplace=True)

        df ["%3 Meses"] = prof_3m
        df ["%6 Meses"] = prof_6m
        df ["%18 Meses"] = prof_18m
        df ["%24 Meses"] = prof_24m

        if "Data" in df.columns:
            df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        if "Cota Líquida" in df.columns:
            df["Cota Líquida"] = pd.to_numeric(df["Cota Líquida"], errors="coerce")
     
        percentage_cols = [col for col in df.columns if col.startswith("%") or col.startswith("Bench %")]
        for col in percentage_cols:
            df[col] = df[col].apply(_parse_percentage_value)

        return df

    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")
        return pd.DataFrame()


# --------------------------
# Market Data (benchmarks) - prices/get  (PAYLOAD CORRETO: instrument_ids)
# --------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _post_market_prices(
    start_date: date,
    end_date: date,
    instrument_ids: List[int],
    headers: Dict[str, str],
) -> pd.DataFrame:
    """
    Payload correto:
      {"start_date":"YYYY-MM-DD","end_date":"YYYY-MM-DD","instrument_ids":[...]}

    Retorna DataFrame: [InstrumentID, Data, Preco]
    """
    if not instrument_ids:
        return pd.DataFrame(columns=["InstrumentID", "Data", "Preco"])

    payload = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "instrument_ids": instrument_ids,
    }

    try:
        r = requests.post(
            f"{BASE_URL_API}/market_data/pricing/prices/get",
            json=payload,
            headers=headers,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()

        root = data.get("objects", data)

        points: Optional[List[dict]] = None
        if isinstance(root, dict):
            for k in ("prices", "data", "items", "results"):
                v = root.get(k)
                if isinstance(v, list):
                    points = v
                    break

        rows: List[Dict[str, Any]] = []

        if isinstance(root, list):
            points = root

        if isinstance(points, list):
            for p in points:
                iid = p.get("instrument_id") or p.get("id") or p.get("instrument")
                dt = p.get("date") or p.get("datetime") or p.get("ref_date")
                pr = p.get("price") or p.get("value") or p.get("close") or p.get("last")
                if iid is None or dt is None or pr is None:
                    continue
                rows.append({"InstrumentID": iid, "Data": dt, "Preco": pr})

        elif isinstance(root, dict):
            for key, pts in root.items():
                if not isinstance(pts, list):
                    continue
                try:
                    iid_key = int(key)
                except Exception:
                    continue

                for p in pts:
                    dt = p.get("date") or p.get("datetime") or p.get("ref_date")
                    pr = p.get("price") or p.get("value") or p.get("close") or p.get("last")
                    if dt is None or pr is None:
                        continue
                    rows.append({"InstrumentID": iid_key, "Data": dt, "Preco": pr})

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["InstrumentID"] = pd.to_numeric(df["InstrumentID"], errors="coerce")
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        df["Preco"] = pd.to_numeric(df["Preco"], errors="coerce")

        df = df.dropna(subset=["InstrumentID", "Data", "Preco"]).copy()
        df["InstrumentID"] = df["InstrumentID"].astype(int)
        df = df.sort_values(["InstrumentID", "Data"]).drop_duplicates(["InstrumentID", "Data"], keep="last")


        return df

    except requests.HTTPError:
        try:
            st.error(f"Erro ao buscar preços de benchmarks: {r.status_code}")
            st.code(r.text)
        except Exception:
            st.error("Erro ao buscar preços de benchmarks (HTTPError sem body).")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Erro ao buscar preços de benchmarks: {e}")
        return pd.DataFrame()


# --------------------------
# Rolling returns - Carteiras
# --------------------------
WINDOW_TO_BACKEND_COL = {
    "1d":  "%Dia",
    "1m":  "%Mês",
    "3m":  "%3 Meses",     # se existir no df; se não existir, você não pode oferecer 3m
    "6m":  "%6 Meses",     # NÃO use %Semestre se você já tem %6 Meses
    "12m": "%12 Meses",    # use o específico, não %Ano
    "18m": "%18 Meses",
    "24m": "%24 Meses",
}



# --------------------------
# Rolling returns - Benchmarks (BACKEND)
# --------------------------
def _compute_benchmark_rolling_returns(
    df_prices: pd.DataFrame,
    instrument_id: int,
    periods_dict: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
) -> Dict[str, float]:
    """Retornos móveis por janela usando a série de Preço do benchmark (as-of)."""
    out: Dict[str, float] = {k: 0.0 for k in periods_dict.keys()}

    if df_prices is None or df_prices.empty:
        return out

    sub = df_prices[df_prices["InstrumentID"] == int(instrument_id)].copy()
    if sub.empty:
        return out

    sub = sub[["Data", "Preco"]].dropna().sort_values("Data")
    series = sub.set_index("Data")["Preco"]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        return out

    last_date = series.index.max()

    for name, (start_ts, end_ts) in periods_dict.items():
        end_ts = min(pd.Timestamp(end_ts), last_date)

        end_key = _asof_on_or_before(series, pd.Timestamp(end_ts))
        start_key = _asof_on_or_before(series, pd.Timestamp(start_ts))

        if end_key is None or start_key is None:
            out[name] = 0.0
            continue

        start_val = series.loc[start_key]
        end_val = series.loc[end_key]

        if pd.notna(start_val) and pd.notna(end_val) and float(start_val) != 0.0:
            out[name] = float(end_val / start_val - 1.0)
        else:
            out[name] = 0.0

    return out


# --------------------------
# Comparação (Carteiras + Benchmarks)
# --------------------------
def _create_comparison_dataframe(
    df_perf: pd.DataFrame,
    selected_carteiras: List[str],
    selected_benchmarks: List[str],
    selected_windows: List[str],
) -> pd.DataFrame:
    """Cria DataFrame comparativo entre carteiras (Cota Líquida) e benchmarks (Preço backend) nas janelas escolhidas."""
    if df_perf.empty:
        return pd.DataFrame()

    windows = [w for w in WINDOW_ORDER if w in set(selected_windows)]
    if not windows:
        return pd.DataFrame()

    max_dt = pd.to_datetime(df_perf["Data"], errors="coerce").dropna().max()
    periods = _periods(max_dt if pd.notna(max_dt) else pd.Timestamp.now(), windows=windows)

    min_start = min([p[0] for p in periods.values()])
    max_end = max([p[1] for p in periods.values()])

    selected_ids = [BENCHMARK_NAME_TO_ID[n] for n in selected_benchmarks if n in BENCHMARK_NAME_TO_ID]
    df_prices = _post_market_prices(
        start_date=min_start.date(),
        end_date=max_end.date(),
        instrument_ids=selected_ids,
        headers=st.session_state.headers,
    )

    rows = []

    for carteira in selected_carteiras:
        sub = df_perf[df_perf["Carteira"] == carteira]
        if sub.empty:
            continue

        row = sub.sort_values("Data").iloc[-1]

        item = {"Nome": carteira, "Tipo": "Carteira"}
        for w in windows:
            col = WINDOW_TO_BACKEND_COL.get(w)
            item[w] = float(row[col]) if col and col in row and pd.notna(row[col]) else np.nan
        rows.append(item)


    for bmk_name in selected_benchmarks:
        iid = BENCHMARK_NAME_TO_ID.get(bmk_name)
        if iid is None:
            continue
        br = _compute_benchmark_rolling_returns(df_prices, iid, periods)
        item = {"Nome": bmk_name, "Tipo": "Benchmark"}
        for w in windows:
            item[w] = float(br.get(w, 0.0))
        rows.append(item)

    out = pd.DataFrame(rows)
    for w in windows:
        out[w] = pd.to_numeric(out[w], errors="coerce")
    return out


# ============================================================
# RESUMO PARA CLIENTE (SEM TABELA)
# ============================================================

def _render_client_summary_multi(
    comparison_df: pd.DataFrame,
    windows: List[str],
    sort_window: str,
    selected_carteiras: List[str],
    selected_benchmarks: List[str],
):
    if comparison_df.empty or not windows:
        st.info("Sem dados para exibir.")
        return

    df = comparison_df.copy()
    for w in windows:
        df[w] = pd.to_numeric(df[w], errors="coerce")

    df_c = df[df["Tipo"] == "Carteira"].copy()
    df_b = df[df["Tipo"] == "Benchmark"].copy()

    if df_c.empty:
        st.info("Sem carteiras para exibir.")
        return
    if df_b.empty:
        st.warning("Selecione ao menos 1 benchmark para comparar.")
        return

    # carteira “principal”
    if len(selected_carteiras) == 1 and selected_carteiras[0] in df_c["Nome"].tolist():
        carteira_row = df_c[df_c["Nome"] == selected_carteiras[0]].head(1)
    else:
        carteira_row = df_c.sort_values(sort_window, ascending=False).head(1)

    carteira_name = str(carteira_row["Nome"].iloc[0])
    st.markdown(f"### {carteira_name} vs Benchmarks")

    c_val_main = float(carteira_row[sort_window].iloc[0]) if sort_window in carteira_row.columns and pd.notna(carteira_row[sort_window].iloc[0]) else np.nan



    # UM BLOCO POR BENCHMARK
    for bmk_name in selected_benchmarks:
        bmk_row = df_b[df_b["Nome"] == bmk_name].head(1)
        if bmk_row.empty:
            continue

        b_val_main = float(bmk_row[sort_window].iloc[0]) if sort_window in bmk_row.columns and pd.notna(bmk_row[sort_window].iloc[0]) else np.nan
        ex_pp_main = (c_val_main - b_val_main) * 100.0 if pd.notna(c_val_main) and pd.notna(b_val_main) else np.nan

        st.markdown(f"#### Vs {bmk_name}")


        # cards por janela (carteira + excesso vs esse benchmark)
        blocks = [windows[i:i+3] for i in range(0, len(windows), 3)]
        for blk in blocks:
            cols = st.columns(len(blk))
            for i, w in enumerate(blk):
                c_val = float(carteira_row[w].iloc[0]) if w in carteira_row.columns and pd.notna(carteira_row[w].iloc[0]) else np.nan
                b_val = float(bmk_row[w].iloc[0]) if w in bmk_row.columns and pd.notna(bmk_row[w].iloc[0]) else np.nan
                ex_pp = (c_val - b_val) * 100.0 if pd.notna(c_val) and pd.notna(b_val) else np.nan

                cols[i].metric(
                    label=f"{WINDOW_LABELS.get(w, w)}",
                    value=_fmt_pct(c_val) if pd.notna(c_val) else "-",
                    delta=_fmt_pp(ex_pp) if pd.notna(ex_pp) else "",
                )
                cols[i].caption(f"{bmk_name}: {_fmt_pct(b_val) if pd.notna(b_val) else '-'}")

        st.divider()



# --------------------------
# Tela – Performance
# --------------------------
def tela_performance() -> None:
    if "df_perf" not in st.session_state:
        st.session_state.df_perf = pd.DataFrame()

    if "headers" not in st.session_state or not st.session_state.headers:
        st.warning("Faça login para consultar os dados.")
        return

    st.markdown("### Performance de Carteiras")
    st.divider()

    # -----------------------------
    # Filtros
    # -----------------------------
    with st.container():
        c_f1, c_f2, c_f3, c_f4, c_f5 = st.columns([1.3, 2.2, 1.8, 1.9, 1.0])

        with c_f1:
            st.markdown("**Data**")
            d_fim = st.date_input(
                "",
                value=date.today(),
                key="perf_data_fim",
                label_visibility="collapsed",
            )

        with c_f2:
            carteira_nome = st.selectbox(
                "**Carteira**",
                sorted(CARTEIRAS.values()),
                index=0,
                key="perf_carteira_base",
            )

        with c_f3:
            st.markdown("**Benchmarks (backend)**")
            bmarks = st.multiselect(
                "",
                options=list(BENCHMARK_NAME_TO_ID.keys()),
                default=["CDI", "IMAB"],
                key="perf_bmarks",
                label_visibility="collapsed",
            )

        with c_f4:
            st.markdown("**Janelas (retornos)**")
            default_windows = ["1m", "3m", "6m", "12m", "18m", "24m"]
            user_windows = st.multiselect(
                "",
                options=WINDOW_ORDER,
                default=[w for w in default_windows if w in WINDOW_ORDER],
                key="perf_user_windows",
                label_visibility="collapsed",
            )

        with c_f5:
            st.markdown(" ")
            carregar = st.button("Buscar", key="perf_btn_carregar", use_container_width=True)

    # carteira id para backend
    carteiras_ids = [k for k, v in CARTEIRAS.items() if v == carteira_nome]
    if carteiras_ids:
        carteira_id = [int(carteiras_ids[0])]
    else:
        carteira_id = []

    # -----------------------------
    # Buscar snapshot (1 dia)
    # -----------------------------
    if carregar:
        if not carteira_id:
            st.error("Carteira inválida.")
            return

        try:
            with st.spinner("Buscando dados (as-of)..."):
                # snapshot: só a data final
                df = _post_positions(d_fim, d_fim, carteira_id, st.session_state.headers)

            if df.empty:
                st.info("Nenhum dado retornado para a data selecionada.")
                return

            st.session_state.df_perf = df
            st.session_state.selected_carteiras = [carteira_nome]
            st.session_state.selected_benchmarks = bmarks
            st.session_state.selected_windows = [w for w in WINDOW_ORDER if w in set(user_windows)]

            st.success("Dados carregados com sucesso!")

        except Exception as e:
            st.error(f"Erro ao buscar dados: {e}")
            return

    # -----------------------------
    # Validação pós-busca
    # -----------------------------
    if st.session_state.df_perf.empty:
        st.info("Selecione carteira, benchmarks e janelas e clique em Buscar.")
        return

    df = st.session_state.df_perf
    selected_carteiras = st.session_state.get("selected_carteiras", [])
    selected_benchmarks = st.session_state.get("selected_benchmarks", [])
    windows = st.session_state.get("selected_windows", [])

    if not selected_carteiras:
        st.info("Selecione ao menos uma carteira e clique em Buscar.")
        return

    if not selected_benchmarks:
        st.warning("Selecione ao menos 1 benchmark para comparar.")
        return

    if not windows:
        st.error("Selecione ao menos uma janela.")
        return

    # -----------------------------
    # Comparação (Carteira + Benchmarks)
    # -----------------------------
    comparison_df = _create_comparison_dataframe(
        df_perf=df,
        selected_carteiras=selected_carteiras,
        selected_benchmarks=selected_benchmarks,
        selected_windows=windows,
    )

    if comparison_df.empty:
        st.info("Sem dados para exibir.")
        return

    # janela principal para ordenar (usa 12m se existir, senão a primeira)
    sort_window = "12m" if "12m" in windows else windows[0]

    # -----------------------------
    # Resumo (cards)
    # -----------------------------
    _render_client_summary_multi(
        comparison_df=comparison_df,
        windows=windows,
        sort_window=sort_window,
        selected_carteiras=selected_carteiras,
        selected_benchmarks=selected_benchmarks,
    )
