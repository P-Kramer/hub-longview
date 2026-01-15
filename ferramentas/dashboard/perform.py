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


def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# --------------------------
# API (posições)
# --------------------------
def _post_positions(start_date: date, end_date: date, portfolio_ids: List[str], headers: Dict[str, str]) -> pd.DataFrame:
    payload = {
    "start_date": str(start_date),
    "end_date": str(start_date),
    "instrument_position_aggregation": 3,
    "portfolio_ids": portfolio_ids,
    "include_profitabilities": True,
}
    
    try:
        r = requests.post(
            f"{BASE_URL_API}/portfolio_position/positions/get",
            json=payload,
            headers=headers,
            timeout=60
        )
        
        r.raise_for_status()
        resultado = r.json()
        dados = resultado.get("objects", {})
        print (dados)
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
        st.dataframe(df)
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


def _get_portfolio_returns_from_backend_df(df: pd.DataFrame, carteira: str, windows: List[str]) -> Dict[str, float]:
    out = {w: np.nan for w in windows}
    sub = df[df["Carteira"] == carteira].copy()
    if sub.empty:
        return out

    # pega o último registro (mesma data, mas ok)
    row = sub.sort_values("Data").iloc[-1] if "Data" in sub.columns else sub.iloc[-1]

    for w in windows:
        col = WINDOW_TO_BACKEND_COL.get(w)
        if not col or col not in df.columns:
            out[w] = np.nan
            continue
        v = row.get(col, np.nan)
        out[w] = float(v) if pd.notna(v) else np.nan

    return out


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
# Barras: retorno por janela (mantido)
# --------------------------
def _render_performance_bars(comparison_df: pd.DataFrame, windows: List[str]):
    if comparison_df.empty or not windows:
        return

    chart_df = comparison_df.set_index("Nome")
    fig = go.Figure()
    colors = px.colors.qualitative.Set3

    for i, w in enumerate(windows):
        fig.add_trace(go.Bar(
            name=w,
            x=chart_df.index,
            y=(chart_df[w] * 100.0),
            text=chart_df[w].apply(lambda x: f"{x*100:.1f}%"),
            textposition="auto",
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        title="Barras: Retorno por Janelas Selecionadas",
        xaxis_title="",
        yaxis_title="Retorno (%)",
        barmode="group",
        height=520,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------
# Equity curves
# --------------------------
def _render_equity_curves(
    df_perf: pd.DataFrame,
    selected_carteiras: List[str],
    benchmark_names: List[str],
    main_window: str
):
    """Equity curve base 100 na janela escolhida. Benchmarks via backend prices/get."""
    if df_perf.empty or not selected_carteiras:
        return

    max_dt = pd.to_datetime(df_perf["Data"], errors="coerce").dropna().max()
    periods = _periods(max_dt if pd.notna(max_dt) else pd.Timestamp.now(), windows=[main_window])
    if main_window not in periods:
        return
    start_ts, end_ts = periods[main_window]

    st.markdown(f"### Equity Curve (base 100) — janela {main_window}")

    fig = go.Figure()

    for carteira in selected_carteiras:
        sub = df_perf[df_perf["Carteira"] == carteira].copy()
        if sub.empty:
            continue
        sub = sub[["Data", "Cota Líquida"]].dropna().sort_values("Data")
        sub = sub[(sub["Data"] >= pd.Timestamp(start_ts)) & (sub["Data"] <= pd.Timestamp(end_ts))]
        if sub.empty:
            continue

        s = sub.set_index("Data")["Cota Líquida"]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if s.empty:
            continue

        base = float(s.iloc[0])
        if base == 0 or pd.isna(base):
            continue

        eq = (s / base) * 100.0
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=f"{carteira}"))

    selected_ids = [BENCHMARK_NAME_TO_ID[n] for n in benchmark_names if n in BENCHMARK_NAME_TO_ID]
    df_prices = _post_market_prices(
        start_date=pd.Timestamp(start_ts).date(),
        end_date=pd.Timestamp(end_ts).date(),
        instrument_ids=selected_ids,
        headers=st.session_state.headers,
    )

    for name in benchmark_names:
        iid = BENCHMARK_NAME_TO_ID.get(name)
        if iid is None:
            continue

        sub = df_prices[df_prices["InstrumentID"] == int(iid)].copy()
        if sub.empty:
            continue

        sub = sub[["Data", "Preco"]].dropna().sort_values("Data")
        s = sub.set_index("Data")["Preco"]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if s.empty:
            continue

        base = float(s.iloc[0])
        if base == 0 or pd.isna(base):
            continue

        eq = (s / base) * 100.0
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=name))

    fig.update_layout(height=520, xaxis_title="Data", yaxis_title="Base 100", legend_title="Séries")
    st.plotly_chart(fig, use_container_width=True)


# --------------------------
# Detalhe por carteira (mantido)
# --------------------------
def _render_detailed_comparison(
    df_perf: pd.DataFrame,
    selected_carteiras: List[str],
    selected_benchmarks: List[str],
    selected_windows: List[str],
):
    if df_perf.empty or not selected_carteiras or not selected_benchmarks:
        return

    windows = [w for w in WINDOW_ORDER if w in set(selected_windows)]
    if not windows:
        return

    st.markdown("### Detalhe por Carteira")

    max_dt = pd.to_datetime(df_perf["Data"], errors="coerce").dropna().max()
    periods = _periods(max_dt if pd.notna(max_dt) else pd.Timestamp.now(), windows=windows)
    preferred = "12m" if "12m" in windows else windows[-1]

    min_start = min([p[0] for p in periods.values()])
    max_end = max([p[1] for p in periods.values()])
    selected_ids = [BENCHMARK_NAME_TO_ID[n] for n in selected_benchmarks if n in BENCHMARK_NAME_TO_ID]
    df_prices = _post_market_prices(
        start_date=min_start.date(),
        end_date=max_end.date(),
        instrument_ids=selected_ids,
        headers=st.session_state.headers,
    )



# --------------------------
# Auditoria (mantida)
# --------------------------
def _create_backend_vs_calculated_df(
    df_perf: pd.DataFrame,
    selected_carteiras: List[str],
    selected_windows: List[str],
) -> pd.DataFrame:
    if df_perf.empty or not selected_carteiras:
        return pd.DataFrame()

    windows = [w for w in WINDOW_ORDER if w in set(selected_windows)]
    if not windows:
        return pd.DataFrame()

    mapping = {
        "1d": "%Dia",
        "1m": "%Mês",
        "6m": "%6 Meses",
        "12m": "%12 Meses",
        "18m": "%18 Meses",
        "24m": "%24 Meses",
    }

    max_dt_global = pd.to_datetime(df_perf["Data"], errors="coerce").dropna().max()
    periods = _periods(max_dt_global if pd.notna(max_dt_global) else pd.Timestamp.now(), windows=windows)

    rows = []
    for carteira in selected_carteiras:
        sub = df_perf[df_perf["Carteira"] == carteira].copy()
        if sub.empty:
            continue

        sub["Data"] = pd.to_datetime(sub["Data"], errors="coerce")
        sub = sub.dropna(subset=["Data"]).sort_values("Data")
        if sub.empty:
            continue

        last = sub.iloc[-1]


        row: Dict[str, Any] = {"Carteira": carteira, "Data": sub["Data"].max().date()}

        for w in windows:
            col = mapping.get(w)
            if not col:
                continue
            backend_val = last.get(col, np.nan)
            backend_val = float(backend_val) if pd.notna(backend_val) else np.nan


        rows.append(row)

    return pd.DataFrame(rows)


def _render_backend_vs_calculated_pretty(df_perf: pd.DataFrame, selected_carteiras: List[str], selected_windows: List[str]):
    st.markdown("### Auditoria: Calculado vs Backend")

    cmp_df = _create_backend_vs_calculated_df(df_perf, selected_carteiras, selected_windows)
    if cmp_df.empty:
        st.info("Sem dados para comparar.")
        return

    show = cmp_df.copy()
    diff_cols = [c for c in show.columns if c.endswith("diff(pp)")]
    calc_cols = [c for c in show.columns if c.endswith(" calc")]
    back_cols = [c for c in show.columns if c.endswith(" back")]

    for c in calc_cols + back_cols:
        show[c] = show[c].apply(lambda x: _fmt_pct(x) if pd.notna(x) else "")

    def _diff_style(v):
        try:
            v = float(v)
        except Exception:
            return ""
        if pd.isna(v):
            return ""
        if abs(v) >= 1.0:
            return "background-color: #ffd6d6"
        if abs(v) >= 0.25:
            return "background-color: #fff3cd"
        return ""

    sty = show.style
    for c in diff_cols:
        sty = sty.format({c: (lambda x: _fmt_pp(x) if pd.notna(x) else "")})
        sty = sty.applymap(_diff_style, subset=[c])

    st.dataframe(sty, use_container_width=True)


# --------------------------
# Tela – Performance
# --------------------------
def tela_performance() -> None:
    if "df_perf" not in st.session_state:
        st.session_state.df_perf = pd.DataFrame()

    if "headers" not in st.session_state or not st.session_state.headers:
        st.warning("Faça login para consultar os dados.")
        return
    df = st.session_state.get("df_perf", pd.DataFrame())
    
    st.markdown("### Performance de Carteiras")
    st.divider()

    with st.container():
        c_f1, c_f2, c_f3,  c_f5 = st.columns([1.2, 2.2, 1.4, 1.0])

        with c_f1:
            st.markdown("**Janela de Consulta (dados brutos)**")
            d_ref = st.date_input(
            "",
            value=date.today(),
            key="perf_data_ref",
            label_visibility="collapsed",
        )
        d_ini = d_ref
        d_fim = d_ref


        with c_f2:
            st.markdown("**Carteiras**")
            carteiras_nomes = st.multiselect(
                "",
                sorted(CARTEIRAS.values()),
                default=[],
                key="perf_carteiras",
                label_visibility="collapsed"
            )

        with c_f3:
            st.markdown("**Benchmarks (backend)**")
            bmarks = st.multiselect(
                "",
                options=list(BENCHMARK_NAME_TO_ID.keys()),
                default=["CDI", "IMAB"],
                key="perf_bmarks",
                label_visibility="collapsed"
            )



        with c_f5:
            st.markdown(" ")
            carregar = st.button("Buscar", key="perf_btn_carregar", use_container_width=True)

    carteiras_ids = [k for k, v in CARTEIRAS.items() if v in carteiras_nomes]
    windows = st.session_state.get("selected_windows", ["1m","3m","6m","12m", "ano", "dia"])
    windows = [w for w in WINDOW_ORDER if w in set(windows)]
    # mapeamento janela -> coluna do df do backend
    WINDOW_TO_BACKEND_COL = {
        "1m": "%Mês",
        "6m": "%6 Meses",
        "12m": "%12 Meses",
        "18m": "%18 Meses",
        "24m": "%24 Meses",
        "ano": "%Ano",
        "dia": "%Dia"
    }

    # carteira base = primeira selecionada
    carteira_base = st.session_state.get("selected_carteiras", [])
    carteira_base = carteira_base[0] if carteira_base else None

    windows_filtradas = []

    if carteira_base:
        sub = df[df.get("Carteira") == carteira_base] if "Carteira" in df.columns else pd.DataFrame()
        if not sub.empty:
            # pega a última linha (mais recente)
            row = sub.sort_values("Data").iloc[-1] if "Data" in sub.columns else sub.iloc[-1]

            for w in windows:
                col = WINDOW_TO_BACKEND_COL.get(w)
                if not col or col not in df.columns:
                    continue

                val = row.get(col, np.nan)

                # >>> FILTRO: só entra se NÃO for NaN
                if pd.notna(val):
                    windows_filtradas.append(w)

    windows = windows_filtradas


    if carregar:
        if not carteiras_ids:
            st.error("Selecione ao menos uma carteira.")
            return


        try:
            with st.spinner("Buscando dados das carteiras..."):
                df = _post_positions(d_ini, d_fim, carteiras_ids, st.session_state.headers)

            if df.empty:
                st.info("Nenhum dado retornado para os filtros informados.")
                return

            st.session_state.df_perf = df
            st.session_state.selected_carteiras = carteiras_nomes
            st.session_state.selected_benchmarks = bmarks
            st.session_state.selected_windows = [w for w in WINDOW_ORDER if w in set(windows)]
            st.success("Dados carregados com sucesso!")

        except Exception as e:
            st.error(f"Erro ao buscar dados: {e}")
            return

    if "df_perf" not in st.session_state or st.session_state.df_perf.empty:
        st.info("Selecione carteiras, benchmarks e janelas e clique em Buscar.")
        return
    print(df)

    windows = ["1m", "3m", "6m", "12m"]
    windows = [w for w in WINDOW_ORDER if w in set(windows)]


    comparison_df = _create_comparison_dataframe(
        df,
        st.session_state.get("selected_carteiras", []),
        st.session_state.get("selected_benchmarks", []),
        windows
    )

    if comparison_df.empty:
        st.info("Sem dados para exibir.")
        return

    main_window = windows[0]

    tab1, tab2 = st.tabs(["Resumo", "Gráficos"])

    with tab1:
        _render_client_summary_multi(
            comparison_df=comparison_df,
            windows=windows,
            sort_window=main_window,
            selected_carteiras=st.session_state.get("selected_carteiras", []),
            selected_benchmarks=st.session_state.get("selected_benchmarks", []),
        )

    with tab2:
        _render_performance_bars(comparison_df, windows)
        st.divider()
        _render_equity_curves(
            df_perf=df,
            selected_carteiras=st.session_state.get("selected_carteiras", []),
            benchmark_names=st.session_state.get("selected_benchmarks", []),
            main_window=main_window
        )

