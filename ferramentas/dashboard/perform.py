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

# yfinance só é usado se você habilitar benchmarks externos específicos
try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False


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
    return "" if pd.isna(x) else f"{x*100:.2f}%"


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
            return float(cleaned) / 100.0
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


def _style_rank_arrows(df: pd.DataFrame, cols: List[str]) -> "pd.io.formats.style.Styler":
    """
    Estilo sem matplotlib: destaca top/bottom 3 por coluna
    (verde claro = top, vermelho claro = bottom).
    """
    sty = df.style

    def _hl_top_bottom(s: pd.Series):
        # s é uma coluna numérica
        s_num = pd.to_numeric(s, errors="coerce")
        top_idx = s_num.nlargest(min(3, s_num.notna().sum())).index
        bot_idx = s_num.nsmallest(min(3, s_num.notna().sum())).index
        out = pd.Series([""] * len(s), index=s.index)

        out.loc[top_idx] = "background-color: #d7f5e6"   # verde claro
        out.loc[bot_idx] = "background-color: #ffd6d6"   # vermelho claro
        return out

    for c in cols:
        if c in df.columns:
            sty = sty.apply(_hl_top_bottom, subset=[c])
    return sty



# --------------------------
# API dos Indicadores Econômicos
# --------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_cdi_data() -> Dict[str, Any]:
    """Busca dados do CDI/SELIC."""
    try:
        url_selic = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json"
        r_selic = requests.get(url_selic, timeout=10)
        r_selic.raise_for_status()
        selic_data = r_selic.json()
        selic_value = float(selic_data[0]["valor"]) if selic_data else 0.0

        current_year = datetime.now().year
        url_cdi_acum = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.4391/dados?formato=json&dataInicial=01/01/{current_year}"
        r_cdi_acum = requests.get(url_cdi_acum, timeout=10)
        r_cdi_acum.raise_for_status()
        cdi_acum_data = r_cdi_acum.json()
        cdi_acum = float(cdi_acum_data[-1]["valor"]) if cdi_acum_data else 0.0

        return {"selic_meta": selic_value, "cdi_acum_ano": cdi_acum, "fonte": "Banco Central do Brasil"}
    except Exception as e:
        st.error(f"Erro ao buscar CDI: {e}")
        return {"selic_meta": 0.0, "cdi_acum_ano": 0.0, "fonte": "Erro"}


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_imab_data() -> Dict[str, Any]:
    """Busca dados do IMA-B via ETF IMAB11 como proxy."""
    try:
        if _HAS_YF:
            imab = yf.Ticker("IMAB11.SA")
            hist = imab.history(period="1y")
            if not hist.empty:
                price_current = hist["Close"].iloc[-1]
                price_prev = hist["Close"].iloc[-2] if len(hist) > 1 else price_current
                daily_change = ((price_current - price_prev) / price_prev) * 100

                current_year = datetime.now().year
                year_start = f"{current_year}-01-01"
                hist_ytd = imab.history(start=year_start)
                ytd_change = ((hist_ytd["Close"].iloc[-1] - hist_ytd["Close"].iloc[0]) / hist_ytd["Close"].iloc[0]) * 100 if len(hist_ytd) > 1 else 0.0

                return {
                    "valor_atual": float(price_current),
                    "variacao_dia": float(daily_change),
                    "variacao_ytd": float(ytd_change),
                    "fonte": "YFinance (IMAB11)"
                }
    except Exception as e:
        st.error(f"Erro ao buscar IMA-B: {e}")

    return {"valor_atual": 0.0, "variacao_dia": 0.0, "variacao_ytd": 0.0, "fonte": "Erro"}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_dolar_data() -> Dict[str, Any]:
    """Busca cotação do dólar (BCB) com fallback Yahoo Finance."""
    try:
        url_bcb = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados/ultimos/1?formato=json"
        r_bcb = requests.get(url_bcb, timeout=10)
        r_bcb.raise_for_status()
        bcb_data = r_bcb.json()

        if bcb_data:
            dolar_bcb = float(bcb_data[0]["valor"])
            url_bcb_prev = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados/ultimos/2?formato=json"
            r_bcb_prev = requests.get(url_bcb_prev, timeout=10)
            r_bcb_prev.raise_for_status()
            bcb_prev_data = r_bcb_prev.json()

            if len(bcb_prev_data) == 2:
                dolar_prev = float(bcb_prev_data[0]["valor"])
                variation = ((dolar_bcb - dolar_prev) / dolar_prev) * 100
            else:
                variation = 0.0

            return {"cotacao": float(dolar_bcb), "variacao": float(variation), "fonte": "Banco Central do Brasil"}
    except Exception:
        if _HAS_YF:
            try:
                usdbrl = yf.Ticker("USDBRL=X")
                hist = usdbrl.history(period="2d")
                if len(hist) >= 2:
                    price_current = hist["Close"].iloc[-1]
                    price_prev = hist["Close"].iloc[-2]
                    variation = ((price_current - price_prev) / price_prev) * 100
                    return {"cotacao": float(price_current), "variacao": float(variation), "fonte": "YFinance (Fallback)"}
            except Exception as yf_error:
                st.error(f"Erro ao buscar dólar YFinance: {yf_error}")

    return {"cotacao": 0.0, "variacao": 0.0, "fonte": "Erro"}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_sp500_data() -> Dict[str, Any]:
    """Busca dados do S&P 500."""
    try:
        if _HAS_YF:
            sp500 = yf.Ticker("^GSPC")
            hist = sp500.history(period="2d")
            if len(hist) >= 2:
                price_current = hist["Close"].iloc[-1]
                price_prev = hist["Close"].iloc[-2]
                daily_change = ((price_current - price_prev) / price_prev) * 100

                current_year = datetime.now().year
                year_start = f"{current_year}-01-01"
                hist_ytd = sp500.history(start=year_start)
                ytd_change = ((hist_ytd["Close"].iloc[-1] - hist_ytd["Close"].iloc[0]) / hist_ytd["Close"].iloc[0]) * 100 if len(hist_ytd) > 1 else 0.0

                return {
                    "valor_atual": float(price_current),
                    "variacao_dia": float(daily_change),
                    "variacao_ytd": float(ytd_change),
                    "fonte": "YFinance"
                }
    except Exception as e:
        st.error(f"Erro ao buscar S&P 500: {e}")

    return {"valor_atual": 0.0, "variacao_dia": 0.0, "variacao_ytd": 0.0, "fonte": "Erro"}


def _display_indicators():
    """Exibe os indicadores econômicos sem HTML/CSS."""
    st.markdown("### Indicadores Econômicos")
    with st.spinner("Atualizando indicadores..."):
        cdi_data = _fetch_cdi_data()
        imab_data = _fetch_imab_data()
        dolar_data = _fetch_dolar_data()
        sp500_data = _fetch_sp500_data()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CDI/SELIC (meta anual)", f"{cdi_data['selic_meta']:.2f}%", f"{cdi_data['cdi_acum_ano']:.2f}% (YTD)")
        st.caption(f"Fonte: {cdi_data['fonte']}")
    with col2:
        st.metric("IMA-B (IMAB11)", f"R$ {imab_data['valor_atual']:.2f}", f"{imab_data['variacao_dia']:.2f}%")
        st.caption(f"YTD: {imab_data['variacao_ytd']:.2f}% | {imab_data['fonte']}")
    with col3:
        st.metric("USD/BRL", f"R$ {dolar_data['cotacao']:.2f}", f"{dolar_data['variacao']:.2f}%")
        st.caption(f"Fonte: {dolar_data['fonte']}")
    with col4:
        st.metric("S&P 500", f"{sp500_data['valor_atual']:.0f}", f"{sp500_data['variacao_dia']:.2f}%")
        st.caption(f"YTD: {sp500_data['variacao_ytd']:.2f}% | {sp500_data['fonte']}")


# --------------------------
# API (posições)
# --------------------------
def _post_positions(start_date: date, end_date: date, portfolio_ids: List[str], headers: Dict[str, str]) -> pd.DataFrame:
    payload = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "instrument_position_aggregation": 3,
        "portfolio_ids": portfolio_ids
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

        # Tipos
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
# Rolling returns - Carteiras
# --------------------------
def _compute_portfolio_rolling_returns(
    df: pd.DataFrame,
    carteira: str,
    periods_dict: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
) -> Dict[str, float]:
    """
    Retornos móveis por janela usando a Cota Líquida.
    Correto:
      - End = último ponto disponível <= end_ts
      - Start = as-of (último <= start_ts)
    """
    out: Dict[str, float] = {k: 0.0 for k in periods_dict.keys()}

    sub = df[df["Carteira"] == carteira].copy()
    if sub.empty or "Data" not in sub.columns or "Cota Líquida" not in sub.columns:
        return out

    sub = sub[["Data", "Cota Líquida"]].dropna().sort_values("Data")
    if sub.empty:
        return out

    series = sub.set_index("Data")["Cota Líquida"]
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
# Benchmarks
# --------------------------
def _calculate_benchmark_returns(periods_dict: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]], benchmark_name: str) -> Dict[str, float]:
    """
    Calcula retornos dos benchmarks para janelas móveis.
    - CDI: usa SELIC meta anual -> taxa diária (aprox) e acumula por dias úteis (mínimo 1).
    - IMAB11, S&P 500, USD/BRL: usa Yahoo Finance (Close) start->end.
    """
    returns: Dict[str, float] = {k: 0.0 for k in periods_dict.keys()}

    try:
        if benchmark_name == "CDI":
            cdi_data = _fetch_cdi_data()
            anual = float(cdi_data.get("selic_meta", 0.0)) / 100.0
            cdi_daily = (1.0 + anual) ** (1.0 / 252.0) - 1.0

            for k, (ini, fim) in periods_dict.items():
                bd = int(np.busday_count(ini.date(), fim.date()))
                bd = max(1, bd)
                returns[k] = (1.0 + cdi_daily) ** bd - 1.0

        elif benchmark_name == "IMA-B (IMAB11)":
            if not _HAS_YF:
                return returns
            t = yf.Ticker("IMAB11.SA")
            for k, (ini, fim) in periods_dict.items():
                hist = t.history(start=ini.date(), end=(fim + pd.Timedelta(days=1)).date())
                if len(hist) >= 2:
                    start_price = float(hist["Close"].iloc[0])
                    end_price = float(hist["Close"].iloc[-1])
                    returns[k] = (end_price / start_price) - 1.0 if start_price else 0.0

        elif benchmark_name == "S&P 500":
            if not _HAS_YF:
                return returns
            t = yf.Ticker("^GSPC")
            for k, (ini, fim) in periods_dict.items():
                hist = t.history(start=ini.date(), end=(fim + pd.Timedelta(days=1)).date())
                if len(hist) >= 2:
                    start_price = float(hist["Close"].iloc[0])
                    end_price = float(hist["Close"].iloc[-1])
                    returns[k] = (end_price / start_price) - 1.0 if start_price else 0.0

        elif benchmark_name == "USD/BRL":
            if not _HAS_YF:
                return returns
            t = yf.Ticker("USDBRL=X")
            for k, (ini, fim) in periods_dict.items():
                hist = t.history(start=ini.date(), end=(fim + pd.Timedelta(days=1)).date())
                if len(hist) >= 2:
                    start_price = float(hist["Close"].iloc[0])
                    end_price = float(hist["Close"].iloc[-1])
                    returns[k] = (end_price / start_price) - 1.0 if start_price else 0.0

    except Exception as e:
        st.error(f"Erro ao calcular {benchmark_name}: {e}")

    return returns


# --------------------------
# Comparação (Carteiras + Benchmarks)
# --------------------------
def _create_comparison_dataframe(
    df_perf: pd.DataFrame,
    selected_carteiras: List[str],
    selected_benchmarks: List[str],
    selected_windows: List[str],
) -> pd.DataFrame:
    """Cria DataFrame comparativo entre carteiras (Cota Líquida) e benchmarks nas janelas escolhidas."""
    if df_perf.empty:
        return pd.DataFrame()

    windows = [w for w in WINDOW_ORDER if w in set(selected_windows)]
    if not windows:
        return pd.DataFrame()

    max_dt = pd.to_datetime(df_perf["Data"], errors="coerce").dropna().max()
    periods = _periods(max_dt if pd.notna(max_dt) else pd.Timestamp.now(), windows=windows)

    rows = []

    for carteira in selected_carteiras:
        roll = _compute_portfolio_rolling_returns(df_perf, carteira, periods)
        item = {"Nome": carteira, "Tipo": "Carteira"}
        for w in windows:
            item[w] = float(roll.get(w, 0.0))
        rows.append(item)

    for bmk in selected_benchmarks:
        br = _calculate_benchmark_returns(periods, bmk)
        item = {"Nome": bmk, "Tipo": "Benchmark"}
        for w in windows:
            item[w] = float(br.get(w, 0.0))
        rows.append(item)

    out = pd.DataFrame(rows)
    for w in windows:
        out[w] = pd.to_numeric(out[w], errors="coerce").fillna(0.0)
    return out


def _render_summary_table(comparison_df: pd.DataFrame, windows: List[str], sort_window: str):
    if comparison_df.empty or not windows:
        return

    st.markdown("### Resumo (ranking)")

    df = _safe_numeric(comparison_df, windows).copy()

    # ordena por janela principal
    if sort_window in windows:
        df = df.sort_values(by=sort_window, ascending=False)

    # carteiras primeiro, depois benchmarks
    df["__ord"] = df["Tipo"].apply(lambda x: 0 if x == "Carteira" else 1)
    df = df.sort_values(["__ord"] + ([sort_window] if sort_window in windows else []),
                        ascending=[True] + ([False] if sort_window in windows else []))
    df = df.drop(columns=["__ord"])

    show = df[["Nome", "Tipo"] + windows].copy()
    show_idx = show.set_index(["Tipo", "Nome"])

    sty = _style_rank_arrows(show_idx, cols=windows)
    sty = sty.format({w: (lambda x: _fmt_pct(x)) for w in windows})

    st.dataframe(sty, use_container_width=True)



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


def _render_equity_curves(
    df_perf: pd.DataFrame,
    selected_carteiras: List[str],
    benchmark_names: List[str],
    main_window: str
):
    """Equity curve base 100 na janela escolhida."""
    if df_perf.empty or not selected_carteiras:
        return

    max_dt = pd.to_datetime(df_perf["Data"], errors="coerce").dropna().max()
    periods = _periods(max_dt if pd.notna(max_dt) else pd.Timestamp.now(), windows=[main_window])
    if main_window not in periods:
        return
    start_ts, end_ts = periods[main_window]

    st.markdown(f"### Equity Curve (base 100) — janela {main_window}")

    fig = go.Figure()

    # Carteiras
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

    # Benchmarks
    if "CDI" in benchmark_names:
        cdi_data = _fetch_cdi_data()
        anual = float(cdi_data.get("selic_meta", 0.0)) / 100.0
        cdi_daily = (1.0 + anual) ** (1.0 / 252.0) - 1.0
        dates = pd.bdate_range(start=start_ts.date(), end=end_ts.date())
        if len(dates) > 0:
            eq = (1.0 + cdi_daily) ** np.arange(len(dates))
            eq = (eq / eq[0]) * 100.0
            fig.add_trace(go.Scatter(x=dates, y=eq, mode="lines", name="CDI (aprox)"))

    if _HAS_YF:
        def _add_yf_curve(symbol: str, name: str):
            try:
                t = yf.Ticker(symbol)
                hist = t.history(start=start_ts.date(), end=(end_ts + pd.Timedelta(days=1)).date())
                if hist is None or hist.empty or "Close" not in hist.columns:
                    return
                s = hist["Close"].dropna()
                if s.empty:
                    return
                base = float(s.iloc[0])
                if base == 0:
                    return
                eq = (s / base) * 100.0
                fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=name))
            except Exception:
                return

        if "IMA-B (IMAB11)" in benchmark_names:
            _add_yf_curve("IMAB11.SA", "IMA-B (IMAB11)")
        if "USD/BRL" in benchmark_names:
            _add_yf_curve("USDBRL=X", "USD/BRL")
        if "S&P 500" in benchmark_names:
            _add_yf_curve("^GSPC", "S&P 500")

    fig.update_layout(height=520, xaxis_title="Data", yaxis_title="Base 100", legend_title="Séries")
    st.plotly_chart(fig, use_container_width=True)


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

    for carteira in selected_carteiras:
        sub = df_perf[df_perf["Carteira"] == carteira].copy()
        if sub.empty:
            continue

        st.subheader(carteira)
        kpis = _compute_portfolio_rolling_returns(df_perf, carteira, periods)

        c1, c2 = st.columns([1.4, 1.0])

        with c1:
            cols = st.columns(min(4, len(windows)))
            for i, w in enumerate(windows):
                with cols[i % len(cols)]:
                    st.metric(label=w, value=_fmt_pct(kpis.get(w, 0.0)))

        with c2:
            st.markdown(f"**Vs Benchmarks ({preferred})**")
            carteira_val = float(kpis.get(preferred, 0.0))
            for bmk in selected_benchmarks:
                br = _calculate_benchmark_returns(periods, bmk)
                diff_pp = (carteira_val - float(br.get(preferred, 0.0))) * 100.0
                st.metric(label=bmk, value=_fmt_pp(diff_pp))


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
        calc = _compute_portfolio_rolling_returns(df_perf, carteira, periods)

        row: Dict[str, Any] = {"Carteira": carteira, "Data": sub["Data"].max().date()}

        for w in windows:
            col = mapping.get(w)
            if not col:
                continue
            backend_val = last.get(col, np.nan)
            backend_val = float(backend_val) if pd.notna(backend_val) else np.nan
            calc_val = float(calc.get(w, np.nan))

            row[f"{w} calc"] = calc_val
            row[f"{w} back"] = backend_val
            row[f"{w} diff(pp)"] = (calc_val - backend_val) * 100.0 if pd.notna(calc_val) and pd.notna(backend_val) else np.nan

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
    if "headers" not in st.session_state or not st.session_state.headers:
        st.warning("Faça login para consultar os dados.")
        return

    st.markdown("### Performance de Carteiras")

    _display_indicators()
    st.divider()

    with st.container():
        c_f1, c_f2, c_f3, c_f4, c_f5 = st.columns([1.2, 2.2, 1.4, 1.6, 1.0])

        with c_f1:
            st.markdown("**Janela de Consulta (dados brutos)**")
            janela = st.date_input(
                "",
                value=[date.today() - timedelta(days=800), date.today()],
                key="perf_janela",
                label_visibility="collapsed"
            )
            if isinstance(janela, list) and len(janela) == 2:
                d_ini, d_fim = janela
            else:
                d_ini = date.today() - timedelta(days=800)
                d_fim = date.today()

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
            st.markdown("**Benchmarks**")
            bmarks = st.multiselect(
                "",
                ["CDI", "IMA-B (IMAB11)", "USD/BRL", "S&P 500"],
                default=["CDI", "IMA-B (IMAB11)"],
                key="perf_bmarks",
                label_visibility="collapsed"
            )

        with c_f4:
            st.markdown("**Janelas**")
            selected_windows = st.multiselect(
                "",
                options=WINDOW_ORDER,
                default=["1m", "3m", "6m", "12m"],
                format_func=lambda w: f"{w} — {WINDOW_LABELS.get(w, w)}",
                key="perf_windows",
                label_visibility="collapsed",
            )

        with c_f5:
            st.markdown(" ")
            carregar = st.button("Buscar", key="perf_btn_carregar", use_container_width=True)

    carteiras_ids = [k for k, v in CARTEIRAS.items() if v in carteiras_nomes]

    if carregar:
        if not carteiras_ids:
            st.error("Selecione ao menos uma carteira.")
            return
        if not selected_windows:
            st.error("Selecione ao menos uma janela de performance.")
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
            st.session_state.selected_windows = [w for w in WINDOW_ORDER if w in set(selected_windows)]
            st.success("Dados carregados com sucesso!")

        except Exception as e:
            st.error(f"Erro ao buscar dados: {e}")
            return

    if "df_perf" not in st.session_state or st.session_state.df_perf.empty:
        st.info("Selecione carteiras, benchmarks e janelas e clique em Buscar.")
        return

    df = st.session_state.df_perf
    windows = st.session_state.get("selected_windows", ["1m", "3m", "6m", "12m"])
    windows = [w for w in WINDOW_ORDER if w in set(windows)]
    if not windows:
        st.warning("Selecione ao menos uma janela.")
        return

    comparison_df = _create_comparison_dataframe(
        df,
        st.session_state.get("selected_carteiras", []),
        st.session_state.get("selected_benchmarks", []),
        windows
    )

    if comparison_df.empty:
        st.info("Sem dados para exibir.")
        return

    main_window = st.selectbox(
        "Janela principal (ranking e destaque)",
        options=windows,
        index=windows.index("12m") if "12m" in windows else len(windows) - 1
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Resumo", "Gráficos", "Detalhe", "Auditoria", "Bruto"])

    with tab1:
        _render_summary_table(comparison_df, windows, sort_window=main_window)

    with tab2:
        _render_performance_bars(comparison_df, windows)
        st.divider()
        _render_equity_curves(
            df_perf=df,
            selected_carteiras=st.session_state.get("selected_carteiras", []),
            benchmark_names=st.session_state.get("selected_benchmarks", []),
            main_window=main_window
        )

    with tab3:
        _render_detailed_comparison(
            df,
            st.session_state.get("selected_carteiras", []),
            st.session_state.get("selected_benchmarks", []),
            windows
        )

    with tab4:
        _render_backend_vs_calculated_pretty(
            df,
            st.session_state.get("selected_carteiras", []),
            windows
        )

    with tab5:
        cols = ["Carteira", "Data", "Cota Líquida", "%Dia", "%Mês", "%Ano", "%12 Meses", "%18 Meses", "%24 Meses"]
        cols = [c for c in cols if c in df.columns]
        display_df = df[cols].copy()
        for col in [c for c in cols if c.startswith("%")]:
            display_df[col] = display_df[col].apply(lambda x: _fmt_pct(x) if pd.notna(x) else "")
        st.dataframe(display_df.sort_values(["Carteira", "Data"]), use_container_width=True)
