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
WINDOW_ORDER = ["1d", "1m", "3m", "6m", "12m", "24m", "36m", "48m","60m"]
WINDOW_LABELS = {
    "1d": "1 dia",
    "1m": "1 mês",
    "3m": "3 meses",
    "6m": "6 meses",
    "12m": "12 meses",
    "24m": "24 meses",
    "36m": "36 meses",
    "48m": "48 meses",
    "60m": "60 meses",
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
        "1m":  (today - pd.DateOffset(months=1), today),
        "3m":  (today - pd.DateOffset(months=3), today),
        "6m":  (today - pd.DateOffset(months=6), today),
        "12m": (today - pd.DateOffset(months=12), today),
        "24m": (today - pd.DateOffset(months=24), today),
        "36m": (today - pd.DateOffset(months=36), today),
        "48m": (today - pd.DateOffset(months=48), today),
        "60m": (today - pd.DateOffset(months=60), today),
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
        "24m": _shift_months_keep_day_clamped(d, 24),
        "36m": _shift_months_keep_day_clamped(d, 36),
        "48m": _shift_months_keep_day_clamped(d, 48),
        "60m": _shift_months_keep_day_clamped(d, 60),
    }
    return {k: nearest_business_day(v) for k, v in raw.items()}
def find_dates(d: date) -> List[str]:
    anchors = anchor_dates(d)  # {"3m": date, "6m": date, ...}

    return [
        anchors["3m"].isoformat(),
        anchors["6m"].isoformat(),
        anchors["24m"].isoformat(),
        anchors["36m"].isoformat(),
        anchors["48m"].isoformat(),
        anchors["60m"].isoformat(),
    ]

def _has_window_coverage(cum: pd.DataFrame, end_ts: pd.Timestamp, window: str, min_coverage: float = 0.9) -> tuple[bool, str]:
    """
    Verifica se há dados suficientes para a janela.
    min_coverage = 0.9 significa: precisa cobrir pelo menos 90% do período.
    Retorna (ok, motivo).
    """
    if cum is None or cum.empty:
        return False, "Sem série acumulada."

    off = WINDOW_TO_OFFSET.get(window)
    if off is None:
        return False, "Janela não suportada."

    end_ts = pd.Timestamp(end_ts)
    start_ts = (end_ts - off).normalize()

    sub = cum.loc[(cum.index >= start_ts) & (cum.index <= end_ts)]
    if sub.empty:
        return False, "Sem pontos no intervalo."

    first = sub.index.min()
    last = sub.index.max()

    expected_days = max(1, (end_ts.normalize() - start_ts).days)
    covered_days = max(0, (last.normalize() - first.normalize()).days)

    coverage = covered_days / expected_days

    if coverage < min_coverage:
        return False, f"Dados insuficientes: cobre ~{coverage*100:.0f}% da janela (início disponível: {first.date()})."

    # Também evita janelas “capengas” com poucos pontos
    if len(sub) < 10:
        return False, f"Poucos pontos ({len(sub)}) para esta janela."

    return True, ""

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
        "custom_dates": custom_dates,
        "use_initial_date_for_investment_reports": True
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

        df_reg = pd.DataFrame(registros)
        # pegue o valor (primeira linha, coluna initial_date)
        initial_date = df_reg.loc[0, "initial_date"]
        st.session_state.perf_initial_date = str(initial_date)

        
        df_valores = pd.json_normalize(registros)
        df_valores = df_valores.filter(like="profitability_by_custom_date.")
        vals = pd.to_numeric(df_valores.iloc[0], errors="coerce").tolist()
        prof_3m, prof_6m, prof_24m, prof_36m, prof_48m, prof_60m = vals

        payload3 = {
        "start_date": str(initial_date),
        "end_date": str(end_date),
        "instrument_position_aggregation": 3,
        "include_profitabilities": False,
        "portfolio_ids": portfolio_ids,
        }

        r3 = requests.post(
            f"{BASE_URL_API}/portfolio_position/positions/get",
            json=payload3,
            headers=headers,
            timeout=60,
        )
        r3.raise_for_status()
        j = r3.json()

        data_graf = r3.json().get("objects", {})
        

        registros: List[dict] = []
        for item in data_graf.values():
            registros.extend(item if isinstance(item, list) else [item])
        st.session_state.df_graf = pd.json_normalize(registros)
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
        df ["%24 Meses"] = prof_24m
        df ["%36 Meses"] = prof_36m
        df ["%48 Meses"] = prof_48m
        df ["%60 Meses"] = prof_60m

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
    "24m": "%24 Meses",
    "36m": "%36 Meses",
    "48m": "%48 Meses",
    "60m": "%60 Meses",
}


def _explode_instrument_positions(df_graf: pd.DataFrame) -> pd.DataFrame:
    """
    Converte df_graf (positions/get normalizado) em um DF longo:
    Data | Instrumento | Qty | MarketValue | UnitPrice
    Tenta descobrir nomes de chaves comuns no dict de cada posição.
    """
    if df_graf is None or df_graf.empty:
        return pd.DataFrame()

    if "instrument_positions" not in df_graf.columns:
        return pd.DataFrame()

    df = df_graf.copy()
    df["Data"] = pd.to_datetime(df.get("date"), errors="coerce")

    rows = []
    for _, r in df.iterrows():
        dt = r.get("Data")
        if pd.isna(dt):
            continue

        pos = r.get("instrument_positions")
        if not isinstance(pos, list):
            continue

        for p in pos:
            if not isinstance(p, dict):
                continue

            # Nome / ID (tentativas)
            name = p.get("instrument_name") or p.get("name") or p.get("symbol") or p.get("ticker") or p.get("instrument")
            iid = p.get("instrument_id") or p.get("id")

            # Quantidade (tentativas)
            qty = (
                p.get("quantity")
                or p.get("shares")
                or p.get("position")
                or p.get("qtd")
            )

            # Valor (tentativas)
            mv = (
                p.get("market_value")
                or p.get("gross_market_value")
                or p.get("asset_value")
                or p.get("value")
                or p.get("financial_value")
            )

            # Preço unitário (tentativas)
            price = (
                p.get("price")
                or p.get("unit_price")
                or p.get("last_price")
                or p.get("market_price")
            )

            # Sanitiza
            try:
                qty_f = float(qty) if qty is not None else np.nan
            except Exception:
                qty_f = np.nan

            try:
                mv_f = float(mv) if mv is not None else np.nan
            except Exception:
                mv_f = np.nan

            try:
                price_f = float(price) if price is not None else np.nan
            except Exception:
                price_f = np.nan

            # Se não veio preço, tenta inferir: market_value / qty
            if (pd.isna(price_f) or price_f == 0.0) and pd.notna(mv_f) and pd.notna(qty_f) and qty_f != 0.0:
                price_f = mv_f / qty_f

            # Nome final
            inst = str(name) if name is not None else (f"ID {iid}" if iid is not None else None)
            if inst is None:
                continue

            rows.append({
                "Data": dt,
                "Instrumento": inst,
                "InstrumentID": iid,
                "Qty": qty_f,
                "MarketValue": mv_f,
                "UnitPrice": price_f,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["UnitPrice"] = pd.to_numeric(out["UnitPrice"], errors="coerce")
    out = out.dropna(subset=["Data", "Instrumento", "UnitPrice"]).sort_values(["Instrumento", "Data"])

    # remove duplicatas por instrumento/dia
    out = out.drop_duplicates(subset=["Instrumento", "Data"], keep="last")
    return out

def _instrument_price_pivot(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long is None or df_long.empty:
        return pd.DataFrame()
    piv = df_long.pivot_table(index="Data", columns="Instrumento", values="UnitPrice", aggfunc="last").sort_index()
    return piv.ffill()

def _rank_best_worst(piv_price: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp, n: int = 5, min_points: int = 5):
    """
    Retorno por ativo no período: last/first - 1 (as-of dentro do recorte).
    Filtra ativos com poucos pontos.
    """
    if piv_price is None or piv_price.empty:
        return pd.DataFrame(), pd.DataFrame()

    sub = piv_price.loc[(piv_price.index >= start_ts) & (piv_price.index <= end_ts)].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    # remove colunas sem dados no recorte
    sub = sub.dropna(axis=1, how="all")
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    # filtra por quantidade mínima de pontos
    counts = sub.notna().sum(axis=0)
    keep = counts[counts >= min_points].index.tolist()
    sub = sub[keep]
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    first = sub.apply(lambda s: s.dropna().iloc[0] if s.dropna().shape[0] else np.nan, axis=0)
    last = sub.apply(lambda s: s.dropna().iloc[-1] if s.dropna().shape[0] else np.nan, axis=0)

    ret = (last / first - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    if ret.empty:
        return pd.DataFrame(), pd.DataFrame()

    top = ret.sort_values(ascending=False).head(n).to_frame("Retorno")
    bot = ret.sort_values(ascending=True).head(n).to_frame("Retorno")
    return top, bot

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

def _nav_from_df_graf(df_graf: pd.DataFrame) -> pd.DataFrame:
    """
    Converte df_graf (normalizado do positions/get) em série diária de cota.
    Espera colunas: 'date' e 'navps'
    """
    df = df_graf.copy()
    df["Data"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["Cota"] = pd.to_numeric(df.get("navps"), errors="coerce")

    df = df.dropna(subset=["Data", "Cota"]).sort_values("Data").drop_duplicates(subset=["Data"], keep="last")
    return df[["Data", "Cota"]]

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

def _post_portfolio_nav_series(
    df
) -> pd.DataFrame:
   

    df["Data"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["Cota"] = pd.to_numeric(df.get("navps"), errors="coerce")  # cota líquida
    df = df.dropna(subset=["Data", "Cota"]).sort_values("Data").drop_duplicates(subset=["Data"], keep="last")

    return df[["Data", "Cota"]]

def _bench_price_pivot(
    start_date: date,
    end_date: date,
    benchmark_names: List[str],
    headers: Dict[str, str],
) -> pd.DataFrame:
    ids = [BENCHMARK_NAME_TO_ID[n] for n in benchmark_names if n in BENCHMARK_NAME_TO_ID]
    if not ids:
        return pd.DataFrame()

    df_prices = _post_market_prices(start_date, end_date, ids, headers)
    if df_prices.empty:
        return pd.DataFrame()

    piv = (
        df_prices.pivot_table(index="Data", columns="InstrumentID", values="Preco", aggfunc="last")
        .sort_index()
        .ffill()
    )
    piv = piv.rename(columns={BENCHMARK_NAME_TO_ID[n]: n for n in benchmark_names if n in BENCHMARK_NAME_TO_ID})
    return piv

def _series_to_daily_returns(port_nav: pd.DataFrame, bench_prices: pd.DataFrame) -> pd.DataFrame:
    # carteira: cota -> retorno diário
    s_port = port_nav.set_index("Data")["Cota"].sort_index()
    ret_port = s_port.pct_change().fillna(0.0).rename("Carteira")

    # benchmarks: preço -> retorno diário
    ret_bench = bench_prices.pct_change().fillna(0.0) if bench_prices is not None and not bench_prices.empty else pd.DataFrame()

    # junta por data
    ret_all = pd.concat([ret_port, ret_bench], axis=1).sort_index()

    # alinhamento: se algum índice faltar, assume 0 retorno naquele dia (não inventa preço)
    ret_all = ret_all.fillna(0.0)

    return ret_all


def _daily_returns_to_cum(ret_all: pd.DataFrame) -> pd.DataFrame:
    cum = (1.0 + ret_all).cumprod() - 1.0
    return cum

WINDOW_TO_OFFSET = {
    "1m":  pd.DateOffset(months=1),
    "3m":  pd.DateOffset(months=3),
    "6m":  pd.DateOffset(months=6),
    "12m": pd.DateOffset(months=12),
    "24m": pd.DateOffset(months=24),
    "36m": pd.DateOffset(months=36),
    "48m": pd.DateOffset(months=48),
    "60m": pd.DateOffset(months=60),
}

def _slice_and_rebase(cum: pd.DataFrame, end_ts: pd.Timestamp, window: str) -> pd.DataFrame:
    if cum.empty:
        return pd.DataFrame()
    off = WINDOW_TO_OFFSET.get(window)
    if off is None:
        return pd.DataFrame()

    start_ts = (pd.Timestamp(end_ts) - off).normalize()
    sub = cum.loc[(cum.index >= start_ts) & (cum.index <= pd.Timestamp(end_ts))].copy()
    if sub.empty:
        return sub

    return sub - sub.iloc[0]


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

    # carteira principal
    if len(selected_carteiras) == 1 and selected_carteiras[0] in df_c["Nome"].tolist():
        carteira_row = df_c[df_c["Nome"] == selected_carteiras[0]].head(1)
    else:
        carteira_row = df_c.sort_values(sort_window, ascending=False).head(1)

    carteira_name = str(carteira_row["Nome"].iloc[0])
    st.markdown(f"### {carteira_name} vs Benchmarks")

    # filtra e ordena benchmarks conforme seleção do usuário
    df_b = df_b[df_b["Nome"].isin(selected_benchmarks)].copy()
    if df_b.empty:
        st.warning("Selecione ao menos 1 benchmark para comparar.")
        return

    df_b["ord"] = df_b["Nome"].apply(lambda x: selected_benchmarks.index(x) if x in selected_benchmarks else 9999)
    df_b = df_b.sort_values("ord").drop(columns=["ord"])

    windows_ordered = [w for w in WINDOW_ORDER if w in set(windows)]
    tabs = st.tabs([WINDOW_LABELS.get(w, w) for w in windows_ordered])

    def fmt_pct(x):
        return _fmt_pct(x) if pd.notna(x) else "-"

    def fmt_pp(x):
        return _fmt_pp(x) if pd.notna(x) else "-"

    for i, w in enumerate(windows_ordered):
        with tabs[i]:
            c_val = float(carteira_row[w].iloc[0]) if w in carteira_row.columns and pd.notna(carteira_row[w].iloc[0]) else np.nan

            # tabela numérica (para lógica)
            rows = []
            for _, bmk_row in df_b.iterrows():
                bmk_name = str(bmk_row["Nome"])
                b_val = float(bmk_row[w]) if w in bmk_row and pd.notna(bmk_row[w]) else np.nan
                ex_pp = (c_val - b_val) * 100.0 if pd.notna(c_val) and pd.notna(b_val) else np.nan

                rows.append({
                    "Benchmark": bmk_name,
                    "Carteira (%)": c_val,
                    "Benchmark (%)": b_val,
                    "Excesso (pp)": ex_pp,  # NUMÉRICO AQUI
                })

            t = pd.DataFrame(rows)

            # tabela formatada (para exibir)
            t_show = t.copy()
            t_show["Carteira (%)"] = t_show["Carteira (%)"].apply(fmt_pct)
            t_show["Benchmark (%)"] = t_show["Benchmark (%)"].apply(fmt_pct)
            t_show["Excesso (pp)"] = t_show["Excesso (pp)"].apply(fmt_pp)

            # estilo por linha usando os valores numéricos de t
            def _style_row(row_idx: int):
                v = t.loc[row_idx, "Excesso (pp)"]
                if pd.isna(v):
                    return [""] * t_show.shape[1]
                if v > 0:
                    color = "color: #1a7f37; font-weight: 700;"
                elif v < 0:
                    color = "color: #b42318; font-weight: 700;"
                else:
                    color = "color: #555555;"
                styles = [""] * t_show.shape[1]
                excesso_col = list(t_show.columns).index("Excesso (pp)")
                styles[excesso_col] = color
                return styles

            styler = t_show.style.apply(lambda _row: _style_row(_row.name), axis=1)
            styler = styler.set_properties(subset=["Benchmark"], **{"font-weight": "600"})

            st.dataframe(
                styler,
                use_container_width=True,
                hide_index=True,
                key=f"tbl_client_{w}",
            )

            st.caption(f"Carteira no período: {fmt_pct(c_val)}")




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
            with st.spinner("Buscando dados (pode demorar um pouco)..."):
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
    modo = st.radio(
        "Exibição",
        options=["Dados", "Gráficos"],
        horizontal=True,
        key="perf_view_mode",
    )

    if modo == "Dados":
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
    else:
        initial_date = st.session_state.get("perf_initial_date")
        if not initial_date:
            st.info("Sem initial_date. Clique em Buscar novamente para carregar a data inicial da carteira.")
            return

        d_ini = pd.to_datetime(initial_date, errors="coerce").date()
        if not d_ini:
            st.error("initial_date inválida.")
            return

        d_fim = st.session_state.get("perf_data_fim", date.today())
        end_ts = pd.Timestamp(d_fim)

        # resolve portfolio_id (você já tem carteira_id acima)
        if not carteira_id:
            st.error("Carteira inválida.")
            return
        portfolio_id = int(carteira_id[0])

        with st.spinner("Buscando séries diárias para gráficos..."):
            df_graf = st.session_state.get("df_graf")
            if df_graf is None or df_graf.empty:
                st.info("Sem df_graf no session_state. Garanta que você carregou o histórico no botão Buscar.")
                return

            # série da carteira (navps)
            port_nav = _nav_from_df_graf(df_graf)
            if port_nav.empty:
                st.error("Não consegui montar série de cota (navps) a partir do df_graf.")
                return

            # benchmarks (preços)
            bench_prices = _bench_price_pivot(d_ini, d_fim, selected_benchmarks, st.session_state.headers)

            # retornos diários + acumulado (carteira + benchmarks)
            ret_all = _series_to_daily_returns(port_nav, bench_prices)
            cum = _daily_returns_to_cum(ret_all)

            # ---- ativos: explode + pivot de preço unitário (uma vez só)
            df_long_inst = _explode_instrument_positions(df_graf)
            piv_inst_price = _instrument_price_pivot(df_long_inst)

        # janelas a plotar (respeita a ordem)
        windows_plot = [w for w in WINDOW_ORDER if w in set(windows)]
        if not windows_plot:
            st.warning("Nenhuma janela selecionada para gráficos.")
            return

        tab_labels = [WINDOW_LABELS.get(w, w) for w in windows_plot]
        tabs = st.tabs(tab_labels)

        for i, w in enumerate(windows_plot):
            with tabs[i]:
                ok, reason = _has_window_coverage(cum, end_ts, w, min_coverage=0.9)
                if not ok:
                    st.info(f"Sem dados suficientes para {WINDOW_LABELS.get(w, w)}. {reason}")
                    continue

                sub = _slice_and_rebase(cum, end_ts, w)
                if sub.empty:
                    st.info(f"Sem dados para {WINDOW_LABELS.get(w, w)}.")
                    continue

                # ----------------
                # Gráfico
                # ----------------
                df_plot = (sub * 100.0).reset_index()
                xcol = df_plot.columns[0]
                ycols = [c for c in df_plot.columns if c != xcol]

                fig = px.line(df_plot, x=xcol, y=ycols)
                fig.update_layout(yaxis_title="Retorno acumulado (%)", xaxis_title="")
                st.plotly_chart(fig, use_container_width=True, key=f"graf_perf_{w}")

                last = (sub.iloc[-1] * 100.0).round(2)
                st.dataframe(last.to_frame("Retorno (%)"), key=f"tbl_perf_{w}")

                # ----------------
                # Top 5 / Bottom 5 ativos
                # ----------------
                st.divider()
                st.markdown("#### Melhores e piores ativos no período")

                off = WINDOW_TO_OFFSET.get(w)
                start_ts = (pd.Timestamp(end_ts) - off).normalize() if off else sub.index.min()

                if piv_inst_price is None or piv_inst_price.empty:
                    st.info("Sem dados de ativos (instrument_positions) para montar ranking.")
                    continue

                top, bot = _rank_best_worst(
                    piv_price=piv_inst_price,
                    start_ts=start_ts,
                    end_ts=pd.Timestamp(end_ts),
                    n=5,
                    min_points=5,
                )

                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("**Top 5**")
                    if top.empty:
                        st.info("Sem dados suficientes para ranking no período.")
                    else:
                        top_show = top.copy()
                        top_show["Retorno (%)"] = (top_show["Retorno"] * 100.0).round(2)
                        top_show = top_show.drop(columns=["Retorno"])
                        st.dataframe(top_show, use_container_width=True, key=f"top5_{w}")

                with c2:
                    st.markdown("**Bottom 5**")
                    if bot.empty:
                        st.info("Sem dados suficientes para ranking no período.")
                    else:
                        bot_show = bot.copy()
                        bot_show["Retorno (%)"] = (bot_show["Retorno"] * 100.0).round(2)
                        bot_show = bot_show.drop(columns=["Retorno"])
                        st.dataframe(bot_show, use_container_width=True, key=f"bot5_{w}")
