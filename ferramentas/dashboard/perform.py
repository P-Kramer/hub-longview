# perform.py
from __future__ import annotations

import io
import math
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils import BASE_URL_API, CARTEIRAS

# yfinance só é usado se você habilitar benchmarks externos específicos
try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False


# --------------------------
# Estilo (herdado do simul.py)
# --------------------------
CSS = """
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }
.card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 12px;
  padding: 14px 14px;
  background: white;
}
.card-muted {
  border: 1px dashed rgba(0,0,0,0.10);
  border-radius: 12px;
  padding: 12px 12px;
  background: rgba(0,0,0,0.02);
}
.h-label {
  font-weight: 600; font-size: 0.95rem; margin: 0 0 6px 0;
}
.help {
  color: #6b7280; font-size: 0.85rem; margin-top: -4px; margin-bottom: 8px;
}
.hr { height: 1px; background: rgba(0,0,0,0.06); margin: 10px 0 14px 0; }
.indicator-card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 12px;
  padding: 16px 14px;
  background: white;
  text-align: center;
}
.comparison-positive { color: #00a86b; font-weight: 600; }
.comparison-negative { color: #ff4b4b; font-weight: 600; }
.comparison-neutral { color: #6b7280; }
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 8px;
  margin: 10px 0;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------
# Utilitários
# --------------------------
def _to_date(x) -> date:
    return pd.to_datetime(x).date()

def _periods(today: Optional[pd.Timestamp] = None) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """Devolve janelas móveis padronizadas terminando em 'today'."""
    if today is None:
        today = pd.Timestamp.now().normalize()
    else:
        today = pd.Timestamp(today).normalize()

    return {
        "1d":  (today - pd.Timedelta(days=1), today),
        "1w":  (today - pd.Timedelta(weeks=1), today),
        "1m":  (today - pd.DateOffset(months=1), today),
        "3m":  (today - pd.DateOffset(months=3), today),
        "6m":  (today - pd.DateOffset(months=6), today),
        "12m": (today - pd.DateOffset(months=12), today),
        "18m": (today - pd.DateOffset(months=18), today),
        "24m": (today - pd.DateOffset(months=24), today),
    }

def _fmt_pct(x) -> str:
    return "" if pd.isna(x) else f"{x*100:.2f}%"

def _fmt_pct_color(x) -> str:
    """Formata porcentagem com cores, convertendo para float primeiro"""
    try:
        if isinstance(x, str):
            x_clean = x.replace('%', '').replace(',', '.').strip()
            x_float = float(x_clean) / 100.0
        else:
            x_float = float(x)
    except (ValueError, TypeError):
        x_float = 0.0
    
    if pd.isna(x_float):
        return ""
    
    color_class = "comparison-positive" if x_float > 0 else "comparison-negative" if x_float < 0 else "comparison-neutral"
    return f'<span class="{color_class}">{x_float*100:.2f}%</span>'

def _fmt_diff_color(x) -> str:
    """Formata diferença com cores, convertendo para float primeiro"""
    try:
        if isinstance(x, str):
            x_clean = x.replace('pp', '').replace(',', '.').strip()
            x_float = float(x_clean) / 100.0
        else:
            x_float = float(x)
    except (ValueError, TypeError):
        x_float = 0.0
    
    if pd.isna(x_float):
        return ""
    
    color_class = "comparison-positive" if x_float > 0 else "comparison-negative" if x_float < 0 else "comparison-neutral"
    symbol = "+" if x_float > 0 else ""
    return f'<span class="{color_class}">{symbol}{x_float*100:.2f}pp</span>'

def _parse_percentage_value(value) -> float:
    """Converte valores de porcentagem (string ou número) para float decimal"""
    if pd.isna(value):
        return 0.0
    try:
        if isinstance(value, str):
            cleaned = value.replace('%', '').replace(',', '.').strip()
            return float(cleaned) / 100.0
        elif isinstance(value, (int, float)):
            # Se já vier em decimal (ex. 0.0123), mantém; se vier como 5.4 (5,4%), trate antes no backend.
            return float(value)
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0

def _equity_curve(ret: pd.Series) -> pd.Series:
    eq = (1.0 + ret.fillna(0)).cumprod()
    return eq/eq.dropna().iloc[0]

def _drawdown(ret: pd.Series) -> pd.Series:
    eq = _equity_curve(ret)
    peak = eq.cummax()
    return (eq/peak) - 1.0

def _first_on_or_after(series: pd.Series, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Primeiro timestamp >= ts na série indexada por datetime."""
    idx = series.index.searchsorted(ts, side="left")
    if idx < len(series.index):
        return series.index[idx]
    return None

def _last_on_or_before(series: pd.Series, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Último timestamp <= ts na série indexada por datetime."""
    idx = series.index.searchsorted(ts, side="right") - 1
    if idx >= 0:
        return series.index[idx]
    return None


# --------------------------
# API dos Indicadores Econômicos
# --------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_cdi_data() -> Dict[str, Any]:
    """Busca dados do CDI/SELIC."""
    try:
        # SELIC meta (aproxima a taxa anual)
        url_selic = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json"
        r_selic = requests.get(url_selic, timeout=10)
        r_selic.raise_for_status()
        selic_data = r_selic.json()
        selic_value = float(selic_data[0]['valor']) if selic_data else 0.0
        
        # CDI acumulado no ano
        current_year = datetime.now().year
        url_cdi_acum = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.4391/dados?formato=json&dataInicial=01/01/{current_year}"
        r_cdi_acum = requests.get(url_cdi_acum, timeout=10)
        r_cdi_acum.raise_for_status()
        cdi_acum_data = r_cdi_acum.json()
        cdi_acum = float(cdi_acum_data[-1]['valor']) if cdi_acum_data else 0.0
            
        return {
            'selic_meta': selic_value,
            'cdi_acum_ano': cdi_acum,
            'fonte': 'Banco Central do Brasil'
        }
    except Exception as e:
        st.error(f"Erro ao buscar CDI: {e}")
        return {'selic_meta': 0.0, 'cdi_acum_ano': 0.0, 'fonte': 'Erro'}

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_imab_data() -> Dict[str, Any]:
    """Busca dados do IMA-B via ETF IMAB11 como proxy."""
    try:
        if _HAS_YF:
            imab = yf.Ticker("IMAB11.SA")
            hist = imab.history(period="1y")
            if not hist.empty:
                price_current = hist['Close'].iloc[-1]
                price_prev = hist['Close'].iloc[-2] if len(hist) > 1 else price_current
                daily_change = ((price_current - price_prev) / price_prev) * 100
                
                # YTD
                current_year = datetime.now().year
                year_start = f"{current_year}-01-01"
                hist_ytd = imab.history(start=year_start)
                if len(hist_ytd) > 1:
                    ytd_change = ((hist_ytd['Close'].iloc[-1] - hist_ytd['Close'].iloc[0]) / hist_ytd['Close'].iloc[0]) * 100
                else:
                    ytd_change = 0.0
                    
                return {
                    'valor_atual': float(price_current),
                    'variacao_dia': float(daily_change),
                    'variacao_ytd': float(ytd_change),
                    'fonte': 'YFinance (IMAB11)'
                }
    except Exception as e:
        st.error(f"Erro ao buscar IMA-B: {e}")
    
    return {'valor_atual': 0.0, 'variacao_dia': 0.0, 'variacao_ytd': 0.0, 'fonte': 'Erro'}

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_dolar_data() -> Dict[str, Any]:
    """Busca cotação do dólar (BCB) com fallback Yahoo Finance)."""
    try:
        url_bcb = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados/ultimos/1?formato=json"
        r_bcb = requests.get(url_bcb, timeout=10)
        r_bcb.raise_for_status()
        bcb_data = r_bcb.json()
        
        if bcb_data:
            dolar_bcb = float(bcb_data[0]['valor'])
            url_bcb_prev = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados/ultimos/2?formato=json"
            r_bcb_prev = requests.get(url_bcb_prev, timeout=10)
            r_bcb_prev.raise_for_status()
            bcb_prev_data = r_bcb_prev.json()
            
            if len(bcb_prev_data) == 2:
                dolar_prev = float(bcb_prev_data[0]['valor'])
                variation = ((dolar_bcb - dolar_prev) / dolar_prev) * 100
            else:
                variation = 0.0
                
            return {
                'cotacao': float(dolar_bcb),
                'variacao': float(variation),
                'fonte': 'Banco Central do Brasil'
            }
    except Exception:
        # Fallback para Yahoo Finance
        if _HAS_YF:
            try:
                usdbrl = yf.Ticker("USDBRL=X")
                hist = usdbrl.history(period="2d")
                if len(hist) >= 2:
                    price_current = hist['Close'].iloc[-1]
                    price_prev = hist['Close'].iloc[-2]
                    variation = ((price_current - price_prev) / price_prev) * 100
                    return {
                        'cotacao': float(price_current),
                        'variacao': float(variation),
                        'fonte': 'YFinance (Fallback)'
                    }
            except Exception as yf_error:
                st.error(f"Erro ao buscar dólar YFinance: {yf_error}")
    
    return {'cotacao': 0.0, 'variacao': 0.0, 'fonte': 'Erro'}

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_sp500_data() -> Dict[str, Any]:
    """Busca dados do S&P 500."""
    try:
        if _HAS_YF:
            sp500 = yf.Ticker("^GSPC")
            hist = sp500.history(period="2d")
            if len(hist) >= 2:
                price_current = hist['Close'].iloc[-1]
                price_prev = hist['Close'].iloc[-2]
                daily_change = ((price_current - price_prev) / price_prev) * 100
                
                current_year = datetime.now().year
                year_start = f"{current_year}-01-01"
                hist_ytd = sp500.history(start=year_start)
                if len(hist_ytd) > 1:
                    ytd_change = ((hist_ytd['Close'].iloc[-1] - hist_ytd['Close'].iloc[0]) / hist_ytd['Close'].iloc[0]) * 100
                else:
                    ytd_change = 0.0
                    
                return {
                    'valor_atual': float(price_current),
                    'variacao_dia': float(daily_change),
                    'variacao_ytd': float(ytd_change),
                    'fonte': 'YFinance'
                }
    except Exception as e:
        st.error(f"Erro ao buscar S&P 500: {e}")
    
    return {'valor_atual': 0.0, 'variacao_dia': 0.0, 'variacao_ytd': 0.0, 'fonte': 'Erro'}

def _display_indicators():
    """Exibe os indicadores econômicos em cards (visão rápida)."""
    st.markdown("### 📊 Indicadores Econômicos")
    with st.spinner("Atualizando indicadores..."):
        cdi_data = _fetch_cdi_data()
        imab_data = _fetch_imab_data()
        dolar_data = _fetch_dolar_data()
        sp500_data = _fetch_sp500_data()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="indicator-card">', unsafe_allow_html=True)
        st.markdown('<div class="h-label">💰 CDI/SELIC</div>', unsafe_allow_html=True)
        st.metric("Meta Anual", f"{cdi_data['selic_meta']:.2f}%", f"{cdi_data['cdi_acum_ano']:.2f}% (YTD)")
        st.caption(f"Fonte: {cdi_data['fonte']}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="indicator-card">', unsafe_allow_html=True)
        st.markdown('<div class="h-label">📈 IMA-B</div>', unsafe_allow_html=True)
        st.metric("Valor", f"R$ {imab_data['valor_atual']:.2f}", f"{imab_data['variacao_dia']:.2f}%")
        st.caption(f"YTD: {imab_data['variacao_ytd']:.2f}% | {imab_data['fonte']}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="indicator-card">', unsafe_allow_html=True)
        st.markdown('<div class="h-label">💵 Dólar</div>', unsafe_allow_html=True)
        st.metric("USD/BRL", f"R$ {dolar_data['cotacao']:.2f}", f"{dolar_data['variacao']:.2f}%")
        st.caption(f"Fonte: {dolar_data['fonte']}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="indicator-card">', unsafe_allow_html=True)
        st.markdown('<div class="h-label">🌎 S&P 500</div>', unsafe_allow_html=True)
        st.metric("Índice", f"{sp500_data['valor_atual']:.0f}", f"{sp500_data['variacao_dia']:.2f}%")
        st.caption(f"YTD: {sp500_data['variacao_ytd']:.2f}% | {sp500_data['fonte']}")
        st.markdown('</div>', unsafe_allow_html=True)


# --------------------------
# API
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
            headers=st.session_state.headers
        )
        r.raise_for_status()
        resultado = r.json()
        dados = resultado.get("objects", {})    
        registros = []
        for item in dados.values():
            if isinstance(item, list):
                registros.extend(item)
            else:
                registros.append(item)  
        df = pd.json_normalize(registros)

        # Guarda bruto para uso complementar
        st.session_state.df = df    

        # Renomeia colunas externas (overview / principais)
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
            'id': 'ID Overview',
            "navps": "Cota Líquida",
            "gross_navps": "Cota bruta",
            "shares" : "Qtd. Cotas",
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
            "attribution.portfolio_beta.financial_value": "PnL Beta",
            "attribution_portfolio_beta_percentage": "zzzzzzzzz_Repetido",
            "attribution_portfolio_beta_financial": "zzzzzzz_Repetido",
            "attribution.portfolio_beta.percentage_value": "PnL % Beta",
            "attribution.total.financial_value": "PnL Total",
            "attribution.total.percentage_value": "PnL % Total",
            "attribution_total_financial": "zzzzzz_Repetido",
            "attribution_total_percentage": "zzzzz_Repetido",
            "attribution.currency.financial_value": "PnL Moeda",
            "attribution.currency.percentage_value": "PnL % Moeda",
            "attribution_currency_financial": "zzzz_Repetido",
            "attribution_currency_percentage": "zzz_Repetido",
            "attribution_maximums.par_price": "PnL Máximo Preço Par",
            "attribution_maximums.portfolio_beta": "PnL Máximo Beta da Carteira",
            "attribution_maximums.total": "PnL Máximo Total",
            "attribution_maximums.total_hedged": "PnL Máximo Total Hedgeado",
            "corp_actions_adjusted_navps": "Cota Líquida Ajustada por Eventos Societários",
            "corp_actions_factor": "Fator de Ajuste por Eventos Societários",
            "equity_exposure": "Exposição em Renda Variável",
            "is_system_generated": "Gerado pelo Sistema",
            "navps_admin_status": "Status Administrativo da Cota Líquida",
            "navps_one_day_return": "Retorno Diário da Cota Líquida",
            "navps_status": "Status da Cota Líquida",
            "net_liabilities_transactions_financial_value": "Valor Financeiro das Transações de Passivo Líquido",
            "overview_status": "Status do Overview",
            "pct_lent_exposure": "Exposição % Doada",
            "portfolio_average_term": "Prazo Médio da Carteira",
            "attribution_maximums.corp_actions": "PnL Máximo Eventos Societários",
            "attribution_maximums.currency": "PnL Máximo  Moeda"
        }, inplace=True)

        # Remove colunas repetidas marcadas
        cols_to_drop = [c for c in df.columns if 'repetido' in c.lower()]
        df = df.drop(columns=cols_to_drop)

        # Tipos e parsing
        if 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        if 'Cota Líquida' in df.columns:
            df['Cota Líquida'] = pd.to_numeric(df['Cota Líquida'], errors='coerce')

        # Converter colunas % (se vieram como string) para decimal
        percentage_cols = [col for col in df.columns if col.startswith('%') or col.startswith('Bench %')]
        for col in percentage_cols:
            df[col] = df[col].apply(_parse_percentage_value)

        return df

    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")
        return pd.DataFrame()


# --------------------------
# Cálculo de retornos móveis (carteiras e benchmarks)
# --------------------------
def _compute_portfolio_rolling_returns(df: pd.DataFrame, carteira: str, periods_dict: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]) -> Dict[str, float]:
    """
    Retornos móveis por janela usando a Cota Líquida.
    Se faltar dado suficiente, retorna 0.0 para a janela.
    """
    out: Dict[str, float] = {k: 0.0 for k in periods_dict.keys()}
    sub = df[df["Carteira"] == carteira].copy()
    if sub.empty or "Data" not in sub or "Cota Líquida" not in sub:
        return out

    sub = sub[['Data', 'Cota Líquida']].dropna().sort_values('Data')
    if sub.empty:
        return out

    series = sub.set_index('Data')['Cota Líquida']
    series = series[~series.index.duplicated(keep='last')].sort_index()

    for name, (start_ts, end_ts) in periods_dict.items():
        # pega o último ponto <= end e o primeiro >= start
        end_key = _last_on_or_before(series, pd.Timestamp(end_ts))
        start_key = _first_on_or_after(series, pd.Timestamp(start_ts))
        if end_key is None or start_key is None:
            out[name] = 0.0
            continue
        start_val = series.loc[start_key]
        end_val = series.loc[end_key]
        if pd.notna(start_val) and pd.notna(end_val) and start_val != 0:
            out[name] = float(end_val / start_val - 1.0)
        else:
            out[name] = 0.0
    return out

def _calculate_benchmark_returns(periods_dict: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]], benchmark_name: str) -> Dict[str, float]:
    """
    Calcula retornos dos benchmarks para janelas móveis.
    - CDI: usa taxa diária aproximada pela SELIC meta anual -> (1+anual)^(1/252)-1 e acumula por dias úteis.
    - IMAB11, S&P 500, USD/BRL: usa Yahoo Finance (Close) start->end.
    """
    returns: Dict[str, float] = {k: 0.0 for k in periods_dict.keys()}

    try:
        if benchmark_name == "CDI":
            cdi_data = _fetch_cdi_data()
            anual = cdi_data.get('selic_meta', 0.0) / 100.0
            cdi_daily = (1.0 + anual) ** (1.0/252.0) - 1.0
            for k, (ini, fim) in periods_dict.items():
                # usa dias úteis Brasil genérico (numpy.busday_count em calendário padrão)
                bd = int(np.busday_count(ini.date(), fim.date())) if hasattr(ini, "date") else 0
                if bd <= 0:
                    # Para 1d (pode cair no mesmo dia), cai no retorno diário
                    returns[k] = cdi_daily
                else:
                    returns[k] = (1.0 + cdi_daily) ** bd - 1.0

        elif benchmark_name == "IMA-B (IMAB11)":
            if not _HAS_YF:
                return returns
            t = yf.Ticker("IMAB11.SA")
            for k, (ini, fim) in periods_dict.items():
                hist = t.history(start=ini.date(), end=(fim + pd.Timedelta(days=1)).date())
                if len(hist) >= 2:
                    start_price = float(hist['Close'].iloc[0])
                    end_price = float(hist['Close'].iloc[-1])
                    returns[k] = (end_price / start_price) - 1.0 if start_price else 0.0
                else:
                    returns[k] = 0.0

        elif benchmark_name == "S&P 500":
            if not _HAS_YF:
                return returns
            t = yf.Ticker("^GSPC")
            for k, (ini, fim) in periods_dict.items():
                hist = t.history(start=ini.date(), end=(fim + pd.Timedelta(days=1)).date())
                if len(hist) >= 2:
                    start_price = float(hist['Close'].iloc[0])
                    end_price = float(hist['Close'].iloc[-1])
                    returns[k] = (end_price / start_price) - 1.0 if start_price else 0.0
                else:
                    returns[k] = 0.0

        elif benchmark_name == "USD/BRL":
            if not _HAS_YF:
                return returns
            t = yf.Ticker("USDBRL=X")
            for k, (ini, fim) in periods_dict.items():
                hist = t.history(start=ini.date(), end=(fim + pd.Timedelta(days=1)).date())
                if len(hist) >= 2:
                    start_price = float(hist['Close'].iloc[0])
                    end_price = float(hist['Close'].iloc[-1])
                    returns[k] = (end_price / start_price) - 1.0 if start_price else 0.0
                else:
                    returns[k] = 0.0

    except Exception as e:
        st.error(f"Erro ao calcular {benchmark_name}: {e}")

    return returns


# --------------------------
# Tabela/Gráfico comparativos
# --------------------------
_JANELAS = ["1d","1w","1m","3m","6m","12m","18m","24m"]

def _create_comparison_dataframe(df_perf: pd.DataFrame, selected_carteiras: List[str], selected_benchmarks: List[str]) -> pd.DataFrame:
    """Cria DataFrame comparativo entre carteiras (retornos móveis por Cota Líquida) e benchmarks."""
    periods = _periods(pd.Timestamp.now())
    rows = []

    # Carteiras
    for carteira in selected_carteiras:
        roll = _compute_portfolio_rolling_returns(df_perf, carteira, periods)
        item = {"Nome": carteira, "Tipo": "Carteira"}
        for w in _JANELAS:
            item[w] = float(roll.get(w, 0.0))
        rows.append(item)

    # Benchmarks
    for bmk in selected_benchmarks:
        br = _calculate_benchmark_returns(periods, bmk)
        item = {"Nome": bmk, "Tipo": "Benchmark"}
        for w in _JANELAS:
            item[w] = float(br.get(w, 0.0))
        rows.append(item)

    comparison_df = pd.DataFrame(rows)
    for w in _JANELAS:
        comparison_df[w] = pd.to_numeric(comparison_df[w], errors='coerce').fillna(0.0)
    return comparison_df

def _render_comparison_table(comparison_df: pd.DataFrame):
    """Renderiza tabela de comparação com cores."""
    if comparison_df.empty:
        return
    display_df = comparison_df.copy()
    for w in _JANELAS:
        display_df[w] = display_df[w].apply(_fmt_pct_color)
    display_df = display_df[["Nome", "Tipo"] + _JANELAS]

    st.markdown("### 📊 Comparação de Performance (Janelas Móveis)")
    st.markdown("""
    <style>
    .dataframe td { text-align: center !important; }
    .dataframe th { text-align: center !important; }
    </style>
    """, unsafe_allow_html=True)
    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

def _render_performance_chart(comparison_df: pd.DataFrame):
    """Gráfico de barras agrupadas por janela."""
    if comparison_df.empty:
        return

    chart_df = comparison_df.set_index("Nome")
    fig = go.Figure()
    colors = px.colors.qualitative.Set3
    for i, w in enumerate(_JANELAS):
        fig.add_trace(go.Bar(
            name=w,
            x=chart_df.index,
            y=(chart_df[w] * 100.0),
            text=chart_df[w].apply(lambda x: f"{x*100:.1f}%"),
            textposition='auto',
            marker_color=colors[i % len(colors)]
        ))
    fig.update_layout(
        title="Performance Comparada por Janelas Móveis",
        xaxis_title="",
        yaxis_title="Retorno (%)",
        barmode='group',
        height=520,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def _render_detailed_comparison(df_perf: pd.DataFrame, selected_carteiras: List[str], selected_benchmarks: List[str]):
    """Visão detalhada: carteiras vs benchmarks (exemplo destaca 12m)."""
    if not selected_carteiras or not selected_benchmarks:
        return
    st.markdown("### 🎯 Análise Detalhada por Carteira")

    periods = _periods(pd.Timestamp.now())

    for carteira in selected_carteiras:
        sub = df_perf[df_perf["Carteira"] == carteira].copy()
        if sub.empty:
            continue

        # KPIs móveis
        kpis = _compute_portfolio_rolling_returns(df_perf, carteira, periods)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f'<div class="card"><h4>📈 {carteira}</h4>', unsafe_allow_html=True)
            metrics_html = f"""
            <div class="stats-grid">
                <div><strong>1d:</strong><br>{_fmt_pct_color(kpis.get('1d',0))}</div>
                <div><strong>1w:</strong><br>{_fmt_pct_color(kpis.get('1w',0))}</div>
                <div><strong>1m:</strong><br>{_fmt_pct_color(kpis.get('1m',0))}</div>
                <div><strong>3m:</strong><br>{_fmt_pct_color(kpis.get('3m',0))}</div>
                <div><strong>6m:</strong><br>{_fmt_pct_color(kpis.get('6m',0))}</div>
                <div><strong>12m:</strong><br>{_fmt_pct_color(kpis.get('12m',0))}</div>
                <div><strong>18m:</strong><br>{_fmt_pct_color(kpis.get('18m',0))}</div>
                <div><strong>24m:</strong><br>{_fmt_pct_color(kpis.get('24m',0))}</div>
            </div>
            """
            st.markdown(metrics_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card"><h4>📊 Vs Benchmarks (12m)</h4>', unsafe_allow_html=True)
            carteira_12m = float(kpis.get("12m", 0.0))
            for bmk in selected_benchmarks:
                br = _calculate_benchmark_returns(periods, bmk)
                diff_12m = carteira_12m - float(br.get("12m", 0.0))
                st.markdown(f"**{bmk}**: {_fmt_diff_color(diff_12m)}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


# --------------------------
# Tela – Performance
# --------------------------
def tela_performance() -> None:
    if "headers" not in st.session_state or not st.session_state.headers:
        st.warning("Faça login para consultar os dados.")
        return

    st.markdown("### 📈 Performance de Carteiras")

    # Indicadores (cards rápidos)
    _display_indicators()
    st.markdown("---")

    # Filtros
    with st.container():
        c_f1, c_f2, c_f3, c_f4 = st.columns([1.1, 2, 1, 1])
        with c_f1:
            st.markdown('<div class="h-label">Janela de Consulta (dados brutos)</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="h-label">Carteiras</div>', unsafe_allow_html=True)
            carteiras_nomes = st.multiselect(
                "",
                sorted(CARTEIRAS.values()),
                default=[],
                key="perf_carteiras",
                label_visibility="collapsed"
            )
        with c_f3:
            st.markdown('<div class="h-label">Ações</div>', unsafe_allow_html=True)
            carregar = st.button("Carregar", key="perf_btn_carregar", use_container_width=True)
        with c_f4:
            st.markdown('<div class="h-label">Benchmarks</div>', unsafe_allow_html=True)
            bmarks = st.multiselect(
                "",
                ["CDI", "IMA-B (IMAB11)", "USD/BRL", "S&P 500"],
                default=["CDI", "IMA-B (IMAB11)"],
                key="perf_bmarks",
                label_visibility="collapsed"
            )

    # Mapeia nomes -> ids
    carteiras_ids = [k for k, v in CARTEIRAS.items() if v in carteiras_nomes]

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
            st.success("Dados carregados com sucesso!")
        except Exception as e:
            st.error(f"Erro ao buscar dados: {e}")
            return

    # Comparação
    if "df_perf" in st.session_state and not st.session_state.df_perf.empty:
        df = st.session_state.df_perf

        # Comparativo (carteiras + benchmarks) nas janelas móveis pedidas
        comparison_df = _create_comparison_dataframe(
            df,
            st.session_state.get("selected_carteiras", []),
            st.session_state.get("selected_benchmarks", [])
        )

        if not comparison_df.empty:
            _render_comparison_table(comparison_df)
            st.markdown("---")
            _render_performance_chart(comparison_df)
            st.markdown("---")
            _render_detailed_comparison(
                df,
                st.session_state.get("selected_carteiras", []),
                st.session_state.get("selected_benchmarks", [])
            )
            st.markdown("---")

            # Dados brutos opcionais
            with st.expander("📋 Visualizar Dados Brutos"):
                display_df = df[['Carteira','Data','Cota Líquida','%Dia','%Mês','%Ano','%12 Meses','%18 Meses','%24 Meses']].copy()
                for col in ['%Dia','%Mês','%Ano','%12 Meses','%18 Meses','%24 Meses']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{float(x)*100:.2f}%" if pd.notna(x) else "")
                st.dataframe(display_df.sort_values(["Carteira","Data"]))
    else:
        st.info("👆 Selecione as carteiras e benchmarks, depois clique em 'Carregar' para ver a análise comparativa.")
