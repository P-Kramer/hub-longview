from datetime import date
import io
import requests
import streamlit as st
import pandas as pd
from utils import BASE_URL_API, CARTEIRAS

# ===== Configs =====
PIZZA_LIMIAR_OUTROS = 0.02  # classes com <2% vão para "Outros"


def _consolidar_outros(agg: pd.DataFrame, limiar: float = PIZZA_LIMIAR_OUTROS) -> pd.DataFrame:
    """Agrupa classes pequenas em 'Outros' com base no percentual."""
    if "pct" not in agg.columns:
        total = agg["asset_value"].sum()
        agg = agg.assign(pct=agg["asset_value"] / total if total else 0)

    grandes = agg[agg["pct"] >= limiar].copy()
    pequenos = agg[agg["pct"] < limiar].copy()

    if pequenos.empty:
        return grandes.sort_values("asset_value", ascending=False).reset_index(drop=True)

    outros = pd.DataFrame({
        "book_name": ["Outros"],
        "asset_value": [pequenos["asset_value"].sum()],
        "pct": [pequenos["asset_value"].sum() / agg["asset_value"].sum() if agg["asset_value"].sum() else 0]
    })

    res = pd.concat([grandes, outros], ignore_index=True)
    return res.sort_values("asset_value", ascending=False).reset_index(drop=True)


def _fig_pizza(agg: pd.DataFrame):
    """Donut chart com percentuais em NEGRITO e hover em pt-BR."""
    import plotly.graph_objects as go

    labels = agg["book_name"].tolist()
    values = agg["asset_value"].tolist()

    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            sort=False,
            direction="clockwise",
            # Exibir somente a % no rótulo, em NEGRITO via texttemplate
            texttemplate="<b>%{percent}</b>",
            textinfo="percent",
            textposition="inside",
            textfont=dict(size=14, family="Arial, DejaVu Sans, sans-serif"),
            insidetextfont=dict(size=14, family="Arial, DejaVu Sans, sans-serif"),
            hovertemplate="<b>%{label}</b><br>Financeiro: R$ %{value:,.2f}<br>% Alocado: %{percent}<extra></extra>",
            showlegend=True
        )]
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        height=600,
        uniformtext_minsize=12,
        uniformtext_mode="show",
    )
    return fig


def tela_alocacao():
    st.header("Alocação por Classe de Ativo")

    if "headers" not in st.session_state or not st.session_state.headers:
        st.warning("Faça login para consultar os dados.")
        return

    # ===== Filtros =====
    c1, c2, c3 = st.columns([1.1, 2, 0.8])
    with c1:
        data_base = st.date_input("Data-base", value=date.today())
    with c2:
        nome_carteira = st.selectbox("Carteira", options=sorted(CARTEIRAS.values()), index=0)
    with c3:
        consultar = st.button("Consultar", use_container_width=True)

    try:
        carteira_id = next(k for k, v in CARTEIRAS.items() if v == nome_carteira)
    except StopIteration:
        st.error("Carteira inválida.")
        return

    if not consultar:
        st.info("Selecione os filtros e clique em Consultar.")
        return

    # ===== Payload =====
    payload = {
        "start_date": str(data_base),
        "end_date": str(data_base),
        "instrument_position_aggregation": 3,
        "portfolio_ids": [carteira_id],
    }

    # ===== Chamada API =====
    try:
        with st.spinner("Buscando posições..."):
            resp = requests.post(
                f"{BASE_URL_API.rstrip('/')}/portfolio_position/positions/get",
                json=payload,
                headers=st.session_state.headers,
                timeout=60,
            )
            resp.raise_for_status()
            resultado = resp.json()

        objetos = resultado.get("objects", {})
        inst_positions_acumulado = []

        if isinstance(objetos, dict):
            iter_values = objetos.values()
        elif isinstance(objetos, list):
            iter_values = objetos
        else:
            iter_values = []

        for obj in iter_values:
            pos = obj.get("instrument_positions") if isinstance(obj, dict) else None
            if not pos:
                continue
            if isinstance(pos, list):
                inst_positions_acumulado.extend(pos)
            elif isinstance(pos, dict):
                inst_positions_acumulado.append(pos)

        if not inst_positions_acumulado:
            st.info("Nenhuma posição encontrada em objects.<ID>.instrument_positions para os filtros selecionados.")
            return

        df = pd.json_normalize(inst_positions_acumulado)

    except Exception as e:
        st.error(f"Erro ao buscar posições: {e}")
        return

    if df.empty:
        st.info("Nenhuma posição encontrada para os filtros selecionados.")
        return

    # ===== Garantir numérico =====
    df["asset_value"] = pd.to_numeric(df.get("asset_value", 0), errors="coerce").fillna(0)

    # ===== Alocação =====
    if {"book_name", "asset_value"}.issubset(df.columns):
        st.subheader("Alocação por Classe")

        agg = df.groupby("book_name", as_index=False)["asset_value"].sum()
        total = float(agg["asset_value"].sum())
        agg["pct"] = agg["asset_value"] / total

        # Tabela de resumo (formatada pt-BR para exibição)
        tabela_resumo = (
            agg.assign(
                Financeiro=lambda x: x["asset_value"].map(lambda v: f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")),
                pct=lambda x: (x["pct"] * 100).round(2).map(lambda v: str(v).replace(".", ","))
            )
            .rename(columns={"book_name": "Classe", "pct": "% Alocado"})
            .drop(columns=["asset_value"])
        )
        st.dataframe(tabela_resumo, use_container_width=True)

        # Donut chart (percentuais em NEGRITO)
        agg_pizza = _consolidar_outros(agg, limiar=PIZZA_LIMIAR_OUTROS)
        fig_pizza = _fig_pizza(agg_pizza)
        st.plotly_chart(fig_pizza, use_container_width=True)

        # ===== Drilldown =====
        st.subheader("Detalhamento por Classe")
        detalhados_frames = []  # para exportação
        for classe, grupo in df.groupby("book_name"):
            subtotal = float(grupo["asset_value"].sum())
            if subtotal == 0:
                continue

            detalhado_num = (
                grupo[["instrument_name", "asset_value"]]
                .rename(columns={"instrument_name": "Nome", "asset_value": "Financeiro"})
                .copy()
            )
            detalhado_num["% Alocado"] = (detalhado_num["Financeiro"] / subtotal * 100).round(2)

            # Guardar versão numérica para Excel
            detalhado_num.insert(0, "Classe", classe)
            detalhados_frames.append(detalhado_num)

            # Versão formatada pt-BR para tela
            detalhado_view = detalhado_num.copy()
            detalhado_view["Financeiro"] = detalhado_view["Financeiro"].map(
                lambda v: f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            )
            detalhado_view["% Alocado"] = detalhado_view["% Alocado"].map(lambda v: str(v).replace(".", ","))

            with st.expander(
                f"Classe: {classe} — Total: {subtotal:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            ):
                st.dataframe(detalhado_view, use_container_width=True)

        # ===== Botão: Baixar Excel (tudo) =====
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                # Resumo por classe (numérico)
                resumo_export = agg[["book_name", "asset_value", "pct"]].rename(
                    columns={"book_name": "Classe", "asset_value": "Financeiro", "pct": "% Alocado"}
                )
                resumo_export.to_excel(writer, index=False, sheet_name="Resumo_Classe")

                # Detalhe (numérico)
                if detalhados_frames:
                    detalhe_export = pd.concat(detalhados_frames, ignore_index=True)
                    detalhe_export.to_excel(writer, index=False, sheet_name="Detalhe")

                # Raw positions também podem ser úteis
                df.to_excel(writer, index=False, sheet_name="Raw_Positions")

                # Formatação leve
                workbook = writer.book
                pct_fmt = workbook.add_format({"num_format": "0.00%"})
                money_fmt = workbook.add_format({"num_format": "#,##0.00"})
                bold_hdr = workbook.add_format({"bold": True})

                # Ajustar Resumo_Classe
                ws_resumo = writer.sheets["Resumo_Classe"]
                ws_resumo.set_row(0, None, bold_hdr)
                # colunas: Classe (0), Financeiro (1), % Alocado (2)
                ws_resumo.set_column(0, 0, 28)
                ws_resumo.set_column(1, 1, 18, money_fmt)
                ws_resumo.set_column(2, 2, 12, pct_fmt)

                # Ajustar Detalhe, se houver
                if "Detalhe" in writer.sheets:
                    ws_det = writer.sheets["Detalhe"]
                    ws_det.set_row(0, None, bold_hdr)
                    # Classe, Nome, Financeiro, % Alocado
                    ws_det.set_column(0, 0, 22)
                    ws_det.set_column(1, 1, 42)
                    ws_det.set_column(2, 2, 18, money_fmt)
                    ws_det.set_column(3, 3, 12, pct_fmt)


            st.download_button(
                label="⬇️ Baixar Excel (Resumo + Detalhe + Raw)",
                data=buffer.getvalue(),
                file_name=f"{nome_carteira}_alocacao_{str(data_base)}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Falha ao gerar Excel: {e}")

    else:
        st.info("Colunas esperadas (book_name, asset_value) não encontradas.")
