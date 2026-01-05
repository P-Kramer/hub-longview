def checar_divergencias(df_at, df_cd):
    import pandas as pd
    import numpy as np
    from rapidfuzz import fuzz, process
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill
    from openpyxl.utils.dataframe import dataframe_to_rows
    import re
    from io import BytesIO
    from openpyxl.styles import Font, PatternFill, Alignment, numbers
    from openpyxl.utils import get_column_letter

    # ---------- helper p/ Excel ----------
    def _sanitize_df_excel(df: pd.DataFrame) -> pd.DataFrame:
        """Troca <NA>/NaN/NaT por None e garante tipos aceitos pelo openpyxl."""
        df = df.copy()
        # boolean/int "nullable" -> object p/ aceitar None
        for col in df.select_dtypes(include=["boolean",
                                             "Int8","Int16","Int32","Int64",
                                             "UInt8","UInt16","UInt32","UInt64"]).columns:
            df[col] = df[col].astype(object)
        # timestamps -> datetime nativo
        for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
            df[col] = pd.to_datetime(df[col]).dt.to_pydatetime()
        # faltantes -> None
        df = df.astype(object).where(pd.notna(df), None)
        return df
    # -------------------------------------

    # Configuração de cor
    YELLOW_FILL = PatternFill(start_color='FFFFF000', end_color='FFFFF000', fill_type='solid')
    HEADER_FILL = PatternFill(start_color='FFDDDDDD', end_color='FFDDDDDD', fill_type='solid')
    HEADER_FONT = Font(bold=True)
    CENTER_ALIGN = Alignment(horizontal='center', vertical='center')

    pareamentos_forcados = {
        "AMERCO /NV/":"UHAL'B",
        "POLEN CAPITAL FOCUS US GR USD INSTL":"PFUGI",
        "JPM US SELECT EQUITY PLUS C (ACC) USD":"SELQZ",
        "AMUNDI FDS US EQ FUNDM GR I2 USD C":"PONVS",
        "MS INVF GLOBAL ENDURANCE I USD ACC":"SMFVZ",
        "PRINCIPAL PREFERRED SECS N INC USD": "PRGPZ",
        "PIMCO GIS INCOME H INSTL USD INC": "PCOAZ"
    }

    def formatar_aba(ws, colunas_monetarias=None, colunas_percentuais=None):
        colunas_monetarias = colunas_monetarias or []
        colunas_percentuais = colunas_percentuais or []
        for cell in ws[1]:
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = CENTER_ALIGN
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                if isinstance(cell.value, (int, float)):
                    cell.alignment = CENTER_ALIGN
        for col in colunas_monetarias:
            for row in range(2, ws.max_row + 1):
                ws.cell(row=row, column=col).number_format = 'R$ #,##0.00'
        for col in colunas_percentuais:
            for row in range(2, ws.max_row + 1):
                ws.cell(row=row, column=col).number_format = '0.00%'
        for col in ws.columns:
            max_length = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass
            ws.column_dimensions[col_letter].width = max_length + 2

    def extrair_cusip(texto):
        match = re.search(r'US([A-Z0-9]{9})', str(texto).upper())
        return match.group(1) if match else None

    def processar_com_dinheiro(df):
        df = df[
            (df['Classe'].isin(['EQUITY', "FIXED INCOME","FLOATING INCOME"]))
        ].copy()
        df = df.rename(columns={'Ativo':'Ticker','Quant.':'Quantidade','Saldo Bruto':'MarketValue'})
        df['TickerBase'] = df['ticker_cmd_puro'].str.split(':').str[-1].str.strip()
        return df[['Descrição','Ticker','TickerBase','Quantidade','MarketValue','Classe']]

    def processar_ativos(df):
        df = df.rename(columns={'Ativo':'Nome','Quantidade Total':'Quantidade','Market Value':'MarketValue'})
        df['TickerBase'] = df['Ticker'].str.extract(r'([A-Z]{2,6}$)')[0].fillna(df['Ticker'])
        return df[['Nome','Ticker','TickerBase','Quantidade','MarketValue','CUSIP']]

    equity_cd = processar_com_dinheiro(df_cd)
    equity_at = processar_ativos(df_at)

    pareamentos_cusip_descricao_forcados = {"J7596PAJ8": "SOFTBANK GROUP 17/UND. 6,875%"}
    forced_cusip_desc_matches = []
    for _, row_at in equity_at.iterrows():
        cusip_at = row_at['CUSIP']
        if pd.notna(cusip_at) and cusip_at in pareamentos_cusip_descricao_forcados:
            descricao_alvo = pareamentos_cusip_descricao_forcados[cusip_at]
            match_cd = equity_cd[equity_cd['Descrição'].str.contains(descricao_alvo, case=False, na=False)]
            if not match_cd.empty:
                row_cd = match_cd.iloc[0]
                forced_cusip_desc_matches.append({
                    'Descrição_CD': row_cd['Descrição'],'Ticker_CD': row_cd['Ticker'],'TickerBase': row_cd['TickerBase'],
                    'Classe': row_cd["Classe"],'Quantidade_CD': row_cd['Quantidade'],'MarketValue_CD': row_cd['MarketValue'],
                    'Nome_MS': row_at['Nome'],'Ticker_MS': row_at['Ticker'],'Quantidade_MS': row_at['Quantidade'],
                    'CUSIP_MS': row_at['CUSIP'],'MarketValue_MS': row_at['MarketValue'],'Similaridade': 100
                })
                equity_cd = equity_cd.drop(row_cd.name)
                equity_at = equity_at.drop(row_at.name)
    forced_cusip_desc_df = pd.DataFrame(forced_cusip_desc_matches)

    equity_cd['CUSIP_EXTRAIDO'] = equity_cd['Ticker'].apply(extrair_cusip)
    equity_at['CUSIP'].replace(['', ' ', ' '], pd.NA, inplace=True)
    cd_com_cusip = equity_cd.dropna(subset=['CUSIP_EXTRAIDO'])
    at_com_cusip = equity_at.dropna(subset=['CUSIP'])
    cd_sem_cusip = equity_cd[equity_cd['CUSIP_EXTRAIDO'].isna()]
    at_sem_cusip = equity_at[equity_at['CUSIP'].isna()]

    cusip_matches = []
    for _, row_cd in cd_com_cusip.iterrows():
        for _, row_at in at_com_cusip.iterrows():
            if row_at['CUSIP'] in row_cd['Ticker']:
                cusip_matches.append({
                    'Descrição_CD': row_cd['Descrição'],'Ticker_CD': row_cd['Ticker'],'TickerBase': row_cd['TickerBase'],
                    'Classe': row_cd["Classe"],'Quantidade_CD': row_cd['Quantidade'],'MarketValue_CD': row_cd['MarketValue'],
                    'Nome_MS': row_at['Nome'],'Ticker_MS': row_at['Ticker'],'Quantidade_MS': row_at['Quantidade'],
                    'CUSIP_MS': row_at['CUSIP'],'MarketValue_MS': row_at['MarketValue'],'Similaridade': 100
                })
                break
    cusip_df = pd.DataFrame(cusip_matches)

    equity_cd = cd_sem_cusip
    equity_at = at_sem_cusip

    exact_match = pd.merge(equity_cd, equity_at, on='TickerBase', suffixes=('_CD','_MS'), how='inner')
    exact_matches_list = [{
        'Descrição_CD': r['Descrição'],'Ticker_CD': r['Ticker_CD'],'TickerBase': r['TickerBase'],'Classe': r['Classe'],
        'Quantidade_CD': r['Quantidade_CD'],'MarketValue_CD': r['MarketValue_CD'],
        'Nome_MS': r['Nome'],'Ticker_MS': r['Ticker_MS'],'Quantidade_MS': r['Quantidade_MS'],
        'CUSIP_MS': r['CUSIP'],'MarketValue_MS': r['MarketValue_MS'],'Similaridade': 100
    } for _, r in exact_match.iterrows()]

    matched_tickers = exact_match['TickerBase'].unique()
    remaining_cd = equity_cd[~equity_cd['TickerBase'].isin(matched_tickers)]
    remaining_at = equity_at[~equity_at['TickerBase'].isin(matched_tickers)]

    forcados = []
    for _, row in remaining_cd.iterrows():
        descricao_upper = row['Descrição'].strip().upper()
        if descricao_upper in {"AMERCO /NV/","POLEN CAPITAL FOCUS US GR USD INSTL","JPM US SELECT EQUITY PLUS C (ACC) USD",
                               "AMUNDI FDS US EQ FUNDM GR I2 USD C","MS INVF GLOBAL ENDURANCE I USD ACC",
                               "PRINCIPAL PREFERRED SECS N INC USD","PIMCO GIS INCOME H INSTL USD INC"}:
            ticker_alvo = {"AMERCO /NV/":"UHAL'B","POLEN CAPITAL FOCUS US GR USD INSTL":"PFUGI",
                           "JPM US SELECT EQUITY PLUS C (ACC) USD":"SELQZ","AMUNDI FDS US EQ FUNDM GR I2 USD C":"PONVS",
                           "MS INVF GLOBAL ENDURANCE I USD ACC":"SMFVZ","PRINCIPAL PREFERRED SECS N INC USD":"PRGPZ",
                           "PIMCO GIS INCOME H INSTL USD INC":"PCOAZ"}[descricao_upper]
            match_row = remaining_at[remaining_at['Ticker'] == ticker_alvo]
            if not match_row.empty:
                m = match_row.iloc[0]
                forcados.append({
                    'Descrição_CD': row['Descrição'],'Ticker_CD': row['Ticker'],'TickerBase': row['TickerBase'],
                    'Classe': row["Classe"],'Quantidade_CD': row['Quantidade'],'MarketValue_CD': row['MarketValue'],
                    'Nome_MS': m['Nome'],'Ticker_MS': m['Ticker'],'Quantidade_MS': m['Quantidade'],
                    'CUSIP_MS': m['CUSIP'],'MarketValue_MS': m['MarketValue'],'Similaridade': 100
                })

    fuzzy_matches = []
    for _, row in remaining_cd.iterrows():
        match, score, _ = process.extractOne(row['Descrição'], remaining_at['Nome'].tolist(), scorer=fuzz.token_set_ratio)
        if score >= 85:
            m = remaining_at[remaining_at['Nome'] == match].iloc[0]
            fuzzy_matches.append({
                'Descrição_CD': row['Descrição'],'Ticker_CD': row['Ticker'],'TickerBase': row['TickerBase'],
                'Classe': row["Classe"],'Quantidade_CD': row['Quantidade'],'MarketValue_CD': row['MarketValue'],
                'Nome_MS': m['Nome'],'Ticker_MS': m['Ticker'],'Quantidade_MS': m['Quantidade'],
                'CUSIP_MS': m['CUSIP'],'MarketValue_MS': m['MarketValue'],'Similaridade': score
            })

    all_matches = pd.concat([
        pd.DataFrame(forced_cusip_desc_matches),
        cusip_df,
        pd.DataFrame(exact_matches_list),
        pd.DataFrame(forcados + fuzzy_matches),
    ], ignore_index=True)

    matched_bases = all_matches['TickerBase'].unique() if not all_matches.empty else []
    extra_cd = equity_cd[~equity_cd['TickerBase'].isin(matched_bases)]
    extra_at = equity_at[~equity_at['TickerBase'].isin(matched_bases)]

    ticker_matches = []
    for _, row_cd in extra_cd.iterrows():
        mr = extra_at[extra_at['Ticker'] == row_cd['TickerBase']]
        if not mr.empty:
            m = mr.iloc[0]
            ticker_matches.append({
                'Descrição_CD': row_cd['Descrição'],'Ticker_CD': row_cd['Ticker'],'TickerBase': row_cd['TickerBase'],
                'Classe': row_cd["Classe"],'Quantidade_CD': row_cd['Quantidade'],'MarketValue_CD': row_cd['MarketValue'],
                'Nome_MS': m['Nome'],'Ticker_MS': m['Ticker'],'Quantidade_MS': m['Quantidade'],
                'CUSIP_MS': m['CUSIP'],'MarketValue_MS': m['MarketValue'],'Similaridade': 100
            })

    cd_restante = extra_cd[~extra_cd['TickerBase'].isin([m['TickerBase'] for m in ticker_matches])]
    at_restante = extra_at[~extra_at['TickerBase'].isin([m['TickerBase'] for m in ticker_matches])]

    palavra_matches = []
    for _, row_cd in cd_restante.iterrows():
        palavra_cd = str(row_cd['Descrição']).strip().split()[0].upper()
        candidatos = at_restante['Nome'].tolist()
        melhores = process.extract(palavra_cd, candidatos, scorer=fuzz.token_sort_ratio, limit=1)
        if melhores:
            match, score, _ = melhores[0]
            if score >= 80:
                m = at_restante[at_restante['Nome'] == match].iloc[0]
                palavra_matches.append({
                    'Descrição_CD': row_cd['Descrição'],'Ticker_CD': row_cd['Ticker'],'TickerBase': row_cd['TickerBase'],
                    'Classe': row_cd["Classe"],'Quantidade_CD': row_cd['Quantidade'],'MarketValue_CD': row_cd['MarketValue'],
                    'Nome_MS': m['Nome'],'Ticker_MS': m['Ticker'],'Quantidade_MS': m['Quantidade'],
                    'CUSIP_MS': m['CUSIP'],'MarketValue_MS': m['MarketValue'],'Similaridade': score
                })

    extra_df = pd.DataFrame(ticker_matches + palavra_matches)
    all_matches = pd.concat([all_matches, extra_df], ignore_index=True)

    if not all_matches.empty:
        all_matches['PrecoUnitario_CD'] = all_matches['MarketValue_CD'] / all_matches['Quantidade_CD']
        all_matches['PrecoUnitario_MS'] = all_matches['MarketValue_MS'] / all_matches['Quantidade_MS']
        all_matches['Diff_Quantidade'] = all_matches['Quantidade_CD'] - all_matches['Quantidade_MS']
        all_matches['Diff_MarketValue'] = all_matches['MarketValue_CD'] - all_matches['MarketValue_MS']
        all_matches['Diff_PrecoUnitario'] = all_matches['PrecoUnitario_CD'] - all_matches['PrecoUnitario_MS']
        all_matches['Pct_Diff_Quantidade'] = all_matches['Diff_Quantidade'] / all_matches['Quantidade_MS']
        all_matches['Pct_Diff_MarketValue'] = all_matches['Diff_MarketValue'] / all_matches['MarketValue_MS']
        all_matches['Pct_Diff_PrecoUnitario'] = all_matches['Diff_PrecoUnitario'] / all_matches['PrecoUnitario_MS']
        all_matches['Destaque'] = np.abs(all_matches['Diff_PrecoUnitario']) > 1

    tickersbase_usados_cd = all_matches['TickerBase'].unique() if not all_matches.empty else []
    tickers_usados_at = all_matches['Ticker_MS'].unique() if not all_matches.empty else []

    non_matched_cd = equity_cd[~equity_cd['TickerBase'].isin(tickersbase_usados_cd)]
    non_matched_at = equity_at[~equity_at['Ticker'].isin(tickers_usados_at)]

    non_matched_consolidado = pd.concat(
        [non_matched_cd.assign(Origem='COM DINHEIRO'),
         non_matched_at.assign(Origem='ATIVOS')],
        ignore_index=True
    )

    col_order = [
        'TickerBase','Ticker_MS','Ticker_CD','CUSIP_MS','Descrição_CD','Nome_MS',
        'Quantidade_CD','Quantidade_MS','Diff_Quantidade',
        'MarketValue_CD','MarketValue_MS','Diff_MarketValue','Pct_Diff_MarketValue','Classe'
    ]
    if not (all_matches.empty):
        all_matches = all_matches[col_order]

    # --------- SANITIZAÇÃO PARA EXCEL (resolve o <NA>) ---------
    all_matches = _sanitize_df_excel(all_matches) if not all_matches.empty else all_matches
    non_matched_consolidado = _sanitize_df_excel(non_matched_consolidado)
    non_matched_cd = _sanitize_df_excel(non_matched_cd)
    non_matched_at = _sanitize_df_excel(non_matched_at)
    # -----------------------------------------------------------

    wb = Workbook(); wb.remove(wb.active)

    ws_pareados = wb.create_sheet("Pareados")
    for r in dataframe_to_rows(all_matches, index=False, header=True):
        ws_pareados.append(r)

    ws_nao_pareados = wb.create_sheet("Não Pareados")
    for r in dataframe_to_rows(non_matched_consolidado, index=False, header=True):
        ws_nao_pareados.append(r)

    ws_so_cd = wb.create_sheet("Só COM DINHEIRO")
    for r in dataframe_to_rows(non_matched_cd, index=False, header=True):
        ws_so_cd.append(r)

    ws_so_at = wb.create_sheet("Só ATIVOS")
    for r in dataframe_to_rows(non_matched_at, index=False, header=True):
        ws_so_at.append(r)

    if not all_matches.empty:
        diff_mv_col = col_order.index('Diff_MarketValue') + 1
        pct_diff_mv_col = all_matches.columns.get_loc('Pct_Diff_MarketValue') + 1
        classe_col = all_matches.columns.get_loc('Classe') + 1
        for idx, row in enumerate(ws_pareados.iter_rows(min_row=2, max_row=ws_pareados.max_row), start=2):
            diff_mv = ws_pareados.cell(row=idx, column=diff_mv_col).value
            pct_diff_mv = ws_pareados.cell(row=idx, column=pct_diff_mv_col).value
            classe = ws_pareados.cell(row=idx, column=classe_col).value
            if classe == "EQUITY":
                destacar = diff_mv is not None and abs(diff_mv) > 1
            else:
                destacar = pct_diff_mv is not None and abs(pct_diff_mv) > 0.01
            if destacar:
                for cell in row:
                    cell.fill = YELLOW_FILL

    col_monetarias = [col_order.index(c) + 1 for c in ['MarketValue_CD','MarketValue_MS','Diff_MarketValue']]
    col_percentuais = [col_order.index('Pct_Diff_MarketValue') + 1]
    if not all_matches.empty:
        formatar_aba(ws_pareados, colunas_monetarias=col_monetarias, colunas_percentuais=col_percentuais)
    formatar_aba(ws_nao_pareados)
    formatar_aba(ws_so_cd)
    formatar_aba(ws_so_at)

    buffer = BytesIO()
    wb.save(buffer); buffer.seek(0)
    return all_matches, buffer
