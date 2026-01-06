def rename (df):
    df.rename(columns={
                    "portfolio_name": "Portfolio",
                    "portfolio_id":"ID Carteira",
                    "compliance_message":"Descrição",
                    "compliance_summary":"Posição",
                    "created_at":"Data de Criação",
                    "id":"ID",
                    "reference_date":"Data",
                    "rule_id":"Regra ID",
                    "rule_name":"Regra",
                    "status":"Status de Compliance",
                    "updated_at":"Última Atualização"

                }, inplace=True)
    
    return df


mapa_compliance = {

 "portfolio_name": "Portfolio",
 "portfolio_id":"ID Carteira",
 "compliance_message":"Descrição",
 "compliance_summary":"Posição",
 "created_at":"Data de Criação",
 "id":"ID",
 "reference_date":"Data",
 "rule_id":"Regra ID",
 "rule_name":"Regra",
 "status":"Status de Compliance",
 "updated_at":"Última Atualização"
}
  
