import pandas as pd
import random

# Padrão de faixa etária do Brasil
faixas_etarias = [0, 10, 12, 14, 16, 18]
#sendo
# 0 - LIVRE PARA TODOS OS PUBLICOS,
# 10- MAIORES DE 10 ANOS,
# 12- MAIORES DE 12 ANOS,
# 14- MAIORES DE 14 ANOS,
# 16- MAIORES DE 16 ANOS,
# 18- MAIORES DE 18 ANOS

canais_marketing = {
    'Instagram': {'custo_base': 2, 'alcance_medio': 0.15},
    'Google Ads': {'custo_base': 3, 'alcance_medio': 0.20},
    'Twitter': {'custo_base': 1.5, 'alcance_medio': 0.12},
    'Facebook': {'custo_base': 2.5, 'alcance_medio': 0.18},
    'LinkedIn': {'custo_base': 4, 'alcance_medio': 0.10},
    'YouTube': {'custo_base': 3.5, 'alcance_medio': 0.22}
}

def gerar_dados_realistas(n):
    dados = []
    for _ in range(n):
        nome = f"Campanha_{random.randint(1, 10000)}"
        titulo = f"Filme_{random.randint(1, 10000)}"
        faixa_etaria = random.choice(faixas_etarias)

        # Determinando o canal de marketing
        canal_marketing = random.choice(list(canais_marketing.keys()))
        custo_base = canais_marketing[canal_marketing]['custo_base']
        alcance_medio = canais_marketing[canal_marketing]['alcance_medio']

        # Definindo o orçamento com base no canal
        budget_total = round(random.uniform(5000, 50000) * custo_base, 2)

        # Expectativa de alcance relacionada ao orçamento e ao canal
        expectativa_alcance = int(budget_total * alcance_medio * random.uniform(0.8, 1.2))

        # ROI inversamente relacionado ao custo base, ajustado pelo orçamento
        roi = round((random.uniform(0.5, 3) / custo_base) * budget_total, 2)

        # CPC ajustado pelo canal e faixa etária
        cpc = round(custo_base * random.uniform(0.8, 1.5), 2)

        # Número de conversões baseado no alcance e uma taxa de conversão realista
        taxa_conversao = random.uniform(0.01, 0.2)
        numero_conversoes = int(expectativa_alcance * taxa_conversao)

        dados.append([
            nome, titulo, faixa_etaria, budget_total,
            expectativa_alcance, canal_marketing, roi, cpc, numero_conversoes
        ])

    return pd.DataFrame(dados, columns=[
        'nome', 'titulo', 'faixa_etaria', 'budget_total',
        'expectativa_alcance', 'canal_marketing', 'roi', 'cpc', 'numero_conversoes'
    ])
