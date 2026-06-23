"""
Gera um dataset sintético de clientes de e-commerce inspirado na estrutura
do Brazilian E-Commerce Public Dataset (Olist), com um rótulo de CHURN.

O churn é definido como: cliente que NÃO realizou nova compra nos últimos
6 meses. A probabilidade de churn é construída a partir de fatores de negócio
realistas (satisfação, atraso de entrega, recência e frequência de compra),
de forma que o problema tenha sinal aprendível — como nos dados reais.

Para usar dados reais: substitua o CSV gerado por features extraídas do
dataset da Olist (Kaggle) seguindo o mesmo esquema de colunas.
"""
import numpy as np
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(42)
N = 6000

categorias = ["cama_mesa_banho", "beleza_saude", "informatica", "moveis_decoracao",
              "esporte_lazer", "brinquedos", "relogios_presentes", "telefonia"]
pagamentos = ["cartao_credito", "boleto", "pix", "cartao_debito"]

# Features comportamentais (RFM + experiência)
recencia_dias = rng.integers(5, 400, N)                      # dias desde a última compra
frequencia = rng.poisson(1.6, N) + 1                         # nº de pedidos
valor_total = np.round(rng.gamma(2.2, 90, N) + 20, 2)        # gasto total (R$)
ticket_medio = np.round(valor_total / frequencia, 2)
tempo_entrega = np.clip(rng.normal(12, 6, N), 2, 60).round(1)  # dias médios de entrega
atraso_medio = np.clip(rng.normal(0.5, 2.5, N), 0, 25).round(1)  # dias de atraso médio
nota_media = np.clip(rng.normal(4.1, 1.0, N), 1, 5).round(1)   # review médio (1-5)
frete_pct = np.clip(rng.normal(15, 6, N), 2, 45).round(1)      # frete sobre valor (%)
parcelas = rng.integers(1, 11, N)
categoria = rng.choice(categorias, N)
pagamento = rng.choice(pagamentos, N, p=[0.55, 0.20, 0.18, 0.07])

# Score latente de propensão ao churn (quanto maior, mais provável sair)
score = (
    0.010 * recencia_dias
    - 0.45 * frequencia
    - 0.55 * nota_media
    + 0.08 * atraso_medio
    + 0.02 * tempo_entrega
    + 0.015 * frete_pct
    - 0.0015 * valor_total
    + rng.normal(0, 0.8, N)
)
prob = 1 / (1 + np.exp(-(score - score.mean()) / score.std()))
churn = (prob > rng.uniform(0.35, 0.65, N)).astype(int)

df = pd.DataFrame({
    "recencia_dias": recencia_dias,
    "frequencia_pedidos": frequencia,
    "valor_total": valor_total,
    "ticket_medio": ticket_medio,
    "tempo_entrega_dias": tempo_entrega,
    "atraso_medio_dias": atraso_medio,
    "nota_media_review": nota_media,
    "frete_pct": frete_pct,
    "parcelas": parcelas,
    "categoria_principal": categoria,
    "metodo_pagamento": pagamento,
    "churn": churn,
})

out = os.path.join(BASE, "dados", "clientes_churn.csv")
df.to_csv(out, index=False)
print(f"Dataset gerado: {out}  ({len(df)} clientes)")
print(f"Taxa de churn: {df['churn'].mean()*100:.1f}%")
