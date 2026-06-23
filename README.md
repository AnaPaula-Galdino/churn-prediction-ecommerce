# Previsão de Churn em E-commerce

Modelo de Machine Learning que estima quais clientes de uma loja online estão prestes a
parar de comprar (churn). A ideia é simples: identificar quem está em risco com antecedência
para dar tempo de agir na retenção, em vez de descobrir a perda depois que ela já aconteceu.

O projeto usa uma base no estilo do dataset da Olist e conversa direto com o meu
[Tech Challenge da pós](https://github.com/AnaPaula-Galdino/tech-challenge-olist-analytics).

**[Ler a análise executiva (PDF)](Analise_Executiva_Churn.pdf)**

## O problema

Conquistar cliente novo custa caro; segurar um que já compra custa bem menos. O desafio é
saber quem vai embora antes que vá. Aqui defini churn como cliente sem nova compra nos
últimos 6 meses, e treinei um modelo para reconhecer os sinais que antecedem essa saída.

## Como construí

1. **Features de comportamento (RFM + experiência):** recência, frequência, valor gasto,
   ticket médio, tempo e atraso de entrega, nota média de avaliação, peso do frete,
   parcelas, categoria e forma de pagamento.
2. **Pré-processamento:** padronização das numéricas e one-hot nas categóricas.
3. **Modelos:** comparei Regressão Logística e Random Forest.
4. **Avaliação:** accuracy, precisão, recall, F1 e ROC AUC, com matriz de confusão e curva ROC.

## Resultado

| Modelo | Acurácia | Precisão | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Regressão Logística | 0,821 | 0,817 | 0,821 | 0,819 | **0,909** |
| Random Forest | 0,810 | 0,803 | 0,815 | 0,809 | 0,902 |

O modelo acerta a saída do cliente em cerca de 82% dos casos, com ROC AUC de 0,91.

## O que move o churn

| | |
|---|---|
| ![Distribuição](imagens/01_distribuicao_churn.png) | ![Satisfação](imagens/02_churn_por_satisfacao.png) |
| ![Recência](imagens/03_churn_por_recencia.png) | ![Importância](imagens/04_importancia_variaveis.png) |
| ![Confusão](imagens/05_matriz_confusao.png) | ![ROC](imagens/06_curva_roc.png) |

Os números falam por si:

- A **recência** é o fator que mais pesa, seguida de satisfação e valor gasto.
- Quem deu nota 1 a 2 tem 77% de churn; quem deu 4,5 a 5, só 39%.
- Clientes parados há mais de 300 dias são os que mais somem.

## Tecnologias

Python 3.10+, scikit-learn, pandas, numpy, matplotlib e reportlab.

## Organização do repositório

```
churn-prediction-ecommerce/
├── README.md
├── Analise_Executiva_Churn.pdf
├── requirements.txt
├── dados/clientes_churn.csv
├── src/
│   ├── gerar_dados.py       # monta a base
│   ├── modelo_churn.py      # treina, avalia e gera os 6 gráficos
│   └── gerar_relatorio.py   # gera o PDF
└── imagens/
```

```bash
pip install -r requirements.txt
python src/gerar_dados.py
python src/modelo_churn.py
python src/gerar_relatorio.py
```

## Sobre os dados

A base (6.000 clientes) foi montada por mim para reproduzir padrões realistas de e-commerce
seguindo a estrutura da Olist, o que deixa o projeto inteiramente reproduzível sem precisar
baixar nada. Para usar dados reais, basta extrair as mesmas colunas do
[dataset da Olist no Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
e apontar o pipeline para o novo CSV.

---

Ana Paula Corrêa Galdino · Data Analytics (POSTECH/FIAP)
[GitHub](https://github.com/AnaPaula-Galdino) · [LinkedIn](https://linkedin.com/in/galdinoana/)
