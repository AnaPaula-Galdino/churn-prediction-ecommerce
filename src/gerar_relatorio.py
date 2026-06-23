import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relatorio_exec import construir

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(BASE, "imagens")
def img(n): return os.path.join(IMG, n)
m = json.load(open(os.path.join(BASE, "metricas.json"), encoding="utf-8"))
best = m["melhor"]; mb = m["metricas"][best]
auc = f"{mb['roc_auc']:.3f}"; acc = f"{mb['accuracy']*100:.0f}%"
fatores = {"recencia_dias":"recência","nota_media_review":"satisfação (nota)","valor_total":"valor gasto",
           "frequencia_pedidos":"frequência","ticket_medio":"ticket médio"}
top = ", ".join(fatores.get(f, f) for f in m["top_fatores"])

config = {
 "titulo": "Previsão de Churn de Clientes",
 "subtitulo": "Machine Learning aplicado à retenção em e-commerce (estrutura estilo Olist)",
 "meta": "Autora: Ana Paula Galdino  •  Pós-graduação em Data Analytics (POSTECH/FIAP)  •  Junho de 2026",
 "fonte": "Dataset: e-commerce (estilo Olist)  |  Modelagem: Ana Paula Galdino",
 "sumario": [
   f"Reter um cliente custa muito menos do que conquistar um novo. Este projeto desenvolve um modelo de "
   f"<b>Machine Learning</b> que prevê quais clientes têm maior probabilidade de <b>churn</b> (deixar de comprar), "
   f"a partir do comportamento de compra e da experiência de entrega de <b>{m['n_clientes']:,}</b> clientes.".replace(",", "."),
   f"O melhor modelo (<b>{best}</b>) atinge <b>ROC AUC de {auc}</b> e acurácia de <b>{acc}</b>, "
   "com forte capacidade de separar clientes em risco dos clientes ativos — permitindo direcionar ações "
   "de retenção antes da perda, e não depois.",
 ],
 "kpis": [
   (auc, f"ROC AUC ({best})"),
   (acc, "acurácia"),
   (f"{m['taxa_churn']:.0f}%", "taxa de churn na base"),
   (f"{m['n_clientes']:,}".replace(",", "."), "clientes analisados"),
 ],
 "secoes": [
   {"titulo": "1. Entendendo o Churn",
    "texto": [
      f"A base apresenta uma taxa de churn de <b>{m['taxa_churn']:.0f}%</b>. O cruzamento com a satisfação é "
      f"revelador: clientes com nota média baixa (1–2) têm <b>{m['churn_baixa_nota']:.0f}%</b> de churn, contra "
      f"apenas <b>{m['churn_alta_nota']:.0f}%</b> entre os mais satisfeitos (4.5–5). A experiência do cliente "
      "é, portanto, um preditor direto de retenção.",
    ],
    "imagens": [(img("01_distribuicao_churn.png"), "Proporção de clientes ativos vs. churn"),
                (img("02_churn_por_satisfacao.png"), "Quanto menor a satisfação, maior o churn")]},
   {"titulo": "2. Drivers de Negócio",
    "texto": [
      "A <b>recência</b> (tempo desde a última compra) é o fator mais associado ao churn: clientes parados há "
      "mais de 300 dias concentram as maiores taxas de saída. Em seguida vêm <b>satisfação</b> e <b>valor "
      f"gasto</b>. Os três principais fatores do modelo são: <b>{top}</b>.",
    ],
    "imagens": [(img("03_churn_por_recencia.png"), "Taxa de churn cresce com a recência"),
                (img("04_importancia_variaveis.png"), "Importância das variáveis no modelo Random Forest")]},
   {"titulo": "3. Desempenho do Modelo",
    "texto": [
      f"Foram comparados dois algoritmos — Regressão Logística e Random Forest. O modelo <b>{best}</b> obteve o "
      f"melhor equilíbrio (ROC AUC {auc}). A matriz de confusão mostra acertos consistentes em ambas as classes, "
      "e a curva ROC confirma a forte separação entre clientes ativos e em risco.",
    ],
    "imagens": [(img("05_matriz_confusao.png"), "Acertos e erros do melhor modelo"),
                (img("06_curva_roc.png"), "Curva ROC — quanto mais perto do topo esquerdo, melhor")]},
 ],
 "conclusao_titulo": "Aplicação e Recomendações",
 "conclusoes": [
   "<b>Score de risco:</b> o modelo gera, para cada cliente, uma probabilidade de churn — base para uma "
   "régua de retenção priorizada.",
   "<b>Aja na recência:</b> clientes inativos há +300 dias devem receber campanhas de reativação imediatas.",
   "<b>Cuide da experiência:</b> como baixa satisfação dobra o churn, reduzir atrasos e melhorar o pós-venda "
   "tem impacto direto na retenção.",
   "<b>Próximo passo:</b> aplicar o pipeline aos dados reais da Olist (Kaggle) usando o mesmo esquema de "
   "features, e estimar o retorno financeiro da retenção.",
 ],
}

if __name__ == "__main__":
    construir(config, os.path.join(BASE, "Analise_Executiva_Churn.pdf"))
