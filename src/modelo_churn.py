"""
Modelo de Previsão de Churn — E-commerce  (Machine Learning)
Autora: Ana Paula Galdino

Pipeline completo: EDA de negócio -> pré-processamento -> modelagem ->
avaliação. Gera 6 visualizações executivas (paleta azul) e salva as métricas.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(BASE, "imagens"); os.makedirs(IMG, exist_ok=True)
C = {"escuro": "#1f4e79", "medio": "#2e6da4", "claro": "#5b9bd5",
     "suave": "#a6c8e0", "destaque": "#4fc3f7", "cinza": "#d9d9d9", "alerta": "#c0392b"}
FONTE = "Dataset: e-commerce (estilo Olist)  |  Modelagem: Ana Paula Galdino"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.edgecolor": "#9aa7b8",
    "axes.grid": True, "grid.color": "#eef2f7", "axes.axisbelow": True, "figure.dpi": 120,
    "savefig.bbox": "tight"})
def rodape(fig): fig.text(0.01, 0.005, FONTE, fontsize=7.5, color="#7a8aa0")

df = pd.read_csv(os.path.join(BASE, "dados", "clientes_churn.csv"))
y = df["churn"]; X = df.drop(columns="churn")
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()
pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

modelos = {"Regressão Logística": LogisticRegression(max_iter=1000),
           "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12,
                                                    random_state=42, n_jobs=-1)}
res = {}
for nome, clf in modelos.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)]); pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]; pred = pipe.predict(X_te)
    res[nome] = {"pipe": pipe, "proba": proba, "pred": pred,
                 "accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred),
                 "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred),
                 "roc_auc": roc_auc_score(y_te, proba)}
melhor = max(res, key=lambda k: res[k]["roc_auc"])

# ---- 01 Distribuição do churn ----
fig, ax = plt.subplots(figsize=(7, 5))
vc = y.value_counts().sort_index(); taxa = y.mean()*100
ax.bar(["Ativo", "Churn"], vc.values, color=[C["claro"], C["escuro"]])
for i, v in enumerate(vc.values): ax.text(i, v, f"{v}\n({v/len(y)*100:.1f}%)", ha="center", va="bottom", fontsize=11)
ax.set_title(f"Distribuição do churn — taxa geral de {taxa:.1f}%", fontweight="bold", color=C["escuro"], fontsize=14, pad=10)
ax.set_ylabel("Nº de clientes"); ax.set_ylim(0, vc.max()*1.18)
rodape(fig); fig.savefig(os.path.join(IMG, "01_distribuicao_churn.png")); plt.close(fig)

# ---- 02 Churn por satisfação (nota) ----
fig, ax = plt.subplots(figsize=(8.5, 5))
faixa = pd.cut(df["nota_media_review"], [0,2,3,4,4.5,5], labels=["1–2","2–3","3–4","4–4.5","4.5–5"])
tx = df.groupby(faixa, observed=True)["churn"].mean()*100
ax.bar(tx.index.astype(str), tx.values, color=C["medio"])
for i, v in enumerate(tx.values): ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=10)
ax.axhline(taxa, color=C["alerta"], ls="--", lw=1.3, label=f"média geral: {taxa:.0f}%")
ax.set_title("Taxa de churn por nota média de avaliação", fontweight="bold", color=C["escuro"], fontsize=14, pad=10)
ax.set_xlabel("Nota média (review)"); ax.set_ylabel("Taxa de churn (%)"); ax.legend(frameon=True)
rodape(fig); fig.savefig(os.path.join(IMG, "02_churn_por_satisfacao.png")); plt.close(fig)

# ---- 03 Churn por recência ----
fig, ax = plt.subplots(figsize=(8.5, 5))
fr = pd.cut(df["recencia_dias"], [0,60,120,200,300,9999], labels=["0–60","60–120","120–200","200–300","300+"])
tx2 = df.groupby(fr, observed=True)["churn"].mean()*100
ax.bar(tx2.index.astype(str), tx2.values, color=C["escuro"])
for i, v in enumerate(tx2.values): ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=10)
ax.axhline(taxa, color=C["alerta"], ls="--", lw=1.3, label=f"média geral: {taxa:.0f}%")
ax.set_title("Taxa de churn por recência (dias desde a última compra)", fontweight="bold", color=C["escuro"], fontsize=14, pad=10)
ax.set_xlabel("Dias desde a última compra"); ax.set_ylabel("Taxa de churn (%)"); ax.legend(frameon=True)
rodape(fig); fig.savefig(os.path.join(IMG, "03_churn_por_recencia.png")); plt.close(fig)

# ---- 04 Importância das variáveis (RF) ----
rf = res["Random Forest"]["pipe"]
ohe = rf.named_steps["pre"].named_transformers_["cat"]
feat = num_cols + list(ohe.get_feature_names_out(cat_cols))
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=feat).sort_values().tail(12)
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.barh(imp.index, imp.values, color=C["medio"])
ax.set_title("Variáveis que mais influenciam o churn", fontweight="bold", color=C["escuro"], fontsize=14, pad=10)
ax.set_xlabel("Importância (Random Forest)")
rodape(fig); fig.savefig(os.path.join(IMG, "04_importancia_variaveis.png")); plt.close(fig)

# ---- 05 Matriz de confusão ----
cm = confusion_matrix(y_te, res[melhor]["pred"])
fig, ax = plt.subplots(figsize=(5.6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["Ativo","Churn"]); ax.set_yticklabels(["Ativo","Churn"])
ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
ax.set_title(f"Matriz de confusão — {melhor}", fontweight="bold", color=C["escuro"], fontsize=13, pad=10)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=15,
                color="white" if cm[i,j] > cm.max()/2 else "#1f4e79")
ax.grid(False)
rodape(fig); fig.savefig(os.path.join(IMG, "05_matriz_confusao.png")); plt.close(fig)

# ---- 06 Curva ROC ----
fig, ax = plt.subplots(figsize=(6.4, 5.4))
for nome, cor in zip(res, [C["escuro"], C["destaque"]]):
    fpr, tpr, _ = roc_curve(y_te, res[nome]["proba"])
    ax.plot(fpr, tpr, lw=2.2, color=cor, label=f"{nome} (AUC={res[nome]['roc_auc']:.3f})")
ax.plot([0,1],[0,1],"--", color=C["cinza"], lw=1.2, label="Aleatório")
ax.set_xlabel("Taxa de falso positivo"); ax.set_ylabel("Taxa de verdadeiro positivo")
ax.set_title("Curva ROC — comparação de modelos", fontweight="bold", color=C["escuro"], fontsize=14, pad=10)
ax.legend(frameon=True, loc="lower right")
rodape(fig); fig.savefig(os.path.join(IMG, "06_curva_roc.png")); plt.close(fig)

# ---- Métricas para o relatório ----
metr = {n: {k: round(float(r[k]),3) for k in ["accuracy","precision","recall","f1","roc_auc"]}
        for n, r in res.items()}
top3 = imp.sort_values(ascending=False).head(3).index.tolist()
resumo = {"melhor": melhor, "taxa_churn": round(float(taxa),1), "n_clientes": int(len(df)),
          "metricas": metr, "top_fatores": top3,
          "churn_baixa_nota": round(float(df[df['nota_media_review']<3]['churn'].mean()*100),0),
          "churn_alta_nota": round(float(df[df['nota_media_review']>=4.5]['churn'].mean()*100),0)}
with open(os.path.join(BASE, "metricas.json"), "w", encoding="utf-8") as f:
    json.dump(resumo, f, ensure_ascii=False, indent=2)

print("=== Resultados ===")
for n, r in res.items():
    print(f"{n:<22} acc={r['accuracy']:.3f} prec={r['precision']:.3f} rec={r['recall']:.3f} f1={r['f1']:.3f} auc={r['roc_auc']:.3f}")
print("Melhor:", melhor, "| Top fatores:", top3)
print("Gráficos:", sorted(x for x in os.listdir(IMG) if x.startswith("0")))
