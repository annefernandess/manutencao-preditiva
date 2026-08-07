# 🔧 Manutenção Preditiva Industrial

## 📌 Objetivo

Projeto de **Aprendizagem de Máquina** desenvolvido para a disciplina da UFPB, com o objetivo de prever se uma máquina industrial vai apresentar **falha** (`Machine failure = 1`) ou **não** (`Machine failure = 0`) a partir de leituras de sensores operacionais.

São utilizados dois modelos de classificação binária:
- **Rede Neural (MLP)** — Perceptron Multicamadas com Keras/TensorFlow
- **Árvore de Decisão** — Scikit-Learn

Os modelos são avaliados e comparados quanto à capacidade de generalização, seguindo a teoria de aprendizagem de máquina (dimensão VC, Regra de Ouro, validação cruzada).

---

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset** — UCI Machine Learning Repository

- **Link:** [https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Registros:** 10.000
- **Features de sensores (preditoras):**
  - `Air temperature [K]` — Temperatura do ar (Kelvin)
  - `Process temperature [K]` — Temperatura do processo (Kelvin)
  - `Rotational speed [rpm]` — Velocidade rotacional (rpm)
  - `Torque [Nm]` — Torque (Newton-metro)
  - `Tool wear [min]` — Desgaste da ferramenta (minutos)
  - `Type` — Tipo de produto (L, M, H — variável categórica)
- **Variável alvo:** `Machine failure` (0 = operação normal, 1 = falha)

> [!WARNING]
> **Desbalanceamento severo:** Apenas ~3,4% dos registros são falhas (339 de 10.000). A acurácia bruta é uma métrica enganosa nesse cenário. As métricas prioritárias são **Precisão, Recall e F1-Score da classe minoritária (falha)**.

> [!CAUTION]
> **Data leakage:** As colunas `TWF`, `HDF`, `PWF`, `OSF` e `RNF` representam os submodos de falha que **compõem** a variável alvo `Machine failure`. Usá-las como features causaria vazamento de dados e resultados artificialmente perfeitos. Essas colunas são **excluídas** do conjunto de features.

---

## 👥 Integrantes

- Anne Fernandes da Costa Oliveira
- João Vitor Pereira Costa
---

## 📁 Estrutura do Repositório

```
manutencao-preditiva/
├── README.md
├── requirements.txt
├── .gitignore
├── dados_brutos/              # Dataset original (ai4i2020.csv)
│   └── ai4i2020.csv
├── dados_tratados/            # Datasets processados (gerados pelos notebooks)
│   └── .gitkeep
├── notebooks/                 # Jupyter Notebooks organizados sequencialmente
│   ├── 01_extracao_e_eda.ipynb
│   ├── 02_rede_neural.ipynb
│   ├── 03_arvore_decisao.ipynb
│   └── 04_comparacao_final.ipynb
├── src/                       # Código-fonte Python reutilizável
│   ├── __init__.py
│   ├── carregamento_dados.py  # Funções para leitura dos dados
│   ├── preprocessamento.py    # Funções para limpeza e pré-processamento
│   ├── rede_neural.py         # Funções para construção e treino da MLP
│   └── utils.py               # Funções utilitárias (plotagem, etc.)
├── modelos/                   # Modelos treinados salvos
│   └── .gitkeep
├── relatorio/                 # Documentos e relatórios finais
│   └── .gitkeep
└── arquivo_projeto_anterior/  # Dados e código do projeto anterior (Censo/IDEB)
```

---

## 🚀 Ordem de Execução dos Notebooks

1. **`01_extracao_e_eda.ipynb`**: Carregamento do dataset AI4I 2020, exploração inicial, EDA e divisão treino/teste e separação de features.
2. **`02_rede_neural.ipynb`**: Construção e treinamento do modelo MLP (dimensão VC, grid search, regularização).
3. **`03_arvore_decisao.ipynb`**: Construção e treinamento do modelo de Árvore de Decisão.
4. **`04_comparacao_final.ipynb`**: Comparação de métricas e conclusões.

---

## 🛠️ Instalação

```bash
# Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

---

## 📄 Licença

Projeto acadêmico — UFPB.
