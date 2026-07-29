# Previsão do Desempenho Escolar no IDEB com Aprendizagem de Máquina

## 🎯 Objetivo do Projeto
Este projeto de Aprendizagem de Máquina foi desenvolvido em grupo (2 pessoas) com o objetivo de **prever se uma escola atinge ou não a meta estabelecida pelo IDEB** (Índice de Desenvolvimento da Educação Básica). A análise é realizada cruzando dados de infraestrutura e perfil das escolas provenientes do **Censo Escolar** com os indicadores de desempenho e metas do **IDEB**, utilizando algoritmos de **Rede Neural** e **Árvore de Decisão**.

---

## 📊 Fontes de Dados
Os dados utilizados são públicos e fornecidos pelo **INEP (Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira)**:

1. **Censo Escolar (Microdados)**: Informações detalhadas sobre infraestrutura, matrículas, docentes e turmas das escolas brasileiras.
   - 🔗 [Acesso aos Dados do Censo Escolar - INEP](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar)
2. **IDEB (Resultados e Metas)**: Indicadores de rendimento escolar (aprovação) e notas nos exames Saeb por escola.
   - 🔗 [Acesso aos Dados do IDEB - INEP](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/ideb)

---

## 👥 Integrantes do Grupo
- **Integrante 1**: Anne Fernandes da Costa Oliveira
- **Integrante 2**: João Vitor Pereira Costa

---

## 🛠️ Como Configurar o Ambiente

### Pré-requisitos
- Python 3.9+ instalado
- `pip` e `virtualenv`

### Passo a Passo

1. **Clonar o repositório e navegar até a pasta:**
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd ideb-previsao
   ```

2. **Criar e ativar o ambiente virtual (`venv`):**
   - No Linux/macOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - No Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Instalar as dependências do projeto:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Iniciar o Jupyter Notebook ou Jupyter Lab:**
   ```bash
   jupyter notebook
   ```

---

## 📁 Estrutura de Pastas

```text
ideb-previsao/
├── README.md                 # Documentação principal do projeto
├── requirements.txt           # Dependências de pacotes Python
├── .gitignore                 # Arquivos e pastas ignorados pelo Git
├── dados_brutos/             # Arquivos brutos baixados do INEP (CSV, XLSX)
│   └── .gitkeep
├── dados_tratados/           # Datasets resultantes da limpeza e merge
│   └── .gitkeep
├── notebooks/                # Jupyter Notebooks organizados sequencialmente
│   ├── 01_extracao_e_cruzamento.ipynb
│   ├── 02_limpeza_e_eda.ipynb
│   ├── 03_rede_neural.ipynb
│   ├── 04_arvore_decisao.ipynb
│   └── 05_comparacao_final.ipynb
├── src/                      # Código-fonte Python reutilizável
│   ├── __init__.py
│   ├── data_loading.py       # Funções para leitura e junção dos dados
│   ├── preprocessing.py      # Funções para limpeza e pré-processamento
│   └── utils.py              # Funções utilitárias (estilo de plots, etc.)
└── relatorio/                # Documentos, gráficos exportados e relatórios finais
    └── .gitkeep
```

---

## 🚀 Ordem de Execução dos Notebooks

Para replicar os resultados, os notebooks na pasta `notebooks/` devem ser executados na ordem numérica:

1. **`01_extracao_e_cruzamento.ipynb`**: Leitura das bases brutas do Censo Escolar e do IDEB e cruzamento via código de entidade escolar (`CO_ENTIDADE`).
2. **`02_limpeza_e_eda.ipynb`**: Tratamento de dados faltantes, engenharia de variáveis e Análise Exploratória de Dados (EDA).
3. **`03_rede_neural.ipynb`**: Construção e treinamento do modelo de Rede Neural (Keras / TensorFlow).
4. **`04_arvore_decisao.ipynb`**: Construção e treinamento do modelo de Árvore de Decisão (Scikit-Learn).
5. **`05_comparacao_final.ipynb`**: Avaliação cruzada, comparação de métricas de desempenho e consolidação das conclusões.
