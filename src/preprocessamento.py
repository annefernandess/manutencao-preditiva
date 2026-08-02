"""
Módulo para pré-processamento, limpeza e engenharia de recursos.

Funções:
    - limpar_coluna_ideb: converte colunas do IDEB de string para float
    - preparar_ideb: limpa e padroniza o DataFrame do IDEB para merge
    - fazer_merge_ano: cruza Censo de um ano com o IDEB já limpo
    - montar_painel: empilha os anos, remove IDEB nulo e cria ATINGIU_META
"""
import pandas as pd
import numpy as np


def limpar_coluna_ideb(s: pd.Series) -> pd.Series:
    """
    Converte uma coluna de valores do IDEB (originalmente string) para float.

    Trata os seguintes casos:
        - Substitui vírgula por ponto como separador decimal.
        - Converte "-", "*", "ND" e strings vazias para NaN.
        - Converte o resultado para numérico (float).

    Parâmetros:
        s: pd.Series com os valores originais.

    Retorna:
        pd.Series numérica (float64).
    """
    if s.dtype == 'O':
        # Quando carregado do XLSX, a coluna pode conter tipos mistos
        # (int/float para valores válidos e strings como "-" para ausentes).
        # Convertemos tudo para string antes de aplicar as transformações.
        s = s.astype(str).str.strip()
        s = s.str.replace(',', '.')
        s = s.replace({'-': np.nan, '*': np.nan, 'ND': np.nan, '': np.nan, 'nan': np.nan, 'None': np.nan})
    return pd.to_numeric(s, errors='coerce')


def preparar_ideb(df_ideb: pd.DataFrame) -> tuple:
    """
    Prepara o DataFrame do IDEB para o merge com o Censo Escolar.

    Etapas:
        1. Remove linhas onde ID_ESCOLA é NaN.
        2. Renomeia ID_ESCOLA para CO_ENTIDADE e converte para int64.
        3. Aplica limpar_coluna_ideb() em todas as colunas VL_OBSERVADO_* e VL_PROJECAO_*.

    Parâmetros:
        df_ideb: DataFrame carregado do arquivo ideb_pb.csv.

    Retorna:
        tuple: (df_ideb_limpo, qtd_linhas_removidas)
            - df_ideb_limpo: DataFrame pronto para merge.
            - qtd_linhas_removidas: int com o número de linhas descartadas por ID_ESCOLA nulo.
    """
    qtd_antes = len(df_ideb)
    df = df_ideb.dropna(subset=['ID_ESCOLA']).copy()
    qtd_removidas = qtd_antes - len(df)

    # Renomear e converter chave
    df = df.rename(columns={'ID_ESCOLA': 'CO_ENTIDADE'})
    df['CO_ENTIDADE'] = df['CO_ENTIDADE'].astype('int64')

    # Limpar colunas numéricas do IDEB
    colunas_obs = [col for col in df.columns if col.startswith("VL_OBSERVADO_")]
    colunas_proj = [col for col in df.columns if col.startswith("VL_PROJECAO_")]

    for col in colunas_obs + colunas_proj:
        df[col] = limpar_coluna_ideb(df[col])

    return df, qtd_removidas


def fazer_merge_ano(df_censo: pd.DataFrame, df_ideb_limpo: pd.DataFrame, ano: int) -> pd.DataFrame:
    """
    Cruza o Censo Escolar de um ano específico com os dados do IDEB.

    Seleciona do IDEB apenas CO_ENTIDADE, VL_OBSERVADO_{ano} (renomeado para IDEB)
    e VL_PROJECAO_{ano} (renomeado para IDEB_META, se existir). Faz inner join
    por CO_ENTIDADE e adiciona a coluna ANO.

    Parâmetros:
        df_censo: DataFrame do Censo Escolar de um ano.
        df_ideb_limpo: DataFrame do IDEB já processado por preparar_ideb().
        ano: int com o ano de referência (2015, 2017, 2019, 2021 ou 2023).

    Retorna:
        pd.DataFrame com o merge realizado e colunas IDEB, IDEB_META e ANO.
    """
    col_observado = f"VL_OBSERVADO_{ano}"
    col_projecao = f"VL_PROJECAO_{ano}"

    # Selecionar colunas relevantes do IDEB
    cols_ideb = ['CO_ENTIDADE', col_observado]
    if col_projecao in df_ideb_limpo.columns:
        cols_ideb.append(col_projecao)

    df_ideb_ano = df_ideb_limpo[cols_ideb].copy()

    # Renomear para nomes padronizados
    rename_dict = {col_observado: 'IDEB'}
    if col_projecao in df_ideb_limpo.columns:
        rename_dict[col_projecao] = 'IDEB_META'
    df_ideb_ano = df_ideb_ano.rename(columns=rename_dict)

    # Merge (Inner Join)
    df_merged = pd.merge(df_censo, df_ideb_ano, on='CO_ENTIDADE', how='inner')
    df_merged['ANO'] = ano

    # Se IDEB_META não existir (ex: 2023 não tem projeção), criar como NaN
    if 'IDEB_META' not in df_merged.columns:
        df_merged['IDEB_META'] = np.nan

    return df_merged


def montar_painel(lista_dfs_anuais: list) -> pd.DataFrame:
    """
    Empilha os DataFrames anuais em um painel, remove linhas sem nota IDEB
    e cria a variável alvo ATINGIU_META.

    Etapas:
        1. Concatena todos os DataFrames anuais (pd.concat).
        2. Remove linhas onde IDEB é NaN (escola sem nota calculada).
        3. Cria ATINGIU_META: 1 se IDEB >= IDEB_META, 0 se menor, NaN se IDEB_META for NaN.

    Parâmetros:
        lista_dfs_anuais: lista de DataFrames retornados por fazer_merge_ano().

    Retorna:
        pd.DataFrame final em formato de painel (empilhado por ano).
    """
    df_painel = pd.concat(lista_dfs_anuais, ignore_index=True)

    # Remover linhas sem nota real do IDEB
    df_painel = df_painel.dropna(subset=['IDEB']).copy()

    # Criar variável alvo
    cond_atingiu = df_painel['IDEB'] >= df_painel['IDEB_META']
    cond_nao_atingiu = df_painel['IDEB'] < df_painel['IDEB_META']

    df_painel['ATINGIU_META'] = np.nan
    df_painel.loc[cond_atingiu, 'ATINGIU_META'] = 1.0
    df_painel.loc[cond_nao_atingiu, 'ATINGIU_META'] = 0.0

    return df_painel
