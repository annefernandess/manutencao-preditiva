"""
Módulo para carregamento e cruzamento dos dados do Censo Escolar e do IDEB.

Funções:
    - encontrar_coluna_uf: identifica automaticamente a coluna de UF em um DataFrame
    - carregar_censo: lê um CSV do Censo Escolar (sep=";", encoding="latin1")
    - carregar_ideb: lê o XLSX do IDEB com cabeçalho na linha 9
    - filtrar_por_uf: filtra um DataFrame mantendo apenas registros de uma UF específica
"""
import pandas as pd


# Nomes conhecidos da coluna de UF nos arquivos do INEP
_COLUNAS_UF_POSSIVEIS = ["SG_UF", "sg_uf", "CO_UF_ESCOLA", "UF"]


def encontrar_coluna_uf(colunas_df):
    """
    Procura o nome exato da coluna de UF dentre as colunas do DataFrame.

    Parâmetros:
        colunas_df: pd.Index ou lista de strings com os nomes das colunas.

    Retorna:
        str: nome da coluna encontrada, ou None se nenhuma candidata for identificada.
    """
    for candidata in _COLUNAS_UF_POSSIVEIS:
        if candidata in colunas_df:
            return candidata
    # Busca case-insensitive como último recurso
    for col in colunas_df:
        if col.strip().upper() in ("SG_UF", "UF"):
            return col
    return None


def carregar_censo(caminho: str) -> pd.DataFrame:
    """
    Carrega um arquivo CSV de microdados do Censo Escolar.

    Parâmetros:
        caminho: caminho absoluto ou relativo para o arquivo .csv.

    Retorna:
        pd.DataFrame com os dados carregados.
    """
    return pd.read_csv(caminho, sep=";", encoding="latin1", low_memory=False)


def carregar_ideb(caminho: str) -> pd.DataFrame:
    """
    Carrega o arquivo XLSX do IDEB (divulgação por escola).

    O arquivo original do INEP possui 9 linhas de cabeçalho textual antes
    da tabela de dados, por isso usamos header=9.

    Parâmetros:
        caminho: caminho absoluto ou relativo para o arquivo .xlsx.

    Retorna:
        pd.DataFrame com os dados carregados.
    """
    return pd.read_excel(caminho, header=9)


def filtrar_por_uf(df: pd.DataFrame, coluna_uf: str, uf_alvo: str) -> pd.DataFrame:
    """
    Filtra um DataFrame mantendo apenas as linhas de uma UF específica.

    Parâmetros:
        df: DataFrame com os dados.
        coluna_uf: nome da coluna que contém a sigla da UF.
        uf_alvo: sigla da UF desejada (ex: "PB").

    Retorna:
        pd.DataFrame filtrado (cópia).
    """
    return df[df[coluna_uf].astype(str).str.strip().str.upper() == uf_alvo.upper()].copy()
