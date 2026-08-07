"""
Módulo para carregamento e validação dos dados do AI4I 2020 Predictive Maintenance.

Funções:
    - carregar_dataset: lê o CSV do dataset AI4I 2020
    - validar_rotulo_falha: verifica consistência entre Machine failure e submodos
"""
import pandas as pd


def carregar_dataset(caminho: str) -> pd.DataFrame:
    """
    Carrega o dataset AI4I 2020 Predictive Maintenance.

    Parâmetros:
        caminho: caminho para o arquivo CSV.

    Retorna:
        pd.DataFrame com os dados carregados.
    """
    df = pd.read_csv(caminho)
    return df


def validar_rotulo_falha(df: pd.DataFrame) -> dict:
    """
    Valida a consistência do rótulo Machine failure com os submodos de falha.

    Verifica se Machine failure == 1 sempre que pelo menos um dos submodos
    (TWF, HDF, PWF, OSF, RNF) é 1, e vice-versa.

    Parâmetros:
        df: DataFrame contendo as colunas Machine failure, TWF, HDF, PWF, OSF, RNF.

    Retorna:
        dict com chaves:
            - total: total de registros
            - falhas: total de Machine failure == 1
            - submodos_cols: lista dos nomes das colunas de submodo
            - algum_submodo: total de linhas com pelo menos um submodo == 1
            - consistentes: total de linhas onde Machine failure == (algum submodo == 1)
            - inconsistentes: total de linhas inconsistentes
            - df_inconsistentes: DataFrame com as linhas inconsistentes (vazio se nenhuma)
    """
    submodos = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    algum_submodo = (df[submodos].sum(axis=1) > 0).astype(int)

    consistente = df['Machine failure'] == algum_submodo
    inconsistentes = df[~consistente]

    return {
        'total': len(df),
        'falhas': int(df['Machine failure'].sum()),
        'submodos_cols': submodos,
        'algum_submodo': int(algum_submodo.sum()),
        'consistentes': int(consistente.sum()),
        'inconsistentes': int((~consistente).sum()),
        'df_inconsistentes': inconsistentes,
    }
