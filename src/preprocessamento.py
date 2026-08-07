"""
Módulo para pré-processamento, seleção de features e preparação dos dados.

Funções:
    - preparar_features: one-hot encoding e remoção de colunas proibidas
"""
import pandas as pd
import numpy as np


def preparar_features(df: pd.DataFrame, colunas_excluir: list,
                      coluna_categorica: str = 'Type',
                      drop_first_dummy: bool = True) -> pd.DataFrame:
    """
    Prepara o DataFrame para modelagem: aplica one-hot encoding na variável
    categórica e remove colunas que não devem ser usadas como features.

    Parâmetros:
        df: DataFrame original.
        colunas_excluir: lista de colunas a remover (ex: UDI, Product ID, submodos).
        coluna_categorica: nome da coluna categórica para one-hot encoding (padrão: 'Type').
        drop_first_dummy: se True, remove a primeira categoria (evita dummy trap).

    Retorna:
        pd.DataFrame processado.
    """
    df = df.copy()

    # One-hot encoding
    if coluna_categorica in df.columns:
        df = pd.get_dummies(df, columns=[coluna_categorica],
                            prefix=[coluna_categorica],
                            drop_first=drop_first_dummy, dtype=float)

    # Remover colunas proibidas
    colunas_a_dropar = [c for c in colunas_excluir if c in df.columns]
    df = df.drop(columns=colunas_a_dropar)

    return df
