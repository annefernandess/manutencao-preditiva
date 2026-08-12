import pandas as pd
import numpy as np


def preparar_features(df: pd.DataFrame, colunas_excluir: list,
                      coluna_categorica: str = 'Type',
                      drop_first_dummy: bool = True) -> pd.DataFrame:

    df = df.copy()

    # One-hot encoding
    if coluna_categorica in df.columns:
        df = pd.get_dummies(df, columns=[coluna_categorica],
                            prefix=[coluna_categorica],
                            drop_first=drop_first_dummy, dtype=float)

    colunas_a_dropar = [c for c in colunas_excluir if c in df.columns]
    df = df.drop(columns=colunas_a_dropar)

    return df
