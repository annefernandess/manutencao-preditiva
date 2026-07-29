"""
Módulo para pré-processamento, limpeza e engenharia de recursos.
"""
import pandas as pd


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Trata valores ausentes no DataFrame."""
    return df.copy()


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica codificação em variáveis categóricas."""
    return df.copy()


def prepare_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Prepara a variável alvo indicando se a escola atingiu ou não a meta do IDEB."""
    return df.copy()
