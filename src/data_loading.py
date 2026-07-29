"""
Módulo para carregamento e cruzamento dos dados do Censo Escolar e do IDEB.
"""
import pandas as pd


def load_censo_data(file_path: str) -> pd.DataFrame:
    """Carrega os dados do Censo Escolar."""
    return pd.read_csv(file_path, low_memory=False)


def load_ideb_data(file_path: str) -> pd.DataFrame:
    """Carrega os dados do IDEB."""
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)


def merge_datasets(censo_df: pd.DataFrame, ideb_df: pd.DataFrame, on_key: str = 'CO_ENTIDADE') -> pd.DataFrame:
    """Realiza a junção entre os dados do Censo Escolar e do IDEB pelo código da escola."""
    return pd.merge(censo_df, ideb_df, on=on_key, how='inner')
