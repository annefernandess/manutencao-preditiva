import pandas as pd

def carregar_dataset(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    return df


def validar_rotulo_falha(df: pd.DataFrame) -> dict:
 
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
