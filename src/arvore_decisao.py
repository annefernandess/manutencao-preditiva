"""
Funções auxiliares para treinamento e avaliação de Árvore de Decisão
e Random Forest no projeto de Manutenção Preditiva.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, fbeta_score


SEED = 42


def f2_scorer():
    """Retorna um scorer do sklearn para F2-Score (beta=2)."""
    return make_scorer(fbeta_score, beta=2, zero_division=0)


def obter_alphas_candidatos(X_train, y_train):
    """
    Usa cost_complexity_pruning_path para obter os valores candidatos
    de ccp_alpha e as impurezas correspondentes.

    Retorna:
        ccp_alphas (array): valores candidatos de alpha.
        impurities (array): impurezas totais da árvore para cada alpha.
    """
    clf_temp = DecisionTreeClassifier(random_state=SEED)
    path = clf_temp.cost_complexity_pruning_path(X_train, y_train)
    return path.ccp_alphas, path.impurities


def cross_val_arvore(X_train, y_train, ccp_alphas, class_weight=None, n_splits=5):
    """
    Executa validação cruzada estratificada para cada alpha candidato,
    usando F2-Score como métrica.

    Parâmetros:
        X_train: features de treino.
        y_train: rótulos de treino.
        ccp_alphas: array de valores de alpha a testar.
        class_weight: None ou 'balanced'.
        n_splits: número de folds (padrão 5, conforme Regra de Ouro K≈N/5).

    Retorna:
        pd.DataFrame com colunas: ccp_alpha, f2_mean, f2_std.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scorer = f2_scorer()

    resultados = []
    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(
            ccp_alpha=alpha,
            class_weight=class_weight,
            random_state=SEED
        )
        scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring=scorer)
        resultados.append({
            'ccp_alpha': alpha,
            'f2_mean': scores.mean(),
            'f2_std': scores.std()
        })

    return pd.DataFrame(resultados)


def grid_search_random_forest(X_train, y_train, param_grid, n_splits=5):
    """
    Grid search com validação cruzada estratificada para Random Forest,
    usando F2-Score como métrica de seleção.

    Parâmetros:
        X_train: features de treino.
        y_train: rótulos de treino.
        param_grid: dicionário com listas de valores para
                    'n_estimators', 'max_depth', 'class_weight'.
        n_splits: número de folds.

    Retorna:
        pd.DataFrame com resultados ordenados por f2_mean descendente.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scorer = f2_scorer()

    resultados = []
    total = (len(param_grid['n_estimators']) *
             len(param_grid['max_depth']) *
             len(param_grid['class_weight']))
    i = 0

    for n_est in param_grid['n_estimators']:
        for md in param_grid['max_depth']:
            for cw in param_grid['class_weight']:
                i += 1
                print(f'  [{i}/{total}] n_estimators={n_est}, max_depth={md}, class_weight={cw}', end=' ... ')
                clf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=md,
                    class_weight=cw,
                    random_state=SEED,
                    n_jobs=-1
                )
                scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring=scorer)
                f2_mean = scores.mean()
                f2_std = scores.std()
                print(f'F2={f2_mean:.4f} ± {f2_std:.4f}')

                resultados.append({
                    'n_estimators': n_est,
                    'max_depth': md,
                    'class_weight': cw,
                    'f2_mean': f2_mean,
                    'f2_std': f2_std
                })

    df = pd.DataFrame(resultados).sort_values('f2_mean', ascending=False).reset_index(drop=True)
    return df
