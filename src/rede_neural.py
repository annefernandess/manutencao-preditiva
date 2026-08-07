"""
Módulo para construção, treinamento e avaliação de Redes Neurais (MLP).

Funções:
    - calcular_arquitetura_mlp: calcula n_max e |W| pela Regra de Ouro da generalização
    - criar_modelo_mlp: constrói um modelo Keras Sequential com a arquitetura definida
    - grid_search_mlp: executa busca em grade de hiperparâmetros com validação
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, fbeta_score


def calcular_arquitetura_mlp(N: int, d: int, margem: int = 0) -> dict:
    """
    Calcula o número máximo de neurônios na camada escondida pela Regra de Ouro
    da generalização (N >= 10 * d_vc) e o total de pesos |W|.

    Para uma rede com uma camada escondida:
        |W| = (d+1)*n + (n+1)
    Isolando n da regra N >= 10*|W|:
        n <= (N - 10) / (10 * (d + 2))

    Parâmetros:
        N: número de amostras de treino.
        d: número de features de entrada (dimensão do input).
        margem: neurônios a subtrair do n_max para margem de segurança (padrão: 0).

    Retorna:
        dict com chaves: n_max_float, n_escolhido, total_pesos, regra_satisfeita.
    """
    n_max_float = (N - 10) / (10 * (d + 2))
    n_escolhido = int(n_max_float) - margem
    if n_escolhido < 1:
        n_escolhido = 1

    total_pesos = (d + 1) * n_escolhido + (n_escolhido + 1)
    regra_satisfeita = N >= 10 * total_pesos

    return {
        'N': N,
        'd': d,
        'n_max_float': n_max_float,
        'n_escolhido': n_escolhido,
        'total_pesos': total_pesos,
        'regra_satisfeita': regra_satisfeita,
    }


def criar_modelo_mlp(d: int, n: int, learning_rate: float = 0.01, l2_lambda: float = 0.0):
    """
    Cria um modelo MLP Keras Sequential para classificação binária.

    Arquitetura:
        - Camada de entrada: d features
        - 1 camada escondida: n neurônios, ativação ReLU, regularização L2
        - Camada de saída: 1 neurônio, ativação sigmoid

    Parâmetros:
        d: número de features de entrada.
        n: número de neurônios na camada escondida.
        learning_rate: taxa de aprendizado para o otimizador Adam.
        l2_lambda: intensidade da regularização L2 (0.0 = sem regularização).

    Retorna:
        modelo Keras compilado.
    """
    import tensorflow as tf
    from tensorflow import keras

    reg = keras.regularizers.l2(l2_lambda) if l2_lambda > 0 else None

    model = keras.Sequential([
        keras.layers.Dense(n, activation='relu', input_shape=(d,),
                           kernel_regularizer=reg, name='camada_escondida'),
        keras.layers.Dense(1, activation='sigmoid',
                           kernel_regularizer=reg, name='saida'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model


def grid_search_mlp(X_train, y_train, X_val, y_val, d: int, n: int,
                    learning_rates: list, batch_sizes: list, l2_lambdas: list,
                    use_class_weights: list = [False], epochs: int = 50, verbose: int = 0) -> pd.DataFrame:
    """
    Executa busca em grade manual de hiperparâmetros para MLP Keras.

    Para cada combinação, treina um modelo e registra as métricas.
    Devido ao desbalanceamento e a assimetria de custos (Falsos Negativos
    são muito mais caros que Falsos Positivos na manutenção preditiva),
    calcula o F1-Score e o F2-Score no conjunto de validação.

    Parâmetros:
        X_train, y_train: dados de treino (sem validação).
        X_val, y_val: dados de validação.
        d: número de features.
        n: número de neurônios na camada escondida.
        learning_rates: lista de taxas de aprendizado a testar.
        batch_sizes: lista de tamanhos de batch a testar.
        l2_lambdas: lista de coeficientes de regularização L2 a testar.
        use_class_weights: lista de booleanos para testar class_weight.
        epochs: número de épocas por treino (padrão: 50).
        verbose: nível de verbosidade do Keras (0=silencioso).

    Retorna:
        pd.DataFrame com resultados ordenado por val_f2 descendente.
    """
    resultados = []

    total = len(learning_rates) * len(batch_sizes) * len(l2_lambdas) * len(use_class_weights)
    i = 0

    for lr in learning_rates:
        for bs in batch_sizes:
            for l2 in l2_lambdas:
                for use_cw in use_class_weights:
                    i += 1
                    cw_str = 'Sim' if use_cw else 'Não'
                    print(f'  [{i}/{total}] lr={lr}, batch={bs}, L2={l2}, cw={cw_str}', end=' ... ')

                    cw = None
                    if use_cw:
                        n_neg = (y_train == 0).sum()
                        n_pos = (y_train == 1).sum()
                        # balanceamento de classes
                        cw = {0: len(y_train)/(2*n_neg), 1: len(y_train)/(2*n_pos)}

                    model = criar_modelo_mlp(d, n, learning_rate=lr, l2_lambda=l2)
                    model.fit(X_train, y_train, epochs=epochs, batch_size=bs,
                              class_weight=cw, verbose=verbose, validation_data=(X_val, y_val))
                    
                    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
                    
                    # Computa F1 Score and F2 Score
                    y_val_pred_prob = model.predict(X_val, verbose=0)
                    y_val_pred = (y_val_pred_prob >= 0.5).astype(int).ravel()
                    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
                    val_f2 = fbeta_score(y_val, y_val_pred, beta=2, zero_division=0)

                    print(f'E_val={val_loss:.4f}, acc_val={val_acc:.4f}, f1_val={val_f1:.4f}, f2_val={val_f2:.4f}')

                    resultados.append({
                        'learning_rate': lr,
                        'batch_size': bs,
                        'l2_lambda': l2,
                        'class_weight': use_cw,
                        'val_loss': val_loss,
                        'val_accuracy': val_acc,
                        'val_f1': val_f1,
                        'val_f2': val_f2
                    })

    df_resultados = pd.DataFrame(resultados).sort_values('val_f2', ascending=False).reset_index(drop=True)
    return df_resultados
