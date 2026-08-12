import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, fbeta_score


def calcular_arquitetura_mlp(N: int, d: int, margem: int = 0) -> dict:
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

                        cw = {0: len(y_train)/(2*n_neg), 1: len(y_train)/(2*n_pos)}

                    model = criar_modelo_mlp(d, n, learning_rate=lr, l2_lambda=l2)
                    model.fit(X_train, y_train, epochs=epochs, batch_size=bs,
                              class_weight=cw, verbose=verbose, validation_data=(X_val, y_val))
                    
                    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
                    
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
