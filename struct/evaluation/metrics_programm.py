import numpy as np

def compute_mae(y_true, y_pred):
    mae_abs = float(np.mean(np.abs(y_true - y_pred)))
    mean_capacity = float(np.mean(y_true))
    mae_pct = mae_abs / mean_capacity * 100 if mean_capacity > 0 else 0.0
    return {'mae_abs': mae_abs,'mae_pct': mae_pct,'mean_capacity': mean_capacity,}

def compute_rmse(y_true, y_pred):
    rmse_abs = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mean_capacity = float(np.mean(y_true))
    rmse_pct = rmse_abs / mean_capacity * 100 if mean_capacity > 0 else 0.0
    return {'rmse_abs': rmse_abs,'rmse_pct': rmse_pct,}


# LIPSCHITZ — УСТОЙЧИВОСТЬ К ШУМУ
def evaluate_lipschitz(model, X, epsilon=0.01, n_trials=10):
    """
    Оценка Lipschitz-константы.

    Идея:
        L ≈ ||f(x + ε·δ) - f(x)|| / (ε · ||δ||)
        усредняем по n_trials шумовым реализациям.

    model: имеет .predict(X) → (N, P)
    X: (N, L, d) — нормализованные данные (Target)
    """
    # Чистое предсказание
    y_clean = model.predict(X)

    lipschitz_values = []

    for _ in range(n_trials):
        delta = np.random.randn(*X.shape)
        delta_norm = np.mean(np.linalg.norm(delta.reshape(len(delta), -1), axis=1))
        X_noisy = X + epsilon * delta
        y_noisy = model.predict(X_noisy)

        y_diff = np.mean(np.linalg.norm(y_clean - y_noisy, axis=1))
        L = y_diff / (epsilon * delta_norm + 1e-12)
        lipschitz_values.append(L)

    L_mean = float(np.mean(lipschitz_values))
    L_std = float(np.std(lipschitz_values))

    return {'L_mean': L_mean,'L_std': L_std,'epsilon': epsilon,
            'n_trials': n_trials,'interpretation': _interpret_lipschitz(L_mean)}


def _interpret_lipschitz(L):
    if L < 1.0:
        return "Очень устойчива"
    elif L < 5.0:
        return "Устойчива"
    elif L < 20.0:
        return "Умеренно чувствительна"
    else:
        return "Чувствительна к шуму"

def evaluate_all(model, X, y_true, epsilon=0.01, n_trials=10,
                 with_lipschitz=True):
    """
    Полная оценка модели на (X, y_true).

    model: имеет .predict(X)
    X: (N, L, d) — нормализованные данные (Target)
    y_true: (N, P) — истинные значения (в исходных единицах)

    Возвращает dict со всеми метриками.
    """
    y_pred = model.predict(X)

    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)

    result = {'mae_abs': mae['mae_abs'],'mae_pct': mae['mae_pct'],'rmse_abs': rmse['rmse_abs'],
              'rmse_pct': rmse['rmse_pct'],'mean_capacity': mae['mean_capacity'],}

    if with_lipschitz:
        L = evaluate_lipschitz(model, X, epsilon=epsilon, n_trials=n_trials)
        result.update({'L_mean': L['L_mean'], 'L_std': L['L_std'], 'L_interpretation': L['interpretation']})

    return result