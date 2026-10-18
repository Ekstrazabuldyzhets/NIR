import numpy as np

from data.file_reader_programm import stack_data
from preprocessing.preprocessing_programm import (compute_mmd, DomainScalers, DomainAligner,)
from models.model_programm import MODEL_REGISTRY
from correctors.corrector_programm import (CORRECTOR_REGISTRY, extract_gpr_features,)
from evaluation.metrics_programm import (compute_mae, compute_rmse, evaluate_lipschitz,)
from adaptation.adaptation_programm import ADAPTER_REGISTRY

# ОБЁРТКА МОДЕЛИ С GPR
class ModelWithGPR:
    # LSTM (с опциональным мостом) + GPR-коррекция
    def __init__(self, model, corrector):
        self.model = model
        self.corrector = corrector

    def predict(self, X_scaled):
        y_pred = self.model.predict(X_scaled) # предсказывает y_pred
        context = self.model.get_context(X_scaled) # достаёт контекст энкодера
        X_gpr = extract_gpr_features(X_scaled, context) # cобирает 13 GPR-признаков.
        correction = self.corrector.predict(X_gpr) # GPR предсказывает коррекцию
        return y_pred + correction.reshape(-1, 1) # cкладывает: y_pred + correction

# Принимает фолд и конфиг. Возвращает словарь.
def run_fold(fold, cfg) -> dict:
    # verbose - определяет что печатать(печатать ли доп инфу)
    verbose = cfg['verbose']
    fold_idx = fold['fold_idx']

    if verbose:
        print(f"\nFold {fold_idx + 1}")

    # превращает список словарей в два массива (X и y) для source и target.
    X_s_raw, y_s_raw = stack_data(fold['source_data'])
    X_t_raw, y_t_raw = stack_data(fold['target_data'])

    if verbose:
        print(f"Source: {X_s_raw.shape}, Target: {X_t_raw.shape}")

    # считает MMD² между source и target
    mmd_before = compute_mmd(X_s_raw, X_t_raw, n_samples=cfg['mmd_n_samples'], verbose=verbose)

    # нормализация
    scalers = DomainScalers()
    scalers.fit(X_s_raw, y_s_raw, X_t_raw, y_t_raw)

    X_s = scalers.transform_X(X_s_raw, role='source')
    y_s = scalers.transform_y(y_s_raw, role='source')
    X_t = scalers.transform_X(X_t_raw, role='target')
    y_t = scalers.transform_y(y_t_raw, role='target')

    if verbose:
        print(f"y_s: mean={y_s.mean():.4f}, std={y_s.std():.4f}")
        print(f"y_t: mean={y_t.mean():.4f}, std={y_t.std():.4f}")

    # CORAL после нормализации source и target имеют mean=0, std=1 по каждому признаку
    # Но связи между признаками (корреляции) - разные
    # линейно преобразуем source так, чтобы его ковариационная матрица совпала с target
    if cfg['use_coral']:
        aligner = DomainAligner(verbose=verbose)
        aligner.fit(X_s, X_t)
        if verbose:
            err = aligner.alignment_error(X_s, X_t)
            print(f"CORAL: mu_diff={err['mu_diff']:.4f}, "
                  f"cov_diff={err['cov_diff']:.4f}")
        X_s = aligner.transform_source(X_s)
    else:
        aligner = None

    # MMD после
    mmd_after = compute_mmd(X_s, X_t, n_samples=cfg['mmd_n_samples'], verbose=verbose)

    # сама модель
    if verbose:
        print(f"    Обучение модели '{cfg['model']}' "
              f"(epochs={cfg['epochs_source']})...")

    ModelClass = MODEL_REGISTRY[cfg['model']] # выбираем модель
    model = ModelClass(input_dim=X_s.shape[-1], output_dim=y_s.shape[-1], hidden_units=cfg['hidden_units'],
                       learning_rate=cfg['learning_rate_source'], verbose=verbose)
    # input_dim=X_s.shape[-1] - последняя размерность X. Для (315, 6, 9) - 9 число признаков на каждом шаге
    # output_dim=y_s.shape[-1] - последняя размерность y. Для (315, 6) - 6 размер целевого вектора (SoC на 6 шагов)
    # hidden_units - 64 - размер скрытого состояния LSTM
    # learning_rate=cfg['learning_rate_source'] - 1e-4. Learning rate для Adam
    model.build()

    # сохраняем Source-данные для адаптеров (Bridge, DANN)
    model._X_source_scaled = X_s
    model._y_source_scaled = y_s

    # обучаем
    history = model.fit(X_s, y_s, epochs=cfg['epochs_source'], batch_size=cfg['batch_size_source'],
                        patience=cfg.get('patience_source', 30),
                        validation_split=cfg.get('validation_split_source', 0.1),
                        verbose=0)

    final_loss = history.history['loss'][-1]
    if verbose:
        print(f" Финальный loss: {final_loss:.6f}")

    # GPR на остатках Source
    if verbose:
        print(f" Обучение GPR на остатках Source...")

    y_pred_s = model.predict(X_s) # LSTM предсказывает на source данных
    residuals_s = y_s - y_pred_s # покомпонентная разница
    residuals_s_mean = residuals_s.mean(axis=1) # усредняем по 6 шагам

    context_s = model.get_context(X_s) #извлекает контекст энкодера - выход encoder_lstm2
    X_gpr_s = extract_gpr_features(X_s, context_s) # извлекает 13 признаков из X и контекста

    CorrectorClass = CORRECTOR_REGISTRY[cfg['corrector']]
    # оздаем объект MultiScaleGPR
    corrector = CorrectorClass(alpha=cfg.get('gpr_alpha', 1e-6), auto_regularize=cfg.get('gpr_auto_regularize', True), verbose=verbose)
    corrector.fit(X_gpr_s, residuals_s_mean)


    # примененяем GPR к target
    y_pred_t_lstm = model.predict(X_t) # LSTM на target
    wrapped = ModelWithGPR(model, corrector) # LSTM + GPR
    y_pred_t_gpr = wrapped.predict(X_t) # LSTM+GPR на target

    # подсчет ошибки на денормализованных данных
    y_t_denorm = scalers.inverse_y(y_t, role='target')
    y_pred_lstm_denorm = scalers.inverse_y(y_pred_t_lstm, role='target')
    y_pred_gpr_denorm = scalers.inverse_y(y_pred_t_gpr, role='target')

    mae_before = compute_mae(y_t_denorm, y_pred_gpr_denorm)
    rmse_before = compute_rmse(y_t_denorm, y_pred_gpr_denorm)

    if verbose:
        print(f" [ДО] MAE (LSTM):"
              f"{compute_mae(y_t_denorm, y_pred_lstm_denorm)['mae_pct']:.2f}%")
        print(f" [ДО] MAE (LSTM + GPR): {mae_before['mae_pct']:.2f}%")

    # адаптация
    if verbose:
        print(f" Адаптация (method='{cfg['adapt_method']}')...")
    # получаем класс адаптера
    AdapterClass = ADAPTER_REGISTRY[cfg['adapt_method']]
    adapter = AdapterClass(model=model, corrector=corrector, scalers=scalers, verbose=verbose)
    # **cfg.get('adapter_kwargs', {}) - распаковка словаря гиперпараметров
    adapter.adapt(X_t, y_t, **cfg.get('adapter_kwargs', {}))

    # переобучение GPR на Target
    if verbose:
        print(f" Переобучение GPR на остатках Target...")

    # адаптированная модель предсказывает на target
    # Для freeze_enc - с дообученным декодером.
    # Для dann - с домен-инвариантным энкодером.
    # Для none - та же, что раньше.
    y_pred_t_new = model.predict(X_t)
    # считает остатки LSTM на target
    residuals_t = y_t - y_pred_t_new
    residuals_t_mean = residuals_t.mean(axis=1)

    # извлекает контекст энкодера адаптированной модели
    context_t = model.get_context(X_t)
    X_gpr_t = extract_gpr_features(X_t, context_t)
    # созадем новый GPR
    corrector_target = CorrectorClass(alpha=cfg.get('gpr_alpha', 1e-6),
                                      auto_regularize=cfg.get('gpr_auto_regularize', True),
                                      verbose=verbose)
    corrector_target.fit(X_gpr_t, residuals_t_mean)

    # оценка после
    wrapped_after = ModelWithGPR(model, corrector_target)
    y_pred_t_after = wrapped_after.predict(X_t)
    y_pred_after_denorm = scalers.inverse_y(y_pred_t_after, role='target')

    mae_after = compute_mae(y_t_denorm, y_pred_after_denorm)
    rmse_after = compute_rmse(y_t_denorm, y_pred_after_denorm)

    if verbose:
        print(f"[ПОСЛЕ] MAE:  {mae_after['mae_abs']:.4f} Ah "
              f"({mae_after['mae_pct']:.2f}%)")
        print(f"[ПОСЛЕ] RMSE: {rmse_after['rmse_abs']:.4f} Ah "
              f"({rmse_after['rmse_pct']:.2f}%)")

    # липсиц - мера чувствительности модели к изменениям входа
    lipschitz = evaluate_lipschitz(wrapped_after, X_t, epsilon=cfg.get('lipschitz_epsilon', 0.01),
                                   n_trials=cfg.get('lipschitz_trials', 10))
    if verbose:
        print(f"Lipschitz L: {lipschitz['L_mean']:.4f} "
              f"→ {lipschitz['interpretation']}")

    #  κ(K) - число обусловленности GPR
    # показывает обусловленность матрицы.
    # Если κ большое - обращение нестабильно. GPR может давать плохие предсказания.
    gpr_stability = corrector_target.get_stability_report()

    # на сколько процентов MAE уменьшилось относительно исходного.
    improvement = ((mae_before['mae_pct'] - mae_after['mae_pct'])/ mae_before['mae_pct'] * 100
                   if mae_before['mae_pct'] > 0 else 0.0)

    return {'fold_idx': fold_idx,
            'mmd_before': mmd_before['mmd_sq'],
            'mmd_after': mmd_after['mmd_sq'],
            'final_loss': float(final_loss),
            'mae_before': mae_before['mae_pct'],
            'mae_after': mae_after['mae_pct'],
            'rmse_before': rmse_before['rmse_pct'],
            'rmse_after': rmse_after['rmse_pct'],
            'improvement': improvement,
            'lipschitz': lipschitz['L_mean'],
            'kappa': gpr_stability['kappa'] if gpr_stability else None,
            'n_source': len(fold['source_data']),
            'n_target': len(fold['target_data'])}
