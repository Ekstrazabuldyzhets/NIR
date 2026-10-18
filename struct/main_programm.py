import os
import numpy as np

from data.file_reader_programm import load_data
from data.cv_splitter_programm import CVSplitter
from orchestration.fold_runner_programm import run_fold

# Путь к данным (можно переопределить через env)
DATA_DIR = os.environ.get('DATA_DIR', '/Users/nierra/Desktop/диплом-2/реальные данные/данные_20_авто/prepared_data')
VEHICLES_PKL = f'{DATA_DIR}/all_vehicles.pkl'

# Метод адаптации: 'none' | 'freeze_enc' | 'bridge' | 'dann'
ADAPT_METHOD = 'dann'
# Задаем гиперпараметры для каждого метода адаптации
ADAPTER_KWARGS = {
    'none': {},
    'freeze_enc': {
        'epochs_freeze': 20,
        'lr_freeze': 1e-3,
        'epochs_unfreeze': 15,
        'lr_unfreeze': 1e-5,
        'batch_size': 16,
    },
    'bridge': {
        'bridge_type': 'rf',
        'n_neighbors': 5,
    },
    'dann': {
        'epochs': 25,
        'lr': 1e-5,
        'alpha_max': 0.05,
        'batch_size': 16,
        'domain_hidden': 64,
    },
}
# Основной конфиг эксперимента
CONFIG = {
    # Данные
    # Сейчас — один файл (cv_mode='same_file'):
    # Source и Target берутся из одного пула, делятся через CV
    # Позже — можно указать два разных файла (cv_mode='separate_files')
    'data_source': {'path': VEHICLES_PKL, 'selector': {'type': 'all'}},
    'data_target': {'path': VEHICLES_PKL, 'selector': {'type': 'all'}},
    # Кросс-валидация
    'cv_mode': 'same_file',      # 'same_file' | 'separate_files'
    'n_folds': 10,               # число фолдов
    'random_state': 42,          # seed для воспроизводимости
    # Preprocessing
    'mmd_n_samples': 500,        # субдискретизация для MMD
    'use_coral': True,           # выравнивание ковариаций Source → Target
    # Модель
    'model': 'lstm_attention',   # имя модели из MODEL_REGISTRY
    'hidden_units': 64,
    'dropout': 0.2,
    'epochs_source': 300,
    'batch_size_source': 32,
    'learning_rate_source': 1e-4,
    'patience_source': 30,       # patience для EarlyStopping
    'validation_split_source': 0.1,
    # Corrector (GPR)
    # имя корректора из CORRECTOR_REGISTRY
    'corrector': 'gpr',
    # шум GPR
    'gpr_alpha': 1e-6,
    # авто-увеличение alpha при κ(K) > 1e10
    'gpr_auto_regularize': True,
    # Lipschitz
    'lipschitz_epsilon': 0.01,
    'lipschitz_trials': 10,
    # Адаптация
    'adapt_method': ADAPT_METHOD,
    'adapter_kwargs': ADAPTER_KWARGS[ADAPT_METHOD],
    # Прочее
    'verbose': True,
}

# Вспомогательные функции
def read_data(cfg):
    # два варианта когда мы берем данные из одного только файла или из двух разных
    if cfg['cv_mode'] == 'same_file':
        pool = load_data(cfg['data_source'], role='Pool', verbose=cfg['verbose'])
        return pool, pool
    source = load_data(cfg['data_source'], role='Source', verbose=cfg['verbose'])
    target = load_data(cfg['data_target'], role='Target', verbose=cfg['verbose'])
    return source, target

# запускаем
def run_all_folds(folds, cfg):
    results = []
    # fold - один фолд: {'fold_idx': 0, 'source_data': [18 авто], 'target_data': [2 авто]}; cfg - конфиг
    for fold in folds:
        results.append(run_fold(fold, cfg))
    return results

def aggregate(fold_results):
    def _vals(key):
        return [r[key] for r in fold_results if r.get(key) is not None]

    def _mean(v):
        return float(np.mean(v)) if v else None

    def _std(v):
        return float(np.std(v)) if v else None

    return {
        'n_folds': len(fold_results),
        'mmd_before_mean': _mean(_vals('mmd_before')),
        'mmd_before_std': _std(_vals('mmd_before')),
        'mmd_after_mean': _mean(_vals('mmd_after')),
        'mmd_after_std': _std(_vals('mmd_after')),
        'loss_mean': _mean(_vals('final_loss')),
        'loss_std': _std(_vals('final_loss')),
        'mae_before_mean': _mean(_vals('mae_before')),
        'mae_before_std': _std(_vals('mae_before')),
        'mae_after_mean': _mean(_vals('mae_after')),
        'mae_after_std': _std(_vals('mae_after')),
        'rmse_before_mean': _mean(_vals('rmse_before')),
        'rmse_after_mean': _mean(_vals('rmse_after')),
        'improvement_mean': _mean(_vals('improvement')),
        'improvement_std': _std(_vals('improvement')),
        'lipschitz_mean': _mean(_vals('lipschitz')),
        'lipschitz_std': _std(_vals('lipschitz')),
        'kappa_mean': _mean(_vals('kappa')),
        'fold_results': fold_results}

def print_results(summary, cfg):
    print(f"  Фолдов:            {summary['n_folds']}")
    print(f"  Режим CV:          {cfg['cv_mode']}")
    print(f"  CORAL:             {cfg['use_coral']}")
    print(f"  Модель:            {cfg['model']}")
    print(f"  Corrector:         {cfg['corrector']}")
    print(f"  Метод адаптации:   {cfg['adapt_method']}")
    print()
    print(f"  MMD² до:           {summary['mmd_before_mean']:.4f} "
          f"± {summary['mmd_before_std']:.4f}")
    print(f"  MMD² после:        {summary['mmd_after_mean']:.4f} "
          f"± {summary['mmd_after_std']:.4f}")
    print(f"  Loss:              {summary['loss_mean']:.6f} "
          f"± {summary['loss_std']:.6f}")
    print()
    print(f"  MAE ДО:            {summary['mae_before_mean']:.2f}% "
          f"± {summary['mae_before_std']:.2f}%")
    print(f"  MAE ПОСЛЕ:         {summary['mae_after_mean']:.2f}% "
          f"± {summary['mae_after_std']:.2f}%")
    print(f"  RMSE ДО:           {summary['rmse_before_mean']:.2f}%")
    print(f"  RMSE ПОСЛЕ:        {summary['rmse_after_mean']:.2f}%")
    print(f"  Улучшение:         {summary['improvement_mean']:.1f}% "
          f"± {summary['improvement_std']:.1f}%")
    print(f"  Lipschitz:         {summary['lipschitz_mean']:.4f} "
          f"± {summary['lipschitz_std']:.4f}")
    if summary['kappa_mean'] is not None:
        print(f"  κ(K):              {summary['kappa_mean']:.2e}")

    # Детали по фолдам
    print(f"\n  Детали по фолдам:")
    print(f"  {'Fold':<6} {'MAE до':<10} {'MAE после':<11} "
          f"{'Улучш.':<10} {'Lipschitz':<10} {'κ(K)':<10}")

    for r in summary['fold_results']:
        kappa_str = f"{r['kappa']:.2e}" if r.get('kappa') else "N/A"
        print(f"  {r['fold_idx'] + 1:<6} "
              f"{r['mae_before']:<10.2f} "
              f"{r['mae_after']:<11.2f} "
              f"{r['improvement']:<10.1f} "
              f"{r['lipschitz']:<10.4f} "
              f"{kappa_str:<10}")

def run_experiment(cfg):
    # данные
    source_data, target_data = read_data(cfg)
    # выполняем кросс-валидация(количество фолдов; 42; какой мод на файлы - из одного файла/из разных файлов; нужно ли вывадить информацию)
    splitter = CVSplitter(n_folds=cfg['n_folds'], random_state=cfg['random_state'], mode=cfg['cv_mode'], verbose=cfg['verbose'],)
    folds = splitter.split(source_data, target_data)
    # фолды
    fold_results = run_all_folds(folds, cfg)
    # агрегация
    return aggregate(fold_results)

# Запуск
if __name__ == '__main__':
    # запускаем эксперимент
    summary = run_experiment(CONFIG)
    print_results(summary, CONFIG)