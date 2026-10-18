"""
Разбиваем АВТО, на окна.
Source и Target меняются на каждом фолде.
Два режима:
    1. same_file:      Source и Target из одного пула авто
    2. separate_files: Source фиксирован, Target разбивается
"""
import numpy as np
from sklearn.model_selection import KFold

class CVSplitter:
    def __init__(self, n_folds: int = 10, random_state: int = 42, mode: str = 'same_file', verbose: bool = True):
        self.n_folds = n_folds
        self.random_state = random_state
        self.mode = mode
        self.verbose = verbose

    # Режим 1: один пул — разбиваем и source, и target
    def split_same_file(self, pool_data):
        n_vehicles = len(pool_data)

        if n_vehicles < self.n_folds:
            raise ValueError(f"Авто ({n_vehicles}) меньше, чем фолдов ({self.n_folds})")
        # берёт индексы [0, 1, 2, ..., 19], перемешивает их, потом делит на 10 групп по 2 индекса.
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        folds = []
        # kf.split(pool_data) - генератор. Даёт 10 пар (train_idx, val_idx)
        # rain_idx - массив 18 индексов.
        # val_idx - массив 2 индексов
        # enumerate(...) - добавляет номер фолда (0, 1, ..., 9).
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(pool_data)):
            fold_source = [pool_data[i] for i in train_idx]
            fold_target = [pool_data[i] for i in val_idx]

            folds.append({'fold_idx': fold_idx, 'source_data': fold_source,'target_data': fold_target})

        return folds

    # Режим 2: source фиксирован, target разбивается
    def split_separate_files(self, source_data, target_data):
        n_target = len(target_data)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(target_data)):
            fold_target = [target_data[i] for i in val_idx]
            folds.append({'fold_idx': fold_idx, 'source_data': source_data, 'target_data': fold_target,})

        return folds

    # Публичный API
    def split(self, source_data, target_data) -> list:
        if self.mode == 'same_file':
            folds = self.split_same_file(source_data)
        else:
            folds = self.split_separate_files(source_data, target_data)

        if self.verbose:
            print(f"Режим: {self.mode}")
            print(f"Фолдов: {len(folds)}")
            for f in folds:
                n_s = len(f['source_data'])
                n_t = len(f['target_data'])
                s_seq = sum(d['X'].shape[0] for d in f['source_data'])
                t_seq = sum(d['X'].shape[0] for d in f['target_data'])
                print(f"    Fold {f['fold_idx'] + 1}: "
                      f"source={n_s} авто ({s_seq} окон), "
                      f"target={n_t} авто ({t_seq} окон)")

        return folds