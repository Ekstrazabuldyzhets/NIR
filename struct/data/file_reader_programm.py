import os
import pickle
import numpy as np
from typing import Optional

def load_data(spec: dict, role: str = 'data', verbose: bool = True) -> list:

    if 'path' not in spec:
        raise ValueError("spec должен содержать ключ 'path'")
    # извлечение параметров
    path = spec['path']
    # берём selector. Если нет - default {'type': 'all'}.
    selector = spec.get('selector', {'type': 'all'})

    # читаем
    raw = load_raw(path, role)

    # нормализуем каждую запись
    normalized = []
    for i, entry in enumerate(raw):
        n = normalize_entry(entry, i)
        if n is not None:
            normalized.append(n)

    # применяем selector
    selected = apply_selector(normalized, selector, role)

    # печатаем сводку
    if verbose and selected:
        print_summary(selected, role)

    return selected

# Склеивает список словарей в два массива (X, y).
def stack_data(data: list) -> tuple:
    # fold['source_data'] - 18 словарей, каждый с X и y
    # [d['X'] for d in data] - список 18 массивов X
    # np.concatenate(..., axis=0) - склеивает их по первой оси
    # Получается один большой массив X

    # 18 авто, у каждого своё число окон:
    # X.shape: (17, 6, 9), (20, 6, 9), (15, 6, 9), ...
    # После concatenate: (315, 6, 9)  ← 17+20+15+... = 315
    # y после concatenate: (315, 6)
    if not data:
        return np.array([]), np.array([])

    X = np.concatenate([d['X'] for d in data], axis=0)
    y = np.concatenate([d['y'] for d in data], axis=0)
    return X, y

# это нужно будет поменять
def load_raw(path: str, role: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{role}] Файл не найден: {path}")
    # В файле all_vehicles.pkl лежит список из 20 словарей.
    # Каждый словарь - один автомобиль. Внутри словаря есть ключи X, y и, возможно, другие поля (vehicle_id, meta).
    # Когда мы вызываем pickle.load(f), Python читает этот файл.
    # Восстанавливает: сначала список, потом 20 словарей, внутри каждого - numpy-массивы X и y.
    # В результате pickle.load(f) возвращает нам тот же список 20 словарей,
    # который кто-то сохранил ранее через pickle.dump.
    with open(path, 'rb') as f:
        return pickle.load(f)

# нужна для работы с pkl нужно будет поменять
def normalize_entry(entry, idx: int) -> Optional[dict]:
    if not isinstance(entry, dict):
        return None

    if 'X' not in entry or 'y' not in entry:
        return None

    # приведение к массиву и приводит тип к float32 для экономии памяти
    # (в pkl могут быть списки или другие типы - надо унифицировать)
    X = np.asarray(entry['X'], dtype=np.float32)
    y = np.asarray(entry['y'], dtype=np.float32)

    # Гарантируем 3D для X и 2D для y
    if X.ndim == 2:
        X = X[None, ...]
    if y.ndim == 1:
        y = y[None, ...]

    if X.ndim != 3 or y.ndim != 2:
        return None
    if X.shape[0] != y.shape[0]:
        return None

    vehicle_id = entry.get('vehicle_id', entry.get('id', idx))
    meta = {k: v for k, v in entry.items() if k not in ('X', 'y')}

    return {
        'vehicle_id': vehicle_id,
        'X': X,
        'y': y,
        'meta': meta,
    }

def apply_selector(entries: list, selector: dict, role: str) -> list:
    # Функция получает полный список записей и правило выбора.
    # Смотрит на type в правиле. Если 'all' - возвращает всё.
    # Если 'indices' - берёт указанные. Если 'range' - берёт диапазон.
    # Если что-то другое - падает с ошибкой.
    # Три варианта - три сценария использования:
    # all - «мне нужны все 20 авто». Удобно, когда разделение делает CV.
    # indices - «мне нужны авто 0, 3, 5». Удобно для специфичных экспериментов.
    # range - «мне нужны авто с 0 по 10». Удобно для «первая половина / вторая половина».
    sel_type = selector.get('type', 'all')

    # Если sel_type == 'all' - возвращаем список как есть.
    # когда нужны все авто. Например, в cv_mode='same_file'
    # мы читаем все 20, а CV сам разделит на source/target.
    if sel_type == 'all':
        return entries

    elif sel_type == 'indices':
        idx = selector['value']
        return [entries[i] for i in idx if 0 <= i < len(entries)]

    elif sel_type == 'range':
        start = selector.get('start', 0)
        end = selector.get('end', len(entries))
        return entries[start:end]

    else:
        raise ValueError(f"[{role}] Неизвестный selector: {sel_type}")

def print_summary(selected: list, role: str):
    n_seq = sum(d['X'].shape[0] for d in selected)
    L = selected[0]['X'].shape[1]
    d = selected[0]['X'].shape[2]
    P = selected[0]['y'].shape[1]

    print(f"[{role:<6}] {len(selected):>2} авто | "
          f"{n_seq:>3} окон | "
          f"X=({n_seq}, {L}, {d}) | y=({n_seq}, {P})")