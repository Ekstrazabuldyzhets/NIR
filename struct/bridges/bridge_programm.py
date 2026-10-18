import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# задаем интерфейс все мосты должны иметь fit и transform
class BaseBridge:
    def fit(self, c_source, y_source, c_target, y_target):
        raise NotImplementedError

    def transform(self, c_target):
        raise NotImplementedError

# мост случайный лес
class RFBridge(BaseBridge):
    def __init__(self, n_estimators=100, max_depth=20, min_samples_split=5, n_neighbors=5,
                 random_state=42, clip_to_source=True, verbose=True):
        self.n_estimators = n_estimators # число деревьев в Random Forest
        self.max_depth = max_depth # максимальная глубина дерева
        self.min_samples_split = min_samples_split
        self.n_neighbors = n_neighbors # сколько ближайших source-окон брать для каждого target
        self.random_state = random_state
        self.clip_to_source = clip_to_source # обрезать ли предсказания в диапазон source-контекстов
        self.verbose = verbose

        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_split=min_samples_split, random_state=random_state, n_jobs=-1)
        # нормализует
        self.scaler_c_s = StandardScaler()
        self.scaler_c_t = StandardScaler()

        self.is_fitted = False # пока False - мост не обучен
        self.context_dim = None # размерность контекста (128)
        self.source_contexts = None # сохраненные source-контексты
        self.train_pairs_count = 0 # число пар

    # построение пар
    def build_pairs(self, c_source, y_source, c_target, y_target):
        # приведение к 2D
        c_source = np.asarray(c_source).reshape(len(c_source), -1)
        c_target = np.asarray(c_target).reshape(len(c_target), -1)
        y_source = np.asarray(y_source).reshape(len(y_source), -1)
        y_target = np.asarray(y_target).reshape(len(y_target), -1)

        pairs_c_t = [] # будет хранить target-контексты (вход моста)
        pairs_c_s = [] # будет хранить source-контексты (выход моста)

        # цикл по всем таргет окнам находя его пару source
        for i in range(len(c_target)):
            y_t = y_target[i] # берем y-вектор target-окна i
            y_diff = np.linalg.norm(y_source - y_t, axis=1) # расстояния от y_t до всех source-y
            # находим индексы n_neighbors ближайших source-окон
            closest_idx = np.argsort(y_diff)[:self.n_neighbors]

            # создание пар(цикл по 5 ближайшим)
            for idx in closest_idx:
                pairs_c_t.append(c_target[i].flatten())
                pairs_c_s.append(c_source[idx].flatten())

        return np.array(pairs_c_t), np.array(pairs_c_s)

    # Обучение
    def fit(self, c_source, y_source, c_target, y_target):
        # приведение к 2D
        c_source = np.asarray(c_source).reshape(len(c_source), -1)
        c_target = np.asarray(c_target).reshape(len(c_target), -1)

        self.context_dim = c_source.shape[-1]
        self.source_contexts = c_source # сохраняем для clip при transform

        # построение пар (n пар возвращает)
        pairs_c_t, pairs_c_s = self.build_pairs(c_source, y_source, c_target, y_target)
        self.train_pairs_count = len(pairs_c_t)

        if self.train_pairs_count == 0:
            raise ValueError("Не удалось построить пары для моста")

        c_t_scaled = self.scaler_c_t.fit_transform(pairs_c_t) # нормализованные target-контексты
        c_s_scaled = self.scaler_c_s.fit_transform(pairs_c_s) # нормализованные source-контексты

        # обучаем случайный лес
        self.model.fit(c_t_scaled, c_s_scaled)
        self.is_fitted = True

        if self.verbose:
            print(f"Мост: обучено пар {self.train_pairs_count}, "
                  f"dim={self.context_dim}")

        return self

    # применение
    # берёт новые target-контексты и превращает их в синтетические source-контексты
    def transform(self, c_target):
        # проверка обучен ли мост
        if not self.is_fitted:
            return c_target

        c_target = np.asarray(c_target).reshape(len(c_target), -1)
        c_t_scaled = self.scaler_c_t.transform(c_target)

        # для каждого target-контекста - похожий source-контекст
        c_s_pred_scaled = self.model.predict(c_t_scaled)
        c_s_pred = self.scaler_c_s.inverse_transform(c_s_pred_scaled)

        # Random Forest может выйти за пределы обучающих данных
        # Декодер не видел таких контекстов. Clip защищает от экстраполяции
        # source_contexts[:, 0] имеет диапазон [0.1, 0.5]
        # c_s_pred[:, 0] = 0.7  ← вне диапазона
        # После clip: 0.5       ← обрезано
        if self.clip_to_source and self.source_contexts is not None:
            c_min = self.source_contexts.min(axis=0)
            c_max = self.source_contexts.max(axis=0)
            c_s_pred = np.clip(c_s_pred, c_min, c_max)

        return c_s_pred

# типы мостов
BRIDGE_REGISTRY = {'rf': RFBridge,}
