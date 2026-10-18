import numpy as np
from sklearn.preprocessing import StandardScaler

# MMD compute_mmd(X_source, X_target, n_samples) -> dict
# MMD² - метрика различия распределений. Чем больше, тем сильнее домены различаются.
def compute_mmd(X_source, X_target, n_samples=500, verbose=True):
    # считает число сэмплов в каждом наборе (315, 6, 9) → 315.
    n_source = len(X_source)
    n_target = len(X_target)

    # превращает (315, 6, 9) в (315, 54).
    # MMD работает в RKHS - пространстве признаков. Форма входа - вектор.
    # Мы сплющиваем окно в вектор 54-мерный.
    X_s = X_source.reshape(n_source, -1)
    X_t = X_target.reshape(n_target, -1)

    # субдискретизация (для скорости) MMD требует одинакового числа сэмплов из P и Q для несмещённой оценки.
    # Иначе формула усложняется. min гарантирует одинаковое n.
    n = min(n_samples, n_source, n_target)
    # случайная выборка несмещённая.
    idx_s = np.random.choice(n_source, n, replace=False)
    idx_t = np.random.choice(n_target, n, replace=False)
    # возвращает новый массив из выбранных строк.
    X_s = X_s[idx_s]
    X_t = X_t[idx_t]

    # gamma через median heuristic
    gamma = median_heuristic(X_s, X_t)

    # ядра RBF измеряет «похожесть» двух точек. Через ядро сравниваются средние распределений.
    K_ss = rbf_kernel(X_s, X_s, gamma) # похожести внутри source
    K_tt = rbf_kernel(X_t, X_t, gamma)
    K_st = rbf_kernel(X_s, X_t, gamma)

    # MMD² = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]
    mmd_sq = float(K_ss.mean() + K_tt.mean() - 2 * K_st.mean())
    interpretation = interpret(mmd_sq)

    if verbose:
        print(f"MMD² = {mmd_sq:.4f}  |  {interpretation}")

    return {'mmd_sq': mmd_sq, 'interpretation': interpretation, 'gamma': gamma}

def median_heuristic(X_s, X_t):
    # gamma = 1 / median(||x_i - x_j||²)
    # медиана попарных расстояний - стандартный способ выбора gamma
    # объединяет X_s и X_t в один массив
    X_all = np.vstack([X_s, X_t])
    # считает квадрат нормы каждой точки
    sq = np.sum(X_all ** 2, axis=1)
    # sq - вектор (351), sq[:, None] - превращает в столбец, sq[None, :] - превращает в строку
    dist_sq = sq[:, None] + sq[None, :] - 2 * X_all @ X_all.T
    # на диагонали dist_sq[i][i] = 0 - расстояние от точки до самой себя.
    # Такие нули испортят медиану: медиана считается только по разным точкам.
    mask = dist_sq > 0
    median_dist = np.median(dist_sq[mask])
    # с защитой от деления на 0
    return 1.0 / (median_dist + 1e-8)

def rbf_kernel(X, Y, gamma):
    # RBF-ядро: k(x, y) = exp(-gamma * ||x - y||²)
    # Использует трюк ||x - y||² = ||x||² + ||y||² - 2*x·y
    # для быстрого вычисления матрицы расстояний

    # считаем квадрат нормы каждой точки X
    XX = np.sum(X ** 2, axis=1)[:, None]     # (n, 1)
    YY = np.sum(Y ** 2, axis=1)[None, :]     # (1, m)
    XY = X @ Y.T                              # (n, m)
    dist_sq = XX + YY - 2 * XY                # (n, m)
    return np.exp(-gamma * dist_sq)

def interpret(mmd_sq):
    if mmd_sq < 0.1:
        return "Домены похожи - alignment НЕ нужен"
    elif mmd_sq < 0.5:
        return "Умеренное различие - alignment полезен"
    else:
        return "Сильное различие - alignment ОБЯЗАТЕЛЕН"

# нормализация (раздельно для доменов)
class DomainScalers:
    ROLES = ('source', 'target')

    def __init__(self):
        self.scaler_X = {r: StandardScaler() for r in self.ROLES}
        self.scaler_y = {r: StandardScaler() for r in self.ROLES}
        self.is_fitted = False

    def _check_role(self, role: str):
        if role not in self.ROLES:
            raise ValueError(f"role должен быть один из {self.ROLES}, "
                             f"получено: {role}")

    # обучает 4 scaler'а.
    # считаем μ и σ по обучающим данным
    def fit(self, X_source, y_source, X_target, y_target):
        self.scaler_X['source'].fit(X_source.reshape(-1, X_source.shape[-1]))
        self.scaler_X['target'].fit(X_target.reshape(-1, X_target.shape[-1]))
        self.scaler_y['source'].fit(y_source.reshape(-1, 1))
        self.scaler_y['target'].fit(y_target.reshape(-1, 1))
        self.is_fitted = True
        return self

    # рименяем формулу (x - μ) / σ с уже посчитанными μ и σ.
    def transform_X(self, X, role: str) -> np.ndarray:
        # проверка что fit был вызван
        self._check_role(role)
        if not self.is_fitted:
            raise RuntimeError("DomainScalers не обучен")
        shape = X.shape
        X_flat = X.reshape(-1, shape[-1])
        X_scaled = self.scaler_X[role].transform(X_flat)
        return X_scaled.reshape(shape).astype(np.float32)

    def transform_y(self, y, role: str) -> np.ndarray:
        self._check_role(role)
        if not self.is_fitted:
            raise RuntimeError("DomainScalers не обучен")
        shape = y.shape
        y_flat = y.reshape(-1, 1)
        y_scaled = self.scaler_y[role].transform(y_flat)
        return y_scaled.reshape(shape).astype(np.float32)

    # денормализация
    # модель предсказывает y. Но предсказание - нормализованное
    # чтобы сравнить с истинным y (в Ah) - нужна денормализация
    def inverse_y(self, y_scaled, role: str) -> np.ndarray:
        self._check_role(role)
        if not self.is_fitted:
            raise RuntimeError("DomainScalers не обучен")
        shape = y_scaled.shape
        y_flat = y_scaled.reshape(-1, 1)
        y_denorm = self.scaler_y[role].inverse_transform(y_flat)
        return y_denorm.reshape(shape).astype(np.float32)

# Coral
def matrix_sqrt(C, power):
    # Считает C^power через SVD.
    U, S, _ = np.linalg.svd(C)
    S = np.maximum(S, 1e-8)
    return U @ np.diag(S ** power) @ U.T

class DomainAligner:
    # X_s_aligned = (X_s - mu_s) · C_s^{-1/2} · C_t^{1/2} + mu_t
    def __init__(self, eps: float = 1e-5, verbose: bool = True):
        self.eps = eps
        self.verbose = verbose

        self.mu_source = None # среднее source
        self.mu_target = None # среднее target
        self.S_sqrt_inv = None
        self.T_sqrt = None

    def fit(self, X_source, X_target):
        n_s = len(X_source)
        n_t = len(X_target)
        # reshape в 2D
        X_s = X_source.reshape(n_s, -1)
        X_t = X_target.reshape(n_t, -1)
        # среднее по строкам
        self.mu_source = X_s.mean(axis=0)
        self.mu_target = X_t.mean(axis=0)
        # центрирование так как ковариация считается по центрированным данным
        X_s_c = X_s - self.mu_source
        X_t_c = X_t - self.mu_target

        # X_s_c.T @ X_s_c - сумма квадратов отклонений
        C_s = (X_s_c.T @ X_s_c) / (n_s - 1) + self.eps * np.eye(X_s.shape[1])
        C_t = (X_t_c.T @ X_t_c) / (n_t - 1) + self.eps * np.eye(X_t.shape[1])

        # C_s^{-1/2}
        self.S_sqrt_inv = matrix_sqrt(C_s, -0.5)
        # C_t^{1/2}
        self.T_sqrt = matrix_sqrt(C_t, 0.5)

        return self

    def transform_source(self, X_source) -> np.ndarray:
        # запоминаем исходную формулу чтоб в конце вернуть в той же форме
        shape = X_source.shape
        X_s = X_source.reshape(shape[0], -1)
        # вычисляем среднее
        X_s_c = X_s - self.mu_source
        X_aligned = X_s_c @ self.S_sqrt_inv @ self.T_sqrt + self.mu_target
        # после преобразования ковариация source совпадёт с target
        return X_aligned.reshape(shape).astype(np.float32)

    # только приводит тип
    def transform_target(self, X_target) -> np.ndarray:
        return X_target.astype(np.float32)

    def alignment_error(self, X_source, X_target) -> dict:
        # метод проверки качества выравнивания
        X_s_aligned = self.transform_source(X_source)
        X_s_flat = X_s_aligned.reshape(len(X_s_aligned), -1)
        X_t_flat = X_target.reshape(len(X_target), -1)

        mu_diff = float(np.linalg.norm(X_s_flat.mean(axis=0) - X_t_flat.mean(axis=0)))
        C_s = np.cov(X_s_flat.T)
        C_t = np.cov(X_t_flat.T)
        cov_diff = float(np.linalg.norm(C_s - C_t, 'fro'))

        return {'mu_diff': mu_diff, 'cov_diff': cov_diff}