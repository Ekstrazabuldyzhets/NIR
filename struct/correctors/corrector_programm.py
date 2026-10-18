import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, WhiteKernel, ConstantKernel)
from sklearn.preprocessing import StandardScaler

def extract_gpr_features(X, context):
    N = X.shape[0]
    features = []

    for i in range(N):
        seq = X[i]
        ctx_last = context[i, -1, :8]  # первые 8 компонент последнего шага

        feat = list(ctx_last)
        feat.append(seq[-1, 0])  # последнее значение 1-го признака
        feat.append(seq[-1, 5])  # последнее значение 6-го
        feat.append(seq[-1, 6])  # последнее значение 7-го
        feat.append(np.mean(seq[:, 0]))  # среднее 1-го признака
        feat.append(np.std(seq[:, 0]))  # std 1-го признака
        features.append(feat)

    return np.array(features, dtype=np.float32) # (315, 13)

# MULTI-SCALE GPR
class MultiScaleGPR:
    """
    GPR с multi-scale RBF ядром и авторегуляризацией.
    k(x, x') = C1·RBF(ℓ=0.5) + C2·RBF(ℓ=2.0) + C3·RBF(ℓ=5.0) + WhiteKernel
    """
    def __init__(self, alpha: float = 1e-6, auto_regularize: bool = True,
                 verbose: bool = True):
        # Multi-scale ядро
        kernel_short = ConstantKernel(1.0) * RBF(length_scale=0.5)
        kernel_medium = ConstantKernel(1.0) * RBF(length_scale=2.0)
        kernel_long = ConstantKernel(1.0) * RBF(length_scale=5.0)
        self.kernel = (kernel_short + kernel_medium + kernel_long + WhiteKernel(0.1))

        self.alpha = alpha
        self.auto_regularize = auto_regularize
        self.verbose = verbose

        self.gpr = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, alpha=alpha,
                                            normalize_y=True, random_state=42)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.is_fitted = False
        self.stability_info = None

    # Проверка числа обусловленности
    def _check_condition_number(self, K):
        """κ(K) = σ_max / σ_min."""
        sigma = np.linalg.svd(K, compute_uv=False)
        kappa = sigma[0] / (sigma[-1] + 1e-15)
        return kappa, sigma[0], sigma[-1]

    def _interpret_kappa(self, kappa):
        if kappa < 1e6:
            return "OK (устойчив)"
        elif kappa < 1e10:
            return "Нужна регуляризация"
        else:
            return "НЕУСТОЙЧИВ"

    # Обучение
    def fit(self, X, y):
        # Обучает GPR на остатках.
        # X: (N, 13) - фичи (из extract_gpr_features)
        # y: (N,) - остатки (средние по 6 шагам)
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Проверка κ(K) считает матрицу ядра и её число обусловленности
        # Если большое - матрица плохо обусловлена, обращение нестабильно
        K = self.gpr.kernel(X_scaled)
        kappa, sigma_max, sigma_min = self._check_condition_number(K)

        self.stability_info = {'kappa': kappa, 'sigma_max': sigma_max, 'sigma_min': sigma_min,
                               'interpretation': self._interpret_kappa(kappa), 'alpha_used': self.alpha}

        # Авторегуляризация если κ(K) > 1e10 - увеличить α (регуляризацию) с 1e-6 до 1e-3
        if self.auto_regularize and kappa > 1e10:
            if self.verbose:
                print(f" κ(K) = {kappa:.2e} > 1e10 → "
                      f"увеличена регуляризация (alpha = 1e-3)")
            self.alpha = 1e-3
            self.gpr.alpha = self.alpha
            self.stability_info['alpha_used'] = self.alpha

        # стандартный sklearn-fit оптимизирует гиперпараметры ядра
        self.gpr.fit(X_scaled, y_scaled)
        self.is_fitted = True

        if self.verbose:
            print(f"    GPR: κ(K) = {kappa:.2e} → "
                  f"{self.stability_info['interpretation']}")

        return self

    # Предсказание
    def predict(self, X):
        if not self.is_fitted:
            return np.zeros(len(X))
        # ормализуем признаки тем же scaler'ом, что при обучении
        X_scaled = self.scaler_X.transform(X)
        #  GPR применяет то, что выучил
        y_pred = self.gpr.predict(X_scaled)
        # денормализуем обратно в исходный масштаб остатков
        return self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    # Отчёт об устойчивости
    def get_stability_report(self):
        return self.stability_info

CORRECTOR_REGISTRY = {'gpr': MultiScaleGPR}
