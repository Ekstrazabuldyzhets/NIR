import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from bridges.bridge_programm import BRIDGE_REGISTRY

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, LSTM, Bidirectional, RepeatVector,Attention, Concatenate)

class BaseAdapter:
    def __init__(self, model, corrector, scalers, verbose=True):
        self.model = model
        self.corrector = corrector
        self.scalers = scalers
        self.verbose = verbose
        self.name = self.__class__.__name__
    # **kwargs в определении функции собирает все именованные аргументы в словарь
    def adapt(self, X_target, y_target, **kwargs):
        raise NotImplementedError

class NoneAdapter(BaseAdapter):
    def adapt(self, X_target, y_target, **kwargs):
        if self.verbose:
            print(f" Адаптация: none (zero-shot)")
        return self

class FreezeEncoderAdapter(BaseAdapter):
    def adapt(self, X_target, y_target, **kwargs):
        epochs_freeze = kwargs.get('epochs_freeze', 20)
        epochs_unfreeze = kwargs.get('epochs_unfreeze', 15)
        lr_freeze = kwargs.get('lr_freeze', 1e-3)
        lr_unfreeze = kwargs.get('lr_unfreeze', 1e-5)
        batch_size = kwargs.get('batch_size', 16)

        if self.verbose:
            print(f"Freeze Encoder: "
                  f"phase2 (lr={lr_freeze}, epochs={epochs_freeze}) → "
                  f"phase3 (lr={lr_unfreeze}, epochs={epochs_unfreeze})")

        # Фаза 2: заморозка энкодера декодер подстраивается под target
        self.model.freeze_encoder()
        self.model.recompile(learning_rate=lr_freeze)

        trainable, total = self.model.count_trainable()
        if self.verbose:
            print(f"[Фаза 2] Обучаемых: {trainable:,}/{total:,}")

        self.model.fit(X_target, y_target, epochs=epochs_freeze, batch_size=batch_size,
                       patience=10, validation_split=0.0, verbose=0)

        # Фаза 3: разморозка + донастройка размораживаем энкодер, дообучаем всю модель с маленьким lr
        self.model.unfreeze_encoder()
        self.model.recompile(learning_rate=lr_unfreeze)

        trainable, total = self.model.count_trainable()
        if self.verbose:
            print(f"[Фаза 3] Обучаемых: {trainable:,}/{total:,}")

        self.model.fit(X_target, y_target, epochs=epochs_unfreeze, batch_size=batch_size,
                       patience=10, validation_split=0.0, verbose=0)
        return self

class BridgeAdapter(BaseAdapter):
    # между енкодером и декодером
    def adapt(self, X_target, y_target, **kwargs):
        bridge_type = kwargs.get('bridge_type', 'rf')
        n_neighbors = kwargs.get('n_neighbors', 5)

        if self.verbose:
            print(f"    Bridge: type='{bridge_type}', "
                  f"n_neighbors={n_neighbors}")

        # получаем Source-данные
        X_source = getattr(self.model, '_X_source_scaled', None)
        y_source = getattr(self.model, '_y_source_scaled', None)

        if X_source is None or y_source is None:
            raise RuntimeError(
                "BridgeAdapter требует model._X_source_scaled и "
                "model._y_source_scaled."
            )

        # извлекаем контексты
        c_source_full = self.model.get_context(X_source)
        c_source = c_source_full[:, -1, :]
        c_target_full = self.model.get_context(X_target)
        c_target = c_target_full[:, -1, :]

        if self.verbose:
            print(f"      c_source: {c_source.shape}, "
                  f"c_target: {c_target.shape}")

        # обучаем мост
        BridgeClass = BRIDGE_REGISTRY[bridge_type]
        bridge = BridgeClass(n_neighbors=n_neighbors, verbose=self.verbose)
        bridge.fit(c_source, y_source, c_target, y_target)

        # созадем decoder-модель
        decoder_model = self.model.build_decoder_from_context()

        # устанавливаем мост
        self.model.set_bridge(bridge, decoder_model)

        return self

# DANN
class DANNAdapter(BaseAdapter):
    def adapt(self, X_target, y_target, **kwargs):
        epochs = kwargs.get('epochs', 25)
        batch_size = kwargs.get('batch_size', 16)
        lr = kwargs.get('lr', 1e-5)
        alpha_max = kwargs.get('alpha_max', 0.05)
        domain_hidden = kwargs.get('domain_hidden', 64)

        if self.verbose:
            print(f"    DANN: epochs={epochs}, lr={lr}, "
                  f"alpha_max={alpha_max}")

        # получить Source-данные
        X_source = getattr(self.model, '_X_source_scaled', None)
        y_source = getattr(self.model, '_y_source_scaled', None)
        if X_source is None or y_source is None:
            raise RuntimeError("DANNAdapter требует model._X_source_scaled и model._y_source_scaled.")

        # собрать DANN-модель
        if self.verbose:
            print(f"      Создание DANN-модели...")

        dann_model = self.model.build_dann(domain_hidden=domain_hidden)

        # отовит данные для adversarial-обучения
        # DANN обучается на обоих доменах, Source и target идут вперемешку
        n_source = len(X_source)
        n_target = len(X_target)

        d_source = np.zeros(n_source, dtype=np.int32)
        d_target = np.ones(n_target, dtype=np.int32)

        X_all = np.concatenate([X_source, X_target], axis=0)
        y_all = np.concatenate([y_source, y_target], axis=0)
        d_all = np.concatenate([d_source, d_target], axis=0)

        # цикл обучения с растущим alpha
        for epoch in range(epochs):
            alpha = alpha_max * (epoch / max(epochs - 1, 1))

            grl_layer = dann_model.get_layer('grl')
            # при обратном проходе градиент от классификатора умножается на - alpha
            grl_layer.set_alpha(alpha)

            #
            dann_model.compile(optimizer=Adam(learning_rate=lr),
                               loss={'soc_output': 'mse', 'domain_output': 'sparse_categorical_crossentropy'},
                               loss_weights={'soc_output': 1.0, 'domain_output': alpha},
                               metrics={'soc_output': ['mae']})

            # при обучении батчами важно, чтобы батчи перемешивались
            # иначе - порядок source, потом target
            idx = np.random.permutation(len(X_all))
            X_shuffled = X_all[idx]
            y_shuffled = y_all[idx]
            d_shuffled = d_all[idx]

            #
            dann_model.fit(X_shuffled, {'soc_output': y_shuffled, 'domain_output': d_shuffled}, epochs=1,
                           batch_size=batch_size, verbose=0)

            if self.verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"      Epoch {epoch + 1}/{epochs}, "
                      f"alpha={alpha:.4f}")

        # пересобрать обычную модель
        self.rebuild_standard_model(dann_model)
        return self
    # Пересборка обычной модели из DANN
    def rebuild_standard_model(self, dann_model):
        """Копирует веса из DANN-модели в основную модель."""
        inp = Input(shape=(None, self.model.input_dim), name='encoder_input')

        enc1 = Bidirectional(LSTM(self.model.hidden_units, return_sequences=True, dropout=0.2),
                             name='encoder_lstm1')(inp)
        enc2 = Bidirectional(LSTM(self.model.hidden_units, return_sequences=True, dropout=0.2),
                             name='encoder_lstm2')(enc1)

        context = enc2[:, -1, :]
        dec_in = RepeatVector(1)(context)

        dec1 = Bidirectional(LSTM(self.model.hidden_units, return_sequences=True, dropout=0.2),
                             name='decoder_lstm1')(dec_in)

        attn = Attention(name='attention')([dec1, enc2])
        comb = Concatenate(name='concat')([dec1, attn])

        dec2 = Bidirectional(LSTM(self.model.hidden_units, return_sequences=False, dropout=0.2),
                             name='decoder_lstm2')(comb)

        out = Dense(self.model.output_dim, name='output')(dec2)

        new_model = Model(inp, out)

        # Копируем веса
        copy_bidirectional(dann_model.get_layer('encoder_lstm1'),new_model.get_layer('encoder_lstm1'))
        copy_bidirectional(dann_model.get_layer('encoder_lstm2'),new_model.get_layer('encoder_lstm2'))
        copy_bidirectional(dann_model.get_layer('dann_decoder_lstm1'),new_model.get_layer('decoder_lstm1'))
        copy_bidirectional(dann_model.get_layer('dann_decoder_lstm2'),new_model.get_layer('decoder_lstm2'))

        new_model.get_layer('output').set_weights(dann_model.get_layer('soc_output').get_weights())

        # Заменяем основную модель
        self.model.model = new_model
        self.model.model.compile(optimizer=Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])
        # Сбрасываем кеши
        self.model._context_model = None
        self.model._decoder_model = None

def copy_bidirectional(src_layer, dst_layer):
    dst_layer.forward_layer.set_weights(src_layer.forward_layer.get_weights())
    dst_layer.backward_layer.set_weights(src_layer.backward_layer.get_weights())

ADAPTER_REGISTRY = {'none': NoneAdapter, 'freeze_enc': FreezeEncoderAdapter,'bridge': BridgeAdapter,'dann': DANNAdapter}