import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, LSTM, Bidirectional, RepeatVector, Attention, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class LSTMAttentionModel:
    ENCODER_LAYERS = ['encoder_lstm1', 'encoder_lstm2']
    DECODER_LAYERS = ['decoder_lstm1', 'attention', 'decoder_lstm2', 'output']

    def __init__(self, input_dim: int, output_dim: int, hidden_units: int = 64, learning_rate: float = 1e-4, verbose: bool = True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.model = None
        self.context_model = None
        self.decoder_model = None
        self.bridge = None
        self.use_bridge = False

    # Сборка
    def build(self):
        inp = Input(shape=(None, self.input_dim), name='encoder_input')
        # энкодер реализуем в два слоя
        # (batch, 6, 9) → (batch, 6, 128);
        # (batch, 6, 128) → (batch, 6, 128);
        # (batch, 128)
        enc1 = Bidirectional(LSTM(self.hidden_units, return_sequences=True, dropout=0.2), name='encoder_lstm1')(inp)
        enc2 = Bidirectional(LSTM(self.hidden_units, return_sequences=True, dropout=0.2), name='encoder_lstm2')(enc1)
        context = enc2[:, -1, :]

        # декодер
        # (batch, 128) → (batch, 1, 128)
        dec_in = RepeatVector(1)(context)
        # (batch, 1, 128) → (batch, 1, 128)
        dec1 = Bidirectional(LSTM(self.hidden_units, return_sequences=True, dropout=0.2), name='decoder_lstm1')(dec_in)
        attn = Attention(name='attention')([dec1, enc2])
        comb = Concatenate(name='concat')([dec1, attn])
        # (batch, 1, 256) → (batch, 128)
        dec2 = Bidirectional(LSTM(self.hidden_units, return_sequences=False, dropout=0.2), name='decoder_lstm2')(comb)
        # (batch, 128) → (batch, 6)
        out = Dense(self.output_dim, name='output')(dec2)

        # компилируем
        self.model = Model(inp, out, name='lstm_attention')
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae'])

        if self.verbose:
            print(f" Модель собрана: параметров "
                  f"{self.model.count_params():,}")

        return self.model

    # Обучение
    def fit(self, X, y, epochs=100, batch_size=32, patience=30, validation_split=0.0, verbose=0):
        # Разбиение: 90% train, 10% val
        # Обучение 300 эпох
        if self.model is None:
            raise RuntimeError("Сначала вызови build()")

        callbacks = []
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=15, min_lr=1e-7, verbose=0))

        if patience and validation_split > 0:
            callbacks.append(tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True))

        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks, verbose=verbose)
        return history

    # Предсказание (с поддержкой моста)
    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Сначала вызови build()")

        # Если установлен мост - используем его
        if self.use_bridge and self.bridge is not None and self.decoder_model is not None:
            c_full = self.get_context(X) # (N, L, 128)
            c_last = c_full[:, -1, :] # (N, 128)
            c_transformed = self.bridge.transform(c_last) # вызываем
            y_pred = self.decoder_model.predict(c_transformed, verbose=0)
            return y_pred

        return self.model.predict(X, verbose=0)

    # Контекст вытаскивает промежуточный выход c encoder_lstm2
    def get_context(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Сначала вызови build()")

        if self.context_model is None:
            self.context_model = Model(inputs=self.model.input, outputs=self.model.get_layer('encoder_lstm2').output)

        return self.context_model.predict(X, verbose=0)

    def get_context_last(self, X) -> np.ndarray:
        return self.get_context(X)[:, -1, :]

    # Имена слоёв
    def get_encoder_layers(self) -> list:
        return list(self.ENCODER_LAYERS)

    def get_decoder_layers(self) -> list:
        return list(self.DECODER_LAYERS)

    # Заморозка / разморозка
    def set_trainable(self, layer_names: list, trainable: bool):
        for name in layer_names:
            self.model.get_layer(name).trainable = trainable

    def freeze_encoder(self):
        self.set_trainable(self.ENCODER_LAYERS, False)

    def unfreeze_encoder(self):
        self.set_trainable(self.ENCODER_LAYERS, True)

    # Перекомпиляция
    def recompile(self, learning_rate=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae'])

    # Decoder-only модель (для BridgeAdapter)
    def build_decoder_from_context(self):
        h = self.hidden_units * 2  # 128

        ctx = Input(shape=(h,), name='synthetic_context')
        dec_in = RepeatVector(1)(ctx)

        # decoder_lstm1
        dec_lstm1_orig = self.model.get_layer('decoder_lstm1')
        x = Bidirectional(LSTM(dec_lstm1_orig.forward_layer.units, return_sequences=True), name='dec_lstm1_copy')(dec_in)

        # Self-attention (вместо attention к энкодеру)
        attn = Attention(name='attn_copy')([x, x])
        comb = Concatenate(name='concat_copy')([x, attn])

        # decoder_lstm2
        dec_lstm2_orig = self.model.get_layer('decoder_lstm2')
        x2 = Bidirectional(LSTM(dec_lstm2_orig.forward_layer.units, return_sequences=False), name='dec_lstm2_copy')(comb)

        out = Dense(self.output_dim, name='output_copy')(x2)

        decoder_model = Model(ctx, out, name='decoder_from_context')

        # Копируем веса
        dec_lstm1_copy = decoder_model.get_layer('dec_lstm1_copy')
        dec_lstm1_copy.forward_layer.set_weights(dec_lstm1_orig.forward_layer.get_weights())
        dec_lstm1_copy.backward_layer.set_weights(dec_lstm1_orig.backward_layer.get_weights())

        dec_lstm2_copy = decoder_model.get_layer('dec_lstm2_copy')
        dec_lstm2_copy.forward_layer.set_weights(dec_lstm2_orig.forward_layer.get_weights())
        dec_lstm2_copy.backward_layer.set_weights(dec_lstm2_orig.backward_layer.get_weights())

        output_layer = self.model.get_layer('output')
        decoder_model.get_layer('output_copy').set_weights(output_layer.get_weights())

        self.decoder_model = decoder_model
        return decoder_model

    # Установка моста (вызывается из BridgeAdapter)
    def set_bridge(self, bridge, decoder_model):
        self.bridge = bridge
        self.decoder_model = decoder_model
        self.use_bridge = True

    # DANN-модель (для DANNAdapter)
    def build_dann(self, domain_hidden=64):
        # Создаёт DANN-модель с двумя головами: SoC + домен.
        # Копирует веса из текущей модели.
        from models.dann_builder import build_dann_model

        self.dann_model = build_dann_model(base_model=self, hidden_units=self.hidden_units, domain_hidden=domain_hidden)
        return self.dann_model

    # Утилита
    def count_trainable(self) -> tuple:
        trainable = int(np.sum([np.prod(w.shape) for w in self.model.trainable_weights]))
        total = int(np.sum([np.prod(w.shape) for w in self.model.weights]))
        return trainable, total

MODEL_REGISTRY = {'lstm_attention': LSTMAttentionModel}
