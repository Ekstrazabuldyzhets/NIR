"""
Сборка DANN-модели на основе обученной базовой модели.
Структура:
    Вход X → [Энкодер] → context
                          ├──→ [Регрессор] → soc_output (batch, 6)
                          └──→ [GRL] → [Классификатор] → domain_output (batch, 2)
"""

from tensorflow.keras.layers import (Input, Dense, Dropout, LSTM, Bidirectional, RepeatVector,
                                     Attention, Concatenate)
from tensorflow.keras.models import Model

from models.dann_layers import GradientReversalLayer
def build_dann_model(base_model, hidden_units=64, domain_hidden=64):
    # строит DANN-модель на основе обученной базовой модели.
    # входной тензор
    encoder_input = base_model.model.input
    # тензор выхода энкодера
    encoder_output_tensor = base_model.model.get_layer('encoder_lstm2').output
    # Context = последний шаг
    context = encoder_output_tensor[:, -1, :]

    # ===== Ветка регрессора (SoC) =====
    decoder_inputs = RepeatVector(1)(context)
    decoder_lstm1 = Bidirectional(LSTM(hidden_units, return_sequences=True, dropout=0.2),
                                  name='dann_decoder_lstm1')(decoder_inputs)
    attention_output = Attention(name='dann_attention')([decoder_lstm1, encoder_output_tensor])
    combined = Concatenate(name='dann_concat')([decoder_lstm1, attention_output])
    decoder_lstm2 = Bidirectional(LSTM(hidden_units, return_sequences=False, dropout=0.2),
                                  name='dann_decoder_lstm2')(combined)
    # регрессор SoC
    soc_output = Dense(6, name='soc_output')(decoder_lstm2)

    # ===== Ветка домен-классификатора (через GRL) =====
    grl = GradientReversalLayer(alpha=0.0, name='grl')
    context_for_grl = grl(context)
    domain_hidden_layer = Dense(domain_hidden, activation='relu', name='domain_hidden')(context_for_grl)
    domain_dropout = Dropout(0.2, name='domain_dropout')(domain_hidden_layer)
    # классификатор домена
    domain_output = Dense(2, activation='softmax', name='domain_output')(domain_dropout)

    # итоговая модель
    dann_model = Model(inputs=encoder_input, outputs=[soc_output, domain_output])
    # копируем веса
    copy_weights(base_model, dann_model)

    return dann_model

def copy_weights(base_model, dann_model):
    """Копирует веса decoder-слоёв и output из base_model в dann_model."""
    # decoder_lstm1
    base_dec1 = base_model.model.get_layer('decoder_lstm1')
    dann_dec1 = dann_model.get_layer('dann_decoder_lstm1')
    dann_dec1.forward_layer.set_weights(base_dec1.forward_layer.get_weights())
    dann_dec1.backward_layer.set_weights(base_dec1.backward_layer.get_weights())
    # decoder_lstm2
    base_dec2 = base_model.model.get_layer('decoder_lstm2')
    dann_dec2 = dann_model.get_layer('dann_decoder_lstm2')
    dann_dec2.forward_layer.set_weights(base_dec2.forward_layer.get_weights())
    dann_dec2.backward_layer.set_weights(base_dec2.backward_layer.get_weights())
    # output
    base_out = base_model.model.get_layer('output')
    dann_model.get_layer('soc_output').set_weights(base_out.get_weights())