import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, RepeatVector, TimeDistributed, Lambda
import numpy as np

class TensorflowLSTMAutoencoder:
    def __init__(self, engine_idx, context_idx, window_size, latent_dim=64):
        self.engine_idx = engine_idx
        self.context_idx = context_idx
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.model = self._build_model()

    def _build_model(self):
        full_input = Input(shape=(self.window_size, len(self.engine_idx) + len(self.context_idx)), name="full_input")

        # Internal Slicing
        engine_input = Lambda(lambda x: tf.gather(x, self.engine_idx, axis=2), name="engine_slice")(full_input)
        context_input = Lambda(lambda x: tf.gather(x, self.context_idx, axis=2), name="context_slice")(full_input)

        # Encoder
        e = LSTM(64, return_sequences=True)(engine_input)
        e = LSTM(32, return_sequences=False)(e)
        c = LSTM(32, return_sequences=False)(context_input)

        # Fusion
        latent = Concatenate()([e, c])
        latent = Dense(self.latent_dim, activation="relu")(latent)
        latent = Dropout(0.1)(latent)

        # Decoder
        d = RepeatVector(self.window_size)(latent)
        d = LSTM(32, return_sequences=True)(d)
        d = LSTM(64, return_sequences=True)(d)

        output = TimeDistributed(Dense(len(self.engine_idx)), name="engine_reconstruction")(d)

        model = Model(inputs=full_input, outputs=output)
        model.compile(optimizer="adam", loss="mae")
        return model

    def train(self, X_train, X_val, epochs=20, batch_size=32):
        # Target is only the engine features
        y_train = X_train[:, :, self.engine_idx]
        y_val = X_val[:, :, self.engine_idx]
        return self.model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                               epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)
