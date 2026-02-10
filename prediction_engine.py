import pandas as pd
import numpy as np
import os
import joblib
from preprocessors import DeepLearningPreprocessor, IsolationForestPreprocessor
from transformer_model import TransformerModel
from iso_forest_model import MarineIsolationForest
from autoencoder_model_tf import TensorflowLSTMAutoencoder

class MarinePredictionEngine:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.models = {}
        self.df = self._load_data()
        self.dl_prep = DeepLearningPreprocessor()
        self.if_prep = IsolationForestPreprocessor()

    def _load_data(self):
        df = pd.read_excel(self.data_path)
        return df.dropna(how='all').loc[:, ~df.columns.str.contains('^Unnamed')]

    def run_training_pipeline(self, cols_target, cols_context):
        print("\nSTARTING PIPELINE (Window Size: 10)")
        
        # 1. Prepare Data
        X_train_dl, X_val_dl = self.dl_prep.prepare_data(self.df, window_size=self.config['window_size'])
        X_train_if, X_val_if = self.if_prep.prepare_data(self.df)

        # 2. Indices for TF Model
        engine_idx = self.dl_prep.get_feature_indices(cols_target)
        context_idx = self.dl_prep.get_feature_indices(cols_context)

        # --- Train Transformer (PyTorch) ---
        self.models['transformer'] = TransformerModel(X_train_dl.shape[2], X_train_dl.shape[1], self.config['transformer'])
        self.models['transformer'].train_model(X_train_dl, X_val_dl, epochs=self.config['epochs'])

        # --- Train TF LSTM AE (New) ---
        self.models['tf_autoencoder'] = TensorflowLSTMAutoencoder(engine_idx, context_idx, self.config['window_size'])
        self.models['tf_autoencoder'].train(X_train_dl, X_val_dl, epochs=self.config['epochs'])

        # --- Train Isolation Forest ---
        self.models['iso_forest'] = MarineIsolationForest(self.config['iso_forest'])
        self.models['iso_forest'].train_model(X_train_if)

        self._save_all_assets()

    def _save_all_assets(self):
        if not os.path.exists('models'): os.makedirs('models')
        self.dl_prep.save_preprocessor('models/dl_preprocessor.pkl')
        self.if_prep.save_preprocessor('models/if_preprocessor.pkl')
        self.models['iso_forest'].save_model('models/iso_forest.pkl')
        self.models['tf_autoencoder'].model.save("models/tf_lstm_ae")

    def get_test_data(self):
        _, X_dl_test = self.dl_prep.prepare_data(self.df, window_size=self.config['window_size'])
        _, X_if_test = self.if_prep.prepare_data(self.df)
        return X_dl_test, X_if_test
