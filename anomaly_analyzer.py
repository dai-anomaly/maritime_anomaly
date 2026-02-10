import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib
from tensorflow.keras.models import load_model

class MaritimeAnomalyAnalyzer:
    def __init__(self, dl_preprocessor_path, if_preprocessor_path, engine_cols):
        """
        Loads saved states and models.
        engine_cols: List of the 45 engine sensors we want to monitor.
        """
        print("Initializing Maritime Anomaly Analyzer...")
        self.dl_state = joblib.load(dl_preprocessor_path)
        self.if_state = joblib.load(if_preprocessor_path)
        self.feature_names = self.dl_state['features']
        self.engine_cols = engine_cols
        self.results = {}
        
        # Load TensorFlow Autoencoder
        try:
            self.tf_ae = load_model("models/tf_lstm_ae")
            print("Successfully loaded TensorFlow LSTM Autoencoder.")
        except Exception as e:
            print(f"Error loading TF model: {e}")

    def run_comparison(self, transformer_model, if_model, X_dl_test, X_if_test):
        """Generates anomaly scores for all three models"""
        print("Running Inference and Comparison...")
        
        # 1. Transformer Error (PyTorch)
        # Preds shape: (Samples, Window, Total_Features)
        trans_out = transformer_model.predict(X_dl_test)
        self.results['transformer_error'] = np.mean(np.abs(X_dl_test - trans_out), axis=(1, 2))
        
        # 2. TensorFlow LSTM Autoencoder Error (Engine only)
        engine_idx = [self.feature_names.index(c) for c in self.engine_cols if c in self.feature_names]
        X_engine_actual = X_dl_test[:, :, engine_idx]
        
        # TF Prediction
        tf_recon = self.tf_ae.predict(X_dl_test)
        
        # Calculate MAE (Actual Engine vs Reconstructed Engine)
        self.results['tf_lstm_ae_error'] = np.mean(np.abs(X_engine_actual - tf_recon), axis=(1, 2))
        
        # 3. Isolation Forest Score 
        if_scores = if_model.predict_score(X_if_test)
        self.results['if_error'] = -if_scores # Invert because lower score = more anomalous

        print("Comparison Complete.")
        return self.results

    def plot_dashboard(self):
        """Creates the comparative health monitoring dashboard"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        config = [
            ('Transformer (Global System)', 'transformer_error', 'blue'),
            ('Isolation Forest (Statistical)', 'if_error', 'orange'),
            ('TF LSTM AE (Contextual Engine)', 'tf_lstm_ae_error', 'purple')
        ]
        
        for i, (name, key, color) in enumerate(config):
            data = self.results[key]
            threshold = np.mean(data) + (3 * np.std(data))
            
            axes[i].plot(data, label=f'{name} Score', color=color, alpha=0.7)
            axes[i].axhline(y=threshold, color='red', linestyle='--', label='3σ Threshold')
            axes[i].set_title(f"System Health: {name}")
            axes[i].set_ylabel("Anomaly Score")
            axes[i].legend(loc='upper right')

        plt.xlabel("Sample Index (Time)")
        plt.tight_layout()
        plt.show()

    def diagnose_anomaly(self, model_obj, input_seq, model_type='transformer'):
        """
        Root Cause: Which features are responsible.
        model_obj: PyTorch Transformer wrapper or self.tf_ae
        input_seq: (Window, Features)
        """
        print(f"Diagnosing Root Cause using {model_type.upper()}...")
        
        if model_type.lower() == 'transformer':
            # PyTorch Logic
            model_obj.model.eval()
            device = next(model_obj.model.parameters()).device
            with torch.no_grad():
                inp = torch.tensor(input_seq).unsqueeze(0).to(device).float()
                out = model_obj.model(inp)
                diff = torch.mean(torch.abs(out - inp), dim=1).squeeze().cpu().numpy()
            feat_labels = self.feature_names
            
        elif model_type.lower() == 'tf_ae':
            # TensorFlow Logic
            inp = np.expand_dims(input_seq, axis=0)
            recon = self.tf_ae.predict(inp).squeeze() # (Window, 45)
            
            # Slicing actual engine data
            engine_idx = [self.feature_names.index(c) for c in self.engine_cols if c in self.feature_names]
            actual_engine = input_seq[:, engine_idx]
            
            diff = np.mean(np.abs(actual_engine - recon), axis=0)
            feat_labels = self.engine_cols

        # Create Ranking
        ranking = pd.Series(diff, index=feat_labels).sort_values(ascending=False)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        ranking.head(10).plot(kind='barh', color='salmon' if model_type=='transformer' else 'mediumpurple')
        plt.title(f"Top 10 Anomaly Contributors ({model_type.upper()})")
        plt.gca().invert_yaxis()
        plt.show()
        
        return ranking.head(10)
