import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib
from sklearn.metrics import mean_absolute_error

class MaritimeAnomalyAnalyzer:
    def __init__(self, dl_preprocessor_path, if_preprocessor_path):
        """Loads the saved states to ensure consistent evaluation"""
        self.dl_state = joblib.load(dl_preprocessor_path)
        self.if_state = joblib.load(if_preprocessor_path)
        self.feature_names = self.dl_state['features']
        self.results = {}

    def get_reconstruction_error(self, original, reconstructed):
        """Calculates error for DL models (MAE per sequence)"""
        # Shape: (Samples, Time, Features)
        return np.mean(np.abs(original - reconstructed), axis=(1, 2))

    def run_comparison(self, transformer_model, ae_model, if_model, X_dl_test, X_if_test):
        """Generates anomaly scores for all three models"""
        print("Running Inference and Comparison...")
        
        # 1. Transformer Error
        trans_out = transformer_model.predict(X_dl_test)
        self.results['transformer_error'] = self.get_reconstruction_error(X_dl_test, trans_out)
        
        # 2. MLP Autoencoder Error
        ae_out = ae_model.predict(X_dl_test)
        self.results['ae_error'] = self.get_reconstruction_error(X_dl_test.mean(axis=1), ae_out)
        
        # 3. Isolation Forest Score 
        # (Lower scores = more anomalous, so we invert it for the graph)
        if_scores = if_model.predict_score(X_if_test)
        self.results['if_error'] = -if_scores 

        return self.results

    def plot_comparative_dashboard(self):
        """Creates the side-by-side health monitoring dashboard"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=False)
        
        models = [
            ('Transformer (Temporal)', 'transformer_error', 'blue'),
            ('MLP Autoencoder (Structural)', 'ae_error', 'green'),
            ('Isolation Forest (Statistical)', 'if_error', 'orange')
        ]
        
        for i, (name, key, color) in enumerate(models):
            data = self.results[key]
            # Calculate a statistical threshold (Mean + 3 Sigma)
            threshold = np.mean(data) + (3 * np.std(data))
            
            axes[i].plot(data, label=f'{name} Score', color=color, alpha=0.7)
            axes[i].axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold')
            axes[i].set_title(f"System Health Monitoring: {name}")
            axes[i].set_ylabel("Error/Anomaly Score")
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    def analyze_root_cause(self, model, input_seq, model_type='transformer'):
        """Identifies which sensor triggered the anomaly"""
        model.model.eval()
        device = next(model.model.parameters()).device
        
        with torch.no_grad():
            inp = torch.tensor(input_seq).unsqueeze(0).to(device)
            out = model.model(inp)
            # MAE per feature across the window
            diff = torch.mean(torch.abs(out - inp), dim=1).squeeze().cpu().numpy()
            
        # Create a ranking
        ranking = pd.Series(diff, index=self.feature_names).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        ranking.head(10).plot(kind='barh', color='salmon')
        plt.title(f"Top 10 Anomaly Contributors ({model_type.upper()})")
        plt.gca().invert_yaxis()
        plt.show()
        
        return ranking.head(10)

# ==========================================
# USAGE EXAMPLE (In your notebook)
# ==========================================
# analyzer = MaritimeAnomalyAnalyzer('models/dl_preprocessor.pkl', 'models/if_preprocessor.pkl')
# scores = analyzer.run_comparison(trained_trans, trained_ae, trained_if, X_dl_test, X_if_test)
# analyzer.plot_comparative_dashboard()