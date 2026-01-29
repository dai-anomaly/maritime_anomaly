import pandas as pd
import numpy as np
import torch
import os
import joblib

# Import your custom classes from the other .py files
from preprocessors import DeepLearningPreprocessor, IsolationForestPreprocessor
from transformer_model import TransformerModel
from autoencoder_model import AutoencoderModel
from iso_forest_model import MarineIsolationForest

class MarinePredictionEngine:
    def __init__(self, data_path, config):
        """
        Orchestrates the training and prediction of all marine models.
        Supports .xlsx and .csv formats.
        """
        self.data_path = data_path
        self.config = config
        self.results = {}
        self.models = {}
        
        # 1. Load Data with Format Check
        self.df = self._load_data()
        
        # 2. Initialize Preprocessors
        self.dl_prep = DeepLearningPreprocessor()
        self.if_prep = IsolationForestPreprocessor()

    def _load_data(self):
        """Handles Excel loading and basic validation"""
        print(f"Loading data from: {self.data_path}")
        try:
            if self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
                df = pd.read_excel(self.data_path, engine='openpyxl')
            else:
                # Fallback to CSV with encoding handling
                try:
                    df = pd.read_csv(self.data_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(self.data_path, encoding='latin1')
            
            # Basic cleanup: remove completely empty rows/columns
            df = df.dropna(how='all').loc[:, ~df.columns.str.contains('^Unnamed')]
            print(f"Successfully loaded data. Shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"Critical Error loading file: {e}")
            raise

    def run_training_pipeline(self):
        print("\n" + "="*50)
        print("STARTING MARINE ANOMALY DETECTION PIPELINE")
        print("="*50)

        # 1. Prepare Data for Deep Learning (Transformer & MLP-AE)
        X_train_dl, X_val_dl = self.dl_prep.prepare_data(
            self.df, 
            window_size=self.config['window_size']
        )
        
        # 2. Prepare Data for Isolation Forest (2D)
        X_train_if, X_val_if = self.if_prep.prepare_data(self.df)

        # --- Model 1: Transformer ---
        print("\n[Phase 1] Training Transformer...")
        self.models['transformer'] = TransformerModel(
            n_features=X_train_dl.shape[2],
            seq_len=X_train_dl.shape[1],
            config=self.config.get('transformer')
        )
        self.results['transformer_hist'] = self.models['transformer'].train_model(
            X_train_dl, X_val_dl, 
            epochs=self.config['epochs']
        )

        # --- Model 2: MLP Autoencoder ---
        print("\n[Phase 2] Training MLP Autoencoder...")
        self.models['autoencoder'] = AutoencoderModel(
            n_features=X_train_dl.shape[2],
            config=self.config.get('autoencoder')
        )
        self.results['ae_hist'] = self.models['autoencoder'].train_model(
            X_train_dl, X_val_dl, 
            epochs=self.config['epochs']
        )

        # --- Model 3: Isolation Forest ---
        print("\n[Phase 3] Training Isolation Forest...")
        self.models['iso_forest'] = MarineIsolationForest(
            config=self.config.get('iso_forest')
        )
        self.models['iso_forest'].train_model(X_train_if)

        # 3. Save All States and Assets
        self._save_all_assets()
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETE: All models and scalers saved to /models folder.")
        print("="*50)

    def _save_all_assets(self):
        """Saves weights, models, and preprocessor states for Class 8"""
        if not os.path.exists('models'): 
            os.makedirs('models')
        
        # Save Scaler states (Crucial for consistent prediction)
        self.dl_prep.save_preprocessor('models/dl_preprocessor.pkl')
        self.if_prep.save_preprocessor('models/if_preprocessor.pkl')
        
        # Save Isolation Forest (Joblib)
        self.models['iso_forest'].save_model('models/iso_forest.pkl')
        
        # Note: PyTorch weights (Transformer/AE) are saved as .pth inside their own train_model calls
        print("Saved all preprocessors and model weights to 'models/' directory.")

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # 1. Define Global Hyperparameters
    full_config = {
        'window_size': 60,
        'epochs': 20,
        'transformer': {
            'd_model': 32, 
            'n_heads': 2, 
            'num_layers': 1, 
            'dropout': 0.2, 
            'lr': 1e-4,
            'weight_decay': 1e-4
        },
        'autoencoder': {
            'bottleneck_dim': 16, 
            'dropout': 0.1, 
            'lr': 1e-4,
            'weight_decay': 1e-5
        },
        'iso_forest': {
            'contamination': 0.01, 
            'n_estimators': 100,
            'random_state': 42
        }
    }

    # 2. Provide the path to your XLSX file
    DATA_FILE = "Dataset/Data.xlsx" 

    # 3. Initialize and Run
    engine = MarinePredictionEngine(data_path=DATA_FILE, config=full_config)
    engine.run_training_pipeline()