from prediction_engine import MarinePredictionEngine
from anomaly_analyzer import MaritimeAnomalyAnalyzer
import numpy as np

# 1. Setup Config (Same as we used before)
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

# 2. Run Training
engine = MarinePredictionEngine(data_path="Dataset/Data.xlsx", config=full_config)
engine.run_training_pipeline()

# 3. ANALYSIS PHASE (Option A)
# Get the model objects from engine.models dictionary
trained_trans = engine.models['transformer']
trained_ae = engine.models['autoencoder']
trained_if = engine.models['iso_forest']

# Get the test data tensors
X_dl_test, X_if_test = engine.get_test_data()

# 4. Initialize and run Analyzer
analyzer = MaritimeAnomalyAnalyzer(
    dl_preprocessor_path='models/dl_preprocessor.pkl', 
    if_preprocessor_path='models/if_preprocessor.pkl'
)

# Run the scores
analyzer.run_comparison(trained_trans, trained_ae, trained_if, X_dl_test, X_if_test)

# Show the Dashboard
analyzer.plot_dashboard()

# Perform Root Cause on the sample with the highest error
max_error_idx = np.argmax(analyzer.results['transformer'])
analyzer.diagnose_anomaly(trained_trans, X_dl_test[max_error_idx], max_error_idx)