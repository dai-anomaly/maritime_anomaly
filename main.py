from prediction_engine import MarinePredictionEngine
from anomaly_analyzer import MaritimeAnomalyAnalyzer
import numpy as np

cols_context = [
    'BridgeAirHumidity', 'BridgeAirPressure', 'BridgeAirTemp', 
    'ERAirHumidity', 'ERAirPressure', 'ERAirTemp',
    'ObsSeaWaterTemp', 'ObsSeaWaterPressure', 'ObsWaterDepth',
    'ObsWindDirRel', 'ObsWindDirTrue', 'ObsWindSpeedRel', 'ObsWindSpeedTrue', 
    'WindForce', 'WindForceTrue', 'DraftAft', 'DraftFwd', 'DraftPort', 
    'DraftStbd', 'Trim', 'SpeedOG', 'SpeedTW', 'IsME1JWPortPumpRun', 
    'IsME1JWPump1Run', 'IsME1JWPump2Run', 'IsME1LOPump1Run', 'IsME1LOPump2Run',
    'MEFuelSupplyLSFO', 'MEFuelSupplyLSFOAct'
]

cols_target = [
    'ME1RPM', 'ME1Load', 'ME1Power', 'ME1Torque',
    'ME1Exh1Temp', 'ME1Exh2Temp', 'ME1Exh3Temp', 'ME1Exh4Temp', 
    'ME1Exh5Temp', 'ME1Exh6Temp', 'ME1Exh7Temp',
    'ME1JCWInTemp', 'ME1JCWT1OutTemp', 'ME1JCWT2OutTemp', 'ME1JCWT3OutTemp', 
    'ME1JCWT4OutTemp', 'ME1JCWT5OutTemp', 'ME1JCWT6OutTemp', 'ME1JCWT7OutTemp',
    'ME1JacketCoolingWaterInletPressure',
    'ME1Bearing1Temp', 'ME1Bearing2Temp', 'ME1Bearing3Temp', 'ME1Bearing4Temp', 
    'ME1Bearing5Temp', 'ME1Bearing6Temp', 'ME1Bearing7Temp', 'ME1Bearing8Temp',
    'ME1TC1RPM', 'ME1TC2RPM', 'ME1TC1ExhaustTempIn', 'ME1TC1ExhaustTempOut', 
    'ME1TC2ExhaustTempIn', 'ME1TC2ExhaustTempOut',
    'ME1FuelMassSupply', 'ME1FuelMassSupplyAct', 'MEFuelSupplyTemp', 
    'MEFuelSupplyDensity', 'MEFuelSupplyViscosity', 'MEFuelSupplyHFO', 
    'MEFuelSupplyHFOAct', 'ME1LOInletPressure', 'ME1StartingAirPressure'
]

# 1. Setup Config
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
    'iso_forest': {
        'contamination': 0.01, 
        'n_estimators': 100,
        'random_state': 42
    }
}

# 2. Run Training
engine = MarinePredictionEngine(data_path="Dataset/Data.xlsx", config=full_config)
engine.run_training_pipeline(cols_target, cols_context)

# 3. ANALYSIS PHASE
trained_trans = engine.models['transformer']
trained_if = engine.models['iso_forest']
trained_ae = engine.models['tf_autoencoder']

X_dl_test, X_if_test = engine.get_test_data()

# 4. Initialize and run Analyzer
analyzer = MaritimeAnomalyAnalyzer(
    dl_preprocessor_path='models/dl_preprocessor.pkl', 
    if_preprocessor_path='models/if_preprocessor.pkl',
    engine_cols=cols_target
)

# run_comparison loads trained_ae internally so no need to pass
analyzer.run_comparison(trained_trans, trained_if, X_dl_test, X_if_test)

# Show the Dashboard
analyzer.plot_dashboard()

# Perform Root Cause on the point with the highest error
max_error_idx = np.argmax(analyzer.results['transformer_error'])

# Diagnosis 
analyzer.diagnose_anomaly(trained_trans, X_dl_test[max_error_idx], model_type='transformer')
