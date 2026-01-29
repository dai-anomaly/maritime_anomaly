import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class BasePreprocessor:
    def __init__(self):
        self.rs_scaler = RobustScaler()
        self.mm_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_names = None

    def clean_data(self, df):
        # 1. Standardize Column Names (Crucial for Excel files)
        df.columns = df.columns.str.strip()
        
        # 2. Drop constant columns (sensors that aren't working/changing)
        df = df.loc[:, (df != df.iloc[0]).any()]
        
        # 3. Triple-Stage Imputation
        df = df.interpolate(method='linear', limit_direction='both')
        return df.ffill().bfill()

    def engineer_marine_features(self, df):
        """
        Adaptive Feature Extraction: 
        Only calculates features if the required sensors exist in the data.
        """
        print("Extracting Engineer-Level Features...")
        df_eng = df.copy()

        # --- 1. ADAPTIVE EXHAUST ANALYSIS ---
        exh_cols = [f'ME1Exh{i}Temp' for i in range(1, 8) if f'ME1Exh{i}Temp' in df_eng.columns]
        
        if exh_cols:
            # Thermal Spread (Difference between hottest and coldest cylinder)
            df_eng['Exh_Temp_Spread'] = df_eng[exh_cols].max(axis=1) - df_eng[exh_cols].min(axis=1)
            
            # If the Average column exists, calculate individual deviations
            if 'ME1ExhAvgTemp' in df_eng.columns:
                for col in exh_cols:
                    df_eng[f'{col}_Dev'] = df_eng[col] - df_eng['ME1ExhAvgTemp']
            else:
                # If Average is missing, we create our own average to use as a baseline
                temp_avg = df_eng[exh_cols].mean(axis=1)
                for col in exh_cols:
                    df_eng[f'{col}_Dev'] = df_eng[col] - temp_avg

        # --- 2. ADAPTIVE FUEL EFFICIENCY ---
        # Look for supply and power to calculate consumption ratio
        if 'MEFuelMassSupplyAct' in df_eng.columns and 'ME1Power' in df_eng.columns:
            df_eng['Fuel_Power_Ratio'] = df_eng['MEFuelMassSupplyAct'] / (df_eng['ME1Power'] + 1e-6)
            
        if 'ME1RPM' in df_eng.columns and 'ME1Power' in df_eng.columns:
            df_eng['Power_Per_RPM'] = df_eng['ME1Power'] / (df_eng['ME1RPM'] + 1e-6)

        # --- 3. ADAPTIVE COOLING HEALTH ---
        if 'ME1JCWOutTemp' in df_eng.columns and 'ME1JCWInletTemp' in df_eng.columns:
            df_eng['JCW_Temp_Delta'] = df_eng['ME1JCWOutTemp'] - df_eng['ME1JCWInletTemp']

        # --- 4. ADAPTIVE PROPULSION EFFICIENCY ---
        if 'ShipSpeedThroughWater' in df_eng.columns and 'ME1RPM' in df_eng.columns:
            df_eng['Speed_RPM_Ratio'] = df_eng['ShipSpeedThroughWater'] / (df_eng['ME1RPM'] + 1e-6)

        return df_eng

    def save_preprocessor(self, path):
        state = {'rs': self.rs_scaler, 'mm': self.mm_scaler, 'features': self.feature_names}
        joblib.dump(state, path)

class DeepLearningPreprocessor(BasePreprocessor):
    def prepare_data(self, df, window_size=60, test_size=0.3):
        # 1. Clean
        df_clean = self.clean_data(df)
        
        # 2. Engineer Features
        df_eng = self.engineer_marine_features(df_clean)
        
        # 3. Handle Time and Numeric conversion
        if 'Time' in df_eng.columns:
            df_eng = df_eng.drop(columns=['Time'])
            
        numeric_df = df_eng.select_dtypes(include=[np.number])
        self.feature_names = numeric_df.columns.tolist()
        
        # 4. Scale
        scaled = self.rs_scaler.fit_transform(numeric_df)
        scaled = self.mm_scaler.fit_transform(scaled)
        
        # 5. Windowing
        sequences = []
        for i in range(len(scaled) - window_size):
            sequences.append(scaled[i : i + window_size])
        
        X = np.array(sequences).astype('float32')
        split_idx = int(len(X) * (1 - test_size))
        return X[:split_idx], X[split_idx:]

class IsolationForestPreprocessor(BasePreprocessor):
    def prepare_data(self, df, test_size=0.3):
        df_clean = self.clean_data(df)
        df_eng = self.engineer_marine_features(df_clean)
        
        if 'Time' in df_eng.columns:
            df_eng = df_eng.drop(columns=['Time'])
            
        numeric_df = df_eng.select_dtypes(include=[np.number])
        self.feature_names = numeric_df.columns.tolist()
        
        scaled = self.rs_scaler.fit_transform(numeric_df)
        scaled = self.mm_scaler.fit_transform(scaled)
        
        split_idx = int(len(scaled) * (1 - test_size))
        return scaled[:split_idx], scaled[split_idx:]