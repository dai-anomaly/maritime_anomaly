import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# Reuse the Dataset wrapper from the previous architecture logic
class ShipDataset(Dataset):
    def __init__(self, data):
        # MLP Autoencoder expects (Batch, Features)
        # If data is 3D (B, T, F), we flatten it or mean-pool it.
        if len(data.shape) == 3:
            data = data.mean(axis=1) # Average sensors over the 60s window
        self.data = torch.tensor(data, dtype=torch.float32)
        
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.data[idx]

# ==========================================
# 1. MLP AUTOENCODER ARCHITECTURE
# ==========================================
class MarineMLPAE(nn.Module):
    def __init__(self, n_features, bottleneck_dim=16, dropout=0.1):
        super(MarineMLPAE, self).__init__()
        
        # Encoder: Compressing sensor data
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU()
        )
        
        # Decoder: Reconstructing sensor data
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_features)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# ==========================================
# 2. MODEL WRAPPER CLASS
# ==========================================
class AutoencoderModel:
    def __init__(self, n_features, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {
            'bottleneck_dim': 16,
            'dropout': 0.1,
            'lr': 1e-4,
            'weight_decay': 1e-5
        }
        
        self.model = MarineMLPAE(
            n_features=n_features,
            bottleneck_dim=self.config['bottleneck_dim'],
            dropout=self.config['dropout']
        ).to(self.device)

    def train_model(self, X_train, X_val, epochs=20, batch_size=32):
        print(f"Training MLP Autoencoder on {self.device}...")
        
        train_loader = DataLoader(ShipDataset(X_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ShipDataset(X_val), batch_size=batch_size, shuffle=False)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        criterion = nn.MSELoss()
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_x)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_x, _ in val_loader:
                    val_x = val_x.to(self.device)
                    val_out = self.model(val_x)
                    val_loss += criterion(val_out, val_x).item()

            avg_train, avg_val = train_loss/len(train_loader), val_loss/len(val_loader)
            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(self.model.state_dict(), 'best_mlp_ae.pth')

            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
        
        return history

    def predict(self, X):
        self.model.eval()
        # Ensure 2D input
        if len(X.shape) == 3: X = X.mean(axis=1)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy()