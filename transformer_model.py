import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import os

# ==========================================
# 1. PYTORCH DATASET WRAPPER
# ==========================================
class ShipDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # In Autoencoders, the target is the input itself
        return self.data[idx], self.data[idx]

# ==========================================
# 2. TRANSFORMER AUTOENCODER ARCHITECTURE
# ==========================================
class MarineTransformerAE(nn.Module):
    def __init__(self, n_features, seq_len, d_model=32, n_heads=2, num_layers=1, dropout=0.2):
        super(MarineTransformerAE, self).__init__()
        
        # Linear Projection: Maps 87+ sensors into hidden space
        self.embedding = nn.Linear(n_features, d_model)
        
        # Learned Positional Encoding: Helps the model understand time order
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # Transformer Encoder Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Reconstruction Head
        self.decoder = nn.Linear(d_model, n_features)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features]
        x = self.embedding(x) + self.pos_emb
        latent = self.transformer_encoder(x)
        out = self.decoder(latent)
        return out

# ==========================================
# 3. MODEL WRAPPER CLASS (THE CDAC STRUCTURE)
# ==========================================
class TransformerModel:
    def __init__(self, n_features, seq_len, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {
            'd_model': 32,
            'n_heads': 2,
            'num_layers': 1,
            'dropout': 0.2,
            'lr': 1e-4,
            'weight_decay': 1e-4
        }
        
        self.model = MarineTransformerAE(
            n_features=n_features,
            seq_len=seq_len,
            d_model=self.config['d_model'],
            n_heads=self.config['n_heads'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)

    def train_model(self, X_train, X_val, epochs=20, batch_size=32):
        print(f"Training Transformer on {self.device}...")
        
        train_loader = DataLoader(ShipDataset(X_train), batch_size=batch_size)
        val_loader = DataLoader(ShipDataset(X_val), batch_size=batch_size)
        
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['lr'], 
            weight_decay=self.config['weight_decay']
        )
        criterion = nn.MSELoss()
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_x)
                
                if torch.isnan(loss):
                    print(f"NaN Loss at epoch {epoch}. Stopping.")
                    return history
                
                loss.backward()
                # Gradient Clipping for Stability
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()

            # Validation Phase
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for val_x, _ in val_loader:
                    val_x = val_x.to(self.device)
                    val_out = self.model(val_x)
                    total_val_loss += criterion(val_out, val_x).item()

            avg_train = total_train_loss / len(train_loader)
            avg_val = total_val_loss / len(val_loader)
            
            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)

            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                self.save_weights("best_transformer.pth")

        return history

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
        return reconstructions.cpu().numpy()

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), filename)
        print(f"Weights saved to {filename}")

    def load_weights(self, filename):
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename, map_location=self.device))

            print(f"Weights loaded from {filename}")
