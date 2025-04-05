import os
import pandas as pd
import torch
import pytorch_lightning as pl
from torch import nn, optim
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader  # PyG DataLoader
from torch_geometric.nn import GCNConv, global_add_pool

class ForexGraphDataset(Dataset):
    def __init__(self, root="historical_data", transform=None, pre_transform=None):
        # Ensure the raw and processed directories exist
        self.root = root
        raw_path = os.path.join(self.root, "raw")
        processed_path = os.path.join(self.root, "processed")
        os.makedirs(raw_path, exist_ok=True)
        os.makedirs(processed_path, exist_ok=True)
        
        self._processed_file_names = []  # Initialize the attribute properly
        super().__init__(root, transform, pre_transform)

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def raw_file_names(self):
        # List all CSV files in the raw folder
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.csv')]

    @property
    def processed_file_names(self):
        # Return the list of processed file names
        return self._processed_file_names

    def process(self):
        for i, file_name in enumerate(self.raw_file_names):
            try:
                df = pd.read_csv(os.path.join(self.raw_dir, file_name))
                df.columns = df.columns.str.lower()

                # Validate and process data
                if 'timestamp' not in df.columns:
                    print(f"Skipping {file_name}: Missing timestamp")
                    continue

                # Ensure 'volume' column exists, if not, set it to 1
                if 'volume' not in df.columns:
                    print(f"Adding missing 'volume' column for {file_name}.")
                    df['volume'] = 1  # Or any other logic for handling missing volume

                # Create graph data: edge_index connects consecutive time steps
                edge_index = torch.stack([
                    torch.arange(len(df) - 1),
                    torch.arange(1, len(df))
                ], dim=0)

                features = df[['open', 'high', 'low', 'close', 'volume']].values
                targets = df['close'].pct_change().fillna(0).values

                # Use the last target value as the graph label
                target_value = targets[-1]

                data = Data(
                    x=torch.tensor(features, dtype=torch.float32),
                    edge_index=edge_index,
                    y=torch.tensor([target_value], dtype=torch.float32)
                )

                # Save the processed file
                processed_filename = f'data_{i}.pt'
                torch.save(data, os.path.join(self.processed_dir, processed_filename))
                self._processed_file_names.append(processed_filename)  # Append to internal list

            except Exception as e:
                print(f"Failed {file_name}: {str(e)}")
                continue

    def len(self):
        # Return the number of processed files
        return len(self._processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, self._processed_file_names[idx]))
    
class GNNModel(pl.LightningModule):
    def __init__(self, input_features=5, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.fc = nn.Linear(hidden_dim // 2, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        pooled = global_add_pool(x, batch)
        return self.fc(pooled)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_fn(y_hat.squeeze(), batch.y.squeeze())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

def main():
    # Prepare dataset. Make sure to place your CSV files in 'historical_data/raw'
    dataset = ForexGraphDataset()
    dataset.process()
    
    # Create data loader using PyG's DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model and trainer
    model = GNNModel()
    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=100,
        log_every_n_steps=10,
        enable_checkpointing=True
    )
    
    # Train model
    trainer.fit(model, loader)
    
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Save final model under the models/ folder
    torch.save(model.state_dict(), os.path.join("models", "forex_gnn_model.pth"))

if __name__ == "__main__":
    main()
    print("GNN training completed successfully!")
