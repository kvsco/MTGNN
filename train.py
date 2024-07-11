import pandas as pd
import numpy as np
import os

from net import gtnet
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch

cols = ['prepare_count', 'active_tx_count', 'gc_count', 'active_db_conn_count',
        'fetch_time', 'gc_time', 'call_count', 'prepare_time', 'sql_count',
        'sql_time', '_target', 'extcall_count', 'fail_count', 'file_count', 'cpu_time',
        'socket_count', 'heap_usage', 'fetch_count', 'tps', 'extcall_time',
        'response_time', 'thread_count', 'cpu_usage']
# 1. Load and preprocess your data
path = os.getcwd()
train = pd.read_csv(f'{path}/data/train.csv', index_col=[0])[cols]
test = pd.read_csv(f'{path}/data/test.csv', index_col=[0])[cols]

# Assuming the last column is the target
data = train.values
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Convert data to PyTorch tensors
data_tensor = torch.tensor(data, dtype=torch.float32)

# Create sequences for multi-step prediction
def create_sequences(data, seq_length, pred_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - pred_length):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length + pred_length)]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


seq_length = 20  # Example sequence length
pred_length = 5  # Example prediction length

X, Y = create_sequences(data_tensor, seq_length, pred_length)

# Create DataLoader
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

num_nodes = data.shape[1]  # Number of features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = gtnet(gcn_true=True, buildA_true=True, gcn_depth=2, device=device, num_nodes=num_nodes,
              dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32,
              residual_channels=32, skip_channels=64, end_channels=128, seq_length=seq_length,
              in_dim=1, out_dim=pred_length, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True).to(device)

# 3. Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

epochs = 10
for epoch in range(epochs):
    model.train()
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_x = batch_x.permute(0, 2, 1).unsqueeze(1)  # Permute and add channel dimension
        batch_y = batch_y.permute(0, 2, 1).unsqueeze(1)  # Permute and add channel dimension

        optimizer.zero_grad()
        outputs = model(batch_x)

        # Reshape outputs and batch_y to be the same shape
        outputs = outputs.view(outputs.size(0), num_nodes, pred_length)
        batch_y = batch_y.view(batch_y.size(0), num_nodes, pred_length)

        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# 4. Make predictions
model.eval()
with torch.no_grad():
    test_inputs = data_tensor[-seq_length:].unsqueeze(0).permute(0, 2, 1).unsqueeze(1).to(
        device)  # Last sequence for prediction
    predictions = model(test_inputs)
    predictions = predictions.view(num_nodes, pred_length).cpu().numpy()  # Convert to numpy and reshape
    predictions = scaler.inverse_transform(predictions.T).T  # Inverse transform to original scale

print("Predictions: ", predictions)