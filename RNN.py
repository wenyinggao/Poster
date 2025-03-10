import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import random

# 1. Define the custom dataset
class VariableSequenceDataset(Dataset):
    def __init__(self, num_samples, max_seq_len):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random sequence length between 1 and max_seq_len
        seq_len = random.randint(1, self.max_seq_len)
        # Create a random input sequence of shape (seq_len, 4)
        x = torch.randn(seq_len, 4)
        # For demonstration, we use the identity mapping: target = input
        y = x.clone()
        return x, y

# 2. Create a collate function for batching and padding variable-length sequences
def collate_fn(batch):
    # Unpack the list of (input, target) tuples
    xs, ys = zip(*batch)
    # Get valid lengths for each sequence in the batch
    lengths = torch.tensor([seq.shape[0] for seq in xs], dtype=torch.long)
    # Pad the sequences (padding value 0.0)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)
    return xs_padded, ys_padded, lengths

# Create dataset and dataloader
dataset = VariableSequenceDataset(num_samples=1000, max_seq_len=10)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# 3. Define a simple RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(SimpleRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        # Pack the padded batch of sequences
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        # Unpack the sequences to get the padded output
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # Apply the fully-connected layer at each time step
        output = self.fc(output)
        return output

# 4. Training loop parameters and instantiation
input_dim = 4
hidden_dim = 8
num_layers = 1
output_dim = 4
learning_rate = 0.001
num_epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleRNN(input_dim, hidden_dim, num_layers, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch in dataloader:
        inputs, targets, lengths = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Forward pass (accounting for valid lengths via packing)
        outputs = model(inputs, lengths)

        # Create a mask to compute loss only on valid (unpadded) time steps
        # The mask shape is (batch_size, max_seq_len)
        mask = torch.arange(inputs.size(1), device=device)[None, :] < lengths[:, None]
        
        # Compute MSE loss only over valid elements
        loss = criterion(outputs[mask], targets[mask])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
