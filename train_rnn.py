import ast
import time
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Special tokens
SOS_token = 0
EOS_token = 1
PAD_token = -1  # 确保填充值不同于有效数据

input_token_len = 2
target_token_len = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 12 + 1

def get_dataloader(batch_size):
    df = pd.read_csv('Art_Poster_Layout_Dataset.csv')
    pairs = []
    for x, y in zip(df["input_sizes"], df["output_layouts"]):
        x = ast.literal_eval(x)
        y = ast.literal_eval(y)
        assert len(x) == len(y)
        pairs.append((x, y))

    n = len(pairs)
    input_ids = np.full((n, MAX_LENGTH, input_token_len), PAD_token, dtype=np.float32)
    target_ids = np.full((n, MAX_LENGTH, target_token_len), PAD_token, dtype=np.float32)
    input_lengths = []

    for idx, (inp, tgt) in enumerate(pairs):
        input_lengths.append(len(inp))  # 记录真实长度
        inp.append([EOS_token] * input_token_len)
        tgt.append([EOS_token] * target_token_len)
        inp_ids = np.array(inp)
        tgt_ids = np.array(tgt)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(
        torch.FloatTensor(input_ids).to(device),
        torch.FloatTensor(target_ids).to(device),
        torch.tensor(input_lengths, dtype=torch.long).to(device)
    )
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, input_lengths):
        embedded = self.dropout(self.embedding(input))
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Linear(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.RNN(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.zeros(batch_size, 1, target_token_len, dtype=torch.float32, device=device)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.detach().float()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attentions = torch.cat(attentions, dim=1)
        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)
        return output, hidden, attn_weights

hidden_size = 128
batch_size = 32
encoder = EncoderRNN(input_size=input_token_len, hidden_size=hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size=hidden_size, output_size=target_token_len).to(device)
train_dataloader = get_dataloader(batch_size)

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    for input_tensor, target_tensor, input_lengths in dataloader:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        mask = (target_tensor != PAD_token).float()
        loss = criterion(decoder_outputs, target_tensor)
        loss = (loss * mask).sum() / mask.sum()
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001, print_every=5):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='none')

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        if epoch % print_every == 0:
            print(f"Epoch {epoch}/{n_epochs}, Loss: {loss:.4f}")

if __name__ == "__main__":
    train(train_dataloader, encoder, decoder, 80, print_every=5)
