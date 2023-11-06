import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import random


### 1. Create random dataset ###
# src seq to tgt seq
# 1, 1, 1, 1, 1, 1, 1, 1 → 1, 1, 1, 1, 1, 1, 1, 1
# 0, 0, 0, 0, 0, 0, 0, 0 → 0, 0, 0, 0, 0, 0, 0, 0
# 1, 0, 1, 0, 1, 0, 1, 0 → 1, 0, 1, 0, 1, 0, 1, 0
# 0, 1, 0, 1, 0, 1, 0, 1 → 0, 1, 0, 1, 0, 1, 0, 1
def generate_random_data(n):
    SOS_token = np.array([2])  # SOS is 2
    EOS_token = np.array([3])  # EOS is 3
    length = 8

    data = []

    # 1,1,1,1,1,1 -> 1,1,1,1,1
    for i in range(n // 3):
        X = np.concatenate((SOS_token, np.ones(length), EOS_token))
        y = np.concatenate((SOS_token, np.ones(length), EOS_token))
        data.append([X, y])

    # 0,0,0,0 -> 0,0,0,0
    for i in range(n // 3):
        X = np.concatenate((SOS_token, np.zeros(length), EOS_token))
        y = np.concatenate((SOS_token, np.zeros(length), EOS_token))
        data.append([X, y])

    # 1,0,1,0 -> 1,0,1,0,1
    for i in range(n // 3):
        X = np.zeros(length)
        start = random.randint(0, 1)

        X[start::2] = 1

        y = np.zeros(length)
        if X[-1] == 0:
            y[::2] = 1
        else:
            y[1::2] = 1

        X = np.concatenate((SOS_token, X, EOS_token))
        y = np.concatenate((SOS_token, y, EOS_token))

        data.append([X, y])

    np.random.shuffle(data)
    return data


train_data = generate_random_data(9000)
test_data = generate_random_data(3000)
print(test_data[0])
print(test_data[1])


class RandomDataset(Dataset):
    def __init__(self, raw_data):
        self.raw_data = raw_data  # raw_data[i]: [src, tgt]

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return (torch.tensor(self.raw_data[idx][0][:-1].astype(np.int64)),
                torch.tensor(self.raw_data[idx][1][1:].astype(np.int64)))


train_dataset = RandomDataset(train_data)
test_dataset = RandomDataset(test_data)
for src, tgt in test_dataset:
    print(src)
    print(tgt)
    break

### 2. Create pytorch dataloader ###
BATCH_SIZE= 16
NUM_TOKEN=4
SEQ_LEN = 9

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
for src, tgt in test_dataloader:
    print(src)
    print(tgt)
    break

### 3. Create the transformer model class ###
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """

    # Constructor
    def __init__(self,
                 num_tokens,  # number of tokens, or dictionary size
                 dim_model,  # model dimension of transformer, like 512
                 num_heads,  # number of head in multi-head attention
                 num_encoder_layers,  # number of encoder layers, like 6
                 num_decoder_layers,  # number of decoder layers, like 6
                 dropout_p,  #
                 ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)  # input and output embedding
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)  # decoder output layer

    def forward(self, src, tgt, tgt_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        N, T, D = src.shape
        assert (D == self.dim_model)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks. Out size = (batch_size, sequence length, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        out = self.out(transformer_out)

        out = out.permute(1, 0, 2)
        assert (out.shape[0] == N)
        assert (out.shape[1] == T)
        assert (out.shape[2] == NUM_TOKEN)
        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(
    num_tokens=NUM_TOKEN, dim_model=8, num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1
).to(device)
print(model)

# test model run
for batch, (X, y) in enumerate(test_dataloader):
    X, y = X.to(device), y.to(device)
    print(X)
    print(y)
    sequence_length = y.size(1)
    tgt_mask = model.get_tgt_mask(sequence_length).to(device)
    print(tgt_mask)
    pred = model(X, y, tgt_mask=tgt_mask)
    print(pred)
    shape = pred.detach().numpy().shape
    assert shape == (BATCH_SIZE, SEQ_LEN, NUM_TOKEN)

    y0 = y[:, 0].detach().numpy()
    pred_0 = pred[:, 0, :].detach().numpy()
    print(pred_0)
    print(y0)
    print(np.argmax(pred_0, axis=1))
    tmp = np.where(np.argmax(pred_0, axis=1) == y0)[0]
    print(len(tmp)) # this is how many data points are predicted correctly in one mini-batch for the first token (total 4)

    break

### 4. Define loss function and optimizer ###
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

### 5. Define training loop ###
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # The total number of data point / batch size
    model.train()  # Change model to "training" state. For example, save gradients.
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # X, y are tensors. By default, they are CPU tensors. to(device) can move them to gpu

        # Compute prediction error
        sequence_length = y.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)
        pred = model(X, y, tgt_mask=tgt_mask)
        loss = loss_fn(pred.permute(0, 2, 1), y) # nn.cross_entropy wants the size dimension (num_token) to be the second dimension

        # Backpropagation
        loss.backward() # Start backpropagation: starting from loss
        optimizer.step() # optimizer does one step (one minibatch)
        optimizer.zero_grad() # zeroing-out gradients for this mini-batch, so it won't affect the next mini-batch

        # report loss and iteration step
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Minibatch loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

### 6. Define test loop ###
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            sequence_length = y.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)
            pred = model(X, y, tgt_mask=tgt_mask) # (BATCH_SIZE, SEQ_LEN, NUM_TOKEN)
            test_loss += loss_fn(pred.permute(0, 2, 1), y).item() # .item converts tensor to numpy array.
            for l in range(NUM_TOKEN):
                pred_l_numpy = pred.numpy()[:, l, :]
                y_l_numpy = y.numpy()[:, l]
                tmp = np.where(np.argmax(pred_l_numpy, axis=1) == y_l_numpy)[0]
                correct += len(tmp)

    test_loss /= num_batches
    correct /= (size * NUM_TOKEN * 1.0)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg minibatch loss: {test_loss:>8f} \n")


### 7. main function ###

# Main function for train and test
epochs = 5 # one epoch visits all data points.
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
#
# # Save model parameters
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")