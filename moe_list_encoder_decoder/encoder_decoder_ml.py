import math
import os
import json

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def load_data(data_path):
    file_list = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(data_path)
        for file in files
    ]
    schemas = [json.load(open(file, "r")) for file in file_list]
    _data = [schema["mo_energies"] for schema in schemas]
    return _data


def process_data(_data):
    _data = [[math.log(-e) for e in sub_list] for sub_list in tqdm(_data)]
    flattened_data = [i for sd in tqdm(_data) for i in sd]
    _mean = np.mean(flattened_data)
    _std = np.std(flattened_data)
    _data = [[(e - _mean) / _std for e in sub_list] for sub_list in tqdm(_data)]
    print("Data Mean:", _mean)
    print("Data STD:", _std)
    return _data


def pad_data(_data, _max_length):
    _data = torch.tensor([seq + [0] * (_max_length - len(seq)) for seq in tqdm(_data)])
    return _data


def get_loaders_train_test_val(
    _data, _train_batch_size, _eval_batch_size, train_frac, test_frac, val_frac
):
    if not (train_frac + test_frac + val_frac) == 1:
        raise ValueError()
    train_idx = int(len(_data) * train_frac)
    test_idx = int(len(_data) * test_frac) + train_idx
    train_data, test_data, val_data = (
        _data[:train_idx],
        _data[train_idx:test_idx],
        _data[test_idx:],
    )
    max_length = max(len(seq) for seq in train_data + test_data + val_data)
    print("Train", len(train_data))
    print("Test", len(test_data))
    print("Val", len(val_data))
    padded_train = pad_data(train_data, max_length)
    padded_test = pad_data(test_data, max_length)
    padded_val = pad_data(val_data, max_length)
    train_dataloader = get_dataloader_uni(padded_train, _train_batch_size, True)
    test_dataloader = get_dataloader_uni(padded_test, _eval_batch_size, False)
    val_dataloader = get_dataloader_uni(padded_val, _eval_batch_size, False)
    return train_dataloader, test_dataloader, val_dataloader


def get_dataloader_uni(_data, _batch_size, shuffle):
    _dataset = TensorDataset(_data, _data)
    _dataloader = DataLoader(_dataset, batch_size=_batch_size, shuffle=shuffle)
    return _dataloader


def get_dataloader(_x, _y, batch_size, shuffle):
    _dataset = TensorDataset(_x, _y)
    _dataloader = DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle)
    return _dataloader


class EncoderDecoder(nn.Module):
    def __init__(self, z_size):
        super(EncoderDecoder, self).__init__()
        self.f1 = nn.Linear(706, 350)
        self.f2 = nn.Linear(350, 175)
        self.f3 = nn.Linear(175, 50)
        self.f4 = nn.Linear(50, 25)
        self.f5 = nn.Linear(25, z_size)
        self.l5 = nn.Linear(z_size, 25)
        self.l4 = nn.Linear(25, 50)
        self.l3 = nn.Linear(50, 175)
        self.l2 = nn.Linear(175, 350)
        self.l1 = nn.Linear(350, 706)

    def encode(self, x):
        x = self.f1(x)
        x = self.f2(torch.relu(x))
        x = self.f3(torch.relu(x))
        x = self.f4(torch.relu(x))
        x = self.f5(torch.relu(x))
        return x

    def decode(self, x):
        x = self.l5(torch.relu(x))
        x = self.l4(torch.relu(x))
        x = self.l3(torch.relu(x))
        x = self.l2(torch.relu(x))
        x = self.l1(torch.relu(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


if __name__ == "__main__":
    data = load_data("polymer_db_full")
    data = process_data(data)

    lr = 1e-3
    train_batch_size = 8
    eval_batch_size = 32
    epochs = 75

    all_training_losses = dict()

    train_dl, test_dl, val_dl = get_loaders_train_test_val(
        data, train_batch_size, eval_batch_size, 0.7, 0.15, 0.15
    )
    for z_size in [4, 8, 16]:
        print("CURRENT Z SIZE", z_size)
        train_losses, test_losses, val_losses, model_performances = (
            list(),
            list(),
            list(),
            dict(),
        )
        model = EncoderDecoder(z_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        try:
            for epoch in tqdm(range(1, epochs + 1)):
                train_loss_sum = 0
                model.train()
                for inputs, targets in train_dl:
                    optimizer.zero_grad()
                    outputs = model(inputs.float())
                    train_loss = criterion(outputs, targets.float())
                    train_loss.backward()
                    optimizer.step()
                    train_loss_sum += train_loss.item()
                train_losses.append(train_loss_sum)

                test_loss_sum = 0
                val_loss_sum = 0
                model.eval()
                with torch.no_grad():
                    for inputs, targets in test_dl:
                        outputs = model(inputs.float())
                        test_loss = criterion(outputs, targets.float())
                        test_loss_sum += test_loss.item()
                    test_losses.append(test_loss_sum)
                    for inputs, targets in val_dl:
                        outputs = model(inputs.float())
                        validation_loss = criterion(outputs, targets.float())
                        val_loss_sum += validation_loss.item()
                    val_losses.append(val_loss_sum)

                if epoch % 25 == 0:
                    print(
                        f"Epoch: {epoch} - Loss: {train_loss_sum} - Validation loss: {val_loss_sum} - Test loss: {test_loss_sum}"
                    )
        except KeyboardInterrupt:
            pass
        finally:
            all_training_losses[z_size] = train_losses
            torch.save(model.state_dict(), f"model_{z_size}_z_4_layer.pt")
            plt.plot(train_losses, label="Train")
            plt.plot(test_losses, label="Test")
            plt.plot(val_losses, label="Val")
            plt.title(
                f"EP {epochs} - LR {lr} - BATCH {train_batch_size} - VAL LOSS {train_losses[-1]:.7f}"
            )
            plt.legend()
            plt.show()

    plt.figure(figsize=(10, 10))
    for z_size, z_train_losses in all_training_losses.items():
        plt.plot(
            z_train_losses,
            label=f"Z {z_size} – Loss {z_train_losses[-1]:.7f}",
        )
    plt.legend()
    plt.show()
