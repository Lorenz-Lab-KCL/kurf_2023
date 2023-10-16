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
    _data = [schema["_mo_energies"] for schema in schemas]
    return _data


def flatten_data(_data):
    return [i for sd in _data for i in sd]


def get_mean_std(_data):
    return np.mean(_data), np.std(_data)


def norm_dist_data(_data):
    mean, std = get_mean_std(_data)
    return [(i - mean) / std for i in _data]


def norm_dist_data_layered(_data):
    mean, std = get_mean_std(flatten_data(_data))
    return [[(i - mean) / std for i in sub_list] for sub_list in _data]


def norm_exp_data_layered(_data):
    return [[math.log(-i) for i in sub_list] for sub_list in _data]


def pad_data(_data, max_length):
    return torch.tensor([i + [0] * (max_length - len(i)) for i in _data])


def split_train_test_val(_data: list, _fractions: tuple):
    train_frac, test_frac, val_frac = _fractions
    if not (train_frac + test_frac + val_frac) == 1:
        raise ValueError()
    train_idx = int(len(_data) * train_frac)
    test_idx = int(len(_data) * test_frac) + train_idx
    return _data[:train_idx], _data[train_idx:test_idx], _data[test_idx:]


def get_dataset(_x_data, _y_data):
    dataset = TensorDataset(_x_data, torch.tensor(np.array(_y_data).reshape(-1, 1)))
    return dataset


def get_dataloader(dataset, batch_size, shuffle):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def get_train_test_val_datasets(_x_data, _y_data, _fractions):
    train_x, test_x, val_x = split_train_test_val(_x_data, _fractions)
    train_y, test_y, val_y = split_train_test_val(_y_data, _fractions)
    # for name, arr in zip(("train", "test", "val"), (train_x, test_x, val_x)):
    #     print(name, len(arr))
    # max_length = max(len(seq) for seq in _x_data)
    max_length = 706
    padded_train_x, padded_test_x, padded_val_x = (
        pad_data(train_x, max_length),
        pad_data(test_x, max_length),
        pad_data(val_x, max_length),
    )
    train_ds, test_ds, val_ds = (
        get_dataset(padded_train_x, train_y),
        get_dataset(padded_test_x, test_y),
        get_dataset(padded_val_x, val_y),
    )
    return train_ds, test_ds, val_ds


def get_train_test_val_dataloaders(_x_data, _y_data, _fractions, _batch_sizes):
    _train_batch_size, _eval_batch_size = _batch_sizes
    train_ds, test_ds, val_ds = get_train_test_val_datasets(
        _x_data, _y_data, _fractions
    )
    train_dataloader = get_dataloader(train_ds, _train_batch_size, True)
    test_dataloader = get_dataloader(test_ds, _eval_batch_size, False)
    val_dataloader = get_dataloader(val_ds, _eval_batch_size, False)
    return train_dataloader, test_dataloader, val_dataloader


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.f1 = nn.Linear(706, 350)
        self.f2 = nn.Linear(350, 128)
        self.f3 = nn.Linear(128, 64)
        self.f4 = nn.Linear(64, 32)
        self.f5 = nn.Linear(32, 16)
        self.f6 = nn.Linear(16, 8)
        self.l6 = nn.Linear(8, 16)
        self.l5 = nn.Linear(16, 32)
        self.l4 = nn.Linear(32, 64)
        self.l3 = nn.Linear(64, 128)
        self.l2 = nn.Linear(128, 350)
        self.l1 = nn.Linear(350, 706)

    def encode(self, x):
        x = self.f1(x)
        x = self.f2(torch.relu(x))
        x = self.f3(torch.relu(x))
        x = self.f4(torch.relu(x))
        x = self.f5(torch.relu(x))
        x = self.f6(torch.relu(x))
        return x

    def decode(self, x):
        x = self.l6(torch.relu(x))
        x = self.l5(torch.relu(x))
        x = self.l4(torch.relu(x))
        x = self.l3(torch.relu(x))
        x = self.l2(torch.relu(x))
        x = self.l1(torch.relu(x))
        return x

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(torch.relu(x))
        x = self.f3(torch.relu(x))
        x = self.f4(torch.relu(x))
        x = self.f5(torch.relu(x))
        x = self.f6(torch.relu(x))
        x = self.l6(torch.relu(x))
        x = self.l5(torch.relu(x))
        x = self.l4(torch.relu(x))
        x = self.l3(torch.relu(x))
        x = self.l2(torch.relu(x))
        x = self.l1(torch.relu(x))
        return x


class LengthPredictor(nn.Module):
    def __init__(self):
        super(LengthPredictor, self).__init__()
        self.f1 = nn.Linear(8, 4)
        self.f2 = nn.Linear(4, 4)
        self.f3 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(self.relu(x))
        x = self.f3(self.relu(x))
        return x


if __name__ == "__main__":
    x_data = load_data("poly_comp_json_small")
    y_data = [len(sub_list) for sub_list in x_data]

    y_data = norm_dist_data(y_data)
    x_data = norm_exp_data_layered(x_data)
    print("Exp norm", x_data[0])
    x_data = norm_dist_data_layered(x_data)

    fractions = 0.7, 0.15, 0.15
    train, test, val = get_train_test_val_datasets(x_data, y_data, fractions)

    encoder = EncoderDecoder()
    encoder.load_state_dict(torch.load("model_full.pt"))

    x = train[0][0]
    list_encoded = encoder.encode(x.float()).detach()
    print("Encoded", list_encoded)

    list_decoded = encoder.decode(list_encoded.float()).detach()
    list_decoded = (list_decoded + 3.8137497243686145) * 1.4538298382843995
    print("Decoded", list_decoded)

    # batch_sizes = 8, 32
    # fractions = 0.7, 0.15, 0.15
    # epochs = 1_000
    # for lr in [1e-3]:
    #     train_losses, test_losses, val_losses = list(), list(), list()
    #     for epoch in tqdm(range(1, epochs + 1)):
    #         model = LengthPredictor()
    #         criterion = nn.MSELoss()
    #         optimizer = optim.SGD(model.parameters(), lr=lr)
    #         model.train()
    #         for inputs, targets in train:
    #             transformed_inputs = encoder.encode(inputs.float()).detach()
    #             print(transformed_inputs)
    #             optimizer.zero_grad()
    #             outputs = model(transformed_inputs.floats())
    #             print(outputs)
    #             print()
    #             train_loss = criterion(outputs, targets.float())
    #             train_loss.backward()
    #             optimizer.step()
    #         train_losses.append(float(train_loss))
    #         model.eval()
    #         with torch.no_grad():
    #             for inputs, targets in test:
    #                 outputs = model(inputs.float())
    #                 test_loss = criterion(outputs, targets.float())
    #             test_losses.append(float(test_loss))
    #
    #             for inputs, targets in val:
    #                 outputs = model(inputs.float())
    #                 validation_loss = criterion(outputs, targets.float())
    #             val_losses.append(float(validation_loss))
    #         if epoch % 100 == 0:
    #             print(
    #                 f"Epoch: {epoch} - Loss: {float(train_losses[-1])} -",
    #                 f"Test loss: {float(test_losses[-1])} -",
    #                 f"Val loss: {float(val_losses[-1])}",
    #             )
    #
    #     # torch.save(model.state_dict(), "model.pt")
    #     plt.plot(train_losses, label="Train")
    #     plt.plot(test_losses, label="Test")
    #     plt.plot(val_losses, label="Val")
    #     plt.title(
    #         f"EP {epochs} - LR {lr} - BATCH {batch_sizes[0]} - VAL LOSS {val_losses[-1]:.7f}"
    #     )
    #     plt.legend()
    #     plt.show()
