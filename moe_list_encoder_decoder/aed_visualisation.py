import json
import math
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from encoder_decoder_ml import load_data, process_data, get_loaders_train_test_val



class BigEncoderDecoder(nn.Module):
    def __init__(self, z_size):
        super(BigEncoderDecoder, self).__init__()
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


class EncoderDecoder(nn.Module):
    def __init__(self, z_size):
        super(EncoderDecoder, self).__init__()
        self.f1 = nn.Linear(706, z_size)
        self.f2 = nn.Linear(z_size, z_size)
        self.f3 = nn.Linear(z_size, z_size)
        self.l3 = nn.Linear(z_size, z_size)
        self.l2 = nn.Linear(z_size, z_size)
        self.l1 = nn.Linear(z_size, 706)

    def encode(self, x):
        x = self.f1(x)
        x = self.f2(torch.relu(x))
        x = self.f3(torch.relu(x))
        return x

    def decode(self, x):
        x = self.l3(torch.relu(x))
        x = self.l2(torch.relu(x))
        x = self.l1(torch.relu(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


if __name__ == "__main__":
    filepath = "polymer_db_test"
    lr = 5e-4
    train_batch_size = 1
    eval_batch_size = 1
    epochs = 1000

    data = load_data(filepath)
    data = process_data(data)
    train_losses, test_losses, val_losses, model_performances = (
        list(),
        list(),
        list(),
        dict(),
    )

    train_dl, test_dl, val_dl = get_loaders_train_test_val(
        data, train_batch_size, eval_batch_size, 0.7, 0.15, 0.15
    )
    my_num = 16
    model_path = f"model_{my_num}_z_4_layer.pt"
    model = BigEncoderDecoder(my_num)
    model.load_state_dict(torch.load(model_path))

    for ind, data in enumerate([i for i in train_dl][:5]):
        y = data[0].numpy().tolist()[0]
        y_hat = (
            model(torch.from_numpy(np.array(y)).detach().float())
            .detach()
            .numpy()
            .tolist()
        )

        plt.title(f"Molecule #{ind} {my_num} z")
        plt.plot(y_hat, label="Decoded")
        plt.plot(y, label="Original")
        plt.legend()
        plt.show()
