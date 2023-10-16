import time
import torch
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from itertools import product

from func import get_model_name
from objects import Data, OutputSpace, Record


def train_test(
    model: nn.Module,
    loaded_data: Data,
    lr: float,
    epochs,
    with_tqdm: bool,
    model_path: str,
    save_model: bool = True,
    print_every_n_epochs: int = 10,
) -> (nn.Module, Record):
    model = model.float()
    train_losses, test_losses = [], []
    criterion, optimizer = nn.MSELoss(), Adam(model.parameters(), lr=lr)
    time.sleep(1)

    for epoch in [tqdm(range(1, epochs + 1)) if with_tqdm else range(1, epochs + 1)][0]:
        train_loss, test_loss = 0, 0

        # Training Loop
        for batch in loaded_data.train:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.float().squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Testing Loop
        x_real, x_pred = [], []
        with torch.no_grad():
            for batch in loaded_data.test:
                out = model(batch)
                x_real.extend(batch.y.numpy().tolist())
                x_pred.extend(out.detach().numpy().tolist())
                loss = criterion(out, batch.y.float().squeeze())
                test_loss += loss.item()

        train_loss = train_loss / len(loaded_data.train.dataset)
        test_loss = test_loss / len(loaded_data.test.dataset)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if epoch % print_every_n_epochs == 0:
            print(
                f"epoch {epoch}/{epochs} - train loss {train_loss} - test loss {test_loss}"
            )
            plt.scatter(
                x_real,
                x_pred,
                label=f"Pearson: {pearsonr(x_real, x_pred)[0]:.5f}",
            )
            plt.title(f"Epoch {epoch}/{epochs} - Pearson's Correlation")
            plt.xlabel("x")
            plt.ylabel("x_pred")
            plt.legend(loc="upper left")
            plt.grid(True)
            plt.show()
        time.sleep(1)
    if save_model:
        torch.save(model.state_dict(), model_path)
    return model, Record(
        get_model_name(model.__class__),
        model.conv,
        model.linear,
        lr,
        train_losses,
        test_losses,
    )


def train_obj_handler(
    models: list,
    loaded_data: Data,
    lr: float,
    epochs: int,
):
    records = []
    for ind, model in enumerate(models):
        print(f"{ind+1}/{len(models)} - conv {model.conv} - linear {model.linear}")
        _, record = train_test(
            model,
            loaded_data,
            lr,
            epochs,
        )
        records.append(record)
    return OutputSpace(records)


# output_space = train_param_handler(
#     hparams=Hyperparameters(
#         models=[GraphOne, GraphNormOne, GraphNormDeepOne],
#         conv_layers=const_search(
#             max_layers=8, min_layers=8, max_depth=6, min_depth=6
#         ),
#         dnn_layers=[0],
#         learning_rates=[0.001],
#         epochs=25,
#     ),
#     loaded_data=data,
# )
