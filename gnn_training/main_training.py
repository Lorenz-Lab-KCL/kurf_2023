from datetime import datetime

import torch
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from collections import OrderedDict
from objects import Data, GraphOne, OutputSpace
from hparam_tuning import train_test

if __name__ == "__main__":
    mode = "homo"
    # mode = "lumo"
    data = Data(
        output_type=mode,
        batch_size=256,
        test_size=0.3,
        plot=False,
    )
    s_normaliser = data.sustained_normaliser
    conv_layers = [256, 256, 256, 256, 256]
    model = GraphOne(data.feature_count, conv_layers)

    # use the code below to train the model
    model_path = f"model_norm_test_normaliser_{mode}.pth"
    # model, record = train_test(
    #     model,
    #     data,
    #     lr=0.001,
    #     epochs=100,
    #     with_tqdm=True,
    #     model_path=model_path,
    #     save_model=True,
    #     print_every_n_epochs=20,
    # )
    # output_space = OutputSpace([record])
    # output_space.save("output_15.json")

    # the code below loads the trained model
    checkpoint = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    # the code below calculates the pearsons coefficient
    # to gauge the accuracy of the model's normaliser
    model.eval()
    x_real, x_pred = [], []
    for batch in data.test:
        out = model(batch)
        reversed_target = s_normaliser.reverse(out)
        x_real.extend(s_normaliser.reverse(batch.y).numpy().tolist())
        x_pred.extend(s_normaliser.reverse(out))
    plt.scatter(
        x_real,
        x_pred,
        label=f"Pearson: {pearsonr(x_real, x_pred)[0]:.5f}",
    )
    plt.title(f"Pearson's Correlation")
    plt.xlabel("x")
    plt.ylabel("x_pred")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()

    # the code below is used to keep the model alive,
    # so you can run custom inputs from the polymer_db_full
    # folder on it
    # type filename for output
    # type end to end loop
    model.eval()
    while True:
        pin = input("Enter the filename you want calculated: ")
        if pin.lower() == "end":
            break
        for batch in s_normaliser.prepare_for_ml(f"polymer_db_full/{pin}.json"):
            now = datetime.now()
            out = model(batch)
            then = datetime.now()
            print("Time taken", then - now)
            print("Real", s_normaliser.reverse(batch.y))
            print("Rev Norm", s_normaliser.reverse(out))

