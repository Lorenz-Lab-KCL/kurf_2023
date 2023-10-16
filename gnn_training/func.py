import math
import numpy as np


def get_x(max_layers, min_layers, depth):
    layers = [round(i) for i in np.linspace(min_layers, max_layers, num=depth)]
    return [layers]


def get_neg_x(max_layers, min_layers, depth):
    layers = [round(i) for i in np.linspace(min_layers, max_layers, num=depth)]
    return [layers[::-1]]


def const_search(max_layers, min_layers, max_depth, min_depth):
    layers = []
    for depth in range(min_depth, max_depth + 1):
        for layer_size in range(min_layers, max_layers + 1):
            layers.append([2**layer_size for i in range(depth)])
    return layers


"""

start=32, depth=4, type="const"
[32]
[32, 32]
[32, 32, 32]
[32, 32, 32, 32]

start=32, depth=4, type="x", max_val=64
[32]
[32, 64]
[32, 64, 64]

start=32, depth=4, type="-x", min_val=16
[32]
[32, 16]
[32, 16, 16]


"""


def get_hill(max_layers, min_layers, depth, max_ratio):
    layers = get_x(int(max_layers * max_ratio), min_layers, math.ceil(depth / 2))[0]
    return [layers + layers[::-1][1:]]


def get_valley(max_layers, min_ratio, depth):
    layers = get_x(max_layers, int(max_layers * min_ratio), math.ceil(depth / 2))[0]
    return [layers[::-1] + layers[1:]]


def get_exp_decay(max_layers, min_layers, depth):
    if depth == 1:
        return [[min_layers]]
    base = math.exp(math.log(min_layers / max_layers) / (depth - 1))
    layers = [round(max_layers * (base**d)) for d in range(depth)]
    return [layers]


layer_functions = {
    "=": const_search,
    "x": get_x,
    "-x": get_neg_x,
    "l": get_exp_decay,
    "<>": get_hill,
    "><": get_valley,
}


def layer_search(
    max_layers,
    min_layers,
    max_depth,
    density,
    min_depth=1,
    max_ratio=1.5,
    min_ratio=0.5,
    shapes=None,
) -> list[list[int]]:
    if not isinstance(shapes, list):
        shapes = ["=", "-x", "l", "<>"]
    layers = []
    for depth in range(min_depth, max_depth + 1):
        for shape in shapes:
            match shape:
                case "=":
                    layers.extend(
                        const_search(
                            max_layers=max_layers,
                            min_layers=min_layers,
                            max_depth=max_depth,
                            min_depth=min_depth,
                        )
                    )
                case "x":
                    layers.extend(
                        get_x(max_layers=max_layers, min_layers=min_layers, depth=depth)
                    )
                case "-x":
                    layers.extend(
                        get_neg_x(
                            max_layers=max_layers, min_layers=min_layers, depth=depth
                        )
                    )
                case "<>":
                    layers.extend(
                        get_hill(
                            max_layers=max_layers,
                            min_layers=min_layers,
                            depth=depth,
                            max_ratio=max_ratio,
                        )
                    )
                case "><":
                    layers.extend(
                        get_valley(
                            max_layers=max_layers, min_ratio=min_ratio, depth=depth
                        )
                    )
                case "l":
                    layers.extend(
                        get_exp_decay(
                            max_layers=max_layers, min_layers=min_layers, depth=depth
                        )
                    )
    unique_list = []
    for layer in layers:
        if layer not in unique_list:
            unique_list.append(layer)
    return unique_list


class Hyperparameters:
    def __init__(
        self,
        models=None,
        conv_layers=None,
        dnn_layers=None,
        learning_rates=None,
        epochs=None,
    ):
        self.models = models
        self.conv = conv_layers
        self.dnn = dnn_layers
        self.lr = learning_rates
        self.ep = epochs


def get_model_name(model):
    return str(model).split()[1][1:-2].split(".")[1]
