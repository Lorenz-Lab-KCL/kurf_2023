import json
import math

from torch import nn
import plotly.graph_objects as go
from torch.nn import functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch.nn import BatchNorm1d, LeakyReLU, Linear

from data_load import get_train_test


class GraphOne(nn.Module):
    def __init__(self, feature_count: int, conv_layers: list, out: int = 1):
        super(GraphOne, self).__init__()
        self.conv = conv_layers
        self.linear = [conv_layers[-1]] + [out]
        self.conv_layers = nn.ModuleList()
        for in_ch, out_ch in zip([feature_count] + conv_layers[:-1], conv_layers):
            self.conv_layers.append(GATConv(in_ch, out_ch))
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(conv_layers[-1], out))
        self.final_layer = nn.Linear(conv_layers[-1], out)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.leaky_relu(x, 0.01)
        x = self.pool(x, batch)
        x = self.final_layer(x)
        return x.squeeze(-1)


class GraphDeepOne(nn.Module):
    def __init__(self, feature_count, conv_layers, dnn_layers, out=1):
        super(GraphDeepOne, self).__init__()
        self.conv = [feature_count] + conv_layers
        self.linear = [conv_layers[-1]] + dnn_layers + [out]
        self.conv_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for in_ch, out_ch in zip([feature_count] + conv_layers[:-1], conv_layers):
            self.conv_layers.append(GATConv(in_ch, out_ch))
        self.pool = AttentionalAggregation(
            gate_nn=nn.Linear(conv_layers[-1], conv_layers[-1])
        )
        for in_ch, out_ch in zip([conv_layers[-1]] + dnn_layers, dnn_layers + [out]):
            self.linear_layers.append(nn.Linear(in_ch, out_ch))

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.leaky_relu(x, 0.01)
        x = self.pool(x, batch)
        for linear in self.linear_layers:
            x = F.leaky_relu(linear(x), 0.01)
        return x.squeeze(-1)


class Data:
    def __init__(
        self,
        output_type: str,
        batch_size: int,
        test_size: float,
        plot: bool,
    ):
        (
            self.train,
            self.test,
            self.feature_count,
            self.sustained_normaliser,
        ) = get_train_test(
            output_type,
            batch_size,
            test_size,
            plot,
        )


class Record:
    def __init__(
        self,
        model_name,
        conv_layer,
        dnn_layer,
        learning_rate,
        train_losses,
        test_losses,
    ):
        self.model_name = model_name
        self.conv = conv_layer
        self.dnn = dnn_layer
        self.lr = learning_rate
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.epochs = len(train_losses)

    @property
    def json(self):
        out = {
            "name": self.model_name,
            "conv": self.conv,
            "dnn": self.dnn,
            "lr": self.lr,
            "epochs": self.epochs,
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
        }
        return out

    @staticmethod
    def load_json(rec):
        rec = Record(
            conv_layer=rec["conv"],
            dnn_layer=rec["dnn"],
            learning_rate=rec["lr"],
            train_losses=rec["train"],
            test_losses=rec["test"],
        )
        return rec


class OutputSpace:
    def __init__(self, records: list[Record]):
        self.records = records

    @staticmethod
    def load(file_path):
        with open(file_path) as json_file:
            return json.load(json_file)["records"]

    def save(self, file_path):
        with open(file_path, "w") as fp:
            json.dump(
                {"records": [record.json for record in self.records]}, fp, indent=4
            )
            return None

    @staticmethod
    def plot_connections_epoch_history_3d(
        records, layer_type: str, std_cutoff=2, colour_scale="viridis"
    ):
        for record in records:
            layers = record[layer_type]
            connections = sum(
                [layers[i] * layers[i + 1] for i in range(len(layers) - 1)]
            )
            record["connections"] = connections
        epochs = list(range(1, records[0]["epochs"] + 1))
        x_values = []
        y_values = []
        z_values_train = []
        z_values_test = []
        hover_texts = []
        for record in records:
            x_values.extend(epochs)
            y_values.extend([record["connections"]] * len(epochs))
            z_values_train.extend(record["train_losses"])
            z_values_test.extend(record["test_losses"])
            hover_texts.extend(
                [f"{str(record[layer_type])} {record['lr']}"] * len(epochs)
            )

        # Calculate mean and standard deviation for train losses
        mean_train_loss = sum(z_values_train) / len(z_values_train)
        std_dev_train_loss = (
            sum([(x - mean_train_loss) ** 2 for x in z_values_train])
            / len(z_values_train)
        ) ** 0.5
        z_values_train = [
            -0.1 if x > mean_train_loss + std_cutoff * std_dev_train_loss else x
            for x in z_values_train
        ]
        # mean_test_loss = sum(z_values_test) / len(z_values_test)
        # std_dev_test_loss = (sum([(x - mean_test_loss) ** 2 for x in z_values_test]) / len(z_values_test)) ** 0.5
        # z_values_test = [0 if x > mean_test_loss + 3 * std_dev_test_loss else x for x in z_values_test]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values_train,
                mode="markers",
                marker=dict(
                    size=6,
                    color=z_values_train,
                    colorscale=colour_scale,
                    opacity=0.8,
                    # cmin=0,
                    # cmax=2,
                ),
                hovertext=hover_texts,
                name="Train Losses",
            )
        )
        # fig.add_trace(
        #     go.Scatter3d(
        #         x=x_values,
        #         y=y_values,
        #         z=z_values_test,
        #         mode="markers",
        #         marker=dict(
        #             size=6,
        #             color=z_values_test,
        #             colorscale=colour_scale,
        #             opacity=0.8,
        #             cmin=0,
        #             cmax=2,
        #         ),
        #         hovertext=hover_texts,
        #         name="Test Losses",
        #     )
        # )
        fig.update_layout(
            title="3D Scatter plot of Train and Test Losses",
            scene=dict(
                xaxis_title="Epoch", yaxis_title="Connections", zaxis_title="Loss"
            ),
        )
        fig.show()

    @staticmethod
    def plot_conv_dnn_last_loss_3d(records, std_cutoff=0.5, colour_scale="viridis"):
        for record in records:
            conv_layers = record["conv"]
            dnn_layers = record["dnn"]

            conv_connections = sum(
                [
                    conv_layers[i] * conv_layers[i + 1]
                    for i in range(len(conv_layers) - 1)
                ]
            )
            dnn_connections = sum(
                [dnn_layers[i] * dnn_layers[i + 1] for i in range(len(dnn_layers) - 1)]
            )

            record["conv_connections"] = conv_connections
            record["dnn_connections"] = dnn_connections

        x_values = []
        y_values = []
        z_values_train = []
        hover_texts = []

        for record in records:
            x_values.append(record["conv_connections"])
            y_values.append(record["dnn_connections"])
            z_values_train.append(record["train_losses"][-1])
            hover_texts.append(
                f"Conv: {str(record['conv'])}, DNN: {str(record['dnn'])}, LR: {record['lr']}"
            )

        mean_train_loss = sum(z_values_train) / len(z_values_train)
        # std_dev_train_loss = (sum([(x - mean_train_loss) ** 2 for x in z_values_train]) / len(z_values_train)) ** 0.5
        # z_values_train = [-0.1 if x > mean_train_loss + std_cutoff * std_dev_train_loss else x for x in z_values_train]
        # z_values_train = [-0.05 if x > 0.03 else x for x in z_values_train]  # manual cutoff
        z_values_train = [1 / x for x in z_values_train]  # manual cutoff

        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values_train,
                mode="markers",
                marker=dict(
                    size=6, color=z_values_train, colorscale=colour_scale, opacity=0.8
                ),
                hovertext=hover_texts,
                name="Train Losses",
            )
        )

        fig.update_layout(
            title="3D Scatter plot of Train Losses",
            scene=dict(
                xaxis_title="Conv Connections",
                yaxis_title="DNN Connections",
                zaxis_title="Loss",
            ),
        )
        fig.show()

    @staticmethod
    def plot_conv_epochs_2d(records):
        train_last_values = []
        test_last_values = []
        connection_counts = []
        for record in records:
            conv = record["conv"]
            connections = sum([conv[i] * conv[i + 1] for i in range(len(conv) - 1)])

            train_last_values.append(
                record["train_losses"][-1]
            )  # Append last train loss
            test_last_values.append(record["test_losses"][-1])  # Append last test loss
            connection_counts.append(connections)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=connection_counts,
                y=train_last_values,
                mode="markers",
                name="Train Last Values",
                marker=dict(size=10, color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=connection_counts,
                y=test_last_values,
                mode="markers",
                name="Test Last Values",
                marker=dict(size=10, color="red"),
            )
        )
        fig.update_layout(
            title="Distribution of Last Train and Test Values vs. Connection Counts",
            xaxis_title="Connection Count",
            yaxis_title="Loss Value",
        )
        fig.show()

    @staticmethod
    def plot_history_2d(record):
        train_values = record["train_losses"]
        test_values = record["test_losses"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(train_values))),
                y=train_values,
                mode="markers",
                name="Train Values",
                marker=dict(size=10, color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(train_values))),
                y=test_values,
                mode="markers",
                name="Test Values",
                marker=dict(size=10, color="red"),
            )
        )
        fig.update_layout(
            title="Distribution of Last Train and Test Values vs. Connection Counts",
            xaxis_title="Connection Count",
            yaxis_title="Loss Value",
        )
        fig.show()
