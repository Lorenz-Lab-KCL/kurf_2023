import os
import json
import torch
import mendeleev
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

RDLogger.DisableLog("rdApp.*")


atom_features_memory = {
    atomic_number: {
        "atomic_number": atomic_number,
        "atomic_weight": el.atomic_weight,
        "atomic_radius": el.atomic_radius,
        "electronegativity": el.electronegativity(scale="pauling"),
        # "group_id": el.group_id,
        "period": el.period,
        # "covalent_radius_bragg": el.covalent_radius_bragg,
        # "covalent_radius_cordero": el.covalent_radius_cordero,
        # "covalent_radius_pyykko": el.covalent_radius_pyykko,
        # "vdw_radius": el.vdw_radius,
        "specific_heat": el.specific_heat,
        "electron_affinity": el.electron_affinity,
        # "electrons": el.electrons,
        # "protons": el.protons,
        # "neutrons": el.neutrons,
    }
    for atomic_number, el in [(n, mendeleev.element(n)) for n in range(1, 40)]
}
atom_features_memory = {
    k: {k: v if v is not None else 0 for k, v in big_v.items()}
    for k, big_v in tqdm(
        atom_features_memory.items(), desc="Loading atom features to memory"
    )
}


class FeatureNormaliser:
    def __init__(self, data_objects: list, feature_count: int):
        self.data_objects = data_objects
        self.column_names = (
            ["occurrences"]
            + list(atom_features_memory[1].keys())
            + [f"ohe_{i}" for i in range(1, feature_count + 1)]
            + ["target"]
        )
        self.transformers = self.fit(data_objects)

    def data_to_df(self, data_objects: list):
        all_data = []
        for data_object in data_objects:
            y_expanded = data_object.y.repeat(data_object.x.size(0), 1)
            combined_data = torch.cat((data_object.x, y_expanded), dim=1)
            all_data.append(combined_data)
        all_data_np = torch.vstack(all_data).detach().numpy()
        return pd.DataFrame(all_data_np, columns=self.column_names)

    def fit(self, data_objects: list):
        transformers = dict()
        df = self.data_to_df(data_objects)
        chosen_columns = [i for i in df.columns if "ohe" not in i]
        df = df[chosen_columns]
        for column_name in tqdm(chosen_columns, desc="Fitting normalisers"):
            transformer = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(len(df[column_name]), 1000),
            )
            transformer.fit(df[column_name].values.reshape(-1, 1))
            transformers[column_name] = transformer
        return transformers

    def transform(self, data_objects: list, run_type: str):
        normalized_data_objects = []
        chosen_columns = [i for i in self.column_names[:-1] if "ohe" not in i]
        # first_run = True
        for data_object in tqdm(data_objects, desc=f"Normalising {run_type}"):
            new_data_object = data_object.clone()
            for i, col_name in enumerate(chosen_columns):  # Excluding 'target'
                transformed_feature = self.transformers.get(col_name).transform(
                    data_object.x[:, i].reshape(-1, 1)
                )
                new_data_object.x[:, i] = torch.tensor(transformed_feature).squeeze()
            transformed_target = self.transformers["target"].transform(
                data_object.y.reshape(-1, 1)
            )
            # if run_type.lower() == "sustained" and first_run:
            #     print("NORMALISER TEST")
            #     print("raw", data_object.y.reshape(-1, 1).numpy())
            #     print("transformed", transformed_target)
            #     print("reversed", self.transformers["target"].inverse_transform(transformed_target))
            #     print()
            #     first_run = False
            new_data_object.y = torch.tensor(transformed_target).squeeze()
            # new_data_object.y = data_object.y
            normalized_data_objects.append(new_data_object)
        return normalized_data_objects

    def plot_distributions(self, data_objects: list, run_type: str):
        df = self.data_to_df(data_objects)
        chosen_columns = [i for i in self.column_names if "ohe" not in i]
        n = len(chosen_columns)
        fig, ax = plt.subplots(n, n, figsize=(20, 20))
        fig.suptitle(run_type, fontsize=16)
        for i, feature_x in tqdm(enumerate(chosen_columns), desc="Plotting"):
            for j, feature_y in enumerate(chosen_columns):
                if i < j:
                    ax[j, i].scatter(df[feature_x], df[feature_y], s=3)
                    ax[j, i].set(xlabel=feature_x, ylabel=feature_y)
                elif i == j:
                    ax[j, i].hist(df[feature_x], bins=30)
                    ax[j, i].set(title=feature_x)
                ax[j, i].grid(True)
        plt.tight_layout()
        plt.show()


def get_x_y(path: str, output_type: str):
    file_list = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(path)
        for file in files
        if file not in (".DS_Store",)
    ]
    schemas = [json.load(open(file, "r")) for file in file_list]
    if output_type.lower() == "homo":
        raw_y = [schema["mo_energies"][-1] for schema in schemas]
    elif output_type.lower() == "lumo":
        raw_y = [schema["excited_states"]["triplet"][0] for schema in schemas]
    else:
        raise ValueError()
    return schemas, raw_y


def get_molecule_stats(schemas: list, run_type: str):
    highest_atomic_num = 0
    all_molecule_occurrences = []
    for schema in tqdm(schemas, desc=f"Verifying {run_type} schemas"):
        molecule_occurrences = defaultdict(int)
        mol = Chem.MolFromSmiles(schema["smiles"])
        atoms = []
        for atom in mol.GetAtoms():
            molecule_occurrences[atom.GetAtomicNum()] += 1
            atoms.append(atom.GetAtomicNum())
        if max(atoms) > highest_atomic_num:
            highest_atomic_num = max(atoms)
        all_molecule_occurrences.append(molecule_occurrences)
    return highest_atomic_num, all_molecule_occurrences


def featurise_atoms(atomic_number: int, highest_atomic_num: int, occurrences: dict):
    one_hot = torch.zeros(highest_atomic_num)
    one_hot[atomic_number - 1] = 1
    occurrence_tensor = torch.tensor([occurrences[atomic_number]])
    features = torch.tensor([i for i in atom_features_memory[atomic_number].values()])
    concat_features = torch.cat([occurrence_tensor, features, one_hot])
    return concat_features


def get_molecules(
    schemas: list, outputs: list, all_occurrences, max_atomic_number: int, run_type: str
):
    molecules = [Chem.MolFromSmiles(schema["smiles"]) for schema in schemas]
    dataset = [
        Data(
            x=torch.stack(
                [
                    featurise_atoms(atom.GetAtomicNum(), max_atomic_number, occurrences)
                    for atom in mol.GetAtoms()
                ]
            ),
            edge_index=torch.tensor(
                [
                    (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    for bond in mol.GetBonds()
                ],
                dtype=torch.long,
            ).t(),
            y=torch.tensor(output),
        )
        for mol, occurrences, output in tqdm(
            [i for i in zip(molecules, all_occurrences, outputs)],
            desc=f"Transforming {run_type} schemas",
        )
    ]

    # print(dataset[0].x)
    # print(dataset[0].edge_index)
    # print(dataset[0].y)
    # print(schemas[0])
    # print([atom.GetAtomicNum() for atom in Chem.MolFromSmiles(schemas[0]["smiles"]).GetAtoms()])
    # raise ValueError

    return dataset


class SustainedNormaliser:
    def __init__(self, feature_normaliser: FeatureNormaliser, occurrences, max_atomic_number):
        self.feature_normaliser = feature_normaliser
        self.occurrences = occurrences
        self.max_atomic_number = max_atomic_number

    def prepare_for_ml(self, filepath: str, output_type: str = "homo"):
        raw_x = json.load(open(filepath, "r"))
        if output_type.lower() == "homo":
            raw_y = [raw_x["mo_energies"][-1]]
        elif output_type.lower() == "lumo":
            raw_y = [raw_x["excited_states"]["triplet"][0]]
        else:
            raise ValueError()
        raw_x = [raw_x]
        dataset = get_molecules(
            raw_x, raw_y, self.occurrences, self.max_atomic_number, "sustained"
        )
        dataset_normalised = self.feature_normaliser.transform(dataset, "sustained")
        dataloader = DataLoader(dataset_normalised, batch_size=1)
        # print("1. real raw", raw_y)
        return dataloader

    def reverse(self, data_y):
        data_y = data_y.detach().numpy().reshape(-1, 1)
        data_y = self.feature_normaliser.transformers["target"].inverse_transform(data_y)
        return torch.tensor(data_y).squeeze()


def get_train_test(
    output_type: str, batch_size: int, test_size: float = 0.2, plot=False
):
    train_x, train_y = get_x_y("polymer_db_test", output_type)
    train_max_an, train_occur = get_molecule_stats(train_x, "train")
    train_dataset = get_molecules(train_x, train_y, train_occur, train_max_an, "train")
    normaliser = FeatureNormaliser(train_dataset, train_max_an)

    raw_x, raw_y = get_x_y("polymer_db_full", output_type)
    train_x, test_x, train_y, test_y = train_test_split(
        raw_x, raw_y, test_size=test_size
    )
    _, train_occur = get_molecule_stats(train_x, "train")
    _, test_occur = get_molecule_stats(test_x, "test")
    train_max_an = 34
    train_dataset = get_molecules(train_x, train_y, train_occur, train_max_an, "train")
    test_dataset = get_molecules(test_x, test_y, test_occur, train_max_an, "test")
    # normaliser = FeatureNormaliser(train_dataset, train_max_an)
    train_dataset_normalised = normaliser.transform(train_dataset, "train")
    test_dataset_normalised = normaliser.transform(test_dataset, "test")

    if plot:
        normaliser.plot_distributions(train_dataset, "raw")
        normaliser.plot_distributions(train_dataset_normalised, "train")
        normaliser.plot_distributions(test_dataset_normalised, "test")

    s_normaliser = SustainedNormaliser(normaliser, train_occur, train_max_an)

    train_loader = DataLoader(train_dataset_normalised, batch_size=batch_size)
    test_loader = DataLoader(test_dataset_normalised, batch_size=batch_size)

    return train_loader, test_loader, train_dataset[0].x.shape[1], s_normaliser
