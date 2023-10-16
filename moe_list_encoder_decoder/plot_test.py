import os
import json
import matplotlib.pyplot as plt
from collections import Counter


def plot_frequency(lst):
    counter = Counter(lst)
    elements, frequencies = zip(*counter.items())

    plt.bar(elements, frequencies)
    plt.xlabel("Elements")
    plt.ylabel("Frequency")
    plt.title("Frequency of Elements")
    plt.show()


file_list = [
    os.path.join(root, file)
    for root, dirs, files in os.walk("poly_comp_json_small")
    for file in files
]

schemas = [json.load(open(file, "r")) for file in file_list]

mo_energies = [schema["_mo_energies"] for schema in schemas]

mo_energies_len = [len(mo_energy) for mo_energy in mo_energies]


plot_frequency(mo_energies_len)
