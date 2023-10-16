import numpy
import openbabel.pybel as pb
import os
from tqdm import tqdm


def correct_smiles(mol_xyz_3d_coords):
    head = f"{len(mol_xyz_3d_coords)}\n"
    numpy.savetxt("temp.xyz", mol_xyz_3d_coords, delimiter=" ", fmt="%s", header=head, comments="")
    mol = next(pb.readfile("xyz", "temp.xyz"))

    # mol.OBMol.DeleteHydrogens()
    mol.OBMol.ConnectTheDots()
    mol.OBMol.PerceiveBondOrders()
    # mol.OBMol.AddHydrogens()

    corrected = mol.write("smi").split("\t")
    os.unlink("temp.xyz")
    return corrected[0]


if __name__ == "__main__":
    results = Fld().seek_files("json")
    df = open("results.dat", "w")

    for i in tqdm(results):
        df.write(str(os.path.basename(i)) + " " + str(correct_smiles(i)) + "\n")

    df.close()
