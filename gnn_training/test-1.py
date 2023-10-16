import json

if __name__ == "__main__":
    # with open("atom.json") as json_file:
    #     val = json.load(json_file)["3d_coords"]
    #     for i in val:
    #         print(i)

    # Read data from TXT file
    with open("atom.txt", "r") as f:
        lines = f.readlines()

    # Process each line to extract data
    data = [eval(line.strip()) for line in lines if line.strip()]

    # print(data[0])
    # print(type(data[0]))

    # Write to XYZ file
    with open("molecule.xyz", "w") as f:
        # Number of atoms
        f.write(str(len(data)) + "\n")
        # Comment line
        f.write("Converted from TXT to XYZ\n")
        # Atom data
        for atom in data:
            f.write(f"{atom[0]} {atom[1]:.6f} {atom[2]:.6f} {atom[3]:.6f}\n")

    print("XYZ file has been created as molecule.xyz")
