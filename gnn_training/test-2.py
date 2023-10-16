import json

if __name__ == "__main__":
    # running = True
    # while running:
    #     user_in = input("Enter your input: ")
    #     if user_in.lower() == "quit":
    #         running = False
    #     else:
    #         print(user_in)

    schema = json.load(open("polymer_db_full/0_1_10.json", "r"))
    print(json.dumps(schema, indent=4))
