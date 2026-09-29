import json

with open("NeonCortrex_03_Model_Training.ipynb", "r") as f:
    nb = json.load(f)

with open("final_training_script.py", "w") as f:
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            f.write("".join(cell.get("source", [])) + "\n\n")
print("Done")
