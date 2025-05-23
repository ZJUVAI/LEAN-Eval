from datasets.json_dataset import JsonlDataset
from datasets.json_dataset import JsonDataset
from datasets.csv_dataset import CsvDataset
from datasets.yaml_dataset import YamlDataset
import os
json_path =  "./data/json/data_1.json"
csv_path = "./data/csv/data_1.csv"
yaml_path = "./data/yaml/data_1.yaml"
json_ds = JsonDataset(json_path)
csv_ds = CsvDataset(csv_path)
yaml_ds = YamlDataset(yaml_path)
# print(len(ds))
# print(ds[0].prompt_ready_stmt)

# easy_ds = ds.filter(difficulty=1)

sufix = ".lean"
folder_path = "LeanProblems"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for index,ques in enumerate(yaml_ds):
    with open(os.path.join(folder_path,str(index)+sufix),"w") as f:
        f.write(ques.prompt_ready_stmt)

