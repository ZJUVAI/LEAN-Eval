# from typing import Dict, Iterator
# import json

# def load_dataset(filename) -> Iterator[Dict[str, str]]:
#     """
#     从jsonl文件中加载数据集

#     Args:
#         None

#     Returns:
#         {
#             'id': int,
#             'question': str,
#             'answer': str
#         }

#     Raises:
#         None
#     """
#     id = 1
#     with open('../data/dataset.jsonl', 'r') as f:
#         for line in f:
#             data = json.loads(line.strip())
#             yield {'id': id, 'question': data['Problem'], 'answer': data['Answer']}
#             id += 1

# if __name__ == '__main__':
#     for data in load_dataset():
#         print(data)
import json
from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader


class LeanQuestionDataset(Dataset):
    """
    从 .jsonl 文件中加载数据集的 PyTorch Dataset 封装类
    每一行数据格式为：
    {"Problem": "...", "Answer": "..."}
    """

    def __init__(self, filepath: str):
        self.data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, start=1):
                obj = json.loads(line.strip())
                self.data.append({
                    'id': idx,
                    'Solution': obj.get('Solution', ''),
                    'question': obj.get('Problem', ''),
                    'answer': obj.get('Answer', '')
                })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, Any]:
        return self.data[index]

if __name__ == '__main__':
    dataset_path = '../data/dataset.jsonl'
    dataset = LeanQuestionDataset(dataset_path)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(dataloader))
    # print(batch)
    print(f"id is :\n{batch['id'][0]}")
    print(f"question is :\n{batch['question'][0]}")
    print(f"Solution is :\n{batch['Solution'][0]}")
    print(f"answer is :\n{batch['answer'][0]}")

        

