from typing import Dict, Iterator
import json

def load_dataset() -> Iterator[Dict[str, str]]:
    """
    从jsonl文件中加载数据集

    Args:
        None

    Returns:
        {
            'id': int,
            'question': str,
            'answer': str
        }

    Raises:
        None
    """
    id = 1
    with open('../data/dataset.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            yield {'id': id, 'question': data['Problem'], 'answer': data['Answer']}
            id += 1

if __name__ == '__main__':
    for data in load_dataset():
        print(data)