from data_loader import LeanQuestionDataset # 导入自定义的数据集类
from torch.utils.data import DataLoader# 导入 PyTorch 的数据加载器工具
from verifier import verify_lean_file
from model_infer import DeepSeekLeanProver# 导入 DeepSeek 模型封装类
from tqdm import tqdm

deepseek_api_key = "sk-c5172f87dfe4418899fefd6cb6ee7309"
dataset_path = '../data/dataset.jsonl'
deepseek_chatbot = DeepSeekLeanProver(api_key=deepseek_api_key)


def load_dataset(dataset_path: str) -> DataLoader:
    """
    加载数据集并返回 DataLoader

    使用自定义 LeanQuestionDataset，
    设置 collate_fn=list 保证每个 batch 是 List[Dict[str, Any]]
    """
    dataset = LeanQuestionDataset(dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=list  # 不进行自动 Tensor 合并，保持原始结构
    )
    return dataloader


def run(loader, model):
    total_shot = 0  # 成功验证的数量计数器

    # 遍历所有 batch
    for batch in tqdm(loader):
        # 遍历每个样本（每个问题）
        for item in tqdm(batch):
            question = item["question"]
            # 调用模型生成 Lean 代码并写入 proof_n.lean 文件中
            model.nl_prove(question)

    # 验证所有 Lean 文件的正确性
    for i in range(1, len(loader.dataset)):
        lean_file = f"../output/proof_{i}.lean"
        result = verify_lean_file(lean_file)
        # 验证成功（returncode 为 0）
        if result[0] == 0:
            total_shot += 1

    print(f"Total successful proofs: {total_shot}/{len(loader.dataset)}")


if __name__ == '__main__':
    # 加载数据
    loader = load_dataset(dataset_path)
    # 执行生成 + 验证流程
    run(loader, deepseek_chatbot)
