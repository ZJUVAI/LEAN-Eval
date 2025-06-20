from LeanEval.datasets.downloader import GitHubDownloader, HuggingFaceDownloader, get_downloader

# 示例 1：下载并处理 miniF2F (HF)
# hf_downloader = HuggingFaceDownloader(
#     url="hoskinson-center/proofnet",
#     output_dir="./data/downloaded"
# )
# hf_downloader.download()
# # hf_downloader.process("./data/json/minif2f_hf.json")
# print("hf download success")

# hf_downloader = HuggingFaceDownloader(
#     url="hoskinson-center/proof-pile",
#     output_dir="./data/downloaded"
# )
# hf_downloader.download()
# hf_downloader.process("./data/json/ProofNet_hf.json")
# print("hf download success")

# # 示例 2：下载并处理 miniF2F (GitHub)
github_downloader = GitHubDownloader(
    url="https://github.com/deepseek-ai/DeepSeek-Prover-V1.5.git",
    output_dir="./data/downloaded"
)
github_downloader.download()
print('github download success')


