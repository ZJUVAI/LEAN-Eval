# LeanEval/datasets/downloader.py
import os
import subprocess
import shutil
import json
from pathlib import Path
from urllib.parse import urlparse, unquote
from abc import ABC, abstractmethod
from typing import List, Dict

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

# 导入处理逻辑 (假设 process_dataset.py 被重构或其函数可导入)
from LeanEval.utils.process_dataset import parse_lean_file
from LeanEval.datasets.schema import LeanItem

class BaseDownloader(ABC):
    """数据集下载器的抽象基类。"""

    def __init__(self, url: str, output_dir: str | Path):
        self.url = url
        self.output_dir = Path(output_dir)
        self.download_path = None # 下载后的具体路径

    @abstractmethod
    def download(self) -> Path:
        """执行下载操作，并返回下载内容的本地路径。"""
        pass

    def _get_repo_name(self) -> str:
        """从 URL 中提取仓库名称。"""
        return unquote(self.url.split('/')[-1].replace(".git", ""))

    def process(self, output_json_path: str | Path) -> None:
        """
        处理下载的数据集，将其转换为 JSON 格式。

        Args:
            output_json_path (str | Path): 输出 JSON 文件的路径。
        """
        if not self.download_path or not self.download_path.exists():
            raise FileNotFoundError("Dataset not downloaded or path not set. Call download() first.")

        lean_files = list(self.download_path.rglob("*.lean"))
        all_items_data = []

        print(f"Found {len(lean_files)} .lean files for processing in {self.download_path}...")

        for lean_file in lean_files:
            items = parse_lean_file(lean_file)
            all_items_data.extend(items)

        print(f"Extracted {len(all_items_data)} potential theorems/lemmas.")

        lean_items = []
        for item_data in all_items_data:
            try:
                if item_data.get("statement"):
                    LeanItem(**item_data)  # Validate with Pydantic
                    lean_items.append(item_data)
            except Exception as e:
                print(f"Skipping item {item_data.get('id')} due to validation error: {e}")

        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving {len(lean_items)} valid items to {output_json_path}...")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(lean_items, f, indent=2, ensure_ascii=False)
        print("Processing complete.")


class GitHubDownloader(BaseDownloader):
    """从 GitHub 下载数据集。"""

    def download(self) -> Path:
        repo_name = self._get_repo_name()
        self.download_path = self.output_dir / repo_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.download_path.exists():
            print(f"Directory '{self.download_path}' already exists. Skipping clone.")
            return self.download_path

        print(f"Cloning '{self.url}' from GitHub to '{self.download_path}'...")
        try:
            subprocess.run(
                ["git", "clone", self.url, str(self.download_path)],
                check=True, capture_output=True, text=True
            )
            print(f"Repository successfully cloned to '{self.download_path}'.")
            return self.download_path
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone GitHub repository '{self.url}'. Error: {e.stderr}")
            raise
        except FileNotFoundError:
            print("Error: Git is not installed or not in system PATH.")
            raise

class HuggingFaceDownloader(BaseDownloader):
    """从 Hugging Face Hub 下载数据集。"""

    def _get_dataset_id(self) -> str:
        """从 URL 或 ID 字符串中获取 Hugging Face ID。"""
        parsed_url = urlparse(self.url)
        if parsed_url.netloc == "huggingface.co":
            path_parts = [part for part in parsed_url.path.split('/') if part]
            if len(path_parts) >= 2 and path_parts[0] == 'datasets':
                return "/".join(path_parts[1:])
            return "/".join(path_parts)
        return self.url

    def _get_repo_name(self) -> str:
        """对 HF 来说，使用 ID 的最后一部分作为目录名。"""
        return self._get_dataset_id().split('/')[-1]

    def download(self) -> Path:
        dataset_id = self._get_dataset_id()
        repo_name = self._get_repo_name()
        self.download_path = self.output_dir / repo_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.download_path.exists() and os.listdir(self.download_path):
            print(f"Directory '{self.download_path}' already exists. Skipping download.")
            return self.download_path

        print(f"Downloading dataset '{dataset_id}' from Hugging Face Hub to '{self.download_path}'...")
        try:
            snapshot_download(
                repo_id=dataset_id,
                repo_type="dataset",
                local_dir=str(self.download_path),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"Dataset successfully downloaded to '{self.download_path}'.")
            return self.download_path
        except HfHubHTTPError as e:
            print(f"Failed to download Hugging Face dataset '{dataset_id}'. HTTP Error: {e}")
            raise
        except Exception as e:
            print(f"An unknown error occurred: {e}")
            raise

def get_downloader(url: str, output_dir: str) -> BaseDownloader:
    """
    根据 URL 类型返回相应的下载器实例。
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()

    if "github.com" in domain and url.endswith(".git"):
        return GitHubDownloader(url, output_dir)
    elif "huggingface.co" in domain or "/" in url: # 简化 HF 判断
        return HuggingFaceDownloader(url, output_dir)
    else:
        raise ValueError(f"Cannot determine downloader type for URL: {url}")
