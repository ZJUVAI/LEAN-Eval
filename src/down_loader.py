# 标准库：命令行参数分析,系统操作,网络链接分析等
import argparse
import os
import subprocess
import shutil
from urllib.parse import urlparse, unquote

# 第三方库：huggingface下载工具包
from huggingface_hub import snapshot_download, logging as hf_logging
from huggingface_hub.utils import HfHubHTTPError

# Disable some verbose logging from huggingface_hub, comment out for detailed logs
hf_logging.set_verbosity_error()

PREDEFINED_DATASETS = {
    """
    提前预设好的数据集,根据朱老师提供的,有miniF2F,ProofNet,Proof-pile三个
    minif2f一种是Hugging Face上的,一种是GitHub上的
    ProofNet一种是Hugging Face上的,一种是GitHub上的
    Proof-pile一种是EleutherAI上的,一种是Hoskinson Center上的
    其中,Hugging Face上的数据集是通过snapshot_download函数下载的,GitHub上的数据集是通过git clone命令下载的
    其中,EleutherAI上的数据集是通过snapshot_download函数下载的,Hoskinson Center上的数据集是通过git clone命令下载的
    """
    "1": {
        "name": "miniF2F (Hugging Face)",
        "url": "leanprover-community/miniF2F", # HF Dataset ID
        "type": "hf"
    },
    "2": {
        "name": "miniF2F (GitHub)",
        "url": "https://github.com/openai/miniF2F.git",
        "type": "github"
    },
    "3": {
        "name": "ProofNet (Hugging Face)",
        "url": "stanford-crfm/ProofNet", # HF Dataset ID
        "type": "hf"
    },
    "4": {
        "name": "ProofNet (GitHub)",
        "url": "https://github.com/albertqjiang/ProofNet.git",
        "type": "github"
    },
    "5": {
        "name": "Proof-pile (EleutherAI - Hugging Face)",
        "url": "EleutherAI/proof-pile", # HF Dataset ID
        "type": "hf"
    },
    "6": {
        "name": "Proof-pile (Hoskinson Center - Hugging Face)",
        "url": "hoskinson-center/proof-pile", # HF Dataset ID
        "type": "hf"
    }
}

def _get_repo_name_from_url(url: str) -> str:
    """
    Extracts repository name from a Git URL.
    """
    return unquote(url.split('/')[-1].replace(".git", ""))

def _download_github_repo(repo_url: str, local_dir_base: str):
    """
    Clones a GitHub repository.
    """
    repo_name = _get_repo_name_from_url(repo_url)
    clone_path = os.path.join(local_dir_base, repo_name)

    if os.path.exists(clone_path):
        print(f"Directory '{clone_path}' already exists. Skipping clone of '{repo_url}'.")
        print("Hint: To update, manually delete the directory or use 'git pull'.")
        return clone_path

    print(f"Cloning '{repo_url}' from GitHub to '{clone_path}'...")
    try:
        subprocess.run(
            ["git", "clone", repo_url, clone_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Repository successfully cloned to '{clone_path}'.")
        return clone_path
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone GitHub repository '{repo_url}'.")
        print(f"Error: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: Git is not installed or not in system PATH. Please install Git.")
        return None

def _download_huggingface_dataset(dataset_id_or_url: str, local_dir_base: str):
    """
    Downloads a dataset from Hugging Face Hub.
    """
    parsed_url = urlparse(dataset_id_or_url)
    if parsed_url.netloc == "huggingface.co":
        path_parts = [part for part in parsed_url.path.split('/') if part]
        if len(path_parts) >= 2 and path_parts[0] == 'datasets':
            dataset_id = "/".join(path_parts[1:])
        else:
            dataset_id = "/".join(path_parts) # Handles model IDs or dataset IDs like 'user/name' from URL
    else:
        dataset_id = dataset_id_or_url # Assumes direct dataset_id

    dataset_name_for_path = dataset_id.split('/')[-1]
    download_path = os.path.join(local_dir_base, dataset_name_for_path)

    if os.path.exists(download_path) and os.listdir(download_path):
        print(f"Directory '{download_path}' already exists and is not empty. Skipping download of '{dataset_id}'.")
        print("Hint: To re-download or update, manually delete the directory.")
        return download_path

    print(f"Downloading dataset '{dataset_id}' from Hugging Face Hub to '{download_path}'...")
    try:
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset", # This assumes it's always a dataset. For models, this would be "model"
            local_dir=download_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"Dataset '{dataset_id}' successfully downloaded to '{download_path}'.")
        return download_path
    except HfHubHTTPError as e:
        print(f"Failed to download Hugging Face dataset '{dataset_id}'.")
        print(f"HTTP Error: {e}")
        return None
    except Exception as e:
        print(f"An unknown error occurred while downloading Hugging Face dataset '{dataset_id}'.")
        print(f"Error: {e}")
        return None

# --- New Helper Function to Process any URL ---
def process_single_url(url_to_process: str, output_dir: str):
    """
    Determines the type of URL and calls the appropriate download function.
    """
    parsed_url = urlparse(url_to_process)
    domain = parsed_url.netloc.lower()

    print(f"\nAttempting to process: {url_to_process}")
    if "github.com" in domain and url_to_process.endswith(".git"):
        _download_github_repo(url_to_process, output_dir)
    # Check for Hugging Face URLs or direct IDs (e.g., "username/datasetname")
    elif "huggingface.co" in domain or (not domain and "/" in url_to_process and "." not in url_to_process.split("/")[-1]):
        # Distinguish between full HF URLs and HF IDs
        # A simple HF ID won't have 'huggingface.co' in netloc
        # and typically doesn't start with '/datasets/' unless it's part of a full URL.
        _download_huggingface_dataset(url_to_process, output_dir)
    else:
        print(f"Cannot determine type for URL '{url_to_process}'. It does not appear to be a supported GitHub (.git) or Hugging Face link/ID.")
        print("Supported formats: https://github.com/user/repo.git, https://huggingface.co/datasets/user/dataset, user/dataset.")

# --- New Function to Download from File ---
def download_from_file(filepath: str, output_dir: str):
    """
    Reads URLs from a file (one URL per line) and downloads them.
    """
    if not os.path.exists(filepath):
        print(f"Error: Link file '{filepath}' not found.")
        return

    print(f"Reading URLs from '{filepath}'...")
    with open(filepath, 'r') as f:
        urls = [line.strip() for line in f if line.strip()] # Read non-empty lines

    if not urls:
        print(f"No URLs found in '{filepath}'.")
        return

    for url in urls:
        process_single_url(url, output_dir)

# --- New Function for Interactive Predefined Dataset Selection ---
def select_and_download_predefined(output_dir: str):
    """
    Lists predefined datasets and lets the user choose which ones to download.
    """
    print("\nAvailable predefined datasets:")
    for key, item in PREDEFINED_DATASETS.items():
        print(f"  {key}. {item['name']} ({item['url']})")

    choices_str = input("Enter the numbers of datasets to download (e.g., '1 3' or '1,3,5'), or 'all': ").strip().lower()

    selected_urls_to_download = []

    if choices_str == 'all':
        selected_urls_to_download = list(PREDEFINED_DATASETS.values())
    else:
        chosen_keys = choices_str.replace(',', ' ').split()
        for key in chosen_keys:
            if key in PREDEFINED_DATASETS:
                selected_urls_to_download.append(PREDEFINED_DATASETS[key])
            else:
                print(f"Warning: Invalid selection '{key}', skipping.")

    if not selected_urls_to_download:
        print("No valid datasets selected.")
        return

    for item in selected_urls_to_download:
        # The process_single_url expects a URL or HF ID, which is item['url']
        process_single_url(item['url'], output_dir)


def main():
    """
    主要有三种下载方式
    第一种是选择预先准备好的三个数据集
    第二种是在命令行参数后面给出下载了链接(主要可以是github下载链接和hugging face)
    第三种是在指定文件(默认为link.txt)中给出下载链接,这种能够用于批量下载
    三种方式的命令行参数分别为
    --select 选择预先准备好的三个数据集
    url 命令行后直接跟下载链接
    --from-file 指定文件(默认为link.txt)中给出下载链接,这种能够用于批量下载
    可以用 '-o' 或者 '--output-dir' 来指定下载的目录,默认为当前目录
    例如:
    python downloader.py --select
    python downloader.py https://github.com/username/repository.git
    python downloader.py --from-file link.txt
    python downloader.py --from-file link.txt -o /path/to/output
    """
    parser = argparse.ArgumentParser(
        description="Dataset download tool. Supports GitHub, Hugging Face Hub, downloading from a list in a file, and selecting predefined datasets.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Modified: url is now optional
    parser.add_argument(
        "url",
        type=str,
        nargs='?', # Makes it optional
        default=None, # Default value if not provided
        help=
        "Optional: The direct URL or Hugging Face ID to download.\n"
        "Examples:\n"
        "  GitHub: https://github.com/username/repository.git\n"
        "  Hugging Face URL: https://huggingface.co/datasets/dataset_name\n"
        "  Hugging Face ID: username/dataset_name"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=".",
        help="Base directory to download datasets to (default: current directory)."
    )
    # New argument for downloading from a file
    parser.add_argument(
        "--from-file",
        type=str,
        nargs='?', # Makes the filename optional
        const="link.txt", # Default filename if flag is present without a value
        default=None, # Default if flag is not present
        help="Download URLs from the specified file (one URL per line). If no filename is given, defaults to 'link.txt'."
    )
    # New argument for interactive selection
    parser.add_argument(
        "--select",
        action="store_true", # Makes it a boolean flag
        help="Show a list of predefined datasets (miniF2F, ProofNet, Proof-pile) to choose from for download."
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"Created output directory: '{args.output_dir}'")
        except OSError as e:
            print(f"Error creating output directory '{args.output_dir}': {e}")
            return # Exit if directory creation fails

    # --- Determine action based on arguments ---
    action_taken = False
    if args.url:
        process_single_url(args.url, args.output_dir)
        action_taken = True
    
    # 此处逻辑是用于判断用户使用哪种下载
    if args.from_file: # This will be true if --from-file is present (with or without a value)
        download_from_file(args.from_file, args.output_dir)
        action_taken = True
    
    if args.select:
        select_and_download_predefined(args.output_dir)
        action_taken = True

    if not action_taken:
        print("No action specified. Please provide a URL, or use --from-file, or --select.")
        parser.print_help()

if __name__ == "__main__":
    main()