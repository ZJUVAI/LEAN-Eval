# LeanEval/utils/process_dataset.py
import os
import re
import json
from pathlib import Path
import sys
import argparse
from LeanEval.datasets import LeanItem
from typing import List, Set # 确保导入 List 和 Set

# ... (LEAN_EVAL_ROOT 和 try-except 导入保持不变) ...

# 用于查找定理/引理及其陈述的正则表达式（简化版）
THEOREM_RE = re.compile(
    r"^(?:theorem|lemma)\s+([\w\.]+)\s*(?:\{.*?\})?\s*(?:\(.*?\))?\s*:\s*(.*?)\s*:=",
    re.MULTILINE | re.DOTALL,
)
# 用于查找 imports 的正则表达式
IMPORT_RE = re.compile(r"^import\s+([\w\.]+)", re.MULTILINE)

def resolve_imports(
    initial_imports: List[str], 
    base_path: Path, 
    already_visited: Set[Path],
    project_root: Path # 新增参数，用于确定项目根，以正确解析相对导入
) -> List[str]:
    """
    递归解析导入语句，展开项目内部的间接导入。

    Args:
        initial_imports: 从当前文件直接提取的导入语句列表。
        base_path: 当前正在解析的 .lean 文件所在的目录。
        already_visited: 已访问过的文件路径集合，用于防止循环导入。
        project_root: 数据集项目的根目录。

    Returns:
        一个包含所有直接和间接导入（非项目内部模块或已解析的）的列表。
    """
    final_imports = set()
    queue = list(initial_imports) # 使用队列处理待解析的导入
    
    processed_in_this_run = set() # 防止在单次 resolve_imports 调用中重复处理同一个 import string

    while queue:
        import_statement = queue.pop(0)
        
        if import_statement in processed_in_this_run:
            continue
        processed_in_this_run.add(import_statement)

        # 假设项目内部模块的导入路径是相对于 project_root 的
        # 例如: import MyProject.Utils.Lemmas -> project_root/MyProject/Utils/Lemmas.lean
        # 我们需要将点分隔的模块路径转换为文件路径
        
        # 尝试判断是否为项目内部模块 (这是一个简化判断，可能需要根据具体项目结构调整)
        # 如果导入路径不包含常见的库前缀（如 Mathlib, Batteries 等），我们可能认为它是项目内的
        # 或者，我们可以检查该路径是否能在 project_root 下找到对应的 .lean 文件
        
        module_parts = import_statement.split('.')
        potential_lean_file_rel_path = Path(*module_parts).with_suffix(".lean")
        potential_lean_file_abs_path = project_root / potential_lean_file_rel_path

        # 另一种方式是检查文件是否存在于 base_path 的相对路径或项目根目录的相对路径
        # current_file_dir = base_path (如果 base_path 是文件自身的目录)
        # potential_local_path = base_path / potential_lean_file_rel_path (不常用，Lean 通常是全项目路径导入)

        if potential_lean_file_abs_path.exists() and potential_lean_file_abs_path.is_file():
            # 这是项目内部模块，并且我们找到了对应的 .lean 文件
            if potential_lean_file_abs_path in already_visited:
                # 防止循环导入
                continue
            
            already_visited.add(potential_lean_file_abs_path)
            # print(f"    Resolving internal import: {import_statement} from {potential_lean_file_abs_path}")

            try:
                internal_content = potential_lean_file_abs_path.read_text(encoding="utf-8")
                nested_imports = IMPORT_RE.findall(internal_content)
                for ni in nested_imports:
                    if ni not in processed_in_this_run and ni not in final_imports: # 避免立即重复添加刚处理过的
                        queue.append(ni) # 将嵌套的导入添加到队列中进一步处理
            except Exception as e:
                print(f"    Warning: Could not read or parse internal import file {potential_lean_file_abs_path}: {e}")
        else:
            # 不是项目内部可解析文件，或未找到，认为是外部库导入
            final_imports.add(import_statement)
            
    return sorted(list(final_imports))


def parse_lean_file(file_path: Path, dataset_root_path: Path) -> list[dict]: # 新增 dataset_root_path 参数
    """
    解析 Lean 文件以提取定理/引理和完全解析后的 imports。
    
    Args:
        file_path: 当前要解析的 .lean 文件的路径。
        dataset_root_path: 该 .lean 文件所属的数据集项目的根目录路径。
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"无法读取 {file_path}: {e}")
        return []

    # 1. 直接从当前文件提取导入
    direct_imports = IMPORT_RE.findall(content)
    
    # 2. 解析并展开所有导入 (包括间接的)
    # already_visited 集合在每次 parse_lean_file 调用时独立，
    # 但对于 resolve_imports 的单次完整调用链是共享的。
    # 我们还需要传入文件自身的路径，以便 resolve_imports 知道当前文件的上下文
    # file_path.parent 是当前文件的目录
    # dataset_root_path 是整个数据集的根，用于定位项目内的其他 .lean 文件
    
    # 对于 resolve_imports，already_visited 应该从解析 file_path 开始
    initial_visited_set = {file_path.resolve()} # 解析自身时，自身已被访问
    
    # print(f"Parsing file: {file_path}")
    # print(f"  Direct imports: {direct_imports}")
    resolved_imports = resolve_imports(direct_imports, file_path.parent, initial_visited_set, dataset_root_path)
    # print(f"  Resolved imports for {file_path.name}: {resolved_imports}")

    theorems = []
    for match in THEOREM_RE.finditer(content):
        full_match_text = match.group(0)
        end_match = re.search(r"\s*:=(?:\s*(?:by|sorry|begin))", full_match_text, re.MULTILINE | re.DOTALL)

        theorem_name = match.group(1)
        unique_id = f"{file_path.stem}_{theorem_name}" # 使用原始文件名和定理名构造ID

        if end_match:
            statement_text = full_match_text[:end_match.start()].strip()
            statement_text = re.sub(r"^(?:theorem|lemma)\s+", "", statement_text, 1).strip()
            theorems.append({
                "id": unique_id,
                "imports": resolved_imports, # 使用解析后的完整导入列表
                "statement": statement_text,
                "source_file": str(file_path.relative_to(dataset_root_path)), # 记录原始文件相对路径
                "difficulty": 1, 
            })
        else:
            # 对于没有明显 "by", "sorry", "begin" 的情况，可能是定义或不完整的片段
            # 这里的statement提取可能需要更仔细，但暂时保持原样
            raw_statement = match.group(0).replace(":=", "").strip()
            # 尝试从 raw_statement 中再次移除 theorem/lemma 关键字，确保干净
            cleaned_statement = re.sub(r"^(?:theorem|lemma)\s+", "", raw_statement, 1).strip()
            theorems.append({
                "id": unique_id,
                "imports": resolved_imports,
                "statement": cleaned_statement, # 使用更干净的陈述
                "source_file": str(file_path.relative_to(dataset_root_path)),
                "difficulty": 1,
            })
            
    return theorems

def process_dataset(download_path: str, output_json_path: str):
    """将下载的 Lean 文件处理成 JSON 数据集。"""
    dataset_root = Path(download_path).resolve() # 获取数据集的绝对根路径
    lean_files = list(dataset_root.rglob("*.lean"))
    all_items = []

    print(f"找到 {len(lean_files)} 个 .lean 文件在 {dataset_root} 中进行处理...")

    for lean_file in lean_files:
        # print(f"Processing file: {lean_file}")
        # 传递 dataset_root 给 parse_lean_file
        items = parse_lean_file(lean_file, dataset_root)
        all_items.extend(items)

    print(f"提取到 {len(all_items)} 个潜在的定理/引理。")

    lean_items_data = []
    for item_data in all_items:
        try:
            if item_data.get("statement"):
                LeanItem(**item_data) 
                lean_items_data.append(item_data)
        except Exception as e:
            print(f"因验证错误跳过项目 {item_data.get('id')} (来自文件 {item_data.get('source_file', 'N/A')}): {e}")

    print(f"正在保存 {len(lean_items_data)} 个有效项目到 {output_json_path}...")
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(lean_items_data, f, indent=2, ensure_ascii=False)
    print("处理完成。")

def extract_imports(imports_string: str) -> List[str]:
    imports = imports_string.split('\n')
    output = []
    for line in imports:
        line = line.strip()
        if line:
            output.append(line)
    return output

def process_jsonl_dataset(download_path: str,ouput_json_path: str):
    """将已有的 JSONl 文件处理成 JSON 数据集"""
    dataset_root = Path(download_path).resolve()

    theorems = []
    try:
        with open(dataset_root) as f:
            for line in f:
                item = json.loads(line)
                header = [] if "header" not in item else extract_imports(item["header"])
                theorems.append({
                    "id":item["name"],
                    "imports": header,
                    "statement":item["formal_statement"],
                    "source_file":str(download_path),
                    "difficulty":1
                })

    except Exception as e:
        print(f"jsonl dataset loading error: {e}")
        return
    
    print(f"提取到 {len(theorems)} 个定理")
    print(f"正在保存 {len(theorems)} 个定理到 {ouput_json_path}...")
    Path(ouput_json_path).parent.mkdir(parents=True,exist_ok=True)
    with open(ouput_json_path,'w',encoding='utf-8') as f:
        json.dump(theorems,f,indent=2,ensure_ascii=False)
    print("处理完成")
