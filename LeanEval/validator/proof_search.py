# LeanEval/validator/proof_search.py

from queue import PriorityQueue
from LeanEval.validator.search_tree import Node, Edge, Status
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from LeanEval.validator.proof_validator import ProofValidator
import json
import os
import re
import time
import ast
# 移除 concurrent.futures 和 threading 的导入
# from concurrent.futures import ThreadPoolExecutor
# import threading
import queue
import sys

# 导入必要的模块
from LeanEval.prompt import PromptBuilder
from LeanEval.utils.handle_lean_result import handle_lean_str
from LeanEval.utils.extract_lean_code import extract_lean_block


class BFSProver:
    def __init__(self, model, prompt_builder: PromptBuilder, tmp_dir: str | Path = "./tmp_tree_proofs", timeout: int = 1200, error_threshold: int = 2, degree: int = 5):
        self.model = model
        self.tmp_dir: str | Path = tmp_dir
        self.timeout = timeout
        self.prompt_builder: PromptBuilder = prompt_builder
        self.error_threshold = error_threshold
        self.degree = degree
        self.proofValidator = ProofValidator(timeout=120)

    def run_node(self, node: Node, tmp_file: Path) -> Tuple[List[str], List[str]]:
        tmp_file.write_text(node.leanCode, encoding="utf-8")
        success, msg = self.proofValidator.validate_file(tmp_file)
        print(f"\nvalidation message: {msg}\n")
        tips, error = handle_lean_str(msg)
        if msg == "Lean验证超时":
            node.status = Status.Error
        elif success:
            node.status = Status.Proved
        elif len(error) > self.error_threshold:
            node.status = Status.Error
        else:
            node.status = Status.Open
        node.Update()
        return tips, error

    def _parse_prioritized_tactics(self, model_output: str) -> List[str]:
        """
        用多种策略从模型输出中解析出策略列表，以提高成功率。
        """
        json_match = re.search(r"```json\s*\n(.*?)\n?\s*```", model_output, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1).strip()
                data = json.loads(json_str)
                if isinstance(data.get("tactics"), list):
                    return [str(t) for t in data["tactics"]]
            except json.JSONDecodeError:
                pass  # 如果解析失败，则继续尝试其他策略

        try:
            start = model_output.find('{')
            end = model_output.rfind('}')
            if start != -1 and end != -1:
                potential_json = model_output[start:end+1]
                data = json.loads(potential_json)
                if isinstance(data.get("tactics"), list):
                    return [str(t) for t in data["tactics"]]
        except (json.JSONDecodeError, TypeError):
            pass

        code_match = re.search(r"```(lean|python|text)?\s*\n(.*?)\n?\s*```", model_output, re.DOTALL)
        content_to_parse = model_output # 默认解析整个输出
        if code_match:
            content_to_parse = code_match.group(2).strip()
        
        if content_to_parse.startswith('[') and content_to_parse.endswith(']'):
            try:
                # ast.literal_eval比eval更安全，只能解析基本数据类型
                parsed_list = ast.literal_eval(content_to_parse)
                if isinstance(parsed_list, list):
                    return [str(item) for item in parsed_list]
            except (ValueError, SyntaxError, MemoryError, TypeError):
                pass # 解析失败，继续


        return []

    # <<< --- 核心修改：将 thread_prove 重构为同步的 prove 方法 --- >>>
    def prove(self, goal: str) -> Dict[str, Any]:
        Root = Node(leanCode=goal, height=0)
        heap = PriorityQueue()
        heap.put_nowait((Root.height, Root))

        tmp_dir = Path(self.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # 每个进程使用唯一的临时文件
        process_id = os.getpid()
        tmp_file = tmp_dir / f"proof_{process_id}.lean"

        start_time = time.time()
        
        ProvedNode: Optional[Node] = None
        deepest_node: Node = Root

        # 使用一个简单的同步 while 循环
        while not heap.empty():
            if time.time() - start_time > self.timeout:
                break
            
            node_tuple = heap.get_nowait()
            node: Node = node_tuple[1]

            # 追踪最深节点
            if node.height > deepest_node.height:
                deepest_node = node
            
            if node.status != Status.Open:
                continue

            tips, error = self.run_node(node, tmp_file)
            print(f"\nNode {node.id} status: {node.status}, height: {node.height}, tips: {tips}, error: {error}\n")
            if Root.status == Status.Proved:
                ProvedNode = node
                break # 找到证明，跳出循环
            
            if node.status == Status.Open:
                chat_prompt = self.prompt_builder.build_chat_for_tactic(node.leanCode, tips,error)
                p = json.dumps(chat_prompt, ensure_ascii=False, indent=2)
                print(f"\nChat prompt for node {node.id}:\n{p}\n")
                # *** 这里的调用现在是安全的，因为是单线程执行 ***
                model_response = self.model.predict(chat_prompt)
                print(f"\nModel response for node {node.id}:\n{model_response}\n")
                tactics = self._parse_prioritized_tactics(model_response)
                tactics_to_use = tactics[:self.degree]

                for t in tactics_to_use:
                    t_clean = extract_lean_block(t) or t
                    if not t_clean: 
                        continue
                    new_edge = Edge(tactic=t_clean, height=node.height, src=node)
                    new_node = Node(leanCode=node.leanCode + "\n  " + t_clean, height=node.height + 1, src=new_edge)
                    new_edge.dst = new_node
                    node.dst.append(new_edge)
                    heap.put((new_node.height, new_node))
        
        # 构建最终返回的字典
        result = {}
        if ProvedNode:
            result['proved'] = True
            result['final_proof_code'] = ProvedNode.leanCode
            result['tactic_path'] = ProvedNode.reconstruct_path()
        else:
            result['proved'] = False
            result['final_proof_code'] = None
            result['tactic_path'] = deepest_node.reconstruct_path() if deepest_node else []
        
        return result