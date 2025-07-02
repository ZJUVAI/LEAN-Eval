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
from concurrent.futures import ThreadPoolExecutor
# from LeanEval.models.base_api import BaseAPIModel # This import might not be needed directly
from LeanEval.prompt import PromptBuilder
from LeanEval.utils.handle_lean_result import handle_lean_str
from LeanEval.utils.extract_lean_code import extract_lean_block
import threading
import queue
import sys

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
        try:
            match = re.search(r"```json\s*\n(.*?)\n?\s*```", model_output, re.DOTALL)
            json_str = match.group(1) if match else model_output
            start = json_str.find('{')
            end = json_str.rfind('}')
            if start != -1 and end != -1:
                json_str = json_str[start:end+1]
            
            data = json.loads(json_str)
            if "tactics" in data and isinstance(data["tactics"], list):
                return [str(t) for t in data["tactics"]]
            return []
        except Exception:
            return []

    def thread_prove(self, goal: str, num_workers: int = os.cpu_count() + 4) -> Dict[str, Any]:
        Root = Node(leanCode=goal, height=0)
        heap = PriorityQueue()
        heap.put_nowait((Root.height, Root))

        tmp_dir = Path(self.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        Proved = threading.Event()
        ProvedNode: Optional[Node] = None
        node_lock = threading.Lock()

        # 用于追踪最深节点的共享变量
        deepest_node_info = {'node': Root, 'lock': threading.Lock()}

        def worker():
            nonlocal ProvedNode # 允许修改外部作用域的 ProvedNode
            while not Proved.is_set():
                if time.time() - start_time > self.timeout: break
                try:
                    node_tuple = heap.get(timeout=0.5)
                    node: Node = node_tuple[1]

                    # 追踪最深节点
                    with deepest_node_info['lock']:
                        if node.height > deepest_node_info['node'].height:
                            deepest_node_info['node'] = node
                    
                    if node.status != Status.Open:
                        heap.task_done()
                        continue

                    tmp_file = tmp_dir / f"proof_{threading.get_ident()}_{node.id}.lean"
                    tips, error = self.run_node(node, tmp_file)
                    
                    if Root.status == Status.Proved:
                        if not Proved.is_set():
                            with node_lock:
                                if ProvedNode is None: ProvedNode = node
                            Proved.set()
                        break
                    
                    if node.status == Status.Open:
                        chat_prompt = self.prompt_builder.build_chat_for_tactic(node.leanCode, tips)
                        print(f"\nChatPrompt:\n{json.dumps(chat_prompt, indent=2)}\n")
                        model_response = self.model.predict(chat_prompt)

                        tactics = self._parse_prioritized_tactics(model_response)
                        tactics_to_use = tactics[:self.degree]

                        for t in tactics_to_use:
                            t_clean = extract_lean_block(t) or t
                            if not t_clean: continue
                            new_edge = Edge(tactic=t_clean, height=node.height, src=node)
                            new_node = Node(leanCode=node.leanCode + "\n  " + t_clean, height=node.height + 1, src=new_edge)
                            new_edge.dst = new_node
                            print(f"\nNewNode LeanCode:\n{new_node.leanCode}\n")
                            node.dst.append(new_edge)
                            heap.put((new_node.height, new_node))
                    
                    heap.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"线程 {threading.get_ident()} 发生致命错误并退出: {e}", file=sys.stderr)
                    break
        
        executor = ThreadPoolExecutor(max_workers=num_workers)
        try:
            futures = [executor.submit(worker) for _ in range(num_workers)]
            while time.time() - start_time < self.timeout:
                if Proved.is_set() or all(f.done() for f in futures): 
                    break
        finally:
            Proved.set()
            executor.shutdown(wait=True)
        
        # --- 构建最终返回的字典 ---
        result = {}
        if ProvedNode:
            result['proved'] = True
            result['final_proof_code'] = ProvedNode.leanCode
            result['tactic_path'] = ProvedNode.reconstruct_path()
        else:
            # 如果证明失败，记录最深路径
            result['proved'] = False
            result['final_proof_code'] = None
            with deepest_node_info['lock']:
                deepest_node = deepest_node_info['node']
            result['tactic_path'] = deepest_node.reconstruct_path() if deepest_node else []
        
        return result