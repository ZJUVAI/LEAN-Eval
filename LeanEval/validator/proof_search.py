# LeanEval/validator/proof_search.py

from queue import PriorityQueue
from LeanEval.validator.search_tree import Node, Edge, Status
from pathlib import Path
from typing import List, Optional, Tuple, Union 
from LeanEval.validator.proof_validator import ProofValidator
import json
import os
import re # 导入 re 模块
import time
from concurrent.futures import ThreadPoolExecutor
from LeanEval.models.base_api import BaseAPIModel
from LeanEval.prompt import PromptBuilder
from LeanEval.utils.handle_lean_result import handle_lean_str
from LeanEval.utils.extract_lean_code import extract_lean_block
import threading
import queue
import sys

class BFSProver:
    def __init__(self, model, prompt_builder: PromptBuilder, tmp_dir: str | Path = "./tmp_tree_proofs", timeout: int = 1200, error_threshold: int = 2, degree: int = 5):
        self.model: BaseAPIModel = model
        self.root_dir = Path(__file__).resolve().parent.parent.parent
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
        """
        从模型的输出中解析出按优先级排序的策略列表。
        模型被要求返回一个JSON对象。
        """
        try:
            # 尝试提取JSON代码块（如果模型用```json包裹了输出）
            match = re.search(r"```json\s*\n(.*?)\n?\s*```", model_output, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # 如果没有代码块，假设整个输出就是JSON
                # 找到第一个 '{' 和最后一个 '}' 之间的内容，以提高鲁棒性
                start = model_output.find('{')
                end = model_output.rfind('}')
                if start != -1 and end != -1:
                    json_str = model_output[start:end+1]
                else:
                    json_str = model_output

            data = json.loads(json_str)
            
            if "tactics" in data and isinstance(data["tactics"], list):
                tactics = [str(t) for t in data["tactics"]]
                print(f"线程 {threading.get_ident()} 解析到 {len(tactics)} 个策略。")
                return tactics
            else:
                print(f"线程 {threading.get_ident()} 警告: JSON响应格式不正确，缺少'tactics'列表。", file=sys.stderr)
                return []
                
        except json.JSONDecodeError:
            print(f"线程 {threading.get_ident()} 错误: 模型返回的不是有效的JSON。输出: {model_output}", file=sys.stderr)
            return []
        except Exception as e:
            print(f"线程 {threading.get_ident()} 解析策略时发生未知错误: {e}", file=sys.stderr)
            return []

    # prove 方法（单线程）也应该更新，这里为保持一致性，仅作简单修改
    def prove(self, goal) -> Tuple[Node, str]:
        Root = Node(leanCode=goal, height=0)
        heap = PriorityQueue()
        heap.put_nowait((Root.height, Root))

        tmp_dir = Path(self.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = tmp_dir / "proof.lean"
        start_time = time.time()

        while True:
            if heap.empty(): return None, "heap is empty"
            if time.time() - start_time > self.timeout: return None, "timeout"

            node: Node = heap.get_nowait()[1]
            if node.status != Status.Open: continue

            tips, error = self.run_node(node, tmp_file)

            if Root.status == Status.Proved: return Root, node.leanCode
            
            if node.status == Status.Open:
                chat_prompt = self.prompt_builder.build_chat_for_tactic(node.leanCode, tips)
                model_response = self.model.predict(chat_prompt)
                tactics = self._parse_prioritized_tactics(model_response)
                tactics_to_use = tactics[:self.degree]

                for t in tactics_to_use:
                    t = extract_lean_block(t)
                    if not t: continue
                    print("get tactic: ", t)
                    new_edge = Edge(tactic=t, height=node.height, src=node)
                    new_node = Node(leanCode=node.leanCode+"\n  "+t, height=node.height + 1, src=new_edge)
                    new_edge.dst = new_node
                    node.dst.append(new_edge)
                    heap.put_nowait((new_node.height, new_node))

    def thread_prove(self, goal: str, num_workers: int = os.cpu_count() + 4) -> Tuple[Optional[Node], Optional[str]]:
        Root = Node(leanCode=goal, height=0)
        heap = PriorityQueue()
        heap.put_nowait((Root.height, Root))

        tmp_dir = Path(self.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        Proved = threading.Event()
        ProvedNode: Optional[Node] = None
        node_lock = threading.Lock()

        def worker():
            while not Proved.is_set():
                try:
                    node_tuple = heap.get(timeout=0.5)
                    node: Node = node_tuple[1]

                    if Proved.is_set() or (time.time() - start_time > self.timeout):
                        break

                    if node.status != Status.Open:
                        heap.task_done()
                        continue

                    tmp_file = tmp_dir / f"proof_{threading.get_ident()}_{node.id}.lean"
                    tips, error = self.run_node(node, tmp_file)
                    print(f"线程 {threading.get_ident()} 处理节点 {node.id} 完成，状态: {node.status}, 错误: {error}")

                    if Root.status == Status.Proved:
                        if not Proved.is_set():
                            with node_lock:
                                if ProvedNode is None: ProvedNode = node
                            Proved.set()
                        break

                    # <<< --- 核心修改部分 --- >>>
                    if node.status == Status.Open:
                        # 1. 使用新的Prompt Builder构建prompt
                        chat_prompt = self.prompt_builder.build_chat_for_tactic(node.leanCode, tips)
                        
                        # 2. 对模型进行单次调用
                        # 假设model.predict是线程安全的，如果不是，需要加锁
                        print("-----------------")
                        print(f"\nprompt:\n{chat_prompt}")
                        print("-----------------")
                        model_response = self.model.predict(chat_prompt)
                        
                        # 3. 解析返回的多个策略
                        tactics = self._parse_prioritized_tactics(model_response)
                        if not tactics:
                            print(f"线程 {threading.get_ident()} 没有解析到有效策略，跳过当前节点。", file=sys.stderr)
                            heap.task_done()
                            continue
                        print(f'线程 {threading.get_ident()} 解析的策略为:\n {tactics}')
                        # 如果设置了degree，可以取前N个策略
                        tactics_to_use = tactics[:self.degree]

                        for t in tactics_to_use:
                            t_clean = extract_lean_block(t) or t
                            if not t_clean: continue
                            print(f"线程 {threading.get_ident()} 获取策略: {t_clean}")
                            new_edge = Edge(tactic=t_clean, height=node.height, src=node)
                            new_node = Node(leanCode=node.leanCode + "\n  " + t_clean, height=node.height + 1, src=new_edge)
                            new_edge.dst = new_node
                            node.dst.append(new_edge)
                            heap.put((new_node.height, new_node))
                    # <<< --- 核心修改结束 --- >>>
                    
                    heap.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"线程 {threading.get_ident()} 发生致命错误并退出: {type(e).__name__}: {e}", file=sys.stderr)
                    break
        
        executor = ThreadPoolExecutor(max_workers=num_workers)
        try:
            futures = [executor.submit(worker) for _ in range(num_workers)]
            while time.time() - start_time < self.timeout:
                if Proved.is_set() or all(f.done() for f in futures):
                    break
                time.sleep(0.1)
        finally:
            Proved.set()
            executor.shutdown(wait=True)

        with node_lock:
            if ProvedNode:
                return Root, ProvedNode.leanCode
            else:
                return None, "Timeout or no solution found"