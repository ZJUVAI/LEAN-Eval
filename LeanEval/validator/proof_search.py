from queue import PriorityQueue #优先队列
from validator.search_tree import Node, Edge, Status #搜索树的定义
from pathlib import Path
from typing import List, Optional, Tuple, Union 
from validator.proof_validator import ProofValidator #导入验证器
import json, os, time
from concurrent.futures import ThreadPoolExecutor #多线程执行
from models.base_api import BaseAPIModel #模型基类
from prompt import PromptBuilder #构建器
from utils.handle_lean_result import handle_lean_str #处理Lean结果字符串的工具函数
from utils.extract import extract_lean_code_after_marker

class BFSProver:
    def __init__(self, model, prompt_builder: PromptBuilder, tmp_dir: str | Path = "./tmp_tree_proofs", timeout: int = 1200, error_threshold: int = 2, degree: int = 5):
        """
        BFS Prover的初始化函数

        Args:
            model (BaseAPIModel): 用于生成证明的模型
            prompt_builder (PromptBuilder): 用于构建提示的构建器
            tmp_dir (str | Path, optional): 临时文件目录. Defaults to "./tmp_tree_proofs".
            timeout (int, optional): 超时时间. Defaults to 1200.
            error_threshold (int, optional): 错误数量阈值. Defaults to 2.
            degree (int, optional): 搜索树的度数. Defaults to 5.
        """
        self.model: BaseAPIModel = model
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.tmp_dir: str | Path = tmp_dir
        self.timeout = timeout
        self.prompt_builder: PromptBuilder = prompt_builder
        self.error_threshold = error_threshold
        self.degree = degree
        self.proofValidator = ProofValidator(timeout=120)

    def run_node(self, node: Node, tmp_file: Path) -> Tuple[List[str], List[str]]:
        """
        运行一个节点

        Args:
            node (Node): 节点

        Returns:
            Tuple[List[str], List[str]]: 有效信息和错误列表
        """
        total_path = "LeanEval" / tmp_file
        tmp_file.write_text(node.leanCode, encoding="utf-8")
        success, msg = self.proofValidator.validate_file(total_path)
        tips, error = handle_lean_str(msg)
        if msg == "Lean验证超时":
            node.status = Status.Error
        elif success:
            node.status = Status.Proved
        elif len(error) > self.error_threshold: #还没处理warning
            node.status = Status.Error
        else:
            node.status = Status.Open
        node.Update()
        return tips, error


    def prove(self, goal) -> Tuple[Node, str]:
        """
        使用 BFS 搜索证明目标,单线程

        Args:
            goal (str): 目标, Lean 代码

        Returns:
            Tuple[Node, str]: 证明树和最终Lean代码 | None和"timeout"
        """
        Root = Node(leanCode=goal, height=0)
        heap = PriorityQueue()
        heap.put_nowait((Root.height, Root))

        tmp_dir = Path(self.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = tmp_dir / "proof.lean"

        start_time = time.time()

        while True:
            if heap.empty():
                return None, "heap is empty"

            if time.time() - start_time > self.timeout:
                return None, "timeout"

            node: Node = heap.get_nowait()[1]
            if node.status != Status.Open:
                continue

            tips, error = self.run_node(node, tmp_file)

            if Root.status == Status.Proved:
                return Root, node.leanCode
            
            if node.status == Status.Open:
                prompts = []
                prompt = self.prompt_builder.build_chat_for_tactic(node.leanCode, tips)
                for _ in range(self.degree):
                    prompts.append(prompt)
                tactics: List[str] = self.model.predict(prompts)
                for t in tactics:
                    t = extract_lean_code_after_marker(t)
                    print("get tactic: ", t)
                    new_edge = Edge(tactic=t, height=node.height, src=node)
                    new_node = Node(leanCode=node.leanCode+"\n"+t, height=node.height + 1, src=new_edge)
                    new_edge.dst = new_node
                    node.dst.append(new_edge)
                    heap.put_nowait((new_node.height, new_node))


    def thread_prove(self, goal, num_workers: int = os.cpu_count() + 4) -> Tuple[Node, str]:
        Root = Node(leanCode=goal, height=0)
        heap = PriorityQueue()
        heap.put_nowait((Root.height, Root))

        tmp_dir = Path(self.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        Proved = False
        ProvedNode: Node = None

        def worker():
            """
            工作线程：从队列取节点，处理后将子节点放回队列
            """
            nonlocal Proved
            nonlocal start_time
            nonlocal tmp_dir
            nonlocal Root
            nonlocal ProvedNode
            nonlocal heap

            while True:
                try:
                    node: Node = heap.get()[1]

                    if node is None:
                        heap.task_done()
                        break

                    if Proved or time.time() - start_time > self.timeout or node.status != Status.Open:
                        heap.task_done()
                        continue

                    tmp_file = tmp_dir / f"{node.id}.lean"

                    tips, error = self.run_node(node, tmp_file)

                    if Root.status == Status.Proved:
                        Proved = True
                        ProvedNode = node
                        heap.task_done()
                        continue

                    if node.status == Status.Open:
                        prompts = []
                        prompt = self.prompt_builder.build_chat_for_tactic(node.leanCode, tips)
                        for _ in range(self.degree):
                            prompts.append(prompt)
                        tactics: List[str] = self.model.predict(prompts)
                        for t in tactics:
                            t = extract_lean_code_after_marker(t)
                            new_edge = Edge(tactic=t, height=node.height, src=node)
                            new_node = Node(leanCode=node.leanCode+"\n"+t, height=node.height + 1, src=new_edge)
                            new_edge.dst = new_node
                            node.dst.append(new_edge)
                            heap.put((new_node.height, new_node))
                    heap.task_done()  # 标记任务完成
                except Exception as e:
                    print(f"线程退出: {e}")
                    break

            
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _ in range(num_workers):
                executor.submit(worker)
            heap.join()

            for _ in range(num_workers):
                heap.put((0, None))
            heap.join()
                    
        if Proved:
            return Root, ProvedNode.leanCode
        else:
            return None, "timeout"
            