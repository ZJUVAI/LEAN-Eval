from queue import PriorityQueue #优先队列
from LeanEval.validator.search_tree import Node, Edge, Status #搜索树的定义
from pathlib import Path
from typing import List, Optional, Tuple, Union 
from LeanEval.validator.proof_validator import ProofValidator #导入验证器
import json, os, time
from concurrent.futures import ThreadPoolExecutor #多线程执行
from LeanEval.models.base_api import BaseAPIModel #模型基类
from LeanEval.prompt import PromptBuilder #构建器
from LeanEval.utils.handle_lean_result import handle_lean_str #处理Lean结果字符串的工具函数
from LeanEval.utils.extract_lean_code import extract_lean_block
import threading
import queue
import sys

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
        tmp_file.write_text(node.leanCode, encoding="utf-8")
        success, msg = self.proofValidator.validate_file(tmp_file)
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
                    t = extract_lean_block(t)
                    print("get tactic: ", t)
                    new_edge = Edge(tactic=t, height=node.height, src=node)
                    new_node = Node(leanCode=node.leanCode+"\n"+t, height=node.height + 1, src=new_edge)
                    new_edge.dst = new_node
                    node.dst.append(new_edge)
                    heap.put_nowait((new_node.height, new_node))


    def thread_prove(self, goal: str, num_workers: int = os.cpu_count() + 4) -> Tuple[Optional[Node], Optional[str]]:
        """
        使用多线程和 BFS 搜索来并行证明目标。
        """
        Root = Node(leanCode=goal, height=0)
        # 优先队列是线程安全的
        heap = PriorityQueue()
        heap.put_nowait((Root.height, Root))

        tmp_dir = Path(self.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        # Proved 事件用于向所有线程广播“任务已完成”的信号
        Proved = threading.Event()
        ProvedNode: Optional[Node] = None
        # node_lock 用于在多线程环境下安全地写入 ProvedNode
        node_lock = threading.Lock()

        def worker():
            """
            工作线程：从队列取节点，处理后将子节点放回队列。
            """
            while not Proved.is_set():
                try:
                    # 1. 从队列中获取任务，设置一个短超时
                    #    这可以防止在队列暂时为空时线程永久阻塞，
                    #    并允许线程周期性地检查 Proved 状态。
                    node_tuple = heap.get(timeout=0.5)
                    node: Node = node_tuple[1]

                    # 2. 冗余检查，确保在等待期间状态没有改变
                    if Proved.is_set() or (time.time() - start_time > self.timeout):
                        log_msg = f"线程 {threading.get_ident()} 发现超时或已证明，退出循环"
                        print(log_msg, file=sys.stderr)
                        break # 如果已证明或超时，立即退出循环

                    if node.status != Status.Open:
                        heap.task_done() # Proved 或 Error
                        continue

                    # 为每个线程和节点创建唯一的临时文件
                    tmp_file = tmp_dir / f"proof_{threading.get_ident()}_{node.id}.lean"
                    tips, error = self.run_node(node, tmp_file)
                    print(f"线程 {threading.get_ident()} 处理节点 {node.id} 完成，状态: {node.status}, 错误: {error}")

                    # 3. 如果根节点状态变为 Proved，则设置事件并记录结果
                    if Root.status == Status.Proved:
                        if not Proved.is_set(): # 避免重复设置和写入
                            with node_lock:
                                # 双重检查锁定，确保只记录第一个成功的证明
                                if ProvedNode is None:
                                    ProvedNode = node
                            Proved.set() # 广播成功信号
                        break # 当前线程完成使命，退出

                    # 4. 如果节点仍然开放，则调用模型生成新的策略
                    if node.status == Status.Open:
                        chat_prompt = self.prompt_builder.build_chat_for_tactic(node.leanCode, tips)
                        
                        string_prompt = self.model.tokenizer.apply_chat_template(
                            chat_prompt, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        
                        prompts_for_batch = [string_prompt] * self.degree
                        tactics: List[str] = self.model.batch_predict(prompts_for_batch)

                        for t in tactics:
                            t = extract_lean_block(t)
                            if not t: 
                                continue
                            print(f"线程 {threading.get_ident()} 获取策略: {t}")
                            new_edge = Edge(tactic=t, height=node.height, src=node)
                            new_node = Node(leanCode=node.leanCode + "\n  " + t, height=node.height + 1, src=new_edge)
                            new_edge.dst = new_node
                            node.dst.append(new_edge)
                            # 将新生成的节点放回队列，供其他线程处理
                            heap.put((new_node.height, new_node))
                    
                    heap.task_done()

                except queue.Empty:
                    # 队列为空是正常情况，继续下一次循环检查 Proved 状态
                    continue
                except Exception as e:
                    print(f"线程 {threading.get_ident()} 发生致命错误并退出: {type(e).__name__}: {e}", file=sys.stderr)
                    # 可以在此设置 Proved 事件来停止所有其他线程
                    # Proved.set() 
                    break
        
        # --- 主线程逻辑 ---
        executor = ThreadPoolExecutor(max_workers=num_workers)
        try:
            # 提交所有工作线程
            futures = [executor.submit(worker) for _ in range(num_workers)]

            # 主线程的等待循环
            while time.time() - start_time < self.timeout:
                if Proved.is_set():
                    break
                # 如果所有工作线程都因为异常或其他原因结束了，主线程也应退出等待
                if all(f.done() for f in futures):
                    break
                time.sleep(0.1)  # 短暂休眠，避免CPU空转
        
        finally:
            # 无论是因为成功、超时还是异常，都要确保设置 Proved 事件
            # 以便所有正在 heap.get() 上等待的线程能够退出它们的循环
            Proved.set()
            
            # 关闭线程池，等待所有线程优雅地退出
            # `shutdown(wait=True)` 会等待所有已提交的任务完成，
            # 因为 Proved 事件已设置，所有 worker 都会很快结束。
            executor.shutdown(wait=True)

        # 最终返回结果
        with node_lock:
            if ProvedNode:
                return Root, ProvedNode.leanCode
            else:
                return None, "Timeout or no solution found"