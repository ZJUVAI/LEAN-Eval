from typing import Tuple
import re
from typing import List

def handle_lean_str(leanResult: str) -> Tuple[List[str], List[str]]:
    """
    处理真实的 Lean 编译器输出结果，返回“待办目标”和“错误”的完整信息块。

    Args:
        leanResult (str): 从 Lean 进程捕获的原始 stdout/stderr 字符串。

    Returns:
        Tuple[List[str], List[str]]: 
            - tips: 包含 'unsolved goals' 的完整信息块列表。
            - errors: 其他 'error' 信息块的列表。
    """
    if not isinstance(leanResult, str) or not leanResult.strip():
        return [], []

    unsolved_goals_tips = []
    error_messages = []

    # 定义一个正则表达式，用于匹配消息的头部
    # 例如: "path/to/file.lean:10:4: error:"
    # 捕获组 ( ... ) 会让 re.split 保留分隔符
    message_header_pattern = re.compile(
        r"^(.*?:\d+:\d+:\s*(?:error|warning|note))", 
        re.MULTILINE
    )

    # 使用正则表达式分割整个输出字符串
    # 结果会是 [before_first_header, header1, content1, header2, content2, ...]
    parts = message_header_pattern.split(leanResult)
    
    # 第一个元素是第一个消息头之前的所有内容，通常为空或无用，我们忽略它
    # 然后以步长2遍历，每次处理一个 (header, content) 对
    for i in range(1, len(parts), 2):
        header = parts[i]
        # content 可能包含多行，所以我们保留它的原始格式
        content = parts[i+1]
        
        # 将头部和内容重新组合成一个完整的消息块
        full_message_block = (header + content).strip()
        
        header_lower = header.lower()

        # 判断消息块的类型
        if "error: unsolved goals" in header_lower:
            # 这是一个包含待办目标的信息块，是我们需要的 "tips"
            unsolved_goals_tips.append(full_message_block)
        elif "error" in header_lower:
            # 其他类型的错误信息块
            error_messages.append(full_message_block)
        
        # 'warning' 和 'note' 类型的消息块被自动忽略

    return unsolved_goals_tips, error_messages
