import re
from typing import Optional

def extract_lean_block(text: str) -> Optional[str]:
    """
    从文本中提取第一个被 ```lean ... ``` 包裹的代码块。
    """
    # 改进正则表达式以处理可选的换行符和不同的代码块后的内容
    match = re.search(r"lean\s*\n(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip() # group(1) 是括号里的内容
    
    # 后备：如果上面没匹配到，尝试匹配没有结尾换行符的情况，或者更宽松的匹配
    match = re.search(r"lean\s*\n?(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    match = re.search(r"lean(.*?)\s*```", text, re.DOTALL) # 更宽松，允许```lean后面直接跟代码
    if match:
        return match.group(1).strip()

    return text