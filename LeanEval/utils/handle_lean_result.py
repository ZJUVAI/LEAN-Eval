from typing import Tuple

def discurd_lean_result(short_str: str, string_array: list[str]) -> list[str]:
    """
    过滤掉列表中包含短字符串的字符串
    
    Args:
    short_str (str): 要检查的短字符串
    string_array (list): 字符串列表
    
    返回:
    list: 不包含short_str的新列表
    """
    return [s for s in string_array if short_str not in s]

def split_lean_result(short_str: str, long_str: str):
    """
    以短字符串作为分隔依据切分长字符串，确保每个切片的开头都是短字符串
    
    Args:
        short_str (str): 作为分隔依据的短字符串（如 "output"）
        long_str (str): 需要切分的长字符串
        
    Returns:
        list[str]: 切分后的字符串列表，每个元素都以 short_str 开头
    """
    parts = long_str.split(short_str)
    
    if parts and not parts[0].startswith(short_str):
        parts = parts[1:]
    
    result = [short_str + part for part in parts if part.strip()]
    
    return result

def handle_lean_result(short_str, string_array) -> Tuple[list[str], list[str]]:
    """
    根据短字符串筛选字符串数组，并返回分类结果
    
    Args:
        short_str (str): 要匹配的短字符串
        string_array (list[str]): 待筛选的字符串数组
        
    Returns:
        str: 包含的和不包含短字符串的字符串列表
    """
    have = []
    havenot = []
    
    for s in string_array:
        if short_str in s:
            have.append(s)
        else:
            havenot.append(s)
    
    return have, havenot

def handle_lean_str(leanResult: str) -> Tuple[list[str], list[str]]:
    """
    处理 Lean 的输出结果，返回有效信息和错误

    Args:
        leanResult (str): Lean 的输出结果

    Returns:
        Tuple[list[str], list[str]]: 有效信息和错误列表
    """
    split_result = split_lean_result("LeanEval", leanResult)
    have, havenot = handle_lean_result("unsolved goals", split_result)
    havenot = discurd_lean_result("warning", havenot)
    return have, havenot

if __name__ == "__main__":
    leanResult = (
        "LeanEval: unsolved goals (1)\n"
        "LeanEval: error (1)\n"
        "LeanEval: unsolved goals (2)\n"
        "LeanEval: warning (1)\n"
        "LeanEval: error (2)\n"
        "LeanEval: warning (2)\n"
    )
    have, havenot = handle_lean_str(leanResult)
    print(have)
    print(havenot)