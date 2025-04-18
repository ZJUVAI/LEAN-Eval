import subprocess #用于执行命令行
from typing import Tuple 

def verify_lean_file(filename) -> Tuple[bool, str]:  
    """
    调用 lean 命令运行 .lean 文件

    Args:
        values (str): 文件路径

    Returns:
        Tuple[bool, str]: 运行结果

    Raises:
        None
    """
    try:
        result = subprocess.run(
            ["lake", "env", "lean", filename],          # 使用 --run 执行 Lean 脚本
            capture_output=True,                        # 捕获标准输出和标准错误
            text=True,                                  # 将输出解码为字符串
            # cwd="../",                                  # 设置工作目录为项目根目录
            timeout=100,                                # 设置超时时间
        )
    except subprocess.TimeoutExpired:
        return False, "Lean 超时"
    if result.returncode:
        print(result.stderr)

    return result.returncode, result.stdout

if __name__ == "__main__":

    # 正例：一个可以成功运行的 Lean 文件
    good_file = "../output/proof_1.lean"  # 这里假设你已经有一个有效的 Lean 文件

    # 反例：一个会报错的 Lean 文件（比如语法错误）
    bad_file = "../output/proof_1.lean"  # 这里假设你有一个无效的 Lean 文件

    print("正例测试:")
    good_output = verify_lean_file(good_file)
    if good_output:
        print("Lean 执行成功，返回信息如下：")
        print("Return code:", good_output[0])
        print("stdout:\n", good_output[1])

    print("\n反例测试:")
    bad_output = verify_lean_file(bad_file)
    if bad_output:
        print("Lean 执行失败，返回信息如下：")
        print("Return code:", bad_output[0])
        print("stdout:\n", bad_output[1])

