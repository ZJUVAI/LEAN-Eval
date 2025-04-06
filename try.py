import subprocess
import json

def run_lean_file():
    # 调用 lean 命令运行 .lean 文件
    result = subprocess.run(
        ["lake", "env", "lean", "--run", "Main.lean"],  # 使用 --json 选项获取 JSON 输出
        capture_output=True,            # 捕获标准输出和标准错误
        text=True                       # 将输出解码为字符串
    )

    # 检查是否成功运行
    if result.returncode != 1:
        print("Error running Lean:")
        print(result.stderr)
        return None
    
    return result

# 测试
#file_path = "../data/test.lean"
output = run_lean_file()
if output:
    print("Lean output:", output)