import subprocess
import json

def run_lean_file(filename):  # ✅ TODO: 改为参数传入文件名
    """
    调用 lean 命令运行 .lean 文件

    Args:
        values (str): 文件路径

    Returns:
        json: 运行结果

    Raises:
        None
    """
    result = subprocess.run(
        ["lake", "env", "lean", filename],  # 使用 --run 执行 Lean 脚本
        capture_output=True,                         # 捕获标准输出和标准错误
        text=True                                    # 将输出解码为字符串
    )

    return result

# ✅ TODO: 添加正例和反例
# 正例：一个可以成功运行的 Lean 文件
good_file = "examples/simple_proof.lean"

# 反例：一个会报错的 Lean 文件（比如语法错误）
bad_file = "examples/simple_proof_wrong.lean"

print("🟢 正例测试:")
good_output = run_lean_file(good_file)
if good_output:
    print("✅ Lean 执行成功，返回信息如下：")
    print("Return code:", good_output.returncode)
    print("stdout:\n", good_output.stdout)
    print("stderr:\n", good_output.stderr)

print("\n🔴 反例测试:")
bad_output = run_lean_file(bad_file)
if bad_output:
    print("Return code:", bad_output.returncode)
    print("❌ Lean 执行失败，返回信息如下：")
    print("stdout:\n", bad_output.stdout)
    print("stderr:\n", bad_output.stderr)

