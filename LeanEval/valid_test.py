from validator.proof_validator import ProofValidator
from utils.handle_lean_result import split_lean_result, handle_lean_result
import json

if __name__ == "__main__":
    validator = ProofValidator(timeout=60)        # 可自定义 lean_cmd、work_dir
    # 单文件验证功能检查（√）
    # ok, log = validator.validate_file("LeanEval/LeanProblems/1.lean")  # 验证单个文件
    # print(ok,"\n",log) 
    # 多文件验证功能检查（√）
    # files = [f"LeanEval/LeanProblems/{index}.lean" for index in range(10)]
    # passed,failed = validator.validate_batch(files)
    # print(passed,failed) 
    # 字符串验证功能检查(√)
    # with open('./LeanProblems/2.lean',"r") as f:
    #     ques = f.read()
    #     ok,log = validator.validate_code(ques, clean_up=False)
    #     print(ok,"\n",log)
    # 目录验证功能检查（√）
    # results = validator.validate_dir("LeanEval/LeanProblems")
    # for result in results:
    #     print(result[1])
    _, result = validator.validate_file("output/proof_1.lean")
    splited_result = split_lean_result("output", result)
    handled_result = handle_lean_result("unsolved goals", splited_result)
    data = json.loads(handled_result)
    for i in data["have"]:
        print(i)
        print("\n")