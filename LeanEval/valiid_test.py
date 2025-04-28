from validator.proof_validator import ProofValidator

validator = ProofValidator(timeout=60)        # 可自定义 lean_cmd、work_dir
# 单文件验证功能检查（√）
# ok, log = validator.validate_file("./LeanProblems/1.lean")  # 验证单个文件
# print(ok,"\n",log) 
# 多文件验证功能检查（√）
# files = [f"{index}.lean" for index in range(20)]
# passed,failed = validator.validate_batch(files)
# print(passed,failed) 
# 字符串验证功能检查
with open('./LeanProblems/1.lean',"r") as f:
    ques = f.read()
    ok,log = validator.validate_code(ques)
    print(ok,log)