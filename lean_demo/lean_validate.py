from lean_dojo import *

local_repo_path = "C:\\Users\\Lenovo\\Desktop\\pythonlab\\lean-eval"
local_file_path = "C:\\Users\\Lenovo\\Desktop\\pythonlab\\lean-eval\\data\\test.lean"
repo = LeanGitRepo(local_repo_path , None)
theorem = Theorem(repo, local_file_path, "add_comm")

with Dojo(theorem) as (dojo, init_state):
  print(init_state)