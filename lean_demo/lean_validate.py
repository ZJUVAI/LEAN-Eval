#要写注释
from lean_dojo import *#不要import*

def main():#这里应该是一个模块函数
  repo = LeanGitRepo("https://github.com/maksymilan/lean-eval" , "6bd0900d470f688ec876134fdd54406b2de4a36e")

  theorem = Theorem(repo, "test.lean", "add_comm")

  with Dojo(theorem) as (dojo, init_state):
    print(init_state)
  #该函数没有返回值，外部无法调用
if __name__ == "__main__":
  main()
