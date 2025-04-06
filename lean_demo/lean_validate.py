from lean_dojo import *

def main():
  repo = LeanGitRepo("https://github.com/maksymilan/lean-eval" , "6bd0900d470f688ec876134fdd54406b2de4a36e")
  theorem = Theorem(repo, "test.lean", "add_comm")

  with Dojo(theorem) as (dojo, init_state):
    print(init_state)

if __name__ == "__main__":
  main()