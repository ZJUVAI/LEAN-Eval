from lean_dojo import *

def main():
  repo = LeanGitRepo("https://github.com/maksymilan/lean-eval" , "7001b84bf1239a35de986044c78708f745203e24")
  theorem = Theorem(repo, "test.lean", "add_comm")

  with Dojo(theorem) as (dojo, init_state):
    print(init_state)

if __name__ == "__main__":
  main()