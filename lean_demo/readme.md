# Lean Evaluation

这个文件夹是跑通lean证明的代码
- main.py是主文件，负责调用模型和验证模型生成的结果是否能在leandojo中通过
- lean_validata.py是用来实现leandojo的验证，输入为题目和模型输出的结果，验证该结果是否正确
- model_generation.py是实现模型的调用，输入题目，输出模型的结果，保证结果是一行一行完整的lean代码 

其中的题目，结果都用json格式表示
