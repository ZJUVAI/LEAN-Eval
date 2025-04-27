from models import ModelRegistry
from models import DeepSeekAPIModel
with ModelRegistry.create(
        "deepseek_api",
        model_name="deepseek-chat",
        api_url="https://api.deepseek.com/beta/chat/completions",
        api_key="sk-c5172f87dfe4418899fefd6cb6ee7309",
        proxies={"http": None, "https": None},
        timeout=20,               # 可以自定义传入Config内未定义的字段
        temperature=0.8,
) as model:
    with open('result.txt','w') as f:
        # pass
        f.write(model.predict("什么是计算机系统"))
