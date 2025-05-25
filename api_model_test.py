# from openai import OpenAI

# api_key = "sk-Knq6X2wZut1jYaOXBc1120C2474141E2885a14F3A0D11fF8"
# base_url = "https://api-zjuvai.newnan.city/v1"

# client = OpenAI(
#     api_key=api_key,
#     base_url=base_url
# )
# completion = client.chat.completions.create(
#     model="gemini-2.0-flash-exp",
#     messages=[
#         {"role":"system","content":"You are a helpful asistant"},
#         {
#             "role":"user",
#             "content":"你好"
#         }
#     ]
# )
# print(completion.choices[0].message.content)
# import time, requests, concurrent.futures as cf
# S = requests.Session()

# for size in (1, 16, 32):                      # size = 并发线程数
#     adapter = requests.adapters.HTTPAdapter(pool_connections=size,
#                                             pool_maxsize=size,
#                                             pool_block=False)
#     S.mount("https://", adapter)

#     def one():
#         t0 = time.time()
#         S.get("https://httpbin.org/delay/3")  # 固定 3s 的回显
#         return time.time() - t0

#     t0 = time.time()
#     with cf.ThreadPoolExecutor(max_workers=size) as pool:
#         list(pool.map(lambda _: one(), range(size)))
#     print(f"并发 {size:>2} 条，用时 {time.time()-t0:.2f}s")
import requests
import json

# 你的API密钥和基础URL
api_key = "sk-Knq6X2wZut1jYaOXBc1120C2474141E2885a14F3A0D11fF8"
base_url = "https://api-zjuvai.newnan.city/v1"

# 1. 构建完整的 URL
# 假设这个API模仿OpenAI，聊天补全的路径是 /chat/completions
# 你需要确认这个路径对于 api-zjuvai.newnan.city 是否正确
chat_completions_path = "/chat/completions"
full_url = base_url.rstrip('/') + chat_completions_path

# 2. 设置请求头
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 3. 构建请求体 (Payload)
payload = {
    "model": "gemini-2.0-flash-exp", # 确保这个模型名称是该端点支持的
    "messages": [
        {"role": "user", "content": "你好"}
    ]
    # 如果API支持其他参数，如 temperature, max_tokens 等，也可以在这里添加
    # "temperature": 0.8,
    # "max_tokens": 150, # 注意：参数名需要与API文档一致
}

try:
    # 4. 发送 POST 请求
    # timeout 参数可以根据需要设置，单位是秒
    response = requests.post(full_url, headers=headers, data=json.dumps(payload), timeout=60)

    # 5. 处理响应
    response.raise_for_status()  # 如果HTTP请求返回了不成功的状态码 (4xx or 5xx)，这会抛出HTTPError异常

    # 解析JSON响应
    completion_data = response.json()

    # 提取需要的内容 (根据OpenAI的格式)
    if completion_data and "choices" in completion_data and len(completion_data["choices"]) > 0:
        message_content = completion_data["choices"][0].get("message", {}).get("content")
        if message_content:
            print(message_content)
        else:
            print("无法从响应中提取消息内容。")
            print("完整响应:", completion_data)
    else:
        print("响应格式不符合预期。")
        print("完整响应:", completion_data)

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP错误: {http_err}")
    print(f"响应内容: {response.text if 'response' in locals() else '无响应内容'}")
except requests.exceptions.ConnectionError as conn_err:
    print(f"连接错误: {conn_err}")
except requests.exceptions.Timeout as timeout_err:
    print(f"请求超时: {timeout_err}")
except requests.exceptions.RequestException as req_err:
    print(f"请求发生错误: {req_err}")
except json.JSONDecodeError:
    print("无法解析响应的JSON内容。")
    print(f"原始响应文本: {response.text if 'response' in locals() else '无响应内容'}")
except Exception as e:
    print(f"发生未知错误: {e}")