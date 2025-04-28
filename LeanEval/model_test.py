from openai import OpenAI

api_key = "sk-Knq6X2wZut1jYaOXBc1120C2474141E2885a14F3A0D11fF8"
base_url = "https://api-zjuvai.newnan.city/v1"

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)
completion = client.chat.completions.create(
    model="gemini-2.0-flash-exp",
    messages=[
        {"role":"system","content":"You are a helpful asistant"},
        {
            "role":"user",
            "content":"你好"
        }
    ]
)
print(completion.choices[0].message.content)
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
