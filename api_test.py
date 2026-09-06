"""FastAPI 服务测试脚本：向 127.0.0.1:8000/predict 发请求验证推理。

运行前先启动后端：python api.py
"""
import time

import requests

URL = "http://127.0.0.1:8000/predict"

tests = [
    "刀死我了这也太虐了吧",
    "哈哈哈哈笑死我了",
    "前排打卡",
]

for text in tests:
    start = time.time()
    res = requests.post(URL, json={"text": text})
    elapsed = time.time() - start
    print(f"{text!r:20} -> {res.json()}  ({elapsed * 1000:.1f} ms)")
