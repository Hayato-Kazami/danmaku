"""FastAPI 服务测试脚本：先查 /health 确认模型路径，再向 /predict 发请求验证推理。

运行前先启动后端：python api.py（本脚本会自动等待后端就绪，无需卡时间）
"""
import time

import requests

BASE = "http://127.0.0.1:8000"
URL = f"{BASE}/predict"


def wait_ready(timeout=90):
    """等后端就绪。api.py 启动时要加载 400MB 权重（约 10-20 s），
    期间端口还没监听，直接请求会拿到连接错误或空响应。"""
    for i in range(timeout):
        try:
            r = requests.get(f"{BASE}/health", timeout=5)
            if r.status_code == 200 and r.text.strip():
                return r.json()
        except requests.RequestException:
            pass
        if i == 0:
            print(f"等待后端就绪（{BASE}）...")
        time.sleep(1)
    raise SystemExit(f"等待 {timeout}s 后端仍未就绪，请确认已运行 `python api.py`")


health = wait_ready()
print(f"status={health['status']}  device={health['device']}")
print(f"bert-base-chinese: {health['bert_model_path']}  exists={health['bert_model_path_exists']}")
for mt, info in health["models"].items():
    flag = "OK  " if info["exists"] else "MISS"
    print(f"  [{flag}] {mt:10s} {str(info['size_mb']):>8s} MB  {info['device']:4s}  {info['path']}")
print()

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
