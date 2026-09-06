"""弹幕情绪分类 FastAPI 服务。

参照 TMF 项目的 API 架构：后端 FastAPI（8000 端口）+ 前端通过 requests 调用。
POST /predict，输入 {"text": "弹幕内容"}，返回 {"text": ..., "pred_class": "正面/中性/负面"}。

运行：python api.py
文档：启动后访问 http://127.0.0.1:8000/docs 看自动生成的 Swagger 接口文档。
"""
import sys
import time
from pathlib import Path

# 把 src/ 加入 sys.path，复用 BERT 预测模块
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from bert_predict import classify_batch, load_model

# 启动时加载模型（400MB，只加载一次，之后常驻内存）
print("加载 BERT 模型...")
load_model()
print("模型加载完成")

app = FastAPI(title="弹幕情绪分类 API", description="输入一条弹幕，返回情绪标签（正面/中性/负面）。")


class PredictRequest(BaseModel):
    text: str
    model_type: str = "teacher"   # teacher / student / quantized


class PredictBatchRequest(BaseModel):
    texts: list[str]
    model_type: str = "teacher"   # teacher / student / quantized


@app.post("/predict")
async def predict_api(req: PredictRequest):
    try:
        pred_class = classify_batch([req.text], model_type=req.model_type)[0]
        return {"text": req.text, "pred_class": pred_class, "model_type": req.model_type}
    except Exception as e:
        return {"error": str(e)}, 400


@app.post("/predict_batch")
async def predict_batch_api(req: PredictBatchRequest):
    try:
        start = time.time()
        pred_classes = classify_batch(req.texts, model_type=req.model_type)
        elapsed_ms = (time.time() - start) * 1000
        avg_ms = elapsed_ms / len(req.texts) if req.texts else 0.0
        return {"pred_classes": pred_classes, "elapsed_ms": elapsed_ms,
                "avg_ms": avg_ms, "count": len(req.texts), "model_type": req.model_type}
    except Exception as e:
        return {"error": str(e)}, 400


class CrawlRequest(BaseModel):
    bvid: str
    fetch_history: bool = False


@app.post("/crawl_danmaku")
async def crawl_danmaku_api(req: CrawlRequest):
    """输入陌生视频 bvid，爬取弹幕（当前弹幕 + 可选历史弹幕）。"""
    try:
        from crawler import crawl_video_danmaku
        return crawl_video_danmaku(req.bvid.strip(), req.fetch_history)
    except Exception as e:
        return {"error": str(e)}, 400


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
