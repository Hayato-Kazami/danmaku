"""弹幕情绪分类 FastAPI 服务。

参照 TMF 项目的 API 架构：后端 FastAPI（8000 端口）+ 前端 Streamlit 通过 requests 调用。

接口：
    POST /predict         单条预测   {"text": "...", "model_type": "teacher"}
    POST /predict_batch   批量预测   {"texts": [...], "model_type": "teacher"}
    POST /crawl_danmaku   按 bvid 爬弹幕
    GET  /health          ⭐ 各模型路径的实际解析结果 + 加载状态（"模型找不到"先看这个）

运行：python api.py          # 从任何目录都可以，所有路径基于 __file__ 解析，与 CWD 无关
文档：http://127.0.0.1:8000/docs

环境变量：
    BERT_MODEL_PATH   覆盖 bert-base-chinese 目录（由 src/config.py 读取）
    PRELOAD_MODELS    启动时预加载的模型，逗号分隔，默认 "teacher"
    API_HOST/API_PORT 监听地址与端口，默认 127.0.0.1:8000

路径来源（单一事实来源，本文件不重复推导任何模型路径）：
    src/config.py            → bert_model_path / *_model_path
    src/bert_predict.py      → _MODEL_PATHS / _MODEL_DEVICES（模型类型 → 权重路径 / 运行设备）
"""
import os
import sys
import time
from pathlib import Path
from typing import Literal

# ===================== 路径引导（与 CWD 无关） =====================
PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"


def _bootstrap_sys_path() -> None:
    """把 src/ 与项目根加入 sys.path —— 追加到末尾，不用 insert(0)。

    追加而非插到最前：src/ 里有 config.py / utils.py 这类通用名，
    插到 sys.path[0] 会遮蔽标准库或第三方同名模块（`import config` 在
    huggingface_hub / transformers 内部也出现过），追加则永远不会抢走别人的模块。
    """
    for d in (SRC_DIR, PROJECT_DIR):
        if not d.is_dir():
            raise RuntimeError(
                f"目录不存在：{d}\n"
                "  → api.py 必须和 src/ 一起在项目根目录下，别单独把 api.py 拷走。"
            )
        p = str(d)
        if p not in sys.path:
            sys.path.append(p)


_bootstrap_sys_path()


def _assert_config_source() -> None:
    """确认 import 到的 config 就是本项目的 src/config.py。

    防止「CWD 或 sys.path 里有个同名 config.py 抢了先」——那会让所有模型路径
    指向别处，且报错完全不像路径问题（下游只会炸 AttributeError）。
    函数定义在此、调用在下游 import 之前（见下方注释）。
    """
    got = Path(_config_mod.__file__).resolve().parent
    if os.path.normcase(str(got)) != os.path.normcase(str(SRC_DIR.resolve())):
        raise ImportError(
            f"导入到的 config 不是本项目的 src/config.py\n"
            f"  期望目录：{SRC_DIR}\n"
            f"  实际来源：{_config_mod.__file__}\n"
            "  → 删掉 CWD 或 sys.path 里的同名 config.py，或始终从项目根目录启动。"
        )


import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config as _config_mod          # 仅用于下面的"导入来源"自检

# ⚠️ 顺序很重要：必须在 import bert_predict 之前跑。
# bert_predict / bert_classifier 内部都 `from config import Config`，
# 若被同名模块遮蔽，它们会先炸出一个莫名其妙的 AttributeError，
# 就轮不到这里给出"到底谁抢了 config"的可读报错了。
_assert_config_source()

import bert_predict as bp             # 模型路径 / 设备的唯一来源

# ===================== 启动自检 =====================


# 模型类型必须与 bert_predict 对齐，否则这里改了名那边还在用旧名（静默 400）
_MODEL_TYPES = ("teacher", "student", "quantized")
if set(_MODEL_TYPES) != set(bp._MODEL_PATHS):
    raise RuntimeError(
        f"模型类型不一致：api.py={sorted(_MODEL_TYPES)}，"
        f"bert_predict.py={sorted(bp._MODEL_PATHS)}"
    )

conf = bp.conf                        # 复用 bert_predict 已构造的 Config，避免重复加载 tokenizer


def _size_of(p: Path) -> str:
    """文件 → 自身大小；目录（预训练权重目录）→ 里面权重文件的大小。"""
    if p.is_file():
        return f"{p.stat().st_size / 1024 / 1024:8.1f} MB"
    if p.is_dir():
        for name in ("model.safetensors", "pytorch_model.bin"):
            f = p / name
            if f.is_file():
                return f"{f.stat().st_size / 1024 / 1024:8.1f} MB"
        return "    目录"
    return "      --"


def _preflight() -> bool:
    """打印每个模型路径解析成了什么、文件在不在。返回"老师模型是否可用"。"""
    print("=" * 72)
    print("模型路径自检（路径全部由 src/config.py 解析，api.py 不自行拼路径）")
    print("-" * 72)
    rows = [("bert-base-chinese", Path(conf.bert_model_path))]
    rows += [(f"{mt} 权重", Path(p)) for mt, p in bp._MODEL_PATHS.items()]
    for name, p in rows:
        mark = " OK " if p.exists() else "MISS"
        kind = "目录" if p.is_dir() else "文件"
        print(f"  [{mark}] {name:16s} {_size_of(p)} {kind}  {p}")
    print("-" * 72)
    return Path(conf.bert_model_path).is_dir() and Path(
        bp._MODEL_PATHS["teacher"]).is_file()


BERT_OK = _preflight()
if not BERT_OK:
    print("  ⚠️ 上表有 MISS：预训练目录或 teacher 权重缺失，/predict 会返回 400 并给出实际路径。")
print("=" * 72)


def _preload(types) -> None:
    """预加载指定模型。失败不阻断启动 —— 服务要能起来，/health 才有机会告诉你缺什么。"""
    for mt in types:
        t0 = time.time()
        try:
            bp.load_model(mt)
            print(f"  [已加载] {mt:10s} {time.time() - t0:6.2f}s  device={bp._MODEL_DEVICES[mt]}")
        except Exception as e:
            print(f"  [失败]   {mt:10s} {type(e).__name__}: {e}")


_PRELOAD = [m.strip() for m in os.environ.get("PRELOAD_MODELS", "teacher").split(",") if m.strip()]
print(f"预加载模型：{_PRELOAD}（可用 PRELOAD_MODELS 环境变量调整）")
_preload(_PRELOAD)
print("=" * 72)

# ===================== FastAPI =====================

app = FastAPI(
    title="弹幕情绪分类 API",
    description="输入弹幕，返回情绪标签（正面/中性/负面）。三种模型：teacher / student / quantized。",
)

ModelType = Literal["teacher", "student", "quantized"]


class PredictRequest(BaseModel):
    text: str
    model_type: ModelType = "teacher"


class PredictBatchRequest(BaseModel):
    texts: list[str]
    model_type: ModelType = "teacher"


class CrawlRequest(BaseModel):
    bvid: str
    fetch_history: bool = False


def _err(e: Exception) -> JSONResponse:
    """统一错误响应。保持 {"error": ...} 契约（app.py 依赖这个 key 取消息）。"""
    msg = f"{type(e).__name__}: {e}"
    print(f"[ERROR] {msg}")
    return JSONResponse(status_code=400, content={"error": msg})


def _model_report() -> dict:
    rep = {}
    for mt, path in bp._MODEL_PATHS.items():
        p = Path(path)
        rep[mt] = {
            "path": str(p),
            "exists": p.is_file(),
            "size_mb": round(p.stat().st_size / 1024 / 1024, 1) if p.is_file() else None,
            "device": str(bp._MODEL_DEVICES[mt]),
            "loaded": mt in bp._models,
        }
    return rep


@app.get("/health")
async def health():
    """模型路径与加载状态。排查"模型找不到"时第一个该看的接口。"""
    models = _model_report()
    ready = [mt for mt, v in models.items() if v["exists"]]
    return {
        "status": "ok" if "teacher" in ready else "degraded",
        "device": str(conf.device),
        "cuda_available": conf.device.type == "cuda",
        "bert_model_path": conf.bert_model_path,
        "bert_model_path_exists": Path(conf.bert_model_path).is_dir(),
        "cwd": os.getcwd(),          # 与路径解析无关，仅方便确认启动位置
        "project_dir": str(PROJECT_DIR),
        "models": models,
    }


@app.post("/predict")
async def predict_api(req: PredictRequest):
    try:
        pred_class = bp.classify_batch([req.text], model_type=req.model_type)[0]
        return {"text": req.text, "pred_class": pred_class, "model_type": req.model_type}
    except Exception as e:
        return _err(e)


@app.post("/predict_batch")
async def predict_batch_api(req: PredictBatchRequest):
    try:
        start = time.time()
        pred_classes = bp.classify_batch(req.texts, model_type=req.model_type)
        elapsed_ms = (time.time() - start) * 1000
        avg_ms = elapsed_ms / len(req.texts) if req.texts else 0.0
        return {"pred_classes": pred_classes, "elapsed_ms": elapsed_ms,
                "avg_ms": avg_ms, "count": len(req.texts), "model_type": req.model_type}
    except Exception as e:
        return _err(e)


@app.post("/crawl_danmaku")
async def crawl_danmaku_api(req: CrawlRequest):
    """输入陌生视频 bvid，爬取弹幕（当前弹幕 + 可选历史弹幕）。"""
    try:
        from crawler import crawl_video_danmaku
        return crawl_video_danmaku(req.bvid.strip(), req.fetch_history)
    except Exception as e:
        return _err(e)


if __name__ == "__main__":
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    print(f"启动服务：http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)
