"""BERT 弹幕预测：情绪三分类。

加载训练好的模型，提供批量预测接口，供 Streamlit 应用 app.py 调用。

情绪三分类支持三种模型（model_type）：
    - teacher   老师 BERT（0.7862，390MB，GPU）
    - student   蒸馏 BiLSTM（0.6741，12.8MB，GPU）
    - quantized int8 量化 BERT（0.7675，145.6MB，仅 CPU）

模型全部懒加载 + 进程内缓存，避免反复加载数百 MB 权重。
（剧透检测已独立成 d:/Code/spoiler_bert 项目。）
"""
from pathlib import Path

import torch

from config import Config
from bert_classifier import TMFBertClassifier
from student_model import BiLSTM

conf = Config()

# 情绪三分类可选模型：类型 -> 权重文件路径
_MODEL_PATHS = {
    "teacher": conf.teacher_best_model_path,
    "student": conf.student_best_model_path,
    "quantized": conf.teacher_quantized_model_path,
}

# 各模型运行的设备：int8 动态量化只能跑 CPU，其余跑 conf.device
_MODEL_DEVICES = {
    "teacher": conf.device,
    "student": conf.device,
    "quantized": torch.device("cpu"),
}

_models = {}          # model_type -> 已加载的情绪模型


def _load_state_dict(path, cls):
    m = cls().to(conf.device)
    m.load_state_dict(torch.load(path, map_location=conf.device))
    m.eval()
    return m


def _load_quantized(path):
    # 量化模型是 torch.save(whole_model) 存的，整体加载；int8 量化仅 CPU 推理
    m = torch.load(path, map_location="cpu", weights_only=False)
    m.eval()
    return m


def load_model(model_type="teacher"):
    """情绪三分类模型（懒加载，进程内缓存）。

    model_type ∈ {"teacher", "student", "quantized"}，默认老师 BERT。
    """
    if model_type not in _MODEL_PATHS:
        raise ValueError(f"未知模型类型 {model_type!r}，可选：{sorted(_MODEL_PATHS)}")
    if model_type not in _models:
        path = _MODEL_PATHS[model_type]
        if not Path(path).exists():
            raise FileNotFoundError(f"{model_type} 模型尚未训练（{path} 不存在）。")
        if model_type == "quantized":
            _models[model_type] = _load_quantized(path)
        else:
            cls = TMFBertClassifier if model_type == "teacher" else BiLSTM
            _models[model_type] = _load_state_dict(path, cls)
    return _models[model_type]


@torch.no_grad()
def _predict_batch(texts, model, class_list, batch_size=64, device=None):
    device = device or conf.device
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = conf.tokenizer(batch, return_tensors='pt', max_length=conf.max_len,
                             padding='max_length', truncation=True)
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1).tolist()
        results.extend([class_list[p] for p in preds])
    return results


def classify_batch(texts, batch_size=64, model_type="teacher"):
    """情绪三分类批量预测，返回 ['正面','中性','负面', ...]。"""
    model = load_model(model_type)
    return _predict_batch(texts, model, conf.class_list, batch_size,
                          device=_MODEL_DEVICES[model_type])


def classify(text):
    """单条情绪预测（兼容旧接口）。"""
    return classify_batch([text])[0]


if __name__ == '__main__':
    samples = ["刀死我了这也太虐了吧", "哈哈哈哈笑死我了", "前排打卡"]
    for mt in ("teacher", "student", "quantized"):
        print(f"[{mt}]")
        for s in samples:
            print(f"  {s!r:20} -> {classify_batch([s], model_type=mt)[0]}")
