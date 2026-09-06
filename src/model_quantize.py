"""BERT 模型 int8 动态量化（弹幕版）。

对训练好的 BERT 老师模型做后训练动态量化：只量化 nn.Linear（embedding / LayerNorm 保持 fp32），
评估量化前后的 F1 与体积变化。动态量化是 CPU 推理，全程在 CPU 上评估。

运行：python model_quantize.py
"""
import os

import torch
import torch.quantization as quant

from config import Config
from utils import build_dataloader
from bert_classifier import TMFBertClassifier
from train import model2eval

conf = Config()

QUANTIZED_PATH = conf.teacher_best_model_path.replace("best.pth", "quantized.pt")


def main():
    _, _, test_dataloader = build_dataloader()
    cpu = torch.device("cpu")

    # 加载 fp32 老师模型到 CPU
    model = TMFBertClassifier()
    model.load_state_dict(torch.load(conf.teacher_best_model_path, map_location="cpu"))
    model.eval()

    # 量化前评估
    f1_before, _, _ = model2eval(test_dataloader, model, cpu)
    print(f"量化前 test F1: {f1_before:.4f}")

    # 动态量化（只量化 Linear）
    quantized = quant.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # 量化后评估
    f1_after, _, _ = model2eval(test_dataloader, quantized, cpu)
    print(f"量化后 test F1: {f1_after:.4f}")
    print(f"F1 损失: {f1_before - f1_after:.4f}")

    # 保存量化模型
    torch.save(quantized, QUANTIZED_PATH)

    # 体积对比
    size_before = os.path.getsize(conf.teacher_best_model_path)
    size_after = os.path.getsize(QUANTIZED_PATH)
    print(f"体积: {size_before / 1e6:.1f}MB -> {size_after / 1e6:.1f}MB "
          f"(约 {size_before / size_after:.2f}x 压缩)")


if __name__ == "__main__":
    main()
