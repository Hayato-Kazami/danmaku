"""BiLSTM 直接训练基线（不蒸馏，弹幕版）。

用途：和蒸馏结果做对照，量化「知识蒸馏」到底带来多少增益。

对照组设计（只差「有没有老师软标签」这一个变量，其余完全一致）：
    - 模型：BiLSTM（同一个 student_model.py）
    - 学习率：conf.student_lr（1e-3，与蒸馏的学生一致）
    - 轮数 / 批大小 / 评估方式：与 model_distill.py 一致
    - 损失：仅 CrossEntropyLoss 硬标签（蒸馏里是 alpha*软损失 + (1-alpha)*硬损失）

预期结果（弹幕 3 分类，类别不平衡、语义主观，蒸馏增益可能比 THUCNews 更明显）：
    teacher BERT      ≈ ?
    student 蒸馏       ≈ ?
    student 直接训练    ≈ ?        <- 本脚本产出
    直接训练 < 蒸馏 < 老师，这个差值就是蒸馏增益。

运行：python model_direct.py
"""
from pathlib import Path

import torch
from tqdm import tqdm

from config import Config
from utils import build_dataloader, set_seed
from student_model import BiLSTM
from torch.nn import CrossEntropyLoss
from train import model2eval

conf = Config()

# 独立保存路径，避免覆盖蒸馏出的 student 模型
_model_dir = Path(conf.teacher_last_model_path).parent
DIRECT_BEST = str(_model_dir / "direct_bilstm_best.pth")
DIRECT_LAST = str(_model_dir / "direct_bilstm_last.pth")


def model2direct(train_loader, dev_loader, model, device):
    loss_fn = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.student_lr)

    best_f1 = 0.
    total_iters = 0

    for epoch in range(conf.epochs):
        model.train()
        total_loss_epoch = 0.

        for i, (input_ids, attention_mask, labels) in enumerate(
                tqdm(train_loader, total=len(train_loader),
                     desc=f"直接训练 Epoch {epoch + 1}")):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            total_iters += 1

            # 每 150 步评估一次（与蒸馏一致），保存最优模型
            if total_iters % 150 == 0:
                f1, report, cm = model2eval(dev_loader, model, device)
                print(f"\nEpoch {epoch + 1}, Iter {i + 1}/{len(train_loader)}, F1: {f1:.4f}")
                model.train()
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), DIRECT_BEST)
                    print(f"保存最佳模型，F1: {best_f1:.4f}")

    torch.save(model.state_dict(), DIRECT_LAST)
    print(f"直接训练完成，最佳 F1: {best_f1:.4f}")
    return best_f1


def main():
    set_seed(conf.seed)
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    model = BiLSTM().to(conf.device)
    model2direct(train_dataloader, dev_dataloader, model, conf.device)

    # 测试集最终评估
    best_model = BiLSTM().to(conf.device)
    best_model.load_state_dict(torch.load(DIRECT_BEST))
    f1, report, cm = model2eval(test_dataloader, best_model, conf.device)
    print(f"直接训练 test F1: {f1:.4f}")


if __name__ == "__main__":
    main()
