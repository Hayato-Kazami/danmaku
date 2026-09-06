import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from utils import build_dataloader, set_seed
from bert_classifier import TMFBertClassifier
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, recall_score, classification_report, confusion_matrix

conf = Config()


class FocalLoss(nn.Module):
    """Focal Loss：聚焦「难分样本」。

    弹幕 3 分类里负面（少数类）往往就是难分样本，容易被中性吞掉。
    gamma 越大，对「预测置信度低」的样本惩罚越重；gamma=0 退化为普通交叉熵。
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)          # 预测正确的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def _evaluate(dataloader, model, device):
    """跑一遍前向，返回 (y_true, y_pred)。"""
    model.eval()
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader,
                                                      total=len(dataloader),
                                                      desc="模型验证中"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask)
            y_pred = torch.argmax(logits, dim=-1)
            y_pred_list.extend(y_pred.tolist())
            y_true_list.extend(labels.tolist())
    return y_true_list, y_pred_list


def compute_metrics(y_true, y_pred):
    """算 weighted F1 / macro F1 / 负面 recall，外加报告和混淆矩阵。

    - weighted_f1：被中性多数类主导，对外报的就是它
    - macro_f1：三类等权，能反映负面这类少数类有没有被拉起来
    - neg_recall：负面召回（漏判率 = 1 - recall）
    """
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    neg_recall = None
    if '负面' in conf.class_list:
        neg_idx = conf.class_list.index('负面')
        neg_recall = recall_score(y_true, y_pred, labels=[neg_idx], average=None, zero_division=0)[0]
    report = classification_report(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return weighted_f1, macro_f1, neg_recall, report, cm


def model2eval(dev_dataloader, model, device):
    """向后兼容：返回 (weighted_f1, report, cm)。

    model_distill.py / model_direct.py / model_quantize.py 仍调用这个 3 元组版本。
    """
    y_true, y_pred = _evaluate(dev_dataloader, model, device)
    weighted_f1, _, _, report, cm = compute_metrics(y_true, y_pred)
    return weighted_f1, report, cm


def _select_metric(weighted_f1, macro_f1, neg_recall):
    """按 conf.selection_metric 返回「选最优 checkpoint」用的分数。"""
    if conf.selection_metric == "macro_f1":
        return macro_f1
    if conf.selection_metric == "negative_recall":
        return neg_recall if neg_recall is not None else -1.0
    return weighted_f1


def model2train(train_dataloader, dev_dataloader, model, device):
    # Focal Loss：配合类别平衡采样，聚焦负面这类「少数 + 难分」样本
    loss_fn = FocalLoss(gamma=conf.focal_gamma)
    optimizer = AdamW(model.parameters(), lr=conf.lr)

    # 学习率调度：warmup + 线性衰减（BERT 微调标准做法，小数据集尤其需要）
    total_steps = len(train_dataloader) * conf.epochs
    warmup_steps = int(total_steps * conf.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_score = 0.0
    total_iters = 0
    for epoch in range(conf.epochs):
        total_loss = 0.
        total_iter_epoch = 0.
        model.train()

        for i, (input_ids, attention_mask, labels) in tqdm(enumerate(train_dataloader),
                                                           total=len(train_dataloader),
                                                           desc=f"模型训练中，当前轮次为：{epoch+1}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，稳定训练
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_iter_epoch += 1
            total_iters += 1

            # 每 20 个迭代打印一次当前轮次的平均损失
            if (i+1) % 20 == 0:
                avg_loss = total_loss / total_iter_epoch
                print(f"当前轮次：{epoch+1}，当前迭代：{i+1}，平均损失：{avg_loss:.4f}")
                with open(conf.log_path, 'a', encoding='utf-8') as f:
                    f.write(f"当前轮次：{epoch+1}，当前迭代：{i+1}，平均损失：{avg_loss:.4f}\n")

            # 每 200 步评估一次，保存最优模型（batch_size=16 时约 731 batch/轮）
            if (i+1) % 200 == 0 or i+1 == len(train_dataloader):
                y_true, y_pred = _evaluate(dev_dataloader, model, device)
                weighted_f1, macro_f1, neg_recall, report, cm = compute_metrics(y_true, y_pred)
                model.train()
                score = _select_metric(weighted_f1, macro_f1, neg_recall)
                if score > best_score:
                    best_score = score
                    torch.save(model.state_dict(), conf.teacher_best_model_path)
                neg_s = f"{neg_recall:.4f}" if neg_recall is not None else "N/A"
                print(f"当前轮次：{epoch+1}，当前迭代：{i+1}，"
                      f"weighted={weighted_f1:.4f}，macro={macro_f1:.4f}，负面recall={neg_s}，"
                      f"最优({conf.selection_metric})={best_score:.4f}")
                with open(conf.log_path, 'a', encoding='utf-8') as f:
                    f.write(f"当前轮次：{epoch+1}，当前迭代：{i+1}，"
                            f"weighted={weighted_f1:.4f}，macro={macro_f1:.4f}，负面recall={neg_s}，"
                            f"最优({conf.selection_metric})={best_score:.4f}\n")
                    f.write(f"验证报告：{report}\n")
                    f.write(f"混淆矩阵：{cm}\n")

    torch.save(model.state_dict(), conf.teacher_last_model_path)


def main():
    set_seed(conf.seed)
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    model = TMFBertClassifier().to(conf.device)
    model2train(train_dataloader, dev_dataloader, model, conf.device)

    # 测试集最终评估
    model_best = TMFBertClassifier().to(conf.device)
    model_best.load_state_dict(torch.load(conf.teacher_best_model_path))
    y_true, y_pred = _evaluate(test_dataloader, model_best, conf.device)
    weighted_f1, macro_f1, neg_recall, report_best, cm_best = compute_metrics(y_true, y_pred)
    neg_s = f"{neg_recall:.4f}" if neg_recall is not None else "N/A"
    print(f"测试集 weighted F1：{weighted_f1:.4f}，macro F1：{macro_f1:.4f}，负面 recall：{neg_s}")
    print("最优报告：")
    print(report_best)
    print("最优混淆矩阵：")
    print(cm_best)
    with open(conf.log_path, 'a', encoding='utf-8') as f:
        f.write(f"测试集 weighted F1：{weighted_f1:.4f}，macro F1：{macro_f1:.4f}，负面 recall：{neg_s}\n")
        f.write("最优报告：")
        f.write(report_best)
        f.write("最优混淆矩阵：")
        f.write(str(cm_best))


if __name__ == '__main__':
    main()
