"""争议样本导出：找出「模型预测」和「LLM 标签」不一致的样本，人工复核定位标签噪声。

思路（active learning）：不随机抽检，只抽「模型和 LLM 有分歧」的样本——
这些样本里要么是模型错、要么是 LLM 标错，是标签噪声最集中的地方。

导出 CSV（Excel 可开），列：text / true_label / pred_label。
人工打开后重点看「真负面 → 判中性」这类，判断谁对，修正错误标签后可重新训练。

运行：python error_analysis.py
"""
import csv
from collections import Counter
from pathlib import Path

import torch

from config import Config
from utils import build_dataloader
from bert_classifier import TMFBertClassifier

conf = Config()

# 导出到 danmaku_bert 项目根目录
OUT_PATH = Path(conf.teacher_best_model_path).parent.parent / 'error_analysis.csv'


def main():
    _, _, test_dataloader = build_dataloader()

    # 重新读 test.txt 拿原始文本（dataloader 只有 id/mask/label，需按顺序对齐）
    texts = []
    with open(conf.test_data_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.strip():
                texts.append(line.split('\t', 1)[0])

    model = TMFBertClassifier().to(conf.device)
    model.load_state_dict(torch.load(conf.teacher_best_model_path))
    model.eval()

    rows = []
    idx = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_dataloader:
            input_ids = input_ids.to(conf.device)
            attention_mask = attention_mask.to(conf.device)
            labels = labels.to(conf.device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            for t, p in zip(labels.tolist(), preds.tolist()):
                if t != p:
                    rows.append((texts[idx], conf.class_list[t], conf.class_list[p]))
                idx += 1

    # 导出 CSV（utf-8-sig 让 Excel 正确显示中文）
    with open(OUT_PATH, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['text', 'true_label', 'pred_label'])
        w.writerows(rows)
    print(f'共 {len(rows)} 条争议样本，已导出：{OUT_PATH}')

    # 按错误类型统计，让人工复核时知道重点看哪类
    err = Counter((t, p) for _, t, p in rows)
    print('\n错误类型分布（真X → 判Y）：')
    for (t, p), c in err.most_common():
        print(f'  真[{t}] → 判[{p}]: {c} 条')


if __name__ == '__main__':
    main()
