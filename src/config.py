import time
from pathlib import Path

import torch
from transformers import BertTokenizer


class Config():
    def __init__(self):
        # 项目根目录：danmaku_bert（本文件位于 danmaku_bert/src 下，向上 1 级）
        project_dir = Path(__file__).resolve().parents[1]
        # 弹幕数据目录：d:\Code\danmaku_data（采样脚本 Scraping/sample.py 的输出）
        data_dir = project_dir.parent / 'danmaku_data'

        # 数据集路径
        self.train_data_path = str(data_dir / 'train.txt')
        self.test_data_path = str(data_dir / 'test.txt')
        self.dev_data_path = str(data_dir / 'dev.txt')
        self.class_path = str(data_dir / 'class.txt')
        with open(self.class_path, 'r', encoding='utf-8') as f:
            self.class_list = f.read().strip().split('\n')

        # 日志保存路径
        self.log_path = str(project_dir / 'logs' / f'{time.strftime("%Y-%m-%d")}_log.txt')

        # 模型保存路径
        self.teacher_best_model_path = str(project_dir / 'model' / 'bert_classification_best.pth')
        self.teacher_last_model_path = str(project_dir / 'model' / 'bert_classification_last.pth')
        self.teacher_quantized_model_path = str(project_dir / 'model' / 'bert_classification_quantized.pt')
        self.student_best_model_path = str(project_dir / 'model' / 'student_bilstm_best_model.pth')
        self.student_last_model_path = str(project_dir / 'model' / 'student_bilstm_last.pth')

        # 确保日志和模型目录存在
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.teacher_best_model_path).parent.mkdir(parents=True, exist_ok=True)

        # BERT 模型路径（复用 TMF 项目已下载的 bert-base-chinese，避免重复下载约 400MB）
        self.bert_model_path = str(Path(__file__).resolve().parents[2]
                                   / 'PythonProject4' / 'TMF' / 'bert' / 'bert-base-chinese')
        self.hidden_dim = 768
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)
        self.vocab_size = self.tokenizer.vocab_size

        # 训练参数
        self.class_num = len(self.class_list)
        self.batch_size = 16
        self.epochs = 15
        self.lr = 3e-5
        self.max_len = 32  # 覆盖 97% 弹幕（p95≈28）；实测 64 覆盖 99.8% 但 F1 -0.7（padding 噪声），故守 32
        self.warmup_ratio = 0.1  # warmup 步数占总步数比例

        # 类别不平衡（负面 recall 优化）
        self.focal_gamma = 2.0            # FocalLoss gamma，调大聚焦难分样本（负面多为难分样本）
        self.selection_metric = "weighted_f1"  # 选最优 checkpoint 的指标：weighted_f1 / macro_f1 / negative_recall

        # 随机种子（保证训练可复现，配合 utils.set_seed 使用）
        self.seed = 42

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 学生模型（BiLSTM）配置，用于知识蒸馏
        self.embedding_dim = 128
        self.lstm_hidden_dim = 256
        self.lstm_num_layers = 2
        self.dropout = 0.2

        # 蒸馏参数
        self.temperature = 2.0
        self.alpha = 0.7
        self.student_lr = 1e-3  # 学生模型从零训练，需要比 BERT 微调更大的学习率


if __name__ == '__main__':
    config = Config()
    print('device:', config.device)
    print('class_list:', config.class_list)
