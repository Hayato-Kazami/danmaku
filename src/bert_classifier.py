from config import Config
import torch.nn as nn
from transformers import BertModel

conf = Config()


class TMFBertClassifier(nn.Module):
    def __init__(self):
        super(TMFBertClassifier, self).__init__()
        # 加载预训练的 BERT 模型
        self.bert = BertModel.from_pretrained(conf.bert_model_path)
        # 分类层（弹幕情绪：3 类）
        self.fc = nn.Linear(conf.hidden_dim, conf.class_num)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(output.pooler_output)
        return logits
