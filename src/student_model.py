import torch
import torch.nn as nn
from config import Config

conf = Config()


class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # 嵌入层
        self.embedding = nn.Embedding(conf.vocab_size, conf.embedding_dim)
        # LSTM 层（双向）
        self.lstm = nn.LSTM(conf.embedding_dim,
                            conf.lstm_hidden_dim,
                            conf.lstm_num_layers,
                            bidirectional=True,
                            dropout=conf.dropout)
        # 全连接层
        self.fc = nn.Linear(conf.lstm_hidden_dim * 2, conf.class_num)

    def forward(self, input_ids, attention_mask):
        # 嵌入
        embedded = self.embedding(input_ids)
        attention_mask = attention_mask.unsqueeze(-1)
        # 清零 padding
        input = embedded * attention_mask
        # 交换维度：LSTM 需要 seq_len 在第一维
        input = input.transpose(0, 1)
        # LSTM 输出
        lstm_out, _ = self.lstm(input)
        # 实验：改回取「最后一个时间步输出」对比（原版 bug，backward 半边在 padding 位≈0）
        out = lstm_out[-1]  # (batch, hidden_dim*2)
        return self.fc(out)
