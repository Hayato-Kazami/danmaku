import random
from collections import Counter

import numpy as np

from config import Config
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch

conf = Config()


def set_seed(seed=42):
    """固定所有随机源，保证训练可复现。

    - WeightedRandomSampler 内部走 torch.multinomial，靠 torch.manual_seed 控制；
    - 模型权重初始化、dropout、BiLSTM 的 cuDNN RNN 都吃这个种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_raw_data(file_path):
    """
    从指定文本文件加载数据，处理成「文本 - 标签索引」元组列表。
    文件每行格式：文本\t标签
    """
    raw_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc='Loading raw data...'):
            line = line.strip()
            if not line:
                continue
            text, label = line.split('\t', 1)
            raw_data.append((text, int(label)))
    return raw_data


class TMFDataset(Dataset):
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, item):
        return self.raw_data[item]


def collate_fn(batch):
    texts, labels = zip(*batch)
    input = conf.tokenizer(texts,
                           padding="max_length",
                           max_length=conf.max_len,
                           truncation=True,
                           return_tensors="pt")
    input_ids = input['input_ids']
    attention_mask = input['attention_mask']
    labels = torch.tensor(labels)
    return input_ids, attention_mask, labels


def build_dataloader():
    train_data = load_raw_data(conf.train_data_path)
    dev_data = load_raw_data(conf.dev_data_path)
    test_data = load_raw_data(conf.test_data_path)

    train_dataset = TMFDataset(train_data)
    dev_dataset = TMFDataset(dev_data)
    test_dataset = TMFDataset(test_data)

    # 类别平衡采样：负面（少数类）被抽中的概率更高，避免被多数类「中性」淹没
    labels = [label for _, label in train_data]
    class_counts = Counter(labels)
    sample_weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=conf.batch_size,
                                  sampler=sampler,
                                  collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=conf.batch_size,
                                shuffle=False,
                                collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=conf.batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn)
    return train_dataloader, dev_dataloader, test_dataloader
