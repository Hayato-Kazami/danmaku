"""传统机器学习基线（TF-IDF + 逻辑回归 / 随机森林），用于和 BERT 对比。

目的：证明「为什么需要 BERT」。用最经典的传统文本分类方法跑一遍弹幕数据，
得到 weighted F1 作为深度学习模型的下界对照——简历上「传统 vs 深度学习」的对比。

方法：jieba 分词 → TfidfVectorizer（词级 1~2 gram）→ 逻辑回归 / 随机森林。
评估指标与 train.py 一致（weighted F1），可直接和 BERT 老师 0.79 对比。

运行：python baseline.py
"""
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from config import Config
from utils import load_raw_data

conf = Config()


def load(path):
    """复用 utils 的 load_raw_data，拆成文本列表和标签列表。"""
    data = load_raw_data(path)  # [(text, int_label), ...]
    texts = [t for t, _ in data]
    labels = [l for _, l in data]
    return texts, labels


def main():
    train_texts, y_train = load(conf.train_data_path)
    dev_texts, y_dev = load(conf.dev_data_path)
    test_texts, y_test = load(conf.test_data_path)

    # jieba 分词（弹幕超短，分词后词频稀疏，用 1~2 gram 捕捉二元搭配）
    print('jieba 分词中...')
    train_words = [' '.join(jieba.lcut(t)) for t in train_texts]
    dev_words = [' '.join(jieba.lcut(t)) for t in dev_texts]
    test_words = [' '.join(jieba.lcut(t)) for t in test_texts]

    # TF-IDF 特征：min_df=2 去掉只出现一次的罕见词，减少噪声
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train_words)
    X_dev = vectorizer.transform(dev_words)
    X_test = vectorizer.transform(test_words)
    print(f'TF-IDF 特征维度: {X_train.shape[1]}')

    # class_weight='balanced'：与 BERT 侧一致，公平处理类别不平衡（负面是少数类）
    models = {
        '逻辑回归(LR)': LogisticRegression(max_iter=1000, n_jobs=-1, class_weight='balanced'),
        # 随机森林在高维稀疏特征上偏慢，参数参考 TMF 的 GridSearch 范围
        '随机森林(RF)': RandomForestClassifier(n_estimators=100, max_depth=15,
                                            n_jobs=-1, random_state=1,
                                            class_weight='balanced'),
    }

    for name, model in models.items():
        print(f'\n===== {name} =====')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        print(f'weighted F1: {f1:.4f}')
        print(classification_report(y_test, y_pred, zero_division=0,
                                     target_names=conf.class_list))


if __name__ == '__main__':
    main()
