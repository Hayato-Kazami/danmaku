# 弹幕情绪分析（danmaku_bert）

基于 BERT 的 B 站弹幕情绪分类项目：输入一个视频，逐秒可视化观众情绪（正面 / 中性 / 负面），标出情绪爆点与冷场拐点。

**核心卖点**：不只是「训练一个 BERT」，而是完整的 **数据 → 训练 → 压缩 → 部署** 闭环：

- 老师 BERT（0.7986，390MB）→ 知识蒸馏到 BiLSTM（0.6991，19.4MB，约 20× 缩小）→ int8 量化（0.7659，145.6MB）
- 三种模型已接入后端，前端侧边栏一键切换，实时对比「精度 vs 体积 vs 速度」
- 传统 ML 基线（TF-IDF + LR/RF）做下界对照——有对照，才有说服力

---

## 技术栈

| 层 | 选型 |
|---|---|
| 模型 | `transformers` + BertModel（bert-base-chinese） |
| 压缩 | 知识蒸馏（KL 软标签）+ int8 动态量化（只量化 Linear） |
| 后端 | FastAPI（:8000） |
| 前端 | Streamlit + Plotly（:8502） |
| 爬虫 | bilibili-api-python（当前弹幕不限流；历史弹幕需 SESSDATA） |
| 部署 | VPS + frp 内网穿透 |

## 项目结构

```
danmaku_bert/
├── api.py                 # FastAPI 后端（3 个接口）
├── app.py                 # Streamlit 前端（情绪时间曲线产品）
├── api_test.py            # 后端冒烟测试
├── src/
│   ├── config.py          # 路径 + 超参集中配置
│   ├── bert_classifier.py # BERT 情绪三分类模型
│   ├── student_model.py   # 蒸馏学生 BiLSTM
│   ├── bert_predict.py    # 预测入口（teacher/student/quantized 注册表）
│   ├── train.py           # BERT 老师训练（FocalLoss）
│   ├── model_distill.py   # 知识蒸馏
│   ├── model_direct.py    # 消融对照：不蒸馏直接训 BiLSTM
│   ├── model_quantize.py  # int8 动态量化
│   ├── baseline.py        # 传统 ML 基线（jieba+TF-IDF+LR/RF）
│   ├── utils.py           # 数据加载 + WeightedRandomSampler
│   ├── crawler.py         # bvid → 弹幕爬取
│   └── error_analysis.py  # 错例分析
├── model/                 # 训练好的权重（gitignore）
├── logs/                  # 训练日志
└── frp/                   # frp 客户端配置（内网穿透）
```

依赖的兄弟目录（在 `d:/Code/` 下）：

- `danmaku_data/`：训练集 `train.txt / dev.txt / test.txt / class.txt`
- `Scraping/`：爬取 + LLM 标注 + 采样脚本，产出 `labeled_*.csv`
- `PythonProject4/TMF/bert/bert-base-chinese`：复用的 BERT 预训练权重（避免重复下载 400MB）

## 快速开始

### 1. 环境

conda 环境 `mldl`（Python 3.10，torch 2.13 + CUDA，RTX 4060）。

```bash
pip install -r requirements.txt
# torch 建议按你的 CUDA 版本单独安装，见 requirements.txt 顶部注释
```

### 2. 数据 & 模型路径

`config.py` 集中配置了路径，默认约定：

- 数据：`d:/Code/danmaku_data/`
- BERT 权重：复用 `d:/Code/PythonProject4/TMF/bert/bert-base-chinese`
- 训练好的模型：`danmaku_bert/model/`（`bert_classification_best.pth` 等）

### 3. 启动

先起后端，再起前端：

```bash
# 后端（若要用「输入 bvid 爬历史弹幕」，先 export 登录凭证）
export BILI_SESSDATA=你的SESSDATA   # 可选，仅历史弹幕需要
python api.py                       # http://127.0.0.1:8000（Swagger: /docs）

# 前端（另开一个终端）
streamlit run app.py                # http://127.0.0.1:8502
```

> 前后端分离，两个都要起。前端按需懒加载所选模型（首次切换有加载延迟）。

### 4. 用起来

侧边栏先选「模型」，再选「数据来源」：

- **从已标注数据选视频**：读 `Scraping/labeled_*.csv`
- **上传 CSV**：至少含「内容」「出现时间秒」两列
- **输入 bvid 爬取**：爬陌生视频弹幕（历史弹幕可选、慢、需 SESSDATA）
- **使用演示数据**：无真实数据时兜底

## 模型对比（真实 test 指标）

| 模型 | 体积 | weighted F1 | 说明 |
|---|---|---|---|
| 老师 BERT | 390 MB | 0.7986 | 精度最高，GPU |
| int8 量化 BERT | 145.6 MB | 0.7659 | 几乎无损（-3.3 点），仅 CPU |
| 蒸馏 BiLSTM | 19.4 MB | 0.6991 | 体积 20× 缩小 |
| BiLSTM 直接训练（消融） | 19.4 MB | 0.6712 | 蒸馏增益 +2.8 点 |
| TF-IDF + 逻辑回归（基线） | — | 0.6370 | 传统方法下界 |
| TF-IDF + 随机森林（基线） | — | 0.6109 | |

> 注意：加权 F1 会被「中性」多数类主导。老师模型负面类 recall 仅 0.55（45% 负面漏判），详见开发文档 7.3。

> 量化模型跑得比老师/学生慢，是**因为它只能上 CPU**（`quantize_dynamic` 的 qint8 没有 CUDA 内核），不是量化拖慢推理——它在 CPU 上反而比原始 BERT 快约 1.5–2×。它的卖点是「体积 145.6 MB + 不占 GPU + 比 CPU 版 BERT 快」。

## 训练复现

```bash
python src/train.py            # 1. 老师 BERT
python src/model_distill.py    # 2. 知识蒸馏
python src/model_quantize.py   # 3. int8 量化
python src/baseline.py         # 4. 传统基线（对照）
```

数据来自 `Scraping/` 里的爬取 + LLM 标注 + 采样脚本（按 bvid 分层 8:1:1 切分）。

## API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/predict` | 单条情绪预测 |
| POST | `/predict_batch` | 批量预测（`model_type`: teacher / student / quantized） |
| POST | `/crawl_danmaku` | 爬 bvid 弹幕 |

## 部署（内网穿透）

本地 GPU 跑推理，公网只暴露前端：本机 frpc 主动连 VPS frps，`frp/frpc.toml` 里的 `danmaku` 代理把 Streamlit :8502 映射到公网。

```bash
frp/frpc.exe -c frp/frpc.toml
```

## 已知限制 / 后续

- **负面召回低**（recall 0.55）：类别不平衡的硬骨头，可调 FocalLoss gamma 或对负面额外上采样。

详细的设计理由、踩坑记录见《弹幕情绪分析项目开发文档 V2.0》。
