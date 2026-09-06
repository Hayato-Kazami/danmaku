"""弹幕情绪时间曲线 —— Streamlit 产品（BERT 预测版）。

运行：streamlit run app.py
依赖：pip install streamlit plotly pandas

功能：
    1. 从已标注数据（Scraping/labeled_*.csv）按 bvid 选一个视频
    2. 用训练好的模型对每条弹幕实时预测情绪（老师 BERT / 蒸馏 BiLSTM / int8 量化，可切换）
    3. 按弹幕「出现时间秒」分箱，画情绪时间曲线（发散堆叠：正面上、负面下、中性居中）
    4. 标出「情绪爆点」：情绪净得分极值 × 弹幕密度峰值
    5. 附：情绪构成占比、高光片段、爆点弹幕抽样

说明：
    情绪标签由可选模型预测（老师 BERT / 蒸馏 BiLSTM / int8 量化），默认老师 BERT。
    模型在后端首次调用时加载，之后进程内缓存。
"""

import html
import math
import random
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:
    st.error("缺少 plotly，请先运行：pip install plotly")
    st.stop()

# 前后端分离：情绪预测通过 requests 调 FastAPI 后端（api.py，8000 端口）
API_URL = "http://127.0.0.1:8000"

# ---------- 配色（dataviz 校验过的发散配色：蓝=正面 / 红=负面 / 灰=中性）----------
COLOR_POS = "#2a78d6"   # 正面 blue
COLOR_NEU = "#c3c2b7"   # 中性 gray
COLOR_NEG = "#e34948"   # 负面 red

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

COLORS = {"正面": COLOR_POS, "中性": COLOR_NEU, "负面": COLOR_NEG}

# 情绪模型可选类型：侧边栏下拉框的显示标签（F1 / 体积供对比）
MODEL_OPTIONS = {
    "teacher": "老师 BERT（F1 0.7986 · 390MB）",
    "student": "蒸馏 BiLSTM（F1 0.6991 · 19.4MB）",
    "quantized": "int8 量化 BERT（F1 0.7659 · 145.6MB）",
}


# ================= 数据读取 =================

def _find_labeled_csv():
    """找 Scraping 目录下最新的 labeled_*.csv。"""
    scraping = Path(__file__).resolve().parent.parent / "Scraping"
    files = [p for p in scraping.glob("labeled_*.csv")
             if "partial" not in p.name]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


# 全量弹幕文件（未抽样、无标签），用于「从爬取全量选视频」数据源
FULL_CSV = Path(__file__).resolve().parent.parent / "danmaku_all_merged.csv"


@st.cache_data(show_spinner=False)
def _load_video_list(path=None):
    """读弹幕 CSV，返回所有 (bvid, 视频标题)。path=None 时读最新 labeled CSV。"""
    if path is None:
        path = _find_labeled_csv()
        if path is None:
            return None, None
    df = pd.read_csv(path, encoding="utf-8-sig")
    videos = df[["bvid", "视频标题"]].drop_duplicates()
    return str(path), videos


def _read_video(csv_path, bvid):
    """读指定视频的弹幕，只保留文本 + 出现时间秒。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[df["bvid"] == bvid].copy()
    df["出现时间秒"] = pd.to_numeric(df["出现时间秒"], errors="coerce")
    df = df.dropna(subset=["出现时间秒"])
    df = df[df["出现时间秒"] >= 0]
    df = df[["内容", "出现时间秒"]]
    return df


# ================= BERT 预测（调后端 API，缓存） =================

@st.cache_data(show_spinner="调用模型预测...")
def _predict_via_api(texts_tuple, model_type):
    """批量调后端 /predict_batch，返回 (pred_classes, elapsed_ms, avg_ms)。"""
    res = requests.post(f"{API_URL}/predict_batch",
                        json={"texts": list(texts_tuple), "model_type": model_type},
                        timeout=300)
    if res.status_code != 200:
        raise RuntimeError(res.json().get("error", "后端 API 返回错误"))
    data = res.json()
    return data["pred_classes"], data["elapsed_ms"], data.get("avg_ms")


@st.cache_data(show_spinner="正在爬取弹幕（拉历史弹幕会更慢）...")
def _crawl_via_api(bvid, fetch_history):
    """调后端 /crawl_danmaku，返回 {title, count, note, danmakus}。"""
    res = requests.post(f"{API_URL}/crawl_danmaku",
                        json={"bvid": bvid, "fetch_history": fetch_history}, timeout=300)
    if res.status_code != 200:
        raise RuntimeError(res.json().get("error", "爬取接口错误"))
    data = res.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


# ================= 演示数据（无真实 CSV 时兜底） =================

_POS_TEXTS = ["哈哈哈哈笑死", "卧槽牛逼", "泪目了家人们", "名场面预定", "这也太燃了吧",
              "yyds", "awsl", "双厨狂喜", "绝了绝了", "太可爱了", "哈哈哈哈哈哈",
              "666666", "神作！", "吹爆这个镜头", "燃起来了", "不错不错", "爱了爱了"]
_NEU_TEXTS = ["前排", "来了来了", "打卡", "有人吗", "这个bgm是啥", "第几集来着",
              "蹲一个", "路过", "这画质", "字幕组辛苦了", "进度条撑住", "到此一游",
              "打卡打卡", "考古", "先赞后看", "已三连"]
_NEG_TEXTS = ["烂尾了", "好无聊", "这就没了？", "刀死我了", "太虐了", "哭死",
              "翻车", "退钱", "无语", "下头", "这结局太拉了", "蚌埠住了",
              "太难了", "差点意思", "失望", "恶心"]


def _demo_probs(t):
    """给定视频进度 t∈[0,1]，返回 (正面概率, 中性概率, 负面概率)。"""
    p_pos = 0.25 + 0.45 * math.exp(-((t - 0.70) ** 2) / 0.015)
    p_neg = 0.12 + 0.40 * math.exp(-((t - 0.92) ** 2) / 0.010)
    total = p_pos + p_neg
    if total > 0.9:
        p_pos = p_pos / total * 0.9
        p_neg = p_neg / total * 0.9
    p_neu = max(0.0, 1.0 - p_pos - p_neg)
    return p_pos, p_neu, p_neg


def make_demo_df(n=800, duration=600.0, seed=42):
    """生成演示数据：弹幕在「名场面」和「结尾刀」处密度更高，情绪也随之转向。"""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        r = rng.random()
        if r < 0.30:
            t = duration * min(0.99, max(0.0, rng.gauss(0.70, 0.06)))
        elif r < 0.45:
            t = duration * min(0.99, max(0.0, rng.gauss(0.92, 0.05)))
        else:
            t = rng.uniform(0, duration)
        p_pos, p_neu, p_neg = _demo_probs(t / duration)
        rr = rng.random()
        if rr < p_pos:
            text = rng.choice(_POS_TEXTS)
        elif rr < p_pos + p_neg:
            text = rng.choice(_NEG_TEXTS)
        else:
            text = rng.choice(_NEU_TEXTS)
        rows.append({"内容": text, "出现时间秒": round(t, 1)})
    return pd.DataFrame(rows)


# ================= 分箱 =================

def _mmss(s):
    s = int(s)
    return f"{s // 60:02d}:{s % 60:02d}"


def bin_danmaku(df, n_bins):
    duration = df["出现时间秒"].max()
    step = duration / n_bins
    if step <= 0:
        step = 1.0

    df = df.copy()
    df["时间片"] = (df["出现时间秒"] / step).astype(int).clip(0, n_bins - 1)

    counts = df.groupby(["时间片", "情绪"]).size().unstack(fill_value=0)
    for c in ("正面", "中性", "负面"):
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[["正面", "中性", "负面"]].reindex(range(n_bins), fill_value=0)
    counts["时间片标签"] = [_mmss(i * step) for i in range(n_bins)]
    counts["总计"] = counts[["正面", "中性", "负面"]].sum(axis=1)
    counts["情绪得分"] = (counts["正面"] - counts["负面"]) / counts["总计"].replace(0, 1)
    return df, counts


# ================= 情绪爆点 =================

def emotion_spikes(counts, threshold=0.25):
    """找情绪爆点：|情绪得分| > threshold 且 弹幕密度不低于平均。"""
    min_count = counts["总计"].mean()
    pos = counts[(counts["情绪得分"] > threshold) & (counts["总计"] >= min_count)]
    neg = counts[(counts["情绪得分"] < -threshold) & (counts["总计"] >= min_count)]
    return pos, neg


def dropout_points(counts, drop_ratio=0.5):
    """找冷场/流失拐点：前一片弹幕密度不低于均值，当前片骤降到前一片的 drop_ratio 以下。

    规则：
    - 只标「从热到冷」的拐点（前一片热、当前片冷）。一旦骤降，下一片的前一片已 < 均值，
      不会再满足条件，所以连续走低只标第一个拐点。
    - 视频末尾 10% 弹幕自然稀疏，不参与检测（避免把结尾衰减误判为流失）。
    """
    total = counts["总计"]
    mean = total.mean()
    n = len(total)
    end_cut = max(int(n * 0.9), 2)
    drops = []
    for i in range(1, end_cut):
        prev, cur = total.iloc[i - 1], total.iloc[i]
        if prev >= mean and cur < prev * drop_ratio and cur < mean:
            drops.append(i)
    return drops


# ================= 图表 =================

def _base_layout(fig, height):
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif",
                  color=INK, size=13),
        margin=dict(l=50, r=20, t=30, b=40),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=INK_SECONDARY)),
    )
    fig.update_xaxes(showgrid=False, linecolor=BASELINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def emotion_time_curve(counts):
    """情绪时间曲线：发散堆叠面积图。正面上、负面下、中性居中。"""
    x = counts["时间片标签"].tolist()
    pos = counts["正面"].tolist()
    neu = counts["中性"].tolist()
    neg = counts["负面"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pos, name="正面", mode="lines",
                             line=dict(width=0), fill="tozeroy", fillcolor=COLOR_POS))
    fig.add_trace(go.Scatter(x=x, y=[a + b / 2 for a, b in zip(pos, neu)], name="中性",
                             mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=COLOR_NEU))
    fig.add_trace(go.Scatter(x=x, y=[-v for v in neg], name="负面", mode="lines",
                             line=dict(width=0), fill="tozeroy", fillcolor=COLOR_NEG))
    fig.add_trace(go.Scatter(x=x, y=[-(a + b / 2) for a, b in zip(neg, neu)],
                             name="", mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=COLOR_NEU, showlegend=False))
    fig.add_hline(y=0, line_color=BASELINE, line_width=1)

    step_tick = max(1, math.ceil(len(x) / 12))
    tickvals = x[::step_tick]

    fig = _base_layout(fig, height=440)
    fig.update_xaxes(tickvals=tickvals, ticktext=tickvals)
    fig.update_yaxes(title="弹幕数（条）")
    fig.update_layout(hovermode="x unified")
    return fig


def emotion_score_curve(counts, pos_spikes, neg_spikes, drops=None):
    """情绪净得分曲线：(正面-负面)/总，标出正负爆点与冷场拐点。"""
    x = counts["时间片标签"].tolist()
    score = counts["情绪得分"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=score, name="情绪净得分", mode="lines",
                             line=dict(color=INK, width=2)))

    if not pos_spikes.empty:
        fig.add_trace(go.Scatter(
            x=pos_spikes["时间片标签"], y=pos_spikes["情绪得分"],
            mode="markers", name="正向爆点",
            marker=dict(symbol="star", size=16, color=COLOR_POS,
                        line=dict(width=1, color="#ffffff"))))
    if not neg_spikes.empty:
        fig.add_trace(go.Scatter(
            x=neg_spikes["时间片标签"], y=neg_spikes["情绪得分"],
            mode="markers", name="负向爆点",
            marker=dict(symbol="star", size=16, color=COLOR_NEG,
                        line=dict(width=1, color="#ffffff"))))
    if drops:
        drop_labels = [counts.loc[i, "时间片标签"] for i in drops]
        drop_scores = [counts.loc[i, "情绪得分"] for i in drops]
        fig.add_trace(go.Scatter(
            x=drop_labels, y=drop_scores,
            mode="markers", name="冷场点",
            marker=dict(symbol="triangle-down", size=14, color=MUTED,
                        line=dict(width=1, color="#ffffff"))))

    fig.add_hline(y=0, line_color=BASELINE, line_width=1)

    step_tick = max(1, math.ceil(len(x) / 12))
    tickvals = x[::step_tick]

    fig = _base_layout(fig, height=260)
    fig.update_xaxes(tickvals=tickvals, ticktext=tickvals)
    fig.update_yaxes(title="净得分", range=[-1, 1])
    fig.update_layout(hovermode="x unified")
    return fig


def emotion_proportion(pos, neu, neg):
    """情绪构成占比：单条横向堆叠柱（part-to-whole）。"""
    total = pos + neu + neg
    if total == 0:
        return None
    parts = [pos, neu, neg]
    pct = [v / total for v in parts]
    names = ["正面", "中性", "负面"]
    colors = [COLOR_POS, COLOR_NEU, COLOR_NEG]

    fig = go.Figure()
    for n, v, p, c in zip(names, parts, pct, colors):
        fig.add_trace(go.Bar(
            y=["情绪构成"], x=[v], orientation="h", name=n,
            marker_color=c,
            text=[f"{n} {p:.0%}"], textposition="inside",
            textfont=dict(color="#ffffff" if n != "中性" else INK),
        ))
    fig.update_layout(barmode="stack", height=130,
                      paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif",
                                color=INK, size=13),
                      margin=dict(l=60, r=20, t=10, b=30),
                      legend=dict(orientation="h", yanchor="bottom", y=1.3,
                                  xanchor="left", x=0, font=dict(color=INK_SECONDARY)))
    fig.update_xaxes(showgrid=False, linecolor=BASELINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=MUTED))
    return fig


# ================= 高光片段 =================

def highlight_moments(df, counts, n=3):
    """按弹幕总量找出情绪最「热」的几个时间片，返回样例文本。"""
    top = counts["总计"].nlargest(n)
    result = []
    for bin_idx in top.index:
        label = counts.loc[bin_idx, "时间片标签"]
        sub = df[df["时间片"] == bin_idx]
        samples = list(zip(sub["内容"].head(4).tolist(),
                           sub["情绪"].head(4).tolist()))
        result.append((label, int(top[bin_idx]), samples))
    return result


def _chip(text, emotion):
    c = COLORS[emotion]
    t = html.escape(str(text))
    return (f'<span style="background:{c}1a;color:{c};border:1px solid {c};'
            f'padding:2px 9px;border-radius:11px;margin:2px 2px 2px 0;'
            f'display:inline-block;font-size:13px">{t}</span>')


def _spike_samples(df, counts, spikes, emotion, n=5):
    """把爆点时间片展开成 [(时间, 得分, [(弹幕, 情绪)])]。

    归因逻辑：只保留与该爆点情绪一致的弹幕，按出现频次取刷屏最凶的 top n
    （刷屏弹幕才是爆点信号，比随机取前几条更有代表性）。
    """
    result = []
    for bin_idx in spikes.index:
        label = counts.loc[bin_idx, "时间片标签"]
        score = counts.loc[bin_idx, "情绪得分"]
        sub = df[df["时间片"] == bin_idx]
        sub_emo = sub[sub["情绪"] == emotion]
        if sub_emo.empty:
            sub_emo = sub  # 兜底：该片没有该情绪弹幕时退回全部
        top_texts = sub_emo["内容"].value_counts().head(n).index.tolist()
        samples = [(t, emotion) for t in top_texts]
        result.append((label, score, samples))
    return result


# ================= 主流程 =================

def main():
    st.set_page_config(page_title="弹幕情绪时间曲线", layout="wide")
    st.title("弹幕情绪时间曲线")
    st.caption(
        "输入一个视频，看它每一秒观众情绪是正、是负、还是中立，并标出情绪爆点。"
        "情绪标签由训练好的 BERT 模型实时预测。"
    )

    with st.sidebar:
        st.header("模型")
        model_type = st.selectbox(
            "情绪模型",
            list(MODEL_OPTIONS),
            format_func=lambda m: MODEL_OPTIONS[m],
            help="老师 BERT 精度最高；蒸馏 BiLSTM 最小（19.4MB）；量化 BERT 精度损失小、体积减半。")
        st.header("数据")
        source = st.radio("数据来源",
                          ["从已标注数据选视频", "从爬取全量选视频", "上传 CSV", "输入 bvid 爬取", "使用演示数据"],
                          index=0)
        st.header("分箱")
        n_bins = st.slider("时间片数量", 10, 100, 40)
        st.header("爆点")
        spike_threshold = st.slider("情绪爆点阈值（净得分）", 0.15, 0.50, 0.25, step=0.05)

    is_demo = False
    if source == "上传 CSV":
        uploaded = st.file_uploader("选择弹幕 CSV", type=["csv"])
        if uploaded is None:
            st.info("请先在侧边栏上传一个弹幕 CSV。")
            st.stop()
        df = pd.read_csv(uploaded, encoding="utf-8-sig")
        if "出现时间秒" not in df.columns or "内容" not in df.columns:
            st.error("CSV 缺少「内容」或「出现时间秒」列。")
            st.stop()
        df["出现时间秒"] = pd.to_numeric(df["出现时间秒"], errors="coerce")
        df = df.dropna(subset=["出现时间秒"])
        df = df[["内容", "出现时间秒"]]

    elif source == "输入 bvid 爬取":
        bvid = st.sidebar.text_input("视频 bvid", placeholder="例如 BV1xx411c7mD")
        fetch_history = st.sidebar.checkbox("拉取历史弹幕（需 SESSDATA，慢）", value=False)
        if not bvid.strip():
            st.info("请在侧边栏输入视频 bvid。")
            st.stop()
        try:
            crawl_data = _crawl_via_api(bvid.strip(), fetch_history)
            if not isinstance(crawl_data, dict):
                # 旧缓存残留（早期版本把爬取结果存成了 list），清缓存后重爬一次
                st.cache_data.clear()
                crawl_data = _crawl_via_api(bvid.strip(), fetch_history)
        except Exception as e:
            st.error(f"爬取失败：{e}")
            st.stop()
        title = crawl_data.get("title", bvid)
        df = pd.DataFrame(crawl_data.get("danmakus", []))
        if df.empty:
            st.warning(f"该视频（{title}）没有弹幕。")
            st.stop()
        df = df.rename(columns={"text": "内容", "dm_time": "出现时间秒"})
        df = df[["内容", "出现时间秒"]]
        df["出现时间秒"] = pd.to_numeric(df["出现时间秒"], errors="coerce")
        df = df.dropna(subset=["出现时间秒"])
        df = df[df["出现时间秒"] >= 0]
        note = crawl_data.get("note")
        if note:
            st.caption(note)
        st.success(f"已爬取 {len(df)} 条弹幕 —— {title}")

    elif source == "从已标注数据选视频":
        csv_path, videos = _load_video_list()
        if csv_path is None:
            st.warning("没找到已标注数据（Scraping/labeled_*.csv），已用演示数据。")
            df = make_demo_df()
            is_demo = True
        else:
            video_labels = dict(zip(videos["bvid"], videos["视频标题"]))
            selected_bv = st.sidebar.selectbox(
                "选择视频",
                videos["bvid"].tolist(),
                key="labeled_bvid",
                format_func=lambda bv: f"{video_labels[bv]} ({bv})")
            df = _read_video(csv_path, selected_bv)

    elif source == "从爬取全量选视频":
        if not FULL_CSV.exists():
            st.warning("没找到全量弹幕文件（danmaku_all_merged.csv），已用演示数据。")
            df = make_demo_df()
            is_demo = True
        else:
            st.caption("全量弹幕量大（单视频最多 2 万+ 条）；老师/蒸馏模型均在 GPU 上较快，量化模型在 CPU 上较慢（2 万条约 2 分钟）。")
            csv_path, videos = _load_video_list(FULL_CSV)
            video_labels = dict(zip(videos["bvid"], videos["视频标题"]))
            selected_bv = st.sidebar.selectbox(
                "选择视频",
                videos["bvid"].tolist(),
                key="full_bvid",
                format_func=lambda bv: f"{video_labels[bv]} ({bv})")
            df = _read_video(csv_path, selected_bv)
    else:
        df = make_demo_df()
        is_demo = True

    if df.empty:
        st.warning("该视频没有弹幕数据。")
        st.stop()

    # ---- 模型预测情绪（调后端 API） ----
    texts = df["内容"].astype(str).tolist()
    try:
        preds, elapsed_ms, avg_ms = _predict_via_api(tuple(texts), model_type)
        df["情绪"] = preds
    except Exception as e:
        st.error(f"无法连接后端 API（{API_URL}）。请先运行 `python api.py`。\n\n错误：{e}")
        st.stop()

    # ---- 分箱 ----
    df, counts = bin_danmaku(df, n_bins)
    pos_spikes, neg_spikes = emotion_spikes(counts, spike_threshold)
    drops = dropout_points(counts)

    # ---- KPI ----
    total = int(len(df))
    pos = int((df["情绪"] == "正面").sum())
    neg = int((df["情绪"] == "负面").sum())
    neu = total - pos - neg
    peak_bin = counts["总计"].idxmax()
    peak_label = counts.loc[peak_bin, "时间片标签"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("总弹幕量", f"{total:,}")
    c2.metric("正面占比", f"{pos / total:.0%}" if total else "—")
    c3.metric("负面占比", f"{neg / total:.0%}" if total else "—")
    c4.metric("弹幕高峰时刻", peak_label)
    c5.metric("总推理耗时", f"{elapsed_ms / 1000:.2f}s" if elapsed_ms else "—",
              help=f"预测 {total} 条弹幕，后端 {model_type} 总推理耗时")
    c6.metric("单条耗时", f"{avg_ms:.2f} ms" if avg_ms else "—",
              help=f"平均单条弹幕 {model_type} 推理耗时（毫秒）")

    # ---- 图表 ----
    st.subheader("情绪时间曲线")
    st.plotly_chart(emotion_time_curve(counts), use_container_width=True,
                    config={"displayModeBar": False})

    st.subheader("情绪净得分 & 爆点")
    st.caption("净得分 = (正面数 - 负面数) / 总弹幕数。★ 为情绪爆点（得分极值 × 密度峰值），▼ 为冷场拐点（弹幕密度骤降）。")
    st.plotly_chart(emotion_score_curve(counts, pos_spikes, neg_spikes, drops),
                    use_container_width=True, config={"displayModeBar": False})

    if drops:
        st.subheader("冷场 / 流失预警")
        st.caption("▼ 弹幕密度骤降点——观众可能在这里划走，UP 主复盘时可重点关注。")
        for i in drops:
            label = counts.loc[i, "时间片标签"]
            prev = int(counts.loc[i - 1, "总计"])
            cur = int(counts.loc[i, "总计"])
            drop_pct = (1 - cur / prev) * 100 if prev else 0
            st.markdown(f"⚠️ **{label}** 弹幕骤降：{prev} 条 → {cur} 条（-{drop_pct:.0f}%）")

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.subheader("情绪构成")
        fig_p = emotion_proportion(pos, neu, neg)
        if fig_p is not None:
            st.plotly_chart(fig_p, use_container_width=True,
                            config={"displayModeBar": False})

    with col_right:
        st.subheader("情绪爆点")
        pos_moments = _spike_samples(df, counts, pos_spikes, "正面")
        neg_moments = _spike_samples(df, counts, neg_spikes, "负面")
        if not pos_moments and not neg_moments:
            st.info("当前阈值下没有检测到情绪爆点。可调低侧边栏「爆点阈值」。")
        for label, score, samples in pos_moments:
            st.markdown(f"🟦 **正向爆点 {label}** · 净得分 {score:+.2f}")
            st.markdown("".join(_chip(t, e) for t, e in samples), unsafe_allow_html=True)
        for label, score, samples in neg_moments:
            st.markdown(f"🟥 **负向爆点 {label}** · 净得分 {score:+.2f}")
            st.markdown("".join(_chip(t, e) for t, e in samples), unsafe_allow_html=True)

        st.subheader("高光片段（弹幕高峰）")
        moments = highlight_moments(df, counts, n=3)
        for label, cnt, samples in moments:
            st.markdown(f"**{label}** · 共 {cnt} 条")
            st.markdown("".join(_chip(t, e) for t, e in samples), unsafe_allow_html=True)

    # ---- 数据预览 ----
    with st.expander("查看抽样数据（前 100 条）"):
        show = df[["内容", "情绪", "出现时间秒"]].head(100)
        st.dataframe(show, use_container_width=True)


if __name__ == "__main__":
    main()
