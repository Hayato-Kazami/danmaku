"""bvid → 弹幕爬取（基于 bilibili-api-python），供后端 /crawl_danmaku 接口调用。

设计：
    - 当前弹幕：不限流、无需登录，单个视频 600~3000 条，是「陌生视频」功能的主路径。
    - 历史弹幕：可选（默认关闭），接口限流极严（-702），且需登录 SESSDATA。
      SESSDATA 从环境变量 BILI_SESSDATA 读取（绝不写死在代码里，避免泄露 / 被提交）。

bilibili_api 只在函数内延迟 import：未安装时后端仍能启动，只有调用爬取接口才报错。
"""
import datetime
import os
import re
import time

# 历史弹幕逐日期拉取的限速间隔（秒），接口限流严，不宜过快
_HISTORY_SLEEP = 0.3
_HISTORY_MONTH_SLEEP = 0.5
# 触发 -702 限流后的长冷却时间（秒）：B 站历史弹幕接口限流是分钟/小时级，快速重试只会火上浇油
_RATE_LIMIT_COOLDOWN = 300
# 历史索引接口也限流时，多等的时间（秒）
_INDEX_RETRY_SLEEP = 5


def _make_video(bvid):
    """构造 video.Video，若环境变量有 SESSDATA 则带上（历史弹幕需要登录）。"""
    from bilibili_api import video, Credential
    sessdata = os.environ.get("BILI_SESSDATA", "").strip()
    credential = Credential(sessdata=sessdata) if sessdata else None
    return video.Video(bvid=bvid, credential=credential)


def _clean_dms(dms, limit):
    """过滤超短/纯符号弹幕，连续重复字压单字后去重，截断到 limit 条。"""
    rows = []
    seen = set()
    for dm in dms:
        text = (dm.text or "").strip()
        if len(text) < 2:
            continue
        if not re.search(r"[A-Za-z0-9一-鿿]", text):
            continue
        key = re.sub(r"(.)\1+", r"\1", text)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"text": text, "dm_time": dm.dm_time})
        if len(rows) >= limit:
            break
    return rows


def _fetch_history_day(v, cid, d):
    """按日期拉历史弹幕，带「限流即长冷却重试」。

    B 站历史弹幕接口（x/v2/dm/web/history/seg.so）触发 -702 后进入分钟/小时级冷却，
    快速重试只会火上浇油。这里遇到「解析响应数据错误」（-702 的伪装）就长等再试一次，
    仍失败则放弃该日（不影响其他日期）。
    """
    from bilibili_api import sync
    try:
        return sync(v.get_danmakus(cid=cid, date=d))
    except Exception as e:
        if "解析响应数据错误" in str(e):
            print(f"    [历史] {d} 触发限流(-702)，冷却 {_RATE_LIMIT_COOLDOWN}s 后重试一次")
            time.sleep(_RATE_LIMIT_COOLDOWN)
            try:
                return sync(v.get_danmakus(cid=cid, date=d))
            except Exception as e2:
                print(f"    [历史] {d} 仍失败，跳过：{e2}")
                return []
        print(f"    [历史] {d} 拉取失败，跳过：{e}")
        return []


def _fetch_history(v, cid, pubdate):
    """逐月枚举历史弹幕日期并拉取。限流严格，失败不中断，只返回拿到的部分。"""
    from bilibili_api import sync
    start = datetime.datetime.fromtimestamp(pubdate).date()
    today = datetime.date.today()
    cur = start.replace(day=1)
    dms = []
    while cur <= today:
        try:
            dates = sync(v.get_history_danmaku_index(date=cur, cid=cid))
        except Exception as e:
            print(f"  [历史] 索引 {cur:%Y-%m} 失败：{e}")
            dates = None
            time.sleep(_INDEX_RETRY_SLEEP)  # 索引接口也限流时，多等一会
        if dates:
            for ds in dates:
                d = datetime.date.fromisoformat(ds)
                dms.extend(_fetch_history_day(v, cid, d))
                time.sleep(_HISTORY_SLEEP)
        cur = (cur + datetime.timedelta(days=32)).replace(day=1)
        time.sleep(_HISTORY_MONTH_SLEEP)
    return dms


def crawl_video_danmaku(bvid: str, fetch_history: bool = False, limit: int = 10000):
    """爬单个视频弹幕。

    返回 dict：{title, bvid, count, note, danmakus:[{text, dm_time}]}
    note 为可选提示（如未配 SESSDATA 只爬了当前弹幕）。
    """
    from bilibili_api import sync

    v = _make_video(bvid)
    info = sync(v.get_info())
    title = info.get("title", bvid)
    cid = info.get("cid")
    pubdate = info.get("pubdate")

    dms = list(sync(v.get_danmakus(0)))  # 当前弹幕：不限流、无需登录
    note = None

    if fetch_history:
        if not os.environ.get("BILI_SESSDATA", "").strip():
            note = "未设置 SESSDATA，仅爬取当前弹幕（历史弹幕需登录）。"
        elif not pubdate:
            note = "视频无发布时间，仅爬取当前弹幕。"
        else:
            dms.extend(_fetch_history(v, cid, pubdate))
            note = "已包含历史弹幕。"

    rows = _clean_dms(dms, limit)
    return {"title": title, "bvid": bvid, "count": len(rows),
            "note": note, "danmakus": rows}
