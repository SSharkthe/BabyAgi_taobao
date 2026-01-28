# tools/taobao.py
# 模拟：根据 keyword “爬到”相关商品列表 -> 保存到 runs/taobao_candidates.json
from __future__ import annotations

import json
import os
import re
import math
import hashlib
import random
from datetime import datetime
from typing import Any, Dict, List, Optional


# 你可以按需扩展的类目词（中文关键字）
CATEGORY_WORDS = [
    "衬衫", "雪纺衫", "T恤", "针织衫", "针织开衫",
    "连衣裙", "半身裙", "短裤", "阔腿裤", "牛仔裤", "防晒衫", "吊带裙"
]

SEASONS = ["春季", "夏季", "秋季", "冬季"]
GENDERS = ["女装", "男装"]

STYLES = ["爆款", "通勤", "法式", "韩版", "复古", "显瘦", "小个子", "气质", "高级感", "简约", "学院风", "度假"]
MATERIALS = ["棉", "真丝", "雪纺", "亚麻", "牛仔", "针织", "缎面", "冰丝", "莫代尔"]
COLORS = ["白色", "黑色", "雾霾蓝", "卡其", "杏色", "浅蓝", "粉色", "薄荷绿", "奶油黄", "灰色"]

SHOP_NAMES = ["晴空服饰", "风铃女装", "云朵衣橱", "南风小店", "轻语衣舍", "岛屿穿搭", "夏日衣集", "蔷薇工坊", "海盐衣橱", "沐光女装"]


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _safe_load_json_list(path: str) -> List[Dict[str, Any]]:
    """容错读取：文件不存在/空文件/坏JSON -> 返回 []"""
    if not os.path.exists(path):
        return []
    try:
        if os.path.getsize(path) == 0:
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _atomic_write_json(path: str, data: Any) -> None:
    """原子写入：先写 tmp 再 replace，避免中途写崩导致 JSONDecodeError"""
    _ensure_dir_for_file(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _hash_seed(text: str) -> int:
    """把 keyword 变成稳定 seed：同一 keyword 每次生成风格一致（便于复现实验）"""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _infer_from_keyword(keyword: str, rng: random.Random) -> Dict[str, Any]:
    """从 keyword 里尽量提取 year/season/gender/category；提不出就给合理默认"""
    merged = keyword.replace(" ", "")

    # year
    year = None
    m = re.search(r"(20\d{2})", merged)
    if m:
        year = int(m.group(1))
    else:
        year = rng.choice([2024, 2025, 2026])

    # season
    season = None
    for s in SEASONS:
        if s in merged or s.replace("季", "") in merged:
            season = s
            break
    if season is None:
        season = rng.choice(SEASONS)

    # gender
    gender = None
    for g in GENDERS:
        if g in merged:
            gender = g
            break
    if gender is None:
        gender = "女装" if "女" in merged else rng.choice(GENDERS)

    # category
    category = None
    for c in CATEGORY_WORDS:
        if c in merged:
            category = c
            break
    if category is None:
        # 兜底：如果有“衬衣”也算衬衫
        if "衬衣" in merged:
            category = "衬衫"
        else:
            category = rng.choice(CATEGORY_WORDS)

    return {"year": year, "season": season, "gender": gender, "category": category}


def _lognormal_int(rng: random.Random, mean: float, sigma: float, min_v: int, max_v: int) -> int:
    """生成更像销量分布的随机数（长尾）"""
    # random.lognormvariate 依赖底层 random，稳定性够用
    v = int(rng.lognormvariate(math.log(mean), sigma))
    return max(min_v, min(max_v, v))


def _generate_item(rng: random.Random, base: Dict[str, Any], idx: int, seed_tag: str) -> Dict[str, Any]:
    year = base["year"]
    season = base["season"]
    gender = base["gender"]
    category = base["category"]

    style = rng.choice(STYLES)
    material = rng.choice(MATERIALS)
    color = rng.choice(COLORS)

    # 用 seed_tag + idx 生成稳定 item_id（同keyword复现实验更友好）
    item_id = f"TB{seed_tag}-{idx:04d}"

    # 标题尽量贴合 keyword（含年份/季节/类目）
    title = f"{year}{season} {style}{material}{category} {color}"

    # 月销量：长尾分布
    monthly_sales = _lognormal_int(rng, mean=450, sigma=0.9, min_v=5, max_v=12000)

    # 利润率：0.10 ~ 0.85（模拟）
    profit_margin = round(rng.uniform(0.10, 0.85), 4)

    # 竞争店铺：5 ~ 600（模拟）
    competitor_shops = int(rng.triangular(5, 600, 120))

    # 价格：按类目给个大致区间
    if category in ("连衣裙", "吊带裙"):
        sale_price = round(rng.uniform(79, 499), 2)
    elif category in ("衬衫", "雪纺衫", "T恤", "针织衫", "针织开衫", "防晒衫"):
        sale_price = round(rng.uniform(39, 399), 2)
    else:
        sale_price = round(rng.uniform(49, 459), 2)

    cost_price = round(sale_price * (1 - profit_margin), 2)

    shop_name = rng.choice(SHOP_NAMES)

    # 评价：与销量弱相关
    rating = round(rng.uniform(3.8, 4.95), 2)
    review_count = int(max(0, rng.gauss(monthly_sales * 0.25, monthly_sales * 0.08)))

    link = f"https://example.com/item/{item_id}"

    tags = ",".join(filter(None, [
        str(year),
        season,
        gender,
        category,
        style,
        material,
        color,
        "淘宝",
        "搜索结果",
        "模拟爬虫"
    ]))

    return {
        "item_id": item_id,
        "title": title,
        "category": category,
        "year": year,
        "season": season,
        "gender": gender,
        "monthly_sales": monthly_sales,
        "profit_margin": profit_margin,
        "competitor_shops": competitor_shops,
        "sale_price": sale_price,
        "cost_price": cost_price,
        "shop_name": shop_name,
        "rating": rating,
        "review_count": review_count,
        "link": link,
        "tags": tags,
        "source": "taobao_mock_crawler",
        "keyword": seed_tag,  # 简短标记：便于调试
        "crawled_at": datetime.utcnow().isoformat() + "Z",
    }


def search_taobao(
    keyword: str,
    n: int = 300,
    save_path: str = "runs/taobao_candidates.json",
    merge: bool = False,
    preview_k: int = 5,
) -> Dict[str, Any]:
    """
    模拟淘宝爬虫：根据 keyword 生成 n 条“相关商品”并保存到 save_path。

    参数：
    - keyword: 搜索关键词（如 "2024 秋季 女装 衬衫"）
    - n: 生成条数（默认 300）
    - save_path: 输出文件路径（默认 runs/taobao_candidates.json）
    - merge: 是否与已有缓存合并（默认 False：覆盖写；True：合并去重）
    - preview_k: 返回预览条数（默认 5）

    返回：
    - dict: {tool, keyword, generated_count, saved_path, preview}
    """
    keyword = (keyword or "").strip()
    if not keyword:
        raise ValueError("keyword 不能为空")

    # 稳定随机：同一个 keyword 每次生成风格一致，方便复现实验
    seed = _hash_seed(keyword)
    rng = random.Random(seed)

    base = _infer_from_keyword(keyword, rng)
    seed_tag = f"{seed % 100000:05d}"

    new_items: List[Dict[str, Any]] = []
    for i in range(int(n)):
        new_items.append(_generate_item(rng, base, i, seed_tag))

    if merge:
        old = _safe_load_json_list(save_path)
        by_id: Dict[str, Dict[str, Any]] = {}
        for x in old:
            if isinstance(x, dict) and x.get("item_id"):
                by_id[x["item_id"]] = x
        for x in new_items:
            by_id[x["item_id"]] = x
        final_items = list(by_id.values())
    else:
        final_items = new_items

    _atomic_write_json(save_path, final_items)

    # 返回摘要给 Agent（不要把几百条全塞回 prompt）
    preview = [
        {
            "item_id": x["item_id"],
            "title": x["title"],
            "year": x["year"],
            "season": x["season"],
            "category": x["category"],
            "monthly_sales": x["monthly_sales"],
            "profit_margin": x["profit_margin"],
            "competitor_shops": x["competitor_shops"],
            "sale_price": x["sale_price"],
            "shop_name": x["shop_name"],
            "link": x["link"],
        }
        for x in final_items[: max(1, int(preview_k))]
    ]

    return {
        "tool": "search_taobao",
        "keyword": keyword,
        "generated_count": len(new_items),
        "saved_count": len(final_items),
        "saved_path": save_path,
        "preview": preview,
        "note": "模拟爬虫：已生成并落盘候选商品数据（供后续 LLM 读文件分析）。",
    }
