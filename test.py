from tools.taobao import search_taobao, filter_and_rank

items = search_taobao("2025 夏季 女装", limit=80)
out = filter_and_rank(items, objective_text='找到 3 款 2025 年夏季女装爆款，要求：月销量 > 1000 件、利润率 > 50%、竞争店铺 < 100 家')
print(out["constraints"])
for p in out["selected"]:
    print(p["title"], p["monthly_sales"], p["profit_margin"], p["competitor_shops"])
print(out["note"])
