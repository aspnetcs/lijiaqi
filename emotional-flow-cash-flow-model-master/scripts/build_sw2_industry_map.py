# -*- coding: utf-8 -*-
"""
一键构建申万二级行业-股票映射表

利用 westock-data sector constituent 接口批量获取所有申万二级板块的成份股,
生成 _stock_industry_map_sw2.json。

运行方式: python scripts/build_sw2_industry_map.py
"""

import sys
import os
import json
import subprocess

# 路径配置
WESTOCK_DIR = r"C:/Users/k/AppData/Local/Programs/WorkBuddy/resources/app.asar.unpacked/resources/builtin-skills/westock-data"
OUTPUT_PATH = "D:/开发/训练数据/sector_money_flow/_stock_industry_map_sw2.json"
NODE_EXE = r"C:/Users/k/.workbuddy/binaries/node/versions/22.12.0/node.exe"
INDEX_JS = os.path.join(WESTOCK_DIR, "scripts", "index.js")

def run_cmd(args):
    """运行 westock-data node 命令并返回 JSON 数据"""
    result = subprocess.run(
        [NODE_EXE, INDEX_JS] + args + ["--raw"],
        cwd=WESTOCK_DIR,
        capture_output=True, text=True, timeout=60,
        encoding="utf-8", errors="replace"
    )
    if not result.stdout.strip():
        raise RuntimeError(f"Empty output for {' '.join(args)}")
    return json.loads(result.stdout)


def main():
    print("=" * 60)
    print("  申万二级行业-股票映射生成器 (westock-data)")
    print("=" * 60)

    # Step 1: 获取所有申万二级行业列表
    print("\n[Step 1] 获取申万二级行业列表...")

    batch = run_cmd(["sector", "list", "industry_list_sw2", "--limit", "200", "--offset", "0"])
    all_sectors = batch if isinstance(batch, list) else batch.get("data", batch.get("items", []))
    if not all_sectors:
        # 如果 batch 是 None，直接退出
        print("  [ERROR] 无法获取行业列表")
        return

    print(f"  共 {len(all_sectors)} 个申万二级行业")
    for s in all_sectors[:5]:
        print(f"    {s['code']}: {s['name']}")
    print("    ...")

    # Step 2: 逐个板块获取成份股
    print(f"\n[Step 2] 获取板块成份股 ({len(all_sectors)} 个板块)...")
    stock_map = {}  # {stock_code: industry_name}
    sector_stats = []

    for i, sector in enumerate(all_sectors):
        code = sector["code"]
        name = sector["name"]
        print(f"  [{i+1}/{len(all_sectors)}] {name} ({code})...", end=" ", flush=True)

        try:
            result = run_cmd(["sector", "constituent", code])
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # 解析成份股 (字段: StockCode="sh688593", StockName="新相微")
        constituents = result if isinstance(result, list) else result.get("data", [])
        count = 0
        for item in constituents:
            if not isinstance(item, dict):
                continue
            stock_code = item.get("StockCode", "")
            # 统一格式: 去掉 sh/sz 前缀, 保留6位数字
            if stock_code.startswith("sh") or stock_code.startswith("sz"):
                stock_code = stock_code[2:]
            
            if stock_code and len(stock_code) == 6 and stock_code.isdigit():
                if stock_code not in stock_map:
                    stock_map[stock_code] = name
                count += 1
        
        sector_stats.append((name, count))
        print(f"{count} stocks")

    # Step 3: 保存
    print(f"\n[Step 3] 保存映射文件...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 转换格式: {code: {"industry_name": str, "industry_code": str}}
    output = {}
    for code, name in stock_map.items():
        output[code] = {
            "industry_name": name,
            "industry_code": "",
        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 统计
    industry_counts = {}
    for v in stock_map.values():
        industry_counts[v] = industry_counts.get(v, 0) + 1

    print(f"\n{'='*60}")
    print(f"  完成!")
    print(f"  覆盖股票: {len(stock_map)}")
    print(f"  行业数: {len(industry_counts)}")
    print(f"  输出文件: {OUTPUT_PATH}")
    print(f"{'='*60}")
    print("\n行业分布 TOP20:")
    for name, count in sorted(industry_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {name}: {count} stocks")


if __name__ == "__main__":
    main()
