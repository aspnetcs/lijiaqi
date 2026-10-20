# -*- coding: utf-8 -*-
"""
预处理融资融券日明细 → 日聚合CSV

输入: D:/开发/训练数据/融资融券交易明细表(日)2022-06-30_2026-06-29.zip
      内含 CHN_Stkmt_ddetails.csv ~ CHN_Stkmt_ddetails3.csv

输出: D:/开发/训练数据/margin_trade/margin_gta.csv
      列: date, mtbalance (全市场融资融券余额合计, 亿元)
"""

import csv
import os
import gzip
import subprocess
import tempfile
from collections import defaultdict


def process_zip(zip_path, output_path):
    """解压CSV并聚合每日融资融券余额"""

    csv_files = [
        "CHN_Stkmt_ddetails.csv",
        "CHN_Stkmt_ddetails1.csv",
        "CHN_Stkmt_ddetails2.csv",
        "CHN_Stkmt_ddetails3.csv",
    ]

    daily_mtbalance = defaultdict(float)
    total_rows = 0

    for csv_name in csv_files:
        print(f"[INFO] Processing {csv_name} ...")

        # 使用 unzip -p 流式读取 (避免解压全部到磁盘)
        process = subprocess.Popen(
            ["unzip", "-p", zip_path, csv_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        # 流式读取CSV行
        reader = csv.DictReader(
            (line.decode('utf-8-sig', errors='replace') for line in process.stdout),
        )

        prev_date = None
        for row in reader:
            total_rows += 1
            if total_rows % 1000000 == 0:
                print(f"  ... processed {total_rows} rows")

            mtdate = (row.get("Mtdate") or "").strip()
            mtbalance_str = (row.get("Mtbalance") or "0").strip()

            if not mtdate:
                continue

            try:
                mtbalance = float(mtbalance_str)
            except (ValueError, TypeError):
                mtbalance = 0.0

            daily_mtbalance[mtdate] += mtbalance / 1e8  # 转为亿元

        process.stdout.close()
        process.wait()

    print(f"[INFO] Total rows: {total_rows}")
    print(f"[INFO] Unique dates: {len(daily_mtbalance)}")

    # 排序并保存
    sorted_dates = sorted(daily_mtbalance.keys())
    print(f"[INFO] Date range: {sorted_dates[0]} ~ {sorted_dates[-1]}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["date", "mtbalance"])
        for d in sorted_dates:
            writer.writerow([d, f"{daily_mtbalance[d]:.2f}"])

    print(f"[INFO] Saved to: {output_path}")
    print(f"[INFO] File size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    zip_path = "D:/开发/训练数据/融资融券交易明细表(日)2022-06-30_2026-06-29.zip"
    output_path = "D:/开发/训练数据/margin_trade/margin_gta.csv"

    process_zip(zip_path, output_path)
