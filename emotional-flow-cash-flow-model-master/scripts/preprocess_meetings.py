# -*- coding: utf-8 -*-
"""
预处理调研纪要 CSV → 日期索引 Pickle

输入:
    D:/开发/xueshuaiDQK/guo_tai_an/调研/IRM_MEETINGMINUTES.csv
    D:/开发/xueshuaiDQK/guo_tai_an/调研/IRM_MEETINGMINUTES1.csv

输出:
    output/data/meeting_minutes_index.pkl  (dict: {date_str: [text1, text2, ...]})
"""

import csv
import pickle
import os
from collections import defaultdict


def preprocess(input_paths, output_path, start_date="2023-06-01", end_date="2026-12-31"):
    """
    读取调研纪要 CSV，按日期分组文本，保存为 pickle。

    每行拼接 Question + Answer 作为分析文本。
    """
    date_texts = defaultdict(list)
    total_rows = 0
    in_range = 0

    for input_path in input_paths:
        print(f"[INFO] Processing: {input_path}")
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                if total_rows % 500000 == 0:
                    print(f"  ... processed {total_rows} rows, in_range={in_range}")

                report_date = (row.get("ReportDate") or "").strip()
                if not report_date:
                    continue

                # 日期过滤
                if report_date < start_date or report_date > end_date:
                    continue

                # 拼接 Question + Answer
                question = (row.get("Question") or "").strip()
                answer = (row.get("Answer") or "").strip()

                if question and answer:
                    text = question + "。 " + answer
                elif question:
                    text = question
                elif answer:
                    text = answer
                else:
                    continue

                if len(text) < 5:
                    continue

                date_texts[report_date].append(text)
                in_range += 1

    print(f"\n[INFO] Total rows scanned: {total_rows}")
    print(f"[INFO] In-range rows kept: {in_range}")
    print(f"[INFO] Unique dates: {len(date_texts)}")

    # 按日期排序
    sorted_dict = dict(sorted(date_texts.items()))

    # 统计每日期望数量
    text_counts = [len(v) for v in sorted_dict.values()]
    if text_counts:
        print(f"[INFO] Texts per day: min={min(text_counts)}, max={max(text_counts)}, "
              f"mean={sum(text_counts)/len(text_counts):.1f}")

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(sorted_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[INFO] Saved to: {output_path} ({file_size_mb:.1f} MB)")
    return sorted_dict


if __name__ == "__main__":
    input_paths = [
        "D:/开发/xueshuaiDQK/guo_tai_an/调研/IRM_MEETINGMINUTES.csv",
        "D:/开发/xueshuaiDQK/guo_tai_an/调研/IRM_MEETINGMINUTES1.csv",
    ]
    output_path = "C:/Users/k/WorkBuddy/2026-06-28-23-32-40/sentiment_capital_system/output/data/meeting_minutes_index.pkl"

    preprocess(input_paths, output_path)
