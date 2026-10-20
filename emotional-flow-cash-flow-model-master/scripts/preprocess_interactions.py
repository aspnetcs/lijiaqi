# -*- coding: utf-8 -*-
"""
预处理投资者互动数据 → 并入调研纪要索引

输入:
    D:/开发/xueshuaiDQK/guo_tai_an/互动/CMS_InvestorInteraction1.csv
    D:/开发/xueshuaiDQK/guo_tai_an/互动/CMS_InvestorInteraction2.csv

处理:
    提取 QuestionContent + ReplyContent, 按日期分组, 合并入 meeting_minutes_index.pkl
"""

import csv
import pickle
import os
from collections import defaultdict


def process_interactions(input_paths, existing_pkl_path, output_pkl_path,
                         start_date="2023-06-01", end_date="2026-12-31"):
    """读取投资者互动CSV，按日期分组文本，合并到已有 pickle"""

    # 加载已有索引 → 转为 defaultdict 允许新增日期
    if os.path.exists(existing_pkl_path):
        print(f"[INFO] Loading existing index: {existing_pkl_path}")
        with open(existing_pkl_path, 'rb') as f:
            existing = pickle.load(f)
        date_texts = defaultdict(list, existing)
        print(f"[INFO] Existing: {len(date_texts)} dates, "
              f"{sum(len(v) for v in date_texts.values())} texts")
    else:
        date_texts = defaultdict(list)

    # 处理投资者互动CSV
    total_rows = 0
    added_rows = 0

    for input_path in input_paths:
        print(f"[INFO] Processing: {input_path}")
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                if total_rows % 300000 == 0:
                    print(f"  ... processed {total_rows} rows, added={added_rows}")

                # 使用 QuestionDate 作为主日期
                qdate_raw = (row.get("QuestionDate") or "").strip()
                rdate_raw = (row.get("ReplyDate") or "").strip()

                qdate = qdate_raw[:10] if qdate_raw else ""
                rdate = rdate_raw[:10] if rdate_raw and rdate_raw != "null" else ""

                # 如果没有 QuestionDate，跳过
                if not qdate or qdate[0].isdigit() == False:
                    continue

                # 日期过滤
                if qdate < start_date or qdate > end_date:
                    continue

                # 拼接 Question + Reply
                question = (row.get("QuestionContent") or "").strip()
                reply = (row.get("ReplyContent") or "").strip()

                text_parts = []
                if question and len(question) >= 5:
                    text_parts.append(question)
                if reply and len(reply) >= 5 and reply != "null":
                    text_parts.append(reply)

                if not text_parts:
                    continue

                text = "。 ".join(text_parts)
                date_texts[qdate].append(text)
                added_rows += 1

                # 如果有 ReplyDate，也加入
                if rdate and rdate >= start_date and rdate <= end_date:
                    date_texts[rdate].append(text)
                    added_rows += 1

    print(f"\n[INFO] Total rows scanned: {total_rows}")
    print(f"[INFO] New texts added: {added_rows}")
    print(f"[INFO] Total unique dates: {len(date_texts)}")

    # 按日期排序
    sorted_dict = dict(sorted(date_texts.items()))

    # 统计
    text_counts = [len(v) for v in sorted_dict.values()]
    if text_counts:
        print(f"[INFO] Texts per day: min={min(text_counts)}, max={max(text_counts)}, "
              f"mean={sum(text_counts)/len(text_counts):.1f}")

    # 保存
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(sorted_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = os.path.getsize(output_pkl_path) / (1024 * 1024)
    print(f"[INFO] Saved to: {output_pkl_path} ({file_size_mb:.1f} MB)")
    return sorted_dict


if __name__ == "__main__":
    input_paths = [
        "D:/开发/xueshuaiDQK/guo_tai_an/互动/CMS_InvestorInteraction1.csv",
        "D:/开发/xueshuaiDQK/guo_tai_an/互动/CMS_InvestorInteraction2.csv",
    ]
    existing_pkl = "C:/Users/k/WorkBuddy/2026-06-28-23-32-40/sentiment_capital_system/output/data/meeting_minutes_index.pkl"
    output_pkl = "C:/Users/k/WorkBuddy/2026-06-28-23-32-40/sentiment_capital_system/output/data/meeting_minutes_index.pkl"

    process_interactions(input_paths, existing_pkl, output_pkl)
