# -*- coding: utf-8 -*-
"""
用 BaoStock 免费接口下载沪深300历史成分股 (替代 Tushare index_weight 的 2000 积分方案)

原理:
    BaoStock query_hs300_stocks(date) 支持按历史日期查询当时的成分股名单,
    每月第一个交易日采样一次, 全部合并存成一个 parquet, 列: trade_date, con_code, code_name
    输出目录与 ConstituentsReader 的 hs300_weight 兼容, get_hs300() 无需改动即可生效。

用法:
    python data/fetch_hs300_history_baostock.py                # 2018-01 至今
    python data/fetch_hs300_history_baostock.py --start 2019
"""
import os
import sys
import argparse
from datetime import date

import pandas as pd

OUT_DIR = "D:/开发/训练数据/constituents/hs300_weight"


def first_trading_days(start_year: int, end_year: int):
    """生成 [start_year, end_year] 每月第一个交易日候选 (逐日试探到返回非空为止)"""
    candidates = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            candidates.append(f"{y:04d}-{m:02d}-01")
    return candidates


def fetch_all(start_year: int) -> pd.DataFrame:
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")

    end_year = date.today().year
    frames = []
    try:
        for base in first_trading_days(start_year, end_year):
            y, m, d = map(int, base.split("-"))
            if date(y, m, d) > date.today():
                continue
            got = None
            # 从当月1号起逐日试探, 最多试探 10 天找到有效交易日
            for offset in range(10):
                try_date = (date(y, m, d) + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                rs = bs.query_hs300_stocks(date=try_date)
                rows = []
                while rs.error_code == "0" and rs.next():
                    rows.append(rs.get_row_data())
                if rows:
                    got = (try_date, rows)
                    break
            if got is None:
                print(f"  [WARN] {base} 起试探 10 天均无数据, 跳过")
                continue
            dt, rows = got
            df = pd.DataFrame(rows, columns=["updateDate", "code", "code_name"])
            df["trade_date"] = dt
            df["con_code"] = df["code"].str.split(".").str[-1]  # sh.600000 -> 600000
            frames.append(df[["trade_date", "con_code", "code_name", "updateDate"]])
            print(f"  {dt}: {len(df)} 只成分股")

        if not frames:
            raise RuntimeError("未获取到任何历史成分数据")
        return pd.concat(frames, ignore_index=True)
    finally:
        bs.logout()


def main():
    ap = argparse.ArgumentParser(description="BaoStock 沪深300历史成分下载")
    ap.add_argument("--start", type=int, default=2018, help="起始年份 (默认 2018)")
    args = ap.parse_args()

    print(f"下载沪深300历史成分 ({args.start} 至今) ...")
    df = fetch_all(args.start)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"hs300_baostock_{args.start}_latest.parquet")
    df.to_parquet(out, index=False)

    n_dates = df["trade_date"].nunique()
    n_codes = df["con_code"].nunique()
    print(f"\n完成: {out}")
    print(f"  采样期数: {n_dates}, 涉及成分股: {n_codes} 只, 总行数: {len(df)}")


if __name__ == "__main__":
    main()
