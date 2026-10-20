# -*- coding: utf-8 -*-
"""
AKShare 统一数据下载器 (替代 fetch_market_data.py)

依据《量化系统数据获取指南》: AKShare 为主源, 覆盖指数/个股/北向/两融/财务/成分股.
每类数据同时落 CSV (兼容旧代码) 和 Parquet (优先读取).
个股日线下载后立即清洗 (涨跌停/停牌/ST/复牌/缺失值) 再落盘.

用法:
  python data/fetch_data_akshare.py --start 20240101 --end 20260131 --types all
  python data/fetch_data_akshare.py --types stock index north margin
  python data/fetch_data_akshare.py --validate --sample 0.05
"""
import os
import sys
import time
import argparse
import logging
import pandas as pd

# 项目根加入 sys.path 以便 import src
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import get_config
from src.data_cleaning import clean_stock_daily

CFG = get_config()
DATA_ROOT = os.path.dirname(CFG.data.stock_daily_dir)  # D:/开发/训练数据

# 列名映射: AKShare 中文 -> 系统英文
STOCK_COL_MAP = {
    "日期": "eob", "开盘": "open", "收盘": "close", "最高": "high",
    "最低": "low", "成交量": "volume", "成交额": "amount",
    "换手率": "turnover", "涨跌幅": "change_pct", "涨跌额": "change", "振幅": "amplitude",
}
INDEX_COL_MAP = {
    "date": "eob", "open": "open", "high": "high", "low": "low",
    "close": "close", "volume": "volume",
}

# 日志
os.makedirs(CFG.data.data_quality_log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(CFG.data.data_quality_log_dir, "fetch_data.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fetch")


def _to_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)


def _to_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _normalize_stock(df: pd.DataFrame) -> pd.DataFrame:
    """AKShare 个股日线列名归一化"""
    df = df.rename(columns={k: v for k, v in STOCK_COL_MAP.items() if k in df.columns})
    # 统一日期格式
    if "eob" in df.columns:
        df["eob"] = pd.to_datetime(df["eob"]).dt.strftime("%Y-%m-%d")
    return df.sort_values("eob").reset_index(drop=True)


# ============================================================
# 指数日线
# ============================================================

def fetch_index(start: str, end: str):
    """沪深300指数日线 (AKShare stock_zh_index_daily)"""
    log.info("[指数] 下载沪深300 %s ~ %s", start, end)
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol="sh000300")
        df = df.rename(columns=INDEX_COL_MAP)
        df["eob"] = pd.to_datetime(df["eob"]).dt.strftime("%Y-%m-%d")
        df = df[(df["eob"] >= start[:4] + "-" + start[4:6] + "-" + start[6:8]) &
                (df["eob"] <= end[:4] + "-" + end[4:6] + "-" + end[6:8])]
        df = df.sort_values("eob").reset_index(drop=True)
        _to_csv(df, os.path.join(CFG.data.index_daily_dir, "000300.csv"))
        _to_parquet(df, os.path.join(CFG.data.index_daily_pq_dir, "000300.parquet"))
        log.info("[指数] 完成 %d 行", len(df))
    except Exception as e:
        log.error("[指数] 失败: %s", e)


# ============================================================
# 成分股
# ============================================================

def fetch_constituents() -> pd.DataFrame:
    """沪深300成分股 (AKShare index_stock_cons_csindex)"""
    log.info("[成分股] 下载沪深300成分股")
    try:
        import akshare as ak
        df = ak.index_stock_cons_csindex(symbol="000300")
        df = df.rename(columns={"成分券代码": "code", "成分券名称": "name"})
        df = df[["code", "name"]].drop_duplicates()
        os.makedirs(CFG.data.constituents_dir, exist_ok=True)
        df.to_csv(os.path.join(CFG.data.constituents_dir, "hs300_current.csv"),
                  index=False, encoding="utf-8-sig")
        log.info("[成分股] 完成 %d 只", len(df))
        return df
    except Exception as e:
        log.error("[成分股] 失败: %s", e)
        # 降级: 从已有 stock_daily 目录推断
        csv_dir = CFG.data.stock_daily_dir
        if os.path.isdir(csv_dir):
            codes = [f[:6] for f in os.listdir(csv_dir) if f.endswith(".csv")]
            return pd.DataFrame({"code": codes, "name": [""] * len(codes)})
        return pd.DataFrame(columns=["code", "name"])


# ============================================================
# 个股日线 (前复权)
# ============================================================

def _sina_kline(symbol: str, beg: str, end: str) -> pd.DataFrame:
    """新浪后端兜底 (ak.stock_zh_a_daily, qfq). 当 eastmoney 不可达时使用.
    优势: 含真实 amount, 全国可访问"""
    import akshare as ak
    pre = "sh" if symbol.startswith(("6", "9")) else "sz"
    try:
        df = ak.stock_zh_a_daily(symbol=f"{pre}{symbol}", start_date=beg,
                                 end_date=end, adjust="qfq")
        if df is not None and len(df) > 0:
            return df.rename(columns={"date": "eob", "turnover": "turnover"})
    except Exception:
        pass
    return pd.DataFrame()


def fetch_stocks(start: str, end: str, constituents: pd.DataFrame = None):
    """个股日线 (前复权 qfq), 下载后清洗落盘.
    数据源优先级: AKShare stock_zh_a_hist(eastmoney) -> sina stock_zh_a_daily
    """
    if constituents is None or len(constituents) == 0:
        constituents = fetch_constituents()
    log.info("[个股] 下载 %d 只成分股日线 (qfq) %s ~ %s", len(constituents), start, end)
    import akshare as ak
    fail_log = open(os.path.join(CFG.data.data_quality_log_dir, "download_failures.log"),
                    "a", encoding="utf-8")
    ok, fail = 0, 0
    for _, row in constituents.iterrows():
        code = str(row["code"]).zfill(6)
        name = str(row.get("name", ""))
        if code.startswith(("688", "689", "4", "8")):
            continue
        df = None
        # 1. eastmoney 主源
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=start, end_date=end, adjust="qfq")
        except Exception:
            pass
        # 2. sina 兜底
        if df is None or len(df) == 0:
            df = _sina_kline(code, start, end)
        if df is None or len(df) == 0:
            fail += 1
            fail_log.write(f"{code},{name},empty\n")
            continue
        df = _normalize_stock(df)
        # amount 校验 (若 amount 全0, 用 sina 重取)
        if "amount" not in df.columns or df["amount"].sum() == 0:
            df2 = _sina_kline(code, start, end)
            if df2 is not None and len(df2) > 0 and "amount" in df2.columns and df2["amount"].sum() > 0:
                df["amount"] = df2["amount"].values
        # 清洗
        df = clean_stock_daily(df, code, stock_name=name)
        _to_csv(df, os.path.join(CFG.data.stock_daily_dir, f"{code}.csv"))
        _to_parquet(df, os.path.join(CFG.data.stock_daily_pq_dir, f"{code}.parquet"))
        ok += 1
        time.sleep(0.15)
    fail_log.close()
    log.info("[个股] 完成 ok=%d fail=%d", ok, fail)


# ============================================================
# 北向资金
# ============================================================

def fetch_north(start: str, end: str):
    """北向资金 (沪股通 + 深股通)"""
    log.info("[北向] 下载北向资金 %s ~ %s", start, end)
    try:
        import akshare as ak
        frames = []
        for symbol, col in [("沪股通", "sh"), ("深股通", "sz")]:
            df = ak.stock_hsgt_hist_em(symbol=symbol)
            df = df.rename(columns={"日期": "date", "当日成交净买额": f"net_buy_{col}"})
            df = df[["date", f"net_buy_{col}"]]
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            frames.append(df)
            time.sleep(0.3)
        merged = frames[0].merge(frames[1], on="date", how="outer").sort_values("date")
        merged["total_net"] = merged.get("net_buy_sh", 0).fillna(0) + merged.get("net_buy_sz", 0).fillna(0)
        # 落 CSV (兼容旧 reader 的 hsgt_沪.csv / hsgt_深.csv)
        _to_csv(frames[0], os.path.join(CFG.data.north_bound_dir, "hsgt_沪.csv"))
        _to_csv(frames[1], os.path.join(CFG.data.north_bound_dir, "hsgt_深.csv"))
        _to_parquet(merged, os.path.join(CFG.data.north_bound_pq_dir, "north_bound.parquet"))
        log.info("[北向] 完成 %d 行", len(merged))
    except Exception as e:
        log.error("[北向] 失败: %s", e)


# ============================================================
# 两融
# ============================================================

def fetch_margin(start: str, end: str):
    """融资融券 (上交所)"""
    log.info("[两融] 下载融资融券 %s ~ %s", start, end)
    try:
        import akshare as ak
        df = ak.stock_margin_sse(start_date=start, end_date=end)
        col_map = {}
        for c in df.columns:
            cs = str(c)
            if "信用交易日期" in cs:
                col_map[c] = "date"
            elif "融资融券余额" in cs:
                col_map[c] = "balance"
            elif "融资余额" in cs:
                col_map[c] = "fin_balance"
        df = df.rename(columns=col_map)
        keep = [c for c in ["date", "balance", "fin_balance"] if c in df.columns]
        df = df[keep]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        _to_csv(df, os.path.join(CFG.data.margin_trade_dir, "margin_sse.csv"))
        _to_parquet(df, os.path.join(CFG.data.margin_trade_pq_dir, "margin_trade.parquet"))
        log.info("[两融] 完成 %d 行", len(df))
    except Exception as e:
        log.error("[两融] 失败: %s", e)


# ============================================================
# 财务数据
# ============================================================

def _statutory_available(report_date: str) -> pd.Timestamp:
    """报告期 -> 法定披露截止日 (PE/PB 因子的真实可得时点, Point-in-Time)"""
    d = pd.Timestamp(report_date)
    m = d.month
    if m == 12:
        return pd.Timestamp(year=d.year + 1, month=4, day=30)
    if m == 3:
        return pd.Timestamp(year=d.year, month=4, day=30)
    if m == 6:
        return pd.Timestamp(year=d.year, month=8, day=31)
    return pd.Timestamp(year=d.year, month=10, day=31)


def compute_pe_pb_from_yjbb(codes: list = None):
    """用业绩报表(每股收益/每股净资产) + 日线收盘价 自算 PE-TTM/PB

    替代失效的 stock_zh_valuation_baidu。中文财报 EPS 为年初至今累计值,
    TTM_EPS = cum(t) - cum(t-4期), 可得时点取法定披露截止日。
    输出 daily_basic_pq/{code}_pe_ttm.parquet / {code}_pb.parquet (date, value)
    """
    yjbb_path = os.path.join(CFG.data.financial_dir, "yjbb_pq", "yjbb.parquet")
    if not os.path.exists(yjbb_path):
        log.warning("[估值] 无业绩报表, 跳过 PE/PB 计算")
        return
    yjbb = pd.read_parquet(yjbb_path)
    if "股票代码" not in yjbb.columns:
        log.warning("[估值] 业绩报表缺少 股票代码 列, 跳过")
        return
    yjbb["code"] = yjbb["股票代码"].astype(str).str.zfill(6)
    if "report_date" in yjbb.columns:
        yjbb["report_date"] = pd.to_datetime(yjbb["report_date"])
    else:
        return

    if codes is None:
        pq_dir = CFG.data.stock_daily_pq_dir
        codes = [f[:-8] for f in os.listdir(pq_dir) if f.endswith(".parquet")]
    out_dir = os.path.join(CFG.data.financial_dir, "daily_basic_pq")
    os.makedirs(out_dir, exist_ok=True)

    n_pe = n_pb = 0
    for code in codes:
        daily_path = os.path.join(CFG.data.stock_daily_pq_dir, f"{code}.parquet")
        if not os.path.exists(daily_path):
            continue
        g = yjbb[yjbb["code"] == code].sort_values("report_date").drop_duplicates("report_date")
        if len(g) == 0:
            continue
        eps = pd.to_numeric(g.get("每股收益"), errors="coerce")
        bps = pd.to_numeric(g.get("每股净资产"), errors="coerce")
        # 中文财报 EPS 为年初至今累计值: 先还原单季 EPS, 再滚动 4 季求和得 TTM
        # Q1 累计值即单季值; 其余季度需与上一报告期(须为上一季度末)相减
        cum = pd.Series(eps.values, index=g["report_date"].values)
        single = cum.diff()
        for i, d in enumerate(cum.index):
            dt = pd.Timestamp(d)
            if dt.month == 3:                     # 一季报: 累计值即单季值
                single.iloc[i] = cum.iloc[i]
            else:
                prev_expected = (dt - pd.DateOffset(months=3)) + pd.offsets.QuarterEnd(0)
                if i == 0 or pd.Timestamp(cum.index[i - 1]) != prev_expected:
                    single.iloc[i] = float("nan")  # 报告期断档, 无法还原单季
        ttm = single.rolling(4, min_periods=4).sum()
        ttm = ttm[ttm > 0]                        # 剔除亏损 (PE 无意义)

        daily = pd.read_parquet(daily_path, columns=["eob", "close"])
        daily["eob"] = pd.to_datetime(daily["eob"])
        idx = daily.set_index("eob")["close"]

        # 对齐到可得时点后前向填充 (年报与一季报披露截止日相同, 按披露日去重取最新)
        pe_s = pd.Series(ttm.values, index=[_statutory_available(d) for d in ttm.index])
        pe_s = pe_s.groupby(level=0).last()
        pe_daily = pe_s.reindex(pe_s.index.union(idx.index)).ffill().reindex(idx.index)
        bps_s = pd.Series(bps.values, index=[_statutory_available(d) for d in g["report_date"]])
        bps_s = bps_s[bps_s > 0].groupby(level=0).last()
        bps_daily = bps_s.reindex(bps_s.index.union(idx.index)).ffill().reindex(idx.index)

        pe_df = pd.DataFrame({"date": idx.index.strftime("%Y-%m-%d"),
                              "value": (idx.values / pe_daily.values).round(4)})
        pb_df = pd.DataFrame({"date": idx.index.strftime("%Y-%m-%d"),
                              "value": (idx.values / bps_daily.values).round(4)})
        pe_df = pe_df.dropna()
        pb_df = pb_df.dropna()
        if len(pe_df) > 0:
            pe_df.to_parquet(os.path.join(out_dir, f"{code}_pe_ttm.parquet"), index=False)
            n_pe += 1
        if len(pb_df) > 0:
            pb_df.to_parquet(os.path.join(out_dir, f"{code}_pb.parquet"), index=False)
            n_pb += 1
    log.info("[估值] 自算 PE-TTM %d 只 / PB %d 只", n_pe, n_pb)


def fetch_financial(start: str, end: str, constituents: pd.DataFrame = None):
    """财务数据: 业绩报表 + 财务指标 + 自算历史估值(PE/PB)"""
    if constituents is None or len(constituents) == 0:
        constituents = fetch_constituents()
    os.makedirs(os.path.join(CFG.data.financial_dir, "yjbb_pq"), exist_ok=True)
    os.makedirs(os.path.join(CFG.data.financial_dir, "indicator_pq"), exist_ok=True)
    os.makedirs(os.path.join(CFG.data.financial_dir, "daily_basic_pq"), exist_ok=True)
    import akshare as ak

    # 1. 业绩报表 (按报告期, 全市场一份; 自 2017 年起以支持 TTM 回看 4 期)
    log.info("[财务] 下载业绩报表")
    try:
        quarters = pd.date_range("2016-12-31", end, freq="QE").strftime("%Y%m%d").tolist()
        yjbb_all = []
        for q in quarters:
            try:
                df = ak.stock_yjbb_em(date=q)
                if df is not None and len(df) > 0:
                    df["report_date"] = q
                    yjbb_all.append(df)
            except Exception:
                pass
            time.sleep(0.3)
        if yjbb_all:
            yjbb = pd.concat(yjbb_all, ignore_index=True)
            _to_parquet(yjbb, os.path.join(CFG.data.financial_dir, "yjbb_pq", "yjbb.parquet"))
            log.info("[财务] 业绩报表 %d 行", len(yjbb))
    except Exception as e:
        log.error("[财务] 业绩报表失败: %s", e)

    # 2. 财务指标 + 历史估值 (抽样, 全市场太慢)
    sample = constituents.head(30)
    log.info("[财务] 下载 %d 只样本股财务指标+估值", len(sample))
    for _, row in sample.iterrows():
        code = str(row["code"]).zfill(6)
        # 财务指标
        try:
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2020")
            if df is not None and len(df) > 0:
                _to_parquet(df, os.path.join(CFG.data.financial_dir, "indicator_pq", f"{code}.parquet"))
        except Exception as e:
            log.warning("[财务] %s 指标失败: %s", code, e)
        time.sleep(0.3)

    # 3. 自算全市场 PE-TTM/PB (百度估值接口已失效, 改用业绩报表+收盘价, 覆盖全部已下载个股)
    compute_pe_pb_from_yjbb()
    log.info("[财务] 完成")


# ============================================================
# 交叉验证 (BaoStock)
# ============================================================

def run_validation(sample_ratio: float = 0.05):
    """AKShare vs BaoStock 收盘价交叉验证"""
    from src.data_quality import CrossValidator
    cv = CrossValidator(threshold=CFG.data_quality.diff_threshold,
                        log_dir=CFG.data.data_quality_log_dir)
    # 取已有 parquet 文件列表
    pq_dir = CFG.data.stock_daily_pq_dir
    if not os.path.isdir(pq_dir):
        log.error("[验证] stock_daily_pq 不存在, 请先下载")
        return
    codes = [f[:6] for f in os.listdir(pq_dir) if f.endswith(".parquet")]
    import random
    random.seed(42)
    n = max(1, int(len(codes) * sample_ratio))
    sample = random.sample(codes, n)
    log.info("[验证] 抽样 %d/%d 只进行 BaoStock 对比", n, len(codes))
    cv.run_batch(sample)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AKShare 统一数据下载器")
    parser.add_argument("--start", default="20240101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default="20260131", help="结束日期 YYYYMMDD")
    parser.add_argument("--types", default="all",
                        help="数据类型: all / stock,index,north,margin,financial,constituents")
    parser.add_argument("--validate", action="store_true", help="下载后运行 BaoStock 交叉验证")
    parser.add_argument("--sample", type=float, default=0.05, help="交叉验证抽样比例")
    args = parser.parse_args()

    types = set(args.types.split(",")) if args.types != "all" else \
        {"stock", "index", "north", "margin", "financial", "constituents"}

    log.info("=== 开始下载 AKShare 数据 %s ~ %s 类型=%s ===", args.start, args.end, types)
    constituents = None
    if "constituents" in types:
        constituents = fetch_constituents()
    if "index" in types:
        fetch_index(args.start, args.end)
    if "stock" in types:
        fetch_stocks(args.start, args.end, constituents)
    if "north" in types:
        fetch_north(args.start, args.end)
    if "margin" in types:
        fetch_margin(args.start, args.end)
    if "financial" in types:
        fetch_financial(args.start, args.end, constituents)

    if args.validate:
        run_validation(args.sample)

    log.info("=== 全部完成 ===")


if __name__ == "__main__":
    main()
