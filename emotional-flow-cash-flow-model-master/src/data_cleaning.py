# -*- coding: utf-8 -*-
"""
数据清洗模块

对个股日线数据进行清洗标记, 解决量化回测中的常见数据问题:
  - 涨跌停检测 (主板±10% / 创业板科创板±20% / ST±5% / 北交所±30%)
  - 停牌检测 (volume==0 或一字且量极小)
  - 复牌首日标记
  - ST 股票标记
  - 缺失值处理 (重建交易日历 + 前向填充 + was_filled 标记)

清洗在下载落库前执行一次; data_reader 读取时作为安全网对旧 CSV 再补跑。
"""
import pandas as pd
import numpy as np


# ============================================================
# 涨跌停检测
# ============================================================

def _limit_pct(code: str, is_st: bool = False) -> float:
    """根据股票代码前缀判断涨跌停幅度"""
    if is_st:
        return 0.05
    code = str(code).split(".")[-1]  # 兼容 "SHSE.600000" / "600000"
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("4", "8")):
        return 0.30
    return 0.10  # 主板


def detect_limit_up_down(df: pd.DataFrame, code: str, is_st: bool = False) -> pd.DataFrame:
    """涨跌停检测, 新增 is_limit_up / is_limit_down 列"""
    if df is None or len(df) == 0:
        return df
    pct = _limit_pct(code, is_st)
    prev_close = df["close"].shift(1)
    # 涨跌停: 收盘价相对前收达到 ±pct (允许 0.1% 容差, 处理四舍五入)
    up_limit = prev_close * (1 + pct)
    down_limit = prev_close * (1 - pct)
    tol = prev_close * 0.001
    df["is_limit_up"] = (df["close"] >= up_limit - tol).fillna(False)
    df["is_limit_down"] = (df["close"] <= down_limit + tol).fillna(False)
    return df


# ============================================================
# 停牌检测
# ============================================================

def detect_suspension(df: pd.DataFrame) -> pd.DataFrame:
    """停牌检测, 新增 is_suspended 列
    规则: volume==0 或 (开=高=低=收 且 成交量极小)
    """
    if df is None or len(df) == 0:
        return df
    vol_zero = (df["volume"].fillna(0) == 0)
    one_word = ((df["open"] == df["high"]) & (df["high"] == df["low"]) &
                (df["low"] == df["close"]) & (df["volume"].fillna(0) < df["volume"].median() * 0.01))
    df["is_suspended"] = (vol_zero | one_word).fillna(False)
    return df


# ============================================================
# 复牌首日标记
# ============================================================

def detect_resume(df: pd.DataFrame) -> pd.DataFrame:
    """复牌首日: 前一日停牌且当日未停牌"""
    if df is None or len(df) == 0:
        return df
    if "is_suspended" not in df.columns:
        df = detect_suspension(df)
    df["is_resume"] = (df["is_suspended"].shift(1).fillna(False) & ~df["is_suspended"])
    return df


# ============================================================
# ST 标记
# ============================================================

def mark_st(df: pd.DataFrame, stock_name: str = "") -> pd.DataFrame:
    """ST 股票标记, 新增 is_st 列 (全列同值, 因名称通常不变)"""
    if df is None or len(df) == 0:
        return df
    is_st = "ST" in stock_name.upper() if stock_name else False
    df["is_st"] = is_st
    return df


# ============================================================
# 缺失值处理
# ============================================================

def fill_missing(df: pd.DataFrame, date_col: str = "eob") -> pd.DataFrame:
    """缺失值处理: 重建交易日历 + 前向填充 + was_filled 标记"""
    if df is None or len(df) == 0:
        return df
    # 重建完整交易日历 (用沪深300交易日对齐需外部数据, 此处用已有日期的工作日序列)
    df = df.copy()
    idx = pd.to_datetime(df[date_col] if date_col in df.columns else df.index)
    if date_col in df.columns:
        df = df.drop(columns=[date_col])
    df = df.set_index(idx).sort_index()
    # 生成完整工作日序列并 reindex
    full_idx = pd.bdate_range(df.index.min(), df.index.max())
    df = df.reindex(full_idx)
    df.index.name = date_col
    # 前向填充价格和成交额, 成交量填 0
    fill_cols = ["open", "high", "low", "close", "amount", "turnover"]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)
    # 标记被填充的行
    df["was_filled"] = df["close"].isna()  # ffill 后仍为 NaN 说明起始段无数据
    df["was_filled"] = df["was_filled"].fillna(False)
    # 去掉全 NaN 行 (非交易日)
    df = df.dropna(subset=["close"])
    df = df.reset_index().rename(columns={"index": date_col})
    return df


# ============================================================
# 一站式清洗入口
# ============================================================

def clean_stock_daily(df: pd.DataFrame, code: str, stock_name: str = "") -> pd.DataFrame:
    """个股日线一站式清洗: 标记 ST → 涨跌停 → 停牌 → 复牌 → 缺失值

    Args:
        df: 日线 DataFrame, 需含 eob/open/high/low/close/volume 列
        code: 股票代码 (6位或带前缀)
        stock_name: 股票名称 (用于 ST 判断)

    Returns:
        带 is_st/is_limit_up/is_limit_down/is_suspended/is_resume/was_filled 列的 DataFrame
    """
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    df = mark_st(df, stock_name)
    df = detect_limit_up_down(df, code, is_st=bool(df["is_st"].iloc[0]) if "is_st" in df else False)
    df = detect_suspension(df)
    df = detect_resume(df)
    # fill_missing 会重建索引, 放最后
    df = fill_missing(df, date_col="eob" if "eob" in df.columns else df.index.name or "eob")
    # fill_missing 之后重新跑涨跌停/停牌 (因为索引变了)
    df = detect_limit_up_down(df, code, is_st=bool(df["is_st"].iloc[0]) if "is_st" in df.columns and len(df) > 0 else False)
    df = detect_suspension(df)
    df = detect_resume(df)
    return df
