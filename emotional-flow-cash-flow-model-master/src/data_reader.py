# -*- coding: utf-8 -*-
"""
数据读取层 (DataReader)

从本地CSV文件加载各类金融数据, 统一对外提供查询接口。
所有数据来自本地磁盘, 无在线依赖。

数据目录结构约定:
    D:\开发\训练数据\
    ├── stock_daily/        个股日线 ({code}.csv)
    ├── money_flow/         个股资金流 ({code}.csv)
    ├── sector_money_flow/  板块资金流面板 ({date}.csv)
    ├── north_bound/        北向资金 (hsgt_沪.csv, hsgt_深.csv)
    ├── margin_trade/       融资融券 (margin_sse.csv)
    ├── index_daily/        指数日线 (000300.csv)
    ├── minute_kline/       分钟K线 ({code}.csv)
    ├── news_data/          新闻舆情 (预留)
    └── sector_money_flow/_stock_industry_map.json  个股-行业映射
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime


# ============================================================
# 辅助函数
# ============================================================

def _parse_date_col(df: pd.DataFrame, col: str = "eob") -> pd.DataFrame:
    """统一处理日期列: 转datetime + 去时区 + 设为索引"""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
        if df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_localize(None)
    return df


def _filter_date(df: pd.DataFrame, start: str, end: str,
                 date_col: Optional[str] = None) -> pd.DataFrame:
    """按日期范围过滤 (date_col=None则使用index)"""
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    if date_col and date_col in df.columns:
        mask = (df[date_col] >= s) & (df[date_col] <= e)
        return df[mask]
    else:
        mask = (df.index >= s) & (df.index <= e)
        return df[mask]


def _safe_read_csv(path: str, encoding: str = "utf-8-sig") -> Optional[pd.DataFrame]:
    """安全读取CSV, 文件不存在时返回None"""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="gbk")
        except Exception:
            return None
    except Exception:
        return None


def _safe_read_parquet(path: str) -> Optional[pd.DataFrame]:
    """安全读取Parquet, 文件不存在或读取失败时返回None"""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


# ============================================================
# 个股日线读取器
# ============================================================

class StockDailyReader:
    """个股日线数据: stock_daily/{code}.csv 或 stock_daily_pq/{code}.parquet

    期望列: eob, open, high, low, close, volume, amount
    Parquet 优先 (含清洗标记列), CSV 降级 (缺标记列时内存中补清洗)
    """

    def __init__(self, data_dir: str, pq_dir: str = None):
        self.data_dir = data_dir
        self.pq_dir = pq_dir or data_dir  # 默认与 csv 同目录

    def load_single(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载单只股票日线

        Args:
            symbol: 如 'SHSE.600000' 或 '600000'
            start_date: 'YYYY-MM-DD'
            end_date:   'YYYY-MM-DD'

        Returns:
            DataFrame 以 eob 为索引, 或 None (数据不存在)
        """
        # 1. 优先 Parquet
        code = symbol.split(".")[-1] if "." in symbol else symbol
        pq_path = os.path.join(self.pq_dir, f"{code}.parquet")
        df = _safe_read_parquet(pq_path)
        # 2. 降级 CSV
        if df is None:
            path = os.path.join(self.data_dir, f"{symbol}.csv")
            df = _safe_read_csv(path)
            if df is None or "eob" not in df.columns:
                # 再试无前缀文件名
                path = os.path.join(self.data_dir, f"{code}.csv")
                df = _safe_read_csv(path)
            if df is None or "eob" not in df.columns:
                return None
            # 旧 CSV 无清洗标记列 → 内存中补跑清洗
            from src.data_cleaning import clean_stock_daily
            if "is_suspended" not in df.columns:
                df = clean_stock_daily(df, code)

        if df is None or "eob" not in df.columns:
            return None

        df = _parse_date_col(df, "eob")
        df = df.set_index("eob").sort_index()
        return _filter_date(df, start_date, end_date)

    def load_batch(self, symbols: List[str], start_date: str,
                   end_date: str) -> Dict[str, pd.DataFrame]:
        """批量加载, 返回 {symbol: DataFrame} (数据缺失的symbol不包含)"""
        result = {}
        for sym in symbols:
            df = self.load_single(sym, start_date, end_date)
            if df is not None and len(df) > 0:
                result[sym] = df
        return result

    def list_symbols(self, exclude_kcb: bool = True,
                     exclude_bj: bool = True) -> List[str]:
        """列出本地数据目录下所有股票代码 (扫描 parquet 和 csv 两个目录)

        返回文件名 stem (如 'SHSE.600000' 或 '600000'), 保持与旧版兼容.
        """
        seen = set()
        codes = []
        for d, ext in ((self.pq_dir, ".parquet"), (self.data_dir, ".csv")):
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if not f.endswith(ext) or f.startswith("_"):
                    continue
                code = f.replace(ext, "")
                base = code.split(".")[-1] if "." in code else code
                if not base.isdigit() or len(base) != 6:
                    continue
                if exclude_kcb and (base.startswith("688") or base.startswith("689")):
                    continue
                if exclude_bj and (base.startswith("4") or base.startswith("8")):
                    continue
                if base in seen:
                    continue
                seen.add(base)
                codes.append(code)
        return codes


# ============================================================
# 资金流读取器
# ============================================================

class MoneyFlowReader:
    """资金流数据读取

    个股资金流: money_flow/{code}.csv
        期望列: eob, cmf, obv, net_inflow_daily
    板块资金流: sector_money_flow/{date}.csv
        期望列: trade_date, industry, cmf_mean  (每行一个行业)
    北向资金: north_bound/hsgt_沪.csv, hsgt_深.csv
        期望列: 日期, 当日成交净买额
    融资融券: margin_trade/margin_sse.csv
        期望列: 信用交易日期, 融资融券余额, 融资余额
    """

    def __init__(self, stock_flow_dir: str, sector_dir: str,
                 north_dir: str, margin_dir: str,
                 north_pq_dir: str = None, margin_pq_dir: str = None):
        self.stock_flow_dir = stock_flow_dir
        self.sector_dir = sector_dir
        self.north_dir = north_dir
        self.margin_dir = margin_dir
        self.north_pq_dir = north_pq_dir
        self.margin_pq_dir = margin_pq_dir

    # ---- 个股资金流 ----

    def load_stock_cmf(self, symbol: str, start_date: str,
                       end_date: str) -> Optional[pd.DataFrame]:
        """加载个股CMF/OBV数据"""
        path = os.path.join(self.stock_flow_dir, f"{symbol}.csv")
        df = _safe_read_csv(path)
        if df is None or "eob" not in df.columns:
            return None

        df = _parse_date_col(df, "eob")
        df = df.set_index("eob").sort_index()

        # 只保留资金流相关列
        keep_cols = []
        for c in ["cmf", "obv", "net_inflow_daily", "net_inflow"]:
            if c in df.columns:
                keep_cols.append(c)
        if not keep_cols:
            return None

        result = df[keep_cols]
        return _filter_date(result, start_date, end_date)

    # ---- 板块资金流 ----

    def load_sector_flow(self, start_date: str, end_date: str) -> pd.DataFrame:
        """构建行业资金流面板矩阵

        扫描 sector_money_flow/ 下所有 {date}.csv,
        每行一个行业, 列: trade_date, industry, cmf_mean

        Returns:
            DataFrame: index=日期, columns=行业, values=cmf_mean
        """
        if not os.path.isdir(self.sector_dir):
            return pd.DataFrame()

        files = sorted([f for f in os.listdir(self.sector_dir)
                       if f.endswith(".csv") and not f.startswith("_")])

        sector_data: Dict[str, List[Tuple[str, float]]] = {}
        all_dates = []

        for fname in files:
            date_str = fname.replace(".csv", "")
            dt = pd.to_datetime(date_str)
            if dt < pd.to_datetime(start_date) or dt > pd.to_datetime(end_date):
                continue

            df = _safe_read_csv(os.path.join(self.sector_dir, fname))
            if df is None:
                continue

            # 适配不同列名
            date_col = None
            for c in ["trade_date", "date", "eob"]:
                if c in df.columns:
                    date_col = c
                    break
            ind_col = None
            for c in ["industry", "sector", "name"]:
                if c in df.columns:
                    ind_col = c
                    break
            cmf_col = None
            for c in ["cmf_mean", "cmf", "net_flow"]:
                if c in df.columns:
                    cmf_col = c
                    break

            if ind_col is None or cmf_col is None:
                continue

            all_dates.append(date_str)

            for _, row in df.iterrows():
                ind = str(row[ind_col])
                val = float(row[cmf_col]) if pd.notna(row[cmf_col]) else 0.0
                if ind not in sector_data:
                    sector_data[ind] = []
                sector_data[ind].append((date_str, val))

        if not all_dates or not sector_data:
            return pd.DataFrame()

        dates = sorted(set(all_dates))
        industries = sorted(sector_data.keys())
        matrix = pd.DataFrame(index=pd.to_datetime(dates),
                             columns=industries, dtype=float)

        for ind, rows in sector_data.items():
            for d, v in rows:
                try:
                    matrix.loc[pd.to_datetime(d), ind] = v
                except Exception:
                    pass

        return matrix.sort_index()

    def load_sector_flow_latest(self) -> Optional[pd.DataFrame]:
        """加载最近一个交易日的板块资金流明细"""
        if not os.path.isdir(self.sector_dir):
            return None

        files = sorted([f for f in os.listdir(self.sector_dir)
                       if f.endswith(".csv") and not f.startswith("_")])
        if not files:
            return None

        latest = files[-1]
        return _safe_read_csv(os.path.join(self.sector_dir, latest))

    # ---- 北向资金 ----

    def load_north_bound(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载北向资金净买额 (沪+深合计)"""
        # 优先 Parquet (已合并沪+深)
        if self.north_pq_dir:
            pq_path = os.path.join(self.north_pq_dir, "north_bound.parquet")
            df = _safe_read_parquet(pq_path)
            if df is not None and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                if "total_net" not in df.columns:
                    df["total_net"] = df.get("net_buy_sh", 0).fillna(0) + df.get("net_buy_sz", 0).fillna(0)
                return _filter_date(df, start_date, end_date)

        sh_path = os.path.join(self.north_dir, "hsgt_沪.csv")
        sz_path = os.path.join(self.north_dir, "hsgt_深.csv")

        df_sh = _safe_read_csv(sh_path)
        df_sz = _safe_read_csv(sz_path)
        if df_sh is None or df_sz is None:
            return None

        for df in [df_sh, df_sz]:
            # 适配列名
            if "日期" in df.columns:
                df.rename(columns={"日期": "date"}, inplace=True)
            if "当日成交净买额" in df.columns:
                df.rename(columns={"当日成交净买额": "net_buy"}, inplace=True)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

        merged = pd.merge(
            df_sh[["date", "net_buy"]],
            df_sz[["date", "net_buy"]],
            on="date", how="outer", suffixes=("_sh", "_sz")
        )
        merged.fillna(0, inplace=True)
        merged["total_net"] = merged["net_buy_sh"] + merged["net_buy_sz"]
        merged = merged.set_index("date").sort_index()

        return _filter_date(merged, start_date, end_date)

    # ---- 融资融券 ----

    def load_margin_trade(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载融资融券余额

        优先读取 margin_trade_pq/margin_trade.parquet
        其次读取 margin_gta.csv (国泰安日聚合数据, 列: date, mtbalance)
        最后读取 margin_sse.csv (旧格式, 列: 信用交易日期, 融资融券余额)
        """
        # 优先 Parquet
        if self.margin_pq_dir:
            pq_path = os.path.join(self.margin_pq_dir, "margin_trade.parquet")
            df = _safe_read_parquet(pq_path)
            if df is not None and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                keep = [c for c in ["balance", "fin_balance"] if c in df.columns]
                if keep:
                    return _filter_date(df[keep], start_date, end_date)

        # 方式一: 国泰安 margin_gta.csv
        gta_path = os.path.join(self.margin_dir, "margin_gta.csv")
        df = _safe_read_csv(gta_path)
        if df is not None and "date" in df.columns and "mtbalance" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.rename(columns={"mtbalance": "balance"})
            df = df.set_index("date").sort_index()
            return _filter_date(df, start_date, end_date)

        # 方式二: 旧格式 margin_sse.csv
        path = os.path.join(self.margin_dir, "margin_sse.csv")
        df = _safe_read_csv(path)
        if df is None:
            return None

        col_map = {}
        for c in df.columns:
            if "信用交易日期" in str(c):
                col_map[c] = "date"
            elif "融资融券余额" in str(c):
                col_map[c] = "balance"
            elif "融资余额" in str(c):
                col_map[c] = "fin_balance"
        if col_map:
            df.rename(columns=col_map, inplace=True)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"].astype(str))
            df = df.set_index("date").sort_index()
            keep = [c for c in ["balance", "fin_balance"] if c in df.columns]
            if keep:
                return _filter_date(df[keep], start_date, end_date)

        return None


# ============================================================
# 分钟线读取器
# ============================================================

class MinuteKlineReader:
    """分钟K线: minute_kline/{code}.csv

    期望列: eob, open, high, low, close, volume
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_minute_kline(self, symbol: str, trade_date: str) -> Optional[pd.DataFrame]:
        """加载单只股票某日的分钟K线"""
        path = os.path.join(self.data_dir, f"{symbol}.csv")
        df = _safe_read_csv(path)
        if df is None or "eob" not in df.columns:
            return None

        df = _parse_date_col(df, "eob")

        # 筛选当日数据
        date = pd.to_datetime(trade_date)
        next_date = date + pd.Timedelta(days=1)
        df_day = df[(df["eob"] >= date) & (df["eob"] < next_date)]

        if len(df_day) == 0:
            return None

        return df_day.set_index("eob").sort_index()

    def load_multi_day(self, symbol: str, trade_dates: List[str]) -> Dict[str, pd.DataFrame]:
        """加载多日分钟线"""
        result = {}
        for d in trade_dates:
            df = self.load_minute_kline(symbol, d)
            if df is not None:
                result[d] = df
        return result


# ============================================================
# 指数/新闻/行业映射读取器
# ============================================================

class IndexDataReader:
    """指数日线: index_daily/{code}.csv 或 index_daily_pq/{code}.parquet"""

    def __init__(self, data_dir: str, pq_dir: str = None):
        self.data_dir = data_dir
        self.pq_dir = pq_dir or data_dir

    def load_index(self, code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载指数日线 (如 '000300' 代表沪深300)"""
        # 优先 parquet
        pq_path = os.path.join(self.pq_dir, f"{code}.parquet")
        df = _safe_read_parquet(pq_path)
        if df is None:
            path = os.path.join(self.data_dir, f"{code}.csv")
            df = _safe_read_csv(path)
        if df is None or "eob" not in df.columns:
            return None

        df = _parse_date_col(df, "eob")
        df = df.set_index("eob").sort_index()
        return _filter_date(df, start_date, end_date)


class NewsDataReader:
    """新闻/舆情数据 (预留)

    期望格式: CSV, 列包含 date, source, title, content
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_news(self, start_date: str, end_date: str,
                  sources: Optional[List[str]] = None) -> pd.DataFrame:
        """加载新闻数据"""
        # 预留: 扫描 news_data/ 下所有CSV
        if not os.path.isdir(self.data_dir):
            return pd.DataFrame()

        all_dfs = []
        for f in os.listdir(self.data_dir):
            if not f.endswith(".csv"):
                continue
            df = _safe_read_csv(os.path.join(self.data_dir, f))
            if df is None:
                continue
            # 尝试找日期列
            for date_col in ["date", "pub_time", "eob"]:
                if date_col in df.columns:
                    df = _parse_date_col(df, date_col)
                    s = pd.to_datetime(start_date)
                    e = pd.to_datetime(end_date)
                    df = df[(df[date_col] >= s) & (df[date_col] <= e)]
                    break
            if len(df) > 0:
                # 来源过滤
                if sources:
                    for src_col in ["source", "src", "type"]:
                        if src_col in df.columns:
                            df = df[df[src_col].isin(sources)]
                            break
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)


class MeetingMinutesReader:
    """调研纪要数据读取器

    数据源: 国泰安 IRM_MEETINGMINUTES (上市公司调研纪要)
    - 每行: ReportID, Symbol, ReportDate, Rank, Question, Answer
    - 预处理后: {date_str: [text1, text2, ...]} pickle

    优势: 覆盖 ~20.3万条/年, 每个交易日平均 ~680条, 远超传统新闻源
    """

    def __init__(self, index_path: str):
        """
        Args:
            index_path: 预处理后的日期索引 pickle 路径
                        (由 scripts/preprocess_meetings.py 生成)
        """
        self.index_path = index_path
        self._index: Optional[Dict[str, List[str]]] = None

    def _load_index(self) -> Dict[str, List[str]]:
        """延迟加载日期索引 (一次性加载, ~400MB)"""
        if self._index is not None:
            return self._index

        if not os.path.exists(self.index_path):
            print(f"[WARNING] Meeting minutes index not found: {self.index_path}")
            self._index = {}
            return self._index

        print(f"[INFO] Loading meeting minutes index from {self.index_path} ...")
        with open(self.index_path, 'rb') as f:
            self._index = pickle.load(f)
        print(f"[INFO] Loaded {len(self._index)} dates, "
              f"{sum(len(v) for v in self._index.values())} total texts")
        return self._index

    def get_texts_by_date(self, date_str: str) -> List[str]:
        """获取某一天的所有调研纪要文本

        Args:
            date_str: 'YYYY-MM-DD'

        Returns:
            文本列表 (Question + Answer 拼接)
        """
        index = self._load_index()
        return index.get(date_str, [])

    def get_texts_by_period(self, start_date: str, end_date: str,
                            max_texts: int = 5000) -> List[str]:
        """获取一段时间内的所有调研纪要文本

        Args:
            start_date: 'YYYY-MM-DD'
            end_date:   'YYYY-MM-DD'
            max_texts:  最大返回文本数 (防止每日文本过多导致性能问题)

        Returns:
            文本列表 (按日期从早到晚排列)
        """
        index = self._load_index()
        texts = []
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        for date_str in sorted(index.keys()):
            dt = pd.to_datetime(date_str)
            if dt < start:
                continue
            if dt > end:
                break
            day_texts = index[date_str]
            texts.extend(day_texts)
            if len(texts) >= max_texts:
                break

        return texts[:max_texts]

    def get_texts_recent(self, date_str: str, lookback_days: int = 7,
                         max_texts: int = 3000) -> List[str]:
        """获取指定日期之前最近 N 天的调研纪要文本

        用于在回测调仓日获取近期舆情信号。

        Args:
            date_str:     目标日期 'YYYY-MM-DD'
            lookback_days: 回看天数
            max_texts:    最大文本数

        Returns:
            文本列表
        """
        end = pd.to_datetime(date_str)
        start = end - pd.Timedelta(days=lookback_days)
        return self.get_texts_by_period(
            start.strftime('%Y-%m-%d'),
            end.strftime('%Y-%m-%d'),
            max_texts=max_texts
        )


class IndustryMapReader:
    """个股-行业映射: sector_money_flow/_stock_industry_map.json"""

    def __init__(self, map_path: str):
        self.map_path = map_path
        self._mapping: Optional[Dict[str, str]] = None

    def load(self) -> Dict[str, str]:
        """加载映射表 {stock_code: industry_name}

        兼容两种格式:
            - 旧格式 (28行业): {"000001": "银行", ...}
            - SW2格式 (70+行业): {"000001": {"industry_name": "半导体", "industry_code": "电子—半导体"}, ...}
        """
        if self._mapping is not None:
            return self._mapping

        if os.path.exists(self.map_path):
            with open(self.map_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            # 检测格式: dict值 → SW2, str值 → 旧格式
            sample_val = list(raw.values())[0] if raw else ""
            if isinstance(sample_val, dict):
                self._mapping = {
                    k: v.get("industry_name", str(v))
                    for k, v in raw.items()
                }
            else:
                self._mapping = raw
        else:
            self._mapping = {}

        return self._mapping

    def get_industry(self, symbol: str) -> Optional[str]:
        """查询单只股票的行业"""
        mapping = self.load()
        return mapping.get(symbol, None)

    def get_stocks_in_industry(self, industry: str) -> List[str]:
        """查询某行业的所有股票"""
        mapping = self.load()
        return [s for s, ind in mapping.items() if ind == industry]


# ============================================================
# 主数据读取器
# ============================================================

class DataReader:
    """统一数据读取入口

    聚合所有子Reader, 提供一致的查询接口。

    使用示例:
        >>> reader = DataReader(config)
        >>> df = reader.get_stock_daily('SHSE.600000', '2024-01-01', '2024-12-31')
        >>> north = reader.get_north_bound('2024-01-01', '2024-12-31')
    """

    def __init__(self, config):
        """
        Args:
            config: SystemConfig 对象 (from src.config)
        """
        cfg = config.data
        self.config = config

        self.stock = StockDailyReader(cfg.stock_daily_dir, pq_dir=cfg.stock_daily_pq_dir)
        self.money = MoneyFlowReader(
            stock_flow_dir=cfg.money_flow_dir,
            sector_dir=cfg.sector_flow_dir,
            north_dir=cfg.north_bound_dir,
            margin_dir=cfg.margin_trade_dir,
            north_pq_dir=cfg.north_bound_pq_dir,
            margin_pq_dir=cfg.margin_trade_pq_dir,
        )
        self.minute = MinuteKlineReader(cfg.minute_kline_dir)
        self.index = IndexDataReader(cfg.index_daily_dir, pq_dir=cfg.index_daily_pq_dir)
        self.news = NewsDataReader(cfg.news_data_dir)
        # 财务与成分股读取器
        from src.financial_reader import FinancialReader
        from src.constituents_reader import ConstituentsReader
        self.financial = FinancialReader(cfg.financial_dir)
        self.constituents = ConstituentsReader(cfg.constituents_dir,
                                               tushare_token=getattr(config, "tushare_token", ""))
        # 优先使用申万二级 (70+行业) 映射, 不存在则回退到一级 (28行业)
        sw2_path = os.path.join(cfg.sector_flow_dir, "_stock_industry_map_sw2.json")
        old_path = os.path.join(cfg.sector_flow_dir, "_stock_industry_map.json")
        if os.path.exists(sw2_path):
            print(f"  [INFO] 使用申万二级行业映射: {sw2_path}")
            self.industry_map = IndustryMapReader(sw2_path)
        else:
            print(f"  [INFO] 使用申万一级行业映射 (运行 scripts/build_sw2_industry_map.py 升级到二级)")
            self.industry_map = IndustryMapReader(old_path)
        self.meetings = MeetingMinutesReader(cfg.meetings_index_path)

    # ---- 日线 ----

    def get_stock_daily(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        return self.stock.load_single(symbol, start, end)

    def get_batch_daily(self, symbols: List[str], start: str,
                        end: str) -> Dict[str, pd.DataFrame]:
        return self.stock.load_batch(symbols, start, end)

    def get_stock_list(self, exclude_kcb: bool = True,
                       exclude_bj: bool = True) -> List[str]:
        return self.stock.list_symbols(exclude_kcb, exclude_bj)

    def get_hs300_constituents(self, date: str = None) -> List[str]:
        """获取沪深300成分股 (有 Tushare 历史权重则按时点取, 否则用当前成分)"""
        if date is None:
            date = pd.Timestamp.now().strftime("%Y-%m-%d")
        return self.constituents.get_hs300(date)

    def get_stock_daily_basic(self, symbol: str, start: str = None,
                              end: str = None) -> Optional[pd.DataFrame]:
        """获取个股历史 PE/PB (来自 financial/daily_basic_pq/)"""
        code = symbol.split(".")[-1] if "." in symbol else symbol
        return self.financial.get_pe_pb(code, start, end)

    # ---- 资金流 ----

    def get_stock_flow(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        return self.money.load_stock_cmf(symbol, start, end)

    def get_sector_flow_matrix(self, start: str, end: str) -> pd.DataFrame:
        return self.money.load_sector_flow(start, end)

    def get_north_bound(self, start: str, end: str) -> Optional[pd.DataFrame]:
        return self.money.load_north_bound(start, end)

    def get_margin_trade(self, start: str, end: str) -> Optional[pd.DataFrame]:
        return self.money.load_margin_trade(start, end)

    # ---- 分钟线 ----

    def get_minute_kline(self, symbol: str, trade_date: str) -> Optional[pd.DataFrame]:
        return self.minute.load_minute_kline(symbol, trade_date)

    # ---- 指数 ----

    def get_index(self, code: str, start: str, end: str) -> Optional[pd.DataFrame]:
        return self.index.load_index(code, start, end)

    # ---- 新闻 ----

    def get_news(self, start: str, end: str,
                 sources: Optional[List[str]] = None) -> pd.DataFrame:
        return self.news.load_news(start, end, sources)

    # ---- 行业映射 ----

    def get_industry(self, symbol: str) -> Optional[str]:
        return self.industry_map.get_industry(symbol)

    def get_stocks_in_industry(self, industry: str) -> List[str]:
        return self.industry_map.get_stocks_in_industry(industry)

    # ---- 调研纪要 ----

    def get_meeting_minutes(self, start: str, end: str,
                            max_texts: int = 5000) -> List[str]:
        return self.meetings.get_texts_by_period(start, end, max_texts)

    def get_recent_minutes(self, date: str, lookback_days: int = 7,
                           max_texts: int = 3000) -> List[str]:
        return self.meetings.get_texts_recent(date, lookback_days, max_texts)

    # ---- 合并数据 (日线 + 资金流) ----

    def get_merged_data(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """合并日K线和资金流数据

        与V2的 load_stock_data() 逻辑一致:
        将 stock_daily/{code}.csv 和 money_flow/{code}.csv 按日期合并
        """
        kline = self.get_stock_daily(symbol, start, end)
        flow = self.get_stock_flow(symbol, start, end)

        if kline is None:
            return None

        if flow is not None:
            merged = kline.join(flow, how="left")
        else:
            merged = kline.copy()

        # 填充资金流缺失值为0
        for col in merged.columns:
            if col not in ["open", "high", "low", "close", "volume", "amount"]:
                merged[col] = merged[col].fillna(0)

        return merged
