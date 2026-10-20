# -*- coding: utf-8 -*-
"""
财务数据读取器

读取 financial/ 下的:
  - daily_basic_pq/{code}_pe_ttm.parquet / {code}_pb.parquet  (历史PE/PB)
  - indicator_pq/{code}.parquet                              (财务分析指标)
  - yjbb_pq/yjbb.parquet                                     (全市场业绩报表)
"""
import os
import pandas as pd
from typing import Optional, Dict


class FinancialReader:
    def __init__(self, financial_dir: str):
        self.dir = financial_dir
        self.daily_basic_dir = os.path.join(financial_dir, "daily_basic_pq")
        self.indicator_dir = os.path.join(financial_dir, "indicator_pq")
        self.yjbb_dir = os.path.join(financial_dir, "yjbb_pq")

    def _read(self, path: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(path):
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def get_pe_pb(self, code: str, start: str = None, end: str = None) -> Optional[pd.DataFrame]:
        """获取个股历史 PE/PB (按日期对齐, pe/pb 起始日期可不同)"""
        pe = self._read(os.path.join(self.daily_basic_dir, f"{code}_pe_ttm.parquet"))
        pb = self._read(os.path.join(self.daily_basic_dir, f"{code}_pb.parquet"))
        if pe is None and pb is None:
            return None
        out = pd.DataFrame()
        if pe is not None:
            out["pe_ttm"] = pd.Series(pe["value"].values, index=pd.to_datetime(pe["date"]))
        if pb is not None:
            out["pb"] = pd.Series(pb["value"].values, index=pd.to_datetime(pb["date"]))
        out = out.sort_index()
        out.index = out.index.strftime("%Y-%m-%d")
        if start:
            out = out[out.index >= start]
        if end:
            out = out[out.index <= end]
        return out

    def get_indicator(self, code: str) -> Optional[pd.DataFrame]:
        """获取个股财务分析指标 (ROE/资产负债率/...)"""
        return self._read(os.path.join(self.indicator_dir, f"{code}.parquet"))

    def get_yjbb(self, code: str = None, report_date: str = None) -> Optional[pd.DataFrame]:
        """获取业绩报表 (可按代码/报告期过滤)"""
        df = self._read(os.path.join(self.yjbb_dir, "yjbb.parquet"))
        if df is None:
            return None
        if code and "股票代码" in df.columns:
            df = df[df["股票代码"].astype(str).str.zfill(6) == str(code).zfill(6)]
        if report_date and "report_date" in df.columns:
            df = df[df["report_date"] == report_date]
        return df.reset_index(drop=True)
