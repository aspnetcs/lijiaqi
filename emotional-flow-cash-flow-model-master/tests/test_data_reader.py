# -*- coding: utf-8 -*-
"""
测试: 数据读取层
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np
from io import StringIO

from data_reader import (
    StockDailyReader, MoneyFlowReader, MinuteKlineReader,
    IndexDataReader, DataReader, IndustryMapReader,
)


# ============================================================
# Mock 数据工厂
# ============================================================

def make_mock_daily(n_days=100):
    """构造模拟日线数据"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    close = 10.0 + np.cumsum(np.random.randn(n_days) * 0.15)
    high = close + np.abs(np.random.randn(n_days) * 0.1)
    low = close - np.abs(np.random.randn(n_days) * 0.1)
    open_p = low + np.random.rand(n_days) * (high - low)
    volume = np.random.randint(1000000, 10000000, n_days)
    amount = close * volume

    df = pd.DataFrame({
        "eob": dates,
        "open": open_p, "high": high, "low": low, "close": close,
        "volume": volume, "amount": amount,
    })
    return df


# ============================================================
# StockDailyReader 测试
# ============================================================

class TestStockDailyReader:
    """测试个股日线读取"""

    def test_load_single(self, tmp_path):
        """正常加载"""
        df = make_mock_daily(60)
        path = tmp_path / "SHSE.600000.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")

        reader = StockDailyReader(str(tmp_path))
        result = reader.load_single("SHSE.600000", "2024-01-01", "2024-06-01")

        assert result is not None
        assert len(result) > 0
        assert "close" in result.columns
        assert result.index.name == "eob"

    def test_file_not_found(self, tmp_path):
        """文件不存在"""
        reader = StockDailyReader(str(tmp_path))
        result = reader.load_single("SHSE.999999", "2024-01-01", "2024-06-01")
        assert result is None

    def test_date_filter(self, tmp_path):
        """日期过滤"""
        df = make_mock_daily(120)
        path = tmp_path / "SHSE.600000.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")

        reader = StockDailyReader(str(tmp_path))
        result = reader.load_single("SHSE.600000", "2024-03-01", "2024-04-30")

        assert result is not None
        start = pd.to_datetime("2024-03-01")
        end = pd.to_datetime("2024-04-30")
        assert result.index.min() >= start
        assert result.index.max() <= end

    def test_list_symbols(self, tmp_path):
        """列出股票列表"""
        for code in ["SHSE.600000", "SHSE.600036", "SZSE.000001",
                     "SHSE.688001", "SHSE.830001"]:
            df = make_mock_daily(30)
            df.to_csv(tmp_path / f"{code}.csv", index=False)

        reader = StockDailyReader(str(tmp_path))
        codes = reader.list_symbols(exclude_kcb=True, exclude_bj=True)

        assert "SHSE.600000" in codes
        assert "SHSE.688001" not in codes  # 科创板被排除
        assert "SHSE.830001" not in codes  # 北交所被排除


# ============================================================
# MoneyFlowReader 测试
# ============================================================

class TestMoneyFlowReader:
    """测试资金流读取"""

    def test_load_north_bound(self, tmp_path):
        """北向资金"""
        north_dir = tmp_path / "north"
        north_dir.mkdir()

        sh = pd.DataFrame({
            "日期": pd.date_range("2024-01-01", periods=60, freq="B"),
            "当日成交净买额": np.random.randn(60) * 10,
        })
        sz = pd.DataFrame({
            "日期": pd.date_range("2024-01-01", periods=60, freq="B"),
            "当日成交净买额": np.random.randn(60) * 8,
        })
        sh.to_csv(north_dir / "hsgt_沪.csv", index=False, encoding="utf-8-sig")
        sz.to_csv(north_dir / "hsgt_深.csv", index=False, encoding="utf-8-sig")

        reader = MoneyFlowReader(
            stock_flow_dir=str(tmp_path),
            sector_dir=str(tmp_path),
            north_dir=str(north_dir),
            margin_dir=str(tmp_path),
        )
        result = reader.load_north_bound("2024-01-01", "2024-03-01")

        assert result is not None
        assert "total_net" in result.columns


# ============================================================
# IndustryMapReader 测试
# ============================================================

class TestIndustryMapReader:
    """测试行业映射"""

    def test_load(self, tmp_path):
        import json
        mapping = {"SHSE.600000": "银行", "SZSE.000001": "银行", "SHSE.600519": "食品饮料"}
        map_path = tmp_path / "map.json"
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f)

        reader = IndustryMapReader(str(map_path))
        assert reader.get_industry("SHSE.600000") == "银行"
        assert reader.get_industry("SHSE.600519") == "食品饮料"
        assert reader.get_industry("NONEXIST") is None
