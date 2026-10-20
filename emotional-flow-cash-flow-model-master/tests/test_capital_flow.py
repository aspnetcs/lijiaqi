# -*- coding: utf-8 -*-
"""
测试: 资金流分析器
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np

from capital_flow import (
    CMFCalculator, OBVCalculator, MicroFactorExtractor,
    SectorFlowAnalyzer, FactorCalculator, CapitalFlowAnalyzer,
)


# ============================================================
# Mock 数据
# ============================================================

def make_mock_daily(n=150):
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    trend = np.linspace(10, 15, n) + np.cumsum(np.random.randn(n) * 0.2)
    close = pd.Series(trend, index=dates)
    high = close + np.abs(np.random.randn(n) * 0.15)
    low = close - np.abs(np.random.randn(n) * 0.15)
    open_p = low + np.random.rand(n) * (high - low)
    volume = pd.Series(np.random.randint(500000, 5000000, n), index=dates)
    amount = close * volume * 0.1

    return pd.DataFrame({
        "open": open_p, "high": high, "low": low,
        "close": close, "volume": volume, "amount": amount,
    }, index=dates)


def make_mock_minute(n=240):
    np.random.seed(43)
    dates = pd.date_range("2024-06-01 09:30", periods=n, freq="1min")
    base = 10.0
    rets = np.random.randn(n) * 0.001
    close = pd.Series(base * np.exp(np.cumsum(rets)), index=dates)
    open_p = close.shift(1).fillna(base)
    high = close.abs() + np.abs(np.random.randn(n) * 0.02)
    low = close.abs() - np.abs(np.random.randn(n) * 0.02)
    volume = pd.Series(np.random.randint(1000, 100000, n), index=dates)

    return pd.DataFrame({
        "open": open_p, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)


# ============================================================
# CMFCalculator 测试
# ============================================================

class TestCMFCalculator:
    """CMF指标测试"""

    def test_compute_range(self):
        """CMF应在[-1, 1]之间"""
        df = make_mock_daily(100)
        cmf = CMFCalculator.compute(df)

        assert cmf is not None
        assert cmf.min() >= -1.0
        assert cmf.max() <= 1.0
        assert not cmf.isna().all()

    def test_compute_length(self):
        """输出长度与输入一致"""
        df = make_mock_daily(80)
        cmf = CMFCalculator.compute(df)
        assert len(cmf) == len(df)

    def test_turn_signal(self):
        """拐头信号计算"""
        df = make_mock_daily(80)
        turn = CMFCalculator.compute_turn_signal(df)

        assert turn is not None
        assert len(turn) == len(df)

    def test_edge_single_row(self):
        """单行数据边界"""
        df = make_mock_daily(3)
        cmf = CMFCalculator.compute(df)
        assert len(cmf) == 3


# ============================================================
# OBVCalculator 测试
# ============================================================

class TestOBVCalculator:
    """OBV指标测试"""

    def test_compute(self):
        df = make_mock_daily(60)
        obv = OBVCalculator.compute(df)

        assert obv is not None
        assert len(obv) == len(df)

    def test_increasing_with_bull_trend(self):
        """牛市趋势OBV应递增"""
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        # 持续上涨
        close = 10.0 + np.arange(50) * 0.2
        volume = pd.Series(1000000, index=dates)

        df = pd.DataFrame({"close": close, "volume": volume}, index=dates)
        obv = OBVCalculator.compute(df)

        assert obv.iloc[-1] > obv.iloc[0]


# ============================================================
# MicroFactorExtractor 测试
# ============================================================

class TestMicroFactorExtractor:
    """微观因子测试"""

    def test_extract_all(self):
        minute = make_mock_minute(240)
        factors = MicroFactorExtractor.extract_all(minute)

        assert len(factors) >= 5
        assert "obi" in factors
        assert "rv" in factors
        assert "amihud_intra" in factors
        assert "rv_kurt" in factors

    def test_insufficient_data(self):
        """数据不足时返回空"""
        minute = make_mock_minute(50)  # < 100条
        factors = MicroFactorExtractor.extract_all(minute)
        assert factors == {}

    def test_obi_range(self):
        """OBI应在[-1, 1]之间"""
        minute = make_mock_minute(300)
        factors = MicroFactorExtractor.extract_all(minute)
        obi = factors.get("obi", 999)
        assert -1.1 <= obi <= 1.1


# ============================================================
# FactorCalculator 测试
# ============================================================

class TestFactorCalculator:
    """因子计算器测试"""

    def test_compute_all(self):
        df = make_mock_daily(100)
        factors = FactorCalculator.compute_all(df)

        expected = ["cmf_20", "cmf_turn", "mom_20d", "vol_20d", "illiq", "vol_rs", "overnight_ret"]
        for f in expected:
            assert f in factors
            assert not pd.isna(factors[f])

    def test_insufficient_data(self):
        """数据不足60行返回空"""
        df = make_mock_daily(30)
        factors = FactorCalculator.compute_all(df)
        assert factors == {}


# ============================================================
# SectorFlowAnalyzer 测试
# ============================================================

class TestSectorFlowAnalyzer:
    """行业资金流测试"""

    def test_compute_scores(self):
        np.random.seed(44)
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        industries = ["电子", "银行", "食品饮料", "医药"]
        matrix = pd.DataFrame(
            np.random.randn(30, 4) * 0.03,
            index=dates, columns=industries
        )

        inflow, outflow, scores = SectorFlowAnalyzer.compute_sector_scores(
            matrix, dates[-1]
        )

        assert isinstance(inflow, list)
        assert isinstance(outflow, list)
        assert isinstance(scores, dict)
        assert len(scores) == len(industries)
