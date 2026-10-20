# -*- coding: utf-8 -*-
"""
测试: 情绪分析引擎 & 融合决策引擎
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
import pandas as pd

from sentiment_engine import LexiconAnalyzer, MarketSentimentProxy, SentimentEngine
from fusion_engine import (
    FusionEngine, WeightManager, ConflictResolver, SignalQualityChecker,
    MarketRegime, SignalDirection, SignalPack,
)


# ============================================================
# LexiconAnalyzer 测试
# ============================================================

class TestLexiconAnalyzer:
    """词典法情感分析"""

    def setup_method(self):
        self.analyzer = LexiconAnalyzer()

    def test_positive_text(self):
        score = self.analyzer.analyze_text("大盘今天大幅上涨，市场一片繁荣")
        assert score > 0

    def test_negative_text(self):
        score = self.analyzer.analyze_text("市场暴跌，恐慌情绪蔓延，大面积跌停")
        assert score < 0

    def test_neutral_text(self):
        score = self.analyzer.analyze_text("今日沪深两市成交额达到一万亿元")
        # 应该是中性，接近于0
        assert abs(score) < 0.2

    def test_negation_handling(self):
        """否定词翻转"""
        # "不是很好" -> "好"被否定 -> 负面
        score_neg = self.analyzer.analyze_text("今天市场不是很好")
        score_pos = self.analyzer.analyze_text("今天市场很好")
        assert score_neg < score_pos

    def test_empty_text(self):
        assert self.analyzer.analyze_text("") == 0.0
        assert self.analyzer.analyze_text("   ") == 0.0

    def test_batch_analysis(self):
        texts = ["市场大涨", "市场大跌", "市场平稳"]
        scores = self.analyzer.analyze_batch(texts)
        assert len(scores) == 3
        assert scores[0] > 0
        assert scores[1] < 0

    def test_score_range(self):
        """所有得分应在[-1, 1]之间"""
        texts = [
            "暴涨", "暴跌", "涨停", "跌停", "牛市", "熊市",
            "大幅上涨利好买入", "恐慌性下跌利空卖出",
        ]
        scores = self.analyzer.analyze_batch(texts)
        for s in scores:
            assert -1.0 <= s <= 1.0


# ============================================================
# MarketSentimentProxy 测试
# ============================================================

class TestMarketSentimentProxy:
    """市场情绪代理"""

    def test_compute_adr(self):
        proxy = MarketSentimentProxy()
        n = 30
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # 前15天涨，后15天跌
        close = pd.Series(
            [10 + i * 0.2 for i in range(15)] + [13 - i * 0.2 for i in range(15)],
            index=dates
        )
        market = pd.DataFrame({"close": close}, index=dates)
        adr = proxy.compute_adr(market)
        assert -1.0 <= adr <= 1.0

    def test_volume_anomaly(self):
        proxy = MarketSentimentProxy()
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        vol = pd.Series(np.random.randint(1, 10, 50) * 1000000, index=dates)
        market = pd.DataFrame({"volume": vol}, index=dates)
        anomaly = proxy.compute_volume_anomaly(market)
        assert -1.0 <= anomaly <= 1.0

    def test_compute_proxy(self):
        proxy = MarketSentimentProxy()
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        close = 10 + np.cumsum(np.random.randn(30) * 0.2)
        vol = np.random.randint(1, 10, 30) * 1000000
        market = pd.DataFrame({"close": close, "volume": vol}, index=dates)
        result = proxy.compute_proxy(market)
        assert -1.0 <= result <= 1.0


# ============================================================
# WeightManager & ConflictResolver 测试
# ============================================================

class TestWeightManager:
    """权重管理器"""

    def test_bull_weights(self):
        mgr = WeightManager()
        w = mgr.get_weights(MarketRegime.BULL)
        assert w["capital_flow"] > w["sentiment"]

    def test_sideways_weights(self):
        mgr = WeightManager()
        w = mgr.get_weights(MarketRegime.SIDEWAYS)
        assert w["sentiment"] == w["capital_flow"]  # 震荡市等权 (v2设计)

    def test_detect_regime(self):
        assert WeightManager.detect_regime(0.7, 0.2) == MarketRegime.BULL
        assert WeightManager.detect_regime(0.2, 0.1) == MarketRegime.BEAR
        assert WeightManager.detect_regime(0.4, 0.1) == MarketRegime.SIDEWAYS
        assert WeightManager.detect_regime(0.4, 0.8) == MarketRegime.EXTREME


class TestConflictResolver:
    """冲突仲裁"""

    def test_no_conflict_both_long(self):
        pack = SignalPack("", 0.5, SignalDirection.LONG,
                         0.3, SignalDirection.LONG, 0.5, MarketRegime.BULL)
        adj, conflict, desc = ConflictResolver.resolve(pack)
        assert not conflict
        assert adj == 1.0

    def test_conflict_opposite(self):
        pack = SignalPack("", 0.4, SignalDirection.LONG,
                         -0.3, SignalDirection.SHORT, 0.5, MarketRegime.SIDEWAYS)
        adj, conflict, desc = ConflictResolver.resolve(pack)
        assert conflict
        assert adj < 1.0


# ============================================================
# FusionEngine 测试
# ============================================================

class TestFusionEngine:
    """融合引擎"""

    def test_fuse_all_positive(self):
        from src.config import SystemConfig
        cfg = SystemConfig()
        engine = FusionEngine(cfg)

        result = engine.fuse(sentiment=0.6, flow=0.5, macro_position=0.7)
        assert result.position_ratio > 0.5
        assert result.composite_signal > 0.5
        assert not result.conflict

    def test_fuse_all_negative(self):
        from src.config import SystemConfig
        cfg = SystemConfig()
        engine = FusionEngine(cfg)

        result = engine.fuse(sentiment=-0.5, flow=-0.4, macro_position=0.2)
        assert result.position_ratio < 0.5

    def test_position_range(self):
        from src.config import SystemConfig
        cfg = SystemConfig()
        engine = FusionEngine(cfg)

        # 各种极端情况
        for sent in [-0.9, -0.5, 0.0, 0.5, 0.9]:
            for flow in [-0.9, -0.5, 0.0, 0.5, 0.9]:
                for macro in [0.1, 0.5, 0.9]:
                    result = engine.fuse(sentiment=sent, flow=flow, macro_position=macro)
                    assert 0.0 <= result.position_ratio <= 1.0
                    assert -1.0 <= result.composite_signal <= 1.0  # 方向信号, 范围[-1,1]
                    assert 0.0 <= result.confidence <= 1.0

    def test_batch_fuse(self):
        from src.config import SystemConfig
        cfg = SystemConfig()
        engine = FusionEngine(cfg)

        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        sentiment = pd.Series(np.sin(np.linspace(0, 2 * np.pi, 20)) * 0.5, index=dates)
        flow = pd.Series(np.cos(np.linspace(0, 2 * np.pi, 20)) * 0.4, index=dates)
        macro = pd.Series(0.5 + np.random.randn(20) * 0.1, index=dates).clip(0, 1)

        df = engine.batch_fuse(sentiment, flow, macro)
        assert len(df) > 0
        assert "position_ratio" in df.columns
        assert "regime" in df.columns
        assert "confidence" in df.columns
