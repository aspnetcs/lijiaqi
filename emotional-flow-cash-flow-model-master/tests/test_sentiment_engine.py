# -*- coding: utf-8 -*-
"""
测试: 投资者情绪分析引擎
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
import pandas as pd

from sentiment_engine import (
    LexiconAnalyzer, MarketSentimentProxy, SentimentEngine, SentimentResult,
    _ema_smooth,
)


class TestLexiconAnalyzer:
    """词典法情感分析"""

    def setup_method(self):
        self.analyzer = LexiconAnalyzer()

    def test_positive_text(self):
        score = self.analyzer.analyze_text("大盘今天大幅上涨，市场一片繁荣")
        assert score > 0, f"Expected positive, got {score}"

    def test_negative_text(self):
        score = self.analyzer.analyze_text("市场暴跌，恐慌情绪蔓延，大面积跌停")
        assert score < 0, f"Expected negative, got {score}"

    def test_neutral_text(self):
        score = self.analyzer.analyze_text("今日沪深两市成交额达到一万亿元")
        assert abs(score) < 0.2, f"Expected neutral, got {score}"

    def test_negation_handling(self):
        """否定词翻转: "不好" 应比 "好" 更负面"""
        score_neg = self.analyzer.analyze_text("今天市场不是很好")
        score_pos = self.analyzer.analyze_text("今天市场很好")
        assert score_neg < score_pos, f"neg={score_neg}, pos={score_pos}"

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
        texts = [
            "暴涨", "暴跌", "涨停", "跌停", "牛市", "熊市",
            "大幅上涨利好买入", "恐慌性下跌利空卖出",
        ]
        scores = self.analyzer.analyze_batch(texts)
        for s in scores:
            assert -1.0 <= s <= 1.0, f"Score {s} out of range"

    def test_extract_keywords(self):
        texts = ["大盘上涨利好买入", "市场下跌利空卖出", "短期反弹可期"]
        keywords = self.analyzer.extract_keywords(texts, top_n=5)
        assert len(keywords) <= 5
        assert isinstance(keywords[0], tuple)


class TestMarketSentimentProxy:
    """市场情绪代理"""

    def test_compute_adr(self):
        proxy = MarketSentimentProxy()
        n = 30
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
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


class TestSentimentEngine:
    """主引擎"""

    def test_analyze_daily(self):
        from src.config import SystemConfig
        cfg = SystemConfig()
        engine = SentimentEngine(cfg)

        news = ["大盘上涨", "市场活跃", "资金流入"]
        dates = pd.date_range("2024-06-01", periods=10, freq="B")
        close = 3000 + np.cumsum(np.random.randn(10) * 10)
        market = pd.DataFrame({"close": close, "volume": np.ones(10) * 1e8}, index=dates)

        result = engine.analyze_daily("2024-06-01", news, market)
        assert isinstance(result, SentimentResult)
        assert -1.0 <= result.composite_sentiment <= 1.0
        assert isinstance(result.sentiment_label, str)

    def test_detect_extreme(self):
        from src.config import SystemConfig
        cfg = SystemConfig()
        engine = SentimentEngine(cfg)

        results = [
            SentimentResult("2024-01-01", 0.9, 0.8, 0.85, "bullish", True, 0.9, 30, []),
            SentimentResult("2024-01-02", 0.8, 0.7, 0.75, "bullish", True, 0.9, 25, []),
            SentimentResult("2024-01-03", 0.3, 0.2, 0.25, "bullish", False, 0.7, 20, []),
        ]
        events = engine.detect_extreme(results)
        assert len(events) >= 1

    def test_get_sentiment_series(self):
        from src.config import SystemConfig
        cfg = SystemConfig()
        engine = SentimentEngine(cfg)

        results = [
            SentimentResult("2024-01-01", 0.5, 0.3, 0.4, "bullish", False, 0.8, 10, []),
            SentimentResult("2024-01-02", -0.3, -0.2, -0.25, "bearish", False, 0.7, 12, []),
        ]
        series = engine.get_sentiment_series(results)
        assert len(series) == 2
        assert series.iloc[0] > series.iloc[1]
