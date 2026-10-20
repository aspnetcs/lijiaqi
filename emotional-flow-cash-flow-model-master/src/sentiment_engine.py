# -*- coding: utf-8 -*-
"""
投资者情绪分析引擎 (SentimentEngine)

双通道情绪分析:
    Channel A -- 词典规则法 (LexiconAnalyzer)
        基于中文金融情感词典, 对新闻/舆情文本进行情感打分。
        支持: 正面词、负面词、程度副词、否定词处理

    Channel B -- 市场隐含情绪 (MarketSentimentProxy)
        基于行情数据构建情绪代理:
        - 涨跌比 (ADR)
        - 涨停比
        - 成交量异常度
        - 波动率偏度

融合:
    composite = alpha * text_sentiment + (1-alpha) * market_sentiment

学术参考:
    - 混合新闻情绪引擎 (ArXiv 2606.03457)
    - 分析师情绪->波动率 (ArXiv 2606.01077)
    - 雪球处置效应 (ArXiv 2605.05814)
"""

import math
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

# jieba 分词 (无GPU需求)
try:
    import jieba
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False
    print("[WARNING] jieba 未安装, 将使用简单字符级分析。安装: pip install jieba")


# ============================================================
# 内置金融情感词典
# ============================================================

# 正面词 (看涨/乐观)
_POSITIVE_WORDS = {
    "上涨", "大涨", "暴涨", "拉升", "走高", "反弹", "突破",
    "利好", "增长", "盈利", "买入", "加仓", "牛市", "繁荣",
    "领涨", "涨停", "新高", "放量", "活跃", "强势", "爆发",
    "逆势", "抄底", "翻红", "收红", "飘红", "普涨", "回暖",
    "增值", "分红", "改善", "优化", "超预期", "稳增", "向好",
    "景气", "扩张", "抢筹", "净流入", "增持", "回购", "做多",
    "涨", "好", "看好", "看涨", "走强", "乐观",
}

# 负面词 (看跌/悲观)
_NEGATIVE_WORDS = {
    "下跌", "大跌", "暴跌", "走低", "下挫", "回调", "破位",
    "利空", "亏损", "卖出", "减仓", "熊市", "衰退", "低迷",
    "领跌", "跌停", "新低", "缩量", "疲软", "弱势", "崩盘",
    "跳水", "收阴", "飘绿", "普跌", "阴跌", "深跌", "加速下跌",
    "减持", "套现", "净流出", "踩踏", "恐慌", "恶化", "下滑",
    "拖累", "压力", "承压", "风暴", "危机", "退市",
    "跌", "坏", "看跌", "走弱", "悲观",
}

# 否定词 (翻转极性)
_NEGATION_WORDS = {
    "不", "不是", "不会", "不能", "不要", "不大",
    "没", "没有", "无", "非", "未", "别", "莫", "勿",
    "难以", "难以避免", "不至于", "不至于说",
}

# 程度副词及其强度系数
_DEGREE_WORDS = {
    "极度": 2.0, "极其": 2.0, "最为": 2.0, "非常": 1.8,
    "很": 1.5, "十分": 1.5, "相当": 1.5, "特别": 1.5,
    "尤其": 1.5, "大幅": 1.5, "大幅上涨": 1.5, "大幅下跌": 1.5,
    "显著": 1.4, "明显": 1.3, "较为": 1.2, "比较": 1.1,
    "稍微": 0.7, "略": 0.7, "略微": 0.7, "微": 0.5,
    "有些": 0.8, "有点": 0.8,
}


# ============================================================
# 结果数据类型
# ============================================================

@dataclass
class SentimentResult:
    """日度情绪分析结果"""
    date: str
    text_sentiment: float           # 文本情绪 [-1, 1]
    market_sentiment: float         # 市场隐含情绪 [-1, 1]
    composite_sentiment: float      # 融合情绪 [-1, 1]
    sentiment_label: str            # 标签: bullish/neutral/bearish
    extreme_flag: bool              # 极端情绪标记
    confidence: float               # 置信度 [0, 1]
    news_count: int                 # 当日新闻数
    top_keywords: List[Tuple[str, int]]  # 高频词 TOP10


# ============================================================
# 词典法情感分析器
# ============================================================

class LexiconAnalyzer:
    """词典法文本情感分析器

    算法流程:
        1. jieba分词
        2. 遍历词序列, 匹配情感词典
        3. 向前扫描窗口内的否定词 (默认窗口=3), 遇否定翻转极性
        4. 向前扫描程度副词, 调整情感强度
        5. 累加加权情感得分
        6. 归一化到 [-1, 1]: score = sum(pos_weighted) / (sum(pos_count) + sum(neg_count))

    示例:
        "大盘今天大幅上涨" → 1个正面词"上涨", 程度"大幅"(x1.5) → 得分偏高
        "市场不是很好"     → "好"是正面, "不"否定翻转 → 得分偏低
    """

    def __init__(self, negation_window: int = 3):
        """
        Args:
            negation_window: 否定词扫描窗口 (词数)
        """
        self.positive = _POSITIVE_WORDS
        self.negative = _NEGATIVE_WORDS
        self.negation = _NEGATION_WORDS
        self.degree = _DEGREE_WORDS
        self.negation_window = negation_window

    def tokenize(self, text: str) -> List[str]:
        """分词"""
        if not text or not isinstance(text, str):
            return []
        if _JIEBA_AVAILABLE:
            return list(jieba.cut(text.strip()))
        else:
            # 降级: 按标点简单切分
            return text.strip().replace(",", " ").replace("，", " ").replace("。", " ").split()

    def analyze_text(self, text: str) -> float:
        """分析单条文本的情感得分

        Returns:
            情感得分 [-1, 1], 正=看涨, 负=看跌, 0=中性
        """
        if not text or not isinstance(text, str) or len(text.strip()) < 2:
            return 0.0

        tokens = self.tokenize(text)
        if not tokens:
            return 0.0

        pos_weighted = 0.0
        neg_weighted = 0.0
        pos_count = 0
        neg_count = 0

        for i, token in enumerate(tokens):
            is_pos = token in self.positive
            is_neg = token in self.negative

            if not is_pos and not is_neg:
                continue

            # 检查否定词 (向前扫描)
            is_negated = False
            for j in range(max(0, i - self.negation_window), i):
                if tokens[j] in self.negation:
                    is_negated = True
                    break

            # 检查程度副词 (向前扫描)
            degree_mult = 1.0
            for j in range(max(0, i - self.negation_window), i):
                if tokens[j] in self.degree:
                    degree_mult = self.degree[tokens[j]]
                    break

            # 累加
            weight = 1.0 * degree_mult
            if is_pos:
                if is_negated:
                    neg_weighted += weight
                    neg_count += 1
                else:
                    pos_weighted += weight
                    pos_count += 1
            elif is_neg:
                if is_negated:
                    pos_weighted += weight
                    pos_count += 1
                else:
                    neg_weighted += weight
                    neg_count += 1

        # 归一化
        total_count = pos_count + neg_count
        if total_count == 0:
            return 0.0

        raw_score = (pos_weighted - neg_weighted) / total_count
        return float(np.clip(raw_score, -1.0, 1.0))

    def analyze_batch(self, texts: List[str]) -> List[float]:
        """批量分析"""
        return [self.analyze_text(t) for t in texts]

    def extract_keywords(self, texts: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """提取高频关键词 (仅统计情感词)"""
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            for t in tokens:
                if t in self.positive or t in self.negative:
                    counter[t] += 1
        return counter.most_common(top_n)


# ============================================================
# 市场隐含情绪代理
# ============================================================

class MarketSentimentProxy:
    """市场隐含情绪代理指标

    基于公开行情数据计算情绪代理, 不依赖文本。
    适合在新闻缺失日作为补充信号。

    指标:
        1. ADR (涨跌比)
        2. 成交量异常度
        3. 波动率偏度
        4. 综合情绪代理
    """

    @staticmethod
    def compute_adr(market_data: pd.DataFrame) -> float:
        """计算涨跌比 (Advance-Decline Ratio)

        ADR = 上涨家数 / (上涨家数 + 下跌家数)
        需要 market_data 包含 'advance'/'decline' 列 或 从个股涨幅推算

        简化版: 用指数日内涨跌方向 + 幅度 近似
        """
        if market_data is None or len(market_data) < 5:
            return 0.5

        if "close" in market_data.columns:
            ret = market_data["close"].pct_change().dropna()
            if len(ret) < 5:
                return 0.5
            up_ratio = (ret > 0).sum() / len(ret)
            # 映射: 0.3~0.7 的 ADR 对应 [-1, 1] 的情绪
            return float(2 * up_ratio - 1)

        return 0.5

    @staticmethod
    def compute_volume_anomaly(market_data: pd.DataFrame, lookback: int = 20) -> float:
        """成交量异常度

        anomaly = (当前量 - MA20) / MA20
        正值 = 放量 (情绪活跃), 负值 = 缩量 (情绪低迷)
        归一化到 [-1, 1]
        """
        if market_data is None or "volume" not in market_data.columns:
            return 0.0
        vol = market_data["volume"]
        if len(vol) < lookback:
            return 0.0
        ma = vol.rolling(lookback).mean()
        anomaly = (vol.iloc[-1] - ma.iloc[-1]) / (ma.iloc[-1] + 1e-8)
        return float(np.clip(anomaly / 2, -1, 1))  # 缩放到[-1,1]

    @staticmethod
    def compute_return_skew(market_data: pd.DataFrame, window: int = 20) -> float:
        """收益率偏度 (情绪方向代理)

        负偏度 = 恐慌式下跌 (大负收益集中)
        正偏度 = 乐观式上涨
        """
        if market_data is None or "close" not in market_data.columns:
            return 0.0
        ret = market_data["close"].pct_change().dropna().tail(window)
        if len(ret) < 10:
            return 0.0
        skew = float(ret.skew())
        return float(np.clip(skew / 3, -1, 1))

    def compute_proxy(self, market_data: pd.DataFrame) -> float:
        """综合计算市场隐含情绪 [-1, 1]

        加权: ADR(0.4) + 量异常(0.3) + 偏度(0.3)
        """
        if market_data is None or len(market_data) < 5:
            return 0.0

        adr = self.compute_adr(market_data)
        vol_anomaly = self.compute_volume_anomaly(market_data)
        skew = self.compute_return_skew(market_data)

        proxy = 0.4 * adr + 0.3 * vol_anomaly + 0.3 * skew
        return float(np.clip(proxy, -1, 1))


# ============================================================
# 情绪分析主引擎
# ============================================================

class SentimentEngine:
    """投资者情绪分析主引擎

    整合词典法文本分析和市场隐含情绪代理, 输出日度情绪指数。

    使用示例:
        >>> engine = SentimentEngine(config)
        >>> results = engine.analyze_period('2024-01-01', '2024-06-30', index_df, news_df)
        >>> for r in results:
        ...     print(f"{r.date}: {r.composite_sentiment:.3f} [{r.sentiment_label}]")
    """

    def __init__(self, config):
        self.config = config
        self.sent_cfg = config.sentiment

        self.lexicon = LexiconAnalyzer()
        self.market_proxy = MarketSentimentProxy()

        self.alpha = self.sent_cfg.sentiment_smooth_alpha
        self.extreme_bull = self.sent_cfg.extreme_bullish
        self.extreme_bear = self.sent_cfg.extreme_bearish

    def analyze_daily(self, date: str,
                      news_texts: List[str],
                      market_data: Optional[pd.DataFrame] = None) -> SentimentResult:
        """分析单日情绪

        Args:
            date:           日期字符串 'YYYY-MM-DD'
            news_texts:     当日新闻文本列表 (标题或正文)
            market_data:    当日行情数据 (可选)

        Returns:
            SentimentResult
        """
        # Channel A: 文本情绪
        text_scores = self.lexicon.analyze_batch(news_texts) if news_texts else []
        text_sentiment = float(np.mean(text_scores)) if text_scores else 0.0

        # Channel B: 市场代理情绪
        market_sentiment = self.market_proxy.compute_proxy(market_data) if market_data is not None else 0.0

        # 融合
        if news_texts and market_data is not None:
            composite = self.alpha * text_sentiment + (1 - self.alpha) * market_sentiment
        elif news_texts:
            composite = text_sentiment
        elif market_data is not None:
            composite = market_sentiment
        else:
            composite = 0.0

        composite = float(np.clip(composite, -1.0, 1.0))

        # 标签
        if composite > 0.15:
            label = "bullish"
        elif composite < -0.15:
            label = "bearish"
        else:
            label = "neutral"

        # 极端检测
        extreme = composite >= self.extreme_bull or composite <= self.extreme_bear

        # 置信度 (基于数据量)
        if news_texts and market_data is not None:
            confidence = 0.9
        elif news_texts or market_data is not None:
            confidence = 0.6
        else:
            confidence = 0.0

        # 关键词
        keywords = self.lexicon.extract_keywords(news_texts, 10) if news_texts else []

        return SentimentResult(
            date=date,
            text_sentiment=text_sentiment,
            market_sentiment=market_sentiment,
            composite_sentiment=composite,
            sentiment_label=label,
            extreme_flag=extreme,
            confidence=confidence,
            news_count=len(news_texts),
            top_keywords=keywords,
        )

    def analyze_period(self, start_date: str, end_date: str,
                       index_data: Optional[pd.DataFrame] = None,
                       news_data: Optional[pd.DataFrame] = None) -> List[SentimentResult]:
        """分析一段时间内每日情绪

        Args:
            start_date:  起始 'YYYY-MM-DD'
            end_date:    截止 'YYYY-MM-DD'
            index_data:  指数日线 (用于市场代理情绪)
            news_data:   新闻数据 (含 date + title/content 列)

        Returns:
            按日期排序的 SentimentResult 列表

        当 news_data 为空时, 全部使用市场代理情绪。
        """
        results = []

        # 建立日期范围
        dates = pd.date_range(start_date, end_date, freq="B")  # 仅交易日

        for date in dates:
            date_str = date.strftime("%Y-%m-%d")

            # 获取当日新闻
            day_news = []
            if news_data is not None and len(news_data) > 0:
                # 查找日期列
                date_col = None
                for c in ["date", "pub_time", "eob"]:
                    if c in news_data.columns:
                        date_col = c
                        break
                if date_col:
                    day_df = news_data[news_data[date_col].dt.strftime("%Y-%m-%d") == date_str]
                    for _, row in day_df.iterrows():
                        for col in ["title", "content", "text"]:
                            if col in row and pd.notna(row[col]):
                                day_news.append(str(row[col]))

            # 获取当日行情
            market = None
            if index_data is not None:
                try:
                    market = index_data.loc[date_str:date_str]
                    if len(market) == 0:
                        market = None
                except Exception:
                    market = None

            result = self.analyze_daily(date_str, day_news, market)
            results.append(result)

        # 对连续结果应用指数平滑
        if len(results) >= 3:
            compos = np.array([r.composite_sentiment for r in results])
            smoothed = _ema_smooth(compos, self.alpha)
            for i, r in enumerate(results):
                r.composite_sentiment = float(smoothed[i])
                # 重新判断标签
                if r.composite_sentiment > 0.15:
                    r.sentiment_label = "bullish"
                elif r.composite_sentiment < -0.15:
                    r.sentiment_label = "bearish"
                else:
                    r.sentiment_label = "neutral"

        return results

    def get_sentiment_series(self, results: List[SentimentResult]) -> pd.Series:
        """转为pandas Series"""
        dates = [r.date for r in results]
        values = [r.composite_sentiment for r in results]
        return pd.Series(values, index=pd.to_datetime(dates), name="sentiment")

    def detect_extreme(self, results: List[SentimentResult]) -> List[Dict]:
        """检测极端情绪事件

        条件:
            1. 情绪超出阈值 (extreme_bullish / extreme_bearish)
            2. 持续时间 >= 2个交易日 (过滤单日噪音)

        Returns:
            [{'start': date, 'end': date, 'type': 'extreme_bullish'/'extreme_bearish', 'peak': float}, ...]
        """
        events = []
        in_event = False
        event_start = None
        event_type = None
        event_peak = 0.0

        for r in results:
            is_extreme = r.extreme_flag
            if is_extreme and not in_event:
                in_event = True
                event_start = r.date
                event_type = "extreme_bullish" if r.composite_sentiment > 0 else "extreme_bearish"
                event_peak = abs(r.composite_sentiment)
            elif is_extreme and in_event:
                event_peak = max(event_peak, abs(r.composite_sentiment))
            elif not is_extreme and in_event:
                if r.date != event_start:  # 至少持续2天
                    events.append({
                        "start": event_start,
                        "end": r.date,
                        "type": event_type,
                        "peak": event_peak,
                    })
                in_event = False

        # 处理末尾仍在事件中
        if in_event and results[-1].date != event_start:
            events.append({
                "start": event_start,
                "end": results[-1].date,
                "type": event_type,
                "peak": event_peak,
            })

        return events


# ============================================================
# 工具函数
# ============================================================

def _ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    """指数移动平均平滑"""
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed
