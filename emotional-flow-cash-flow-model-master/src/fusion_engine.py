# -*- coding: utf-8 -*-
"""
情绪-资金流融合决策引擎 (FusionEngine)

架构 v2 (修复结构性偏多问题):
    宏观信号 = 仓位上限 (Cap)
    情绪 + 资金流 = 执行比率 (Execution Ratio)
    position = macro_cap * execution_ratio(direction)

修复要点:
    1. 信号保留原始 [-1, 1] 范围, 不归一化到 [0, 1]
    2. 宏观不再是 "三分之一投票", 而是仓位天花板
    3. 极度看空的情绪/资金流可将仓位压至 0
    4. min_position = 0.0, 允许完全空仓
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# 枚举 & 数据类型
# ============================================================

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    EXTREME = "extreme"


class SignalDirection(Enum):
    LONG = 1
    NEUTRAL = 0
    SHORT = -1


@dataclass
class SignalPack:
    """三通道信号包"""
    date: str
    sentiment: float             # [-1, 1]
    sentiment_dir: SignalDirection
    capital_flow: float          # [-1, 1]
    capital_flow_dir: SignalDirection
    macro: float                 # [0, 1]
    regime: MarketRegime


@dataclass
class FusionResult:
    """融合决策结果"""
    date: str
    composite_signal: float      # [-1, 1]
    position_ratio: float        # [0, 1]
    sentiment_contrib: float
    flow_contrib: float
    macro_contrib: float
    weights: Dict[str, float]
    regime: MarketRegime
    conflict: bool
    conflict_desc: str
    confidence: float


# ============================================================
# 权重管理器
# ============================================================

class WeightManager:
    """动态权重管理器

    v2 架构: 权重仅用于情绪与资金流之间的方向融合 (macro 不再参与加权, 而是作为 cap)

              情绪    资金流
    BULL      0.30    0.70   (趋势, 资金流主导方向)
    BEAR      0.60    0.40   (防守, 情绪更敏感)
    SIDEWAYS  0.50    0.50   (震荡, 等权)
    EXTREME   0.70    0.30   (恐慌/狂热, 情绪主导)
    """

    REGIME_WEIGHTS = {
        MarketRegime.BULL:      {"sentiment": 0.30, "capital_flow": 0.70},
        MarketRegime.BEAR:      {"sentiment": 0.60, "capital_flow": 0.40},
        MarketRegime.SIDEWAYS:  {"sentiment": 0.50, "capital_flow": 0.50},
        MarketRegime.EXTREME:   {"sentiment": 0.70, "capital_flow": 0.30},
    }

    def get_weights(self, regime: MarketRegime) -> Dict[str, float]:
        return self.REGIME_WEIGHTS.get(regime,
                                       {"sentiment": 0.50, "capital_flow": 0.50})

    @staticmethod
    def detect_regime(macro_score: float, sentiment: float) -> MarketRegime:
        """检测市场状态

        规则:
            macro > 0.45 -> BULL (降低门槛, 更早识别)
            macro < 0.25 -> BEAR
            |sentiment| > 0.7 -> EXTREME
            否则 -> SIDEWAYS
        """
        if abs(sentiment) > 0.7:
            return MarketRegime.EXTREME
        if macro_score > 0.45:
            return MarketRegime.BULL
        if macro_score < 0.25:
            return MarketRegime.BEAR
        return MarketRegime.SIDEWAYS


# ============================================================
# 冲突仲裁器
# ============================================================

class ConflictResolver:
    """信号冲突仲裁器

    情绪与资金流方向相反时的处理:
        情绪看多 + 资金流看空 -> 半仓 (等资金流确认)
        情绪看空 + 资金流看多 -> 半仓 (等情绪修复)
        情绪极端 + 任何方向   -> 以情绪为准
        资金流极端 + 任何方向 -> 以资金流为准
    """

    @staticmethod
    def _to_direction(value: float) -> SignalDirection:
        if value > 0.1:
            return SignalDirection.LONG
        elif value < -0.1:
            return SignalDirection.SHORT
        return SignalDirection.NEUTRAL

    @staticmethod
    def resolve(pack: SignalPack) -> Tuple[float, bool, str]:
        """仲裁

        Returns:
            (调整系数, 是否冲突, 描述)
        """
        s_dir = pack.sentiment_dir
        f_dir = pack.capital_flow_dir

        # 无冲突: 同向或中性
        if s_dir == f_dir or s_dir == SignalDirection.NEUTRAL or f_dir == SignalDirection.NEUTRAL:
            return 1.0, False, ""

        # 冲突: 方向相反
        if abs(pack.sentiment) > 0.6:
            # 极端情绪 -> 以情绪为准
            return 0.7, True, "extreme_sentiment_override"

        if abs(pack.capital_flow) > 0.6:
            # 极端资金流 -> 以资金流为准
            return 0.7, True, "extreme_flow_override"

        # 一般冲突 -> 半仓
        return 0.5, True, "general_conflict"


# ============================================================
# 信号质量校验器
# ============================================================

class SignalQualityChecker:
    """信号质量校验器

    校验维度:
        1. 信号持续性: 单日反转信号可信度低
        2. 信号极端性: 极端信号需交叉验证
    """

    @staticmethod
    def check(sentiment: float, flow: float, macro_position: float,
              hist_sentiment: Optional[List[float]] = None) -> float:
        """校验质量

        Returns:
            质量得分 [0, 1]
        """
        quality = 1.0

        # 极端信号折扣
        if abs(sentiment) > 0.8:
            quality *= 0.8
        if abs(flow) > 0.8:
            quality *= 0.8

        # 三通道不一致折扣
        # macro_position > 0.5 → 偏多方向, < 0.3 → 偏空方向
        macro_sign = 1 if macro_position > 0.5 else (-1 if macro_position < 0.3 else 0)
        signs = [np.sign(sentiment), np.sign(flow), macro_sign]
        signs_clean = [s for s in signs if s != 0]
        if len(signs_clean) >= 2 and max(signs_clean) != min(signs_clean):
            quality *= 0.7

        # 持续性检查
        if hist_sentiment and len(hist_sentiment) >= 3:
            recent_dir = np.sign(hist_sentiment[-3:])
            if len(set(recent_dir)) > 1:
                quality *= 0.85  # 近期情绪不稳定

        return np.clip(quality, 0.0, 1.0)


# ============================================================
# 融合决策主引擎
# ============================================================

class FusionEngine:
    """情绪-资金流融合决策引擎

    融合流程:
        1. 构建 SignalPack
        2. 检测市场状态
        3. 获取动态权重
        4. 信号质量校验
        5. 冲突检测与仲裁
        6. 加权融合: composite = w_s*sent + w_f*flow + w_m*macro
        7. 信号->仓位映射
    """

    def __init__(self, config):
        fusion_cfg = config.fusion
        self.min_position = fusion_cfg.min_position
        self.max_position = fusion_cfg.max_position

        self.weight_mgr = WeightManager()
        self.conflict_resolver = ConflictResolver()
        self.quality_checker = SignalQualityChecker()

    def fuse(self, sentiment: float, flow: float, macro_position: float,
             date: str = "",
             hist_sentiment: Optional[List[float]] = None) -> FusionResult:
        """执行信号融合决策 (v4: regime仓位锚点 + 情绪资金流执行)

        v4 架构:
            macro_position = 仓位锚点 (regime固定值: bull=0.70, neutral=0.40, bear=0.10)
            direction       = w_s * sentiment + w_f * flow  (情绪+资金流方向)
            execution       = (direction + 1) / 2            (方向→执行比率)
            position        = macro_position * execution     (最终仓位)

            bull: 执行下限0.55 → 最低仓位0.70*0.55=0.385
            bear: 执行上限0.40 → 最高仓位0.10*0.40=0.040

        Args:
            sentiment:       情绪得分 [-1, 1]
            flow:            资金流得分 [-1, 1]
            macro_position:  宏观仓位 (MacroSignal.position_ratio) [0, 1]
            date:            日期
            hist_sentiment:  近期情绪序列 (用于持续性检查)

        Returns:
            FusionResult
        """
        # 防御: 输入 NaN → 全中性
        if np.isnan(sentiment):
            sentiment = 0.0
        if np.isnan(flow):
            flow = 0.0
        if np.isnan(macro_position):
            macro_position = 0.3

        # 1. 方向判断
        s_dir = ConflictResolver._to_direction(sentiment)
        f_dir = ConflictResolver._to_direction(flow)

        # 2. 市场状态 (用 macro_position 而非 sentiment+macro)
        regime = self.weight_mgr.detect_regime(macro_position, sentiment)

        # 3. 信号包
        pack = SignalPack(
            date=date, sentiment=sentiment, sentiment_dir=s_dir,
            capital_flow=flow, capital_flow_dir=f_dir,
            macro=macro_position, regime=regime,
        )

        # 4. 动态权重 (仅用于情绪+资金流的方向融合)
        weights = self.weight_mgr.get_weights(regime)

        # 5. 信号质量 (三通道一致性校验)
        quality = self.quality_checker.check(sentiment, flow, macro_position, hist_sentiment)

        # 6. 冲突仲裁
        conflict_adj, has_conflict, conflict_desc = self.conflict_resolver.resolve(pack)

        # 7. v2 核心: 宏观=仓位上限, 情绪+资金流=执行比率
        # 方向信号 (保留原始 [-1, 1] 范围)
        direction = (
            weights["sentiment"] * sentiment +
            weights["capital_flow"] * flow
        )
        direction = float(np.clip(direction, -1.0, 1.0))

        # 执行比率: direction [-1, +1] → execution [0, 1]
        # -1(极度看空) → 0,  0(中性) → 0.5,  +1(极度看多) → 1.0
        execution = (direction + 1.0) / 2.0

        # 冲突调整
        execution = execution * conflict_adj + 0.5 * (1 - conflict_adj)

        # 低质量信号向 0.5 回归 (减少极端仓位)
        if quality < 0.5:
            execution = execution * quality + 0.5 * (1 - quality)

        execution = float(np.clip(execution, 0.0, 1.0))

        # v4: regime特有执行约束
        # bull: 执行下限0.55, 确保不踏空 (0.70*0.55=0.385)
        # bear: 执行上限0.40, 加强防守 (0.10*0.40=0.040)
        if regime == MarketRegime.BULL:
            execution = max(execution, 0.55)
        elif regime == MarketRegime.BEAR:
            execution = min(execution, 0.40)
            execution = max(execution, 0.10)  # bear下限: 至少10%执行

        # 最终仓位 = 宏观上限 * 执行比率
        position = macro_position * execution

        # 在极端市场中限制仓位
        if regime == MarketRegime.EXTREME:
            position = min(position, 0.5)

        position = float(np.clip(position, self.min_position, self.max_position))

        # composite_signal (保留兼容性, 范围 [-1, 1])
        composite = 2.0 * position / max(macro_position, 0.01) - 1.0
        composite = float(np.clip(composite, -1.0, 1.0))

        return FusionResult(
            date=date,
            composite_signal=composite,
            position_ratio=position,
            sentiment_contrib=weights["sentiment"] * sentiment,
            flow_contrib=weights["capital_flow"] * flow,
            macro_contrib=macro_position,
            weights=weights,
            regime=regime,
            conflict=has_conflict,
            conflict_desc=conflict_desc if has_conflict else "",
            confidence=quality,
        )

    def batch_fuse(self,
                   sentiment_series: pd.Series,
                   flow_series: pd.Series,
                   macro_series: pd.Series) -> pd.DataFrame:
        """批量融合 (对整个时间序列)

        Args:
            sentiment_series: index=date, values=情绪得分 [-1, 1]
            flow_series:      index=date, values=资金流得分 [-1, 1]
            macro_series:     index=date, values=宏观仓位 [0, 1]

        Returns:
            DataFrame: [date, composite_signal, position_ratio, weights, regime, confidence]
        """
        dates = sentiment_series.index.intersection(flow_series.index).intersection(macro_series.index)
        dates = sorted(dates)

        results = []
        hist_sent = []

        for date in dates:
            sent = sentiment_series.loc[date]
            flow = flow_series.loc[date]
            macro = macro_series.loc[date]

            if pd.isna(sent) or pd.isna(flow) or pd.isna(macro):
                continue

            result = self.fuse(
                sentiment=float(sent),
                flow=float(flow),
                macro_position=float(macro),
                date=date.strftime("%Y-%m-%d"),
                hist_sentiment=hist_sent[-10:] if hist_sent else None,
            )

            hist_sent.append(float(sent))
            results.append(result)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame([{
            "date": r.date,
            "composite_signal": r.composite_signal,
            "position_ratio": r.position_ratio,
            "sentiment_contrib": r.sentiment_contrib,
            "flow_contrib": r.flow_contrib,
            "macro_contrib": r.macro_contrib,
            "regime": r.regime.value,
            "conflict": r.conflict,
            "confidence": r.confidence,
        } for r in results])

        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")
