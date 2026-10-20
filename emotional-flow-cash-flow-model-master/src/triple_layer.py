# -*- coding: utf-8 -*-
"""
三层递进决策模块

L1 宏观择时 -> L2 板块筛选 -> L3 个股选股

全部核心算法继承自 V2 资金流策略:
    D:\开发\quant_strategy\backtest\backtest_v2.py
    D:\开发\quant_strategy\backtest\backtest_v2_gm.py
    D:\开发\quant_strategy\backtest\causal_network.py
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize

from .capital_flow import CMFCalculator, FactorCalculator


# ============================================================
# L1: 宏观择时
# ============================================================

@dataclass
class MacroSignal:
    """宏观择时信号"""
    date: str
    position_ratio: float          # 仓位 [0.05, 1.00]
    north_score: float
    margin_score: float
    bubble_risk: bool
    market_regime: str             # 'bull' / 'bear' / 'sideways'


class MacroLayer:
    """L1 宏观择时 -- 北向/两融/泡沫 -> 仓位信号

    核心逻辑 (继承V2):
        base = 0.3 + 0.2*north_score + 0.2*margin_score
        if bubble: position *= 0.3
        final = clip(position, 0.05, 1.0)
    """

    def __init__(self, north_df: pd.DataFrame, margin_df: pd.DataFrame,
                 benchmark: Optional[pd.DataFrame] = None):
        """
        Args:
            north_df:  北向资金DataFrame (index=date, columns=[total_net])
            margin_df: 融资融券DataFrame (index=date, columns=[balance])
            benchmark: 基准指数日线 (用于LPPL泡沫检测)
        """
        self.north = north_df
        self.margin = margin_df
        self.benchmark = benchmark
        self._lppl_cache = {}

    # ---- 北向评分 ----

    def _north_score(self, date: pd.Timestamp) -> float:
        """北向资金方向评分 (连续化, 减少离散跳变)

        MA背离度 = (MA5 - MA20) / |MA20|   [-1, 1] 范围
        MA趋势 = clip(MA20 / 历史均值, -1, 1)
        score = clip(背离度 * 3 + 趋势 * 0.3, -1, 1)
        """
        if self.north is None:
            return 0.0

        past = self.north.loc[:date]
        if len(past) < 25:
            return 0.0

        total = past["total_net"]
        ma5 = total.rolling(5).mean().iloc[-1]
        ma20 = total.rolling(20).mean().iloc[-1]
        hist_mean = total.abs().mean()

        if pd.isna(ma5) or pd.isna(ma20):
            return 0.0

        # MA背离度 (MA5与MA20的相对距离)
        denom = max(abs(ma20), hist_mean * 0.1, 1e-8)
        if denom < 1e-8:
            return 0.0
        divergence = (ma5 - ma20) / denom

        # 长期趋势 (MA20方向)
        trend = np.clip(ma20 / max(hist_mean, 1e-6), -1.0, 1.0)

        score = divergence * 3.0 + trend * 0.3
        return float(np.clip(score, -1.0, 1.0))

    # ---- 两融评分 ----

    def _margin_score(self, date: pd.Timestamp) -> float:
        """融资融券情绪评分 (连续化)

        用余额5日变化率直接映射到 [-1, 1]:
            +5%变化 → +1.0,  0% → 0,  -5% → -1.0
        """
        if self.margin is None:
            return 0.0

        past = self.margin.loc[:date]
        if len(past) < 10 or "balance" not in past.columns:
            return 0.0

        chg = past["balance"].pct_change(5).iloc[-1]
        if pd.isna(chg):
            return 0.0

        # 连续化: 5%变化映射到1.0
        score = chg / 0.05
        return float(np.clip(score, -1.0, 1.0))

    # ---- 价格趋势评分 (v4.1: 用基准指数趋势补充北向/两融) ----

    def _price_trend_score(self, date: pd.Timestamp) -> float:
        """基准指数价格趋势评分

        用20日收益率作为趋势强度信号:
            +10%  → +1.0
            0%    →  0
            -10%  → -1.0
        """
        if self.benchmark is None:
            return 0.0

        past = self.benchmark.loc[:date]
        if len(past) < 25 or "close" not in past.columns:
            return 0.0

        close = past["close"]
        ret_20 = close.pct_change(20).iloc[-1]
        if pd.isna(ret_20):
            return 0.0

        return float(np.clip(ret_20 / 0.10, -1.0, 1.0))

    # ---- LPPL 泡沫检测 ----

    def _is_bubble(self, date: pd.Timestamp) -> bool:
        """LPPL泡沫检测 (继承V2 Layer1Macro._fit_lppl)"""
        if self.benchmark is None:
            return False

        date_str = date.strftime("%Y-%m-%d")
        if date_str in self._lppl_cache:
            return self._lppl_cache[date_str]

        past = self.benchmark.loc[:date]
        if len(past) < 200 or "close" not in past.columns:
            self._lppl_cache[date_str] = False
            return False

        # 取最近500天
        bm = past.tail(500)
        result = self._fit_lppl(bm["close"], pd.Series(range(len(bm)), index=bm.index))
        self._lppl_cache[date_str] = result
        return result

    def _fit_lppl(self, prices: pd.Series, t_relative: pd.Series) -> bool:
        """拟合LPPL模型"""

        y = np.log(prices.values)
        x = t_relative.values
        if len(y) < 100:
            return False

        def lppl_loss(params):
            A, B, C, m, omega, phi = params
            dt = x[-1] + 50 - x
            dt = np.maximum(dt, 1e-6)
            pred = A + B * dt**m + C * dt**m * np.cos(omega * np.log(dt) + phi)
            return np.sqrt(((y - pred)**2).mean())

        best_loss = np.inf
        best_fit = None

        for tc_guess in [30, 60, 90, 120, 180]:
            dt = tc_guess - x
            dt = np.maximum(dt, 1e-6)
            try:
                res = minimize(
                    lppl_loss, [y[-1], -0.1, 0.05, 0.5, 6.0, 0.0],
                    method="Nelder-Mead",
                    options={"maxiter": 2000, "xatol": 1e-6},
                    bounds=[(None, None), (None, 0), (None, 0.2),
                            (0.1, 0.9), (3, 13), (0, 2 * np.pi)]
                )
                if res.fun < best_loss:
                    best_fit = res
                    best_loss = res.fun
            except Exception:
                pass

        if best_fit is None:
            return False

        m_val = best_fit.x[3]
        c_b_ratio = abs(best_fit.x[2]) / (abs(best_fit.x[1]) + 1e-6)
        return m_val < 0.8 and c_b_ratio > 0.15

    # ---- 主计算 ----

    def compute_signal(self, date: pd.Timestamp,
                       sentiment_score: float = 0.0) -> MacroSignal:
        """计算宏观择时信号 (v4: regime固定仓位, 不再线性公式)

        v4架构:
            宏观层统一信号 → 决定regime
            regime → 固定仓位锚点 (bull=0.70, neutral=0.40, bear=0.10)
            情绪+资金流 → 执行比率 (在锚点附近调节)

        Args:
            date:            分析日期
            sentiment_score: 当日情绪得分 [-1, 1] (用于微调)

        Returns:
            MacroSignal
        """
        date = pd.to_datetime(date).normalize().tz_localize(None)

        north_s = self._north_score(date)
        margin_s = self._margin_score(date)

        # 防御性 NaN 保护
        if np.isnan(north_s):
            north_s = 0.0
        if np.isnan(margin_s):
            margin_s = 0.0
        if np.isnan(sentiment_score):
            sentiment_score = 0.0

        trend_s = self._price_trend_score(date)
        if np.isnan(trend_s):
            trend_s = 0.0

        # 宏观综合得分 (北向+两融等权, 价格趋势辅助)
        macro_composite = 0.4 * north_s + 0.4 * margin_s + 0.2 * trend_s

        # 泡沫风险 (提前判定, 用于regime降级)
        is_bubble = self._is_bubble(date)

        # ---- v4.1核心: regime决定仓位锚点 (阈值降低, 趋势辅助) ----
        if macro_composite > 0.10 and not is_bubble:
            regime = "bull"
            base_pos = 0.80
        elif macro_composite < -0.10 or is_bubble:
            regime = "bear"
            base_pos = 0.10
        else:
            regime = "sideways"
            base_pos = 0.40

        # 情绪微调 (在锚点附近 ±0.15)
        sentiment_adj = sentiment_score * 0.15
        base_pos = np.clip(base_pos + sentiment_adj, 0.05, 1.0)

        return MacroSignal(
            date=date.strftime("%Y-%m-%d"),
            position_ratio=float(base_pos),
            north_score=north_s,
            margin_score=margin_s,
            bubble_risk=is_bubble,
            market_regime=regime,
        )


# ============================================================
# L2: 板块筛选
# ============================================================

@dataclass
class SectorSignal:
    """板块筛选信号"""
    date: str
    inflow_sectors: List[str]
    outflow_sectors: List[str]
    leading_sectors: List[str]


class SectorLayer:
    """L2 板块筛选 -- 行业资金流 + 因果网络

    继承V2 Layer2Sector的逻辑:
        1. 从板块CMF面板计算评分
        2. 正向CMF且加速的板块 -> 流入
        3. 负向CMF且减速的板块 -> 流出
    """

    def __init__(self):
        pass

    @staticmethod
    def select_sectors(flow_matrix: pd.DataFrame,
                       date: pd.Timestamp) -> SectorSignal:
        """选择板块

        参考: backtest_v2.py Layer2Sector.compute_scores()
              和 backtest_v2_gm.py compute_sector_scores()

        Returns:
            SectorSignal
        """
        if flow_matrix is None or len(flow_matrix) == 0:
            return SectorSignal(date=date.strftime("%Y-%m-%d"),
                               inflow_sectors=[], outflow_sectors=[], leading_sectors=[])

        date = pd.to_datetime(date).normalize().tz_localize(None)
        if date not in flow_matrix.index:
            available = flow_matrix.index[flow_matrix.index <= date]
            if len(available) == 0:
                return SectorSignal(date=date.strftime("%Y-%m-%d"),
                                   inflow_sectors=[], outflow_sectors=[], leading_sectors=[])
            date = available[-1]

        recent = flow_matrix.loc[:date]
        if len(recent) < 10:
            return SectorSignal(date=date.strftime("%Y-%m-%d"),
                               inflow_sectors=[], outflow_sectors=[], leading_sectors=[])

        scores = {}
        for ind in flow_matrix.columns:
            if ind not in recent.columns:
                continue
            col = recent[ind].dropna().tail(10)
            if len(col) < 5:
                continue

            recent_4 = col.iloc[-4:].mean() if len(col) >= 4 else col.mean()
            prior_4 = col.iloc[-8:-4].mean() if len(col) >= 8 else col.iloc[:4].mean()
            delta = recent_4 - prior_4

            accel = (col.iloc[-2:].mean() - col.iloc[-4:].mean()) if len(col) >= 4 else 0.0

            scores[ind] = delta * 0.6 + accel * 0.4

        sorted_scores = pd.Series(scores).sort_values(ascending=False)

        inflow = sorted_scores[sorted_scores > 0].head(5).index.tolist()
        outflow = sorted_scores[sorted_scores < 0].tail(5).index.tolist()

        # 龙头: 排名前3且得分 > 0.02
        leaders = sorted_scores[sorted_scores > 0.02].head(3).index.tolist()

        return SectorSignal(
            date=date.strftime("%Y-%m-%d"),
            inflow_sectors=inflow,
            outflow_sectors=outflow,
            leading_sectors=leaders,
        )


# ============================================================
# L3: 个股选股
# ============================================================

@dataclass
class StockScore:
    """个股评分"""
    symbol: str
    total_score: float
    factor_scores: Dict[str, float]
    is_selected: bool


class StockLayer:
    """L3 个股选股 -- 多因子截面rank + 双验证过滤

    因子权重基于 factor_ic.py 实证结果 (IC绝对值归一化):
        cmf_turn:      0.35  (资金拐头, IC正)
        vol_20d:       0.18  (低波动, IC负)
        mom_20d:       0.15  (反转, IC负)
        overnight_ret: 0.10  (隔夜, IC正)
        vol_rs:        0.10  (缩量, IC负)
        cmf_20:        0.07  (低CMF, IC负)
        illiq:         0.05  (流动性, IC正)

    因子方向 (来自V2 FACTOR_DIRECTIONS):
        "" 表示正 IC 方向 (值越大越好), "+" 表示反转 (值越小越好)
    """

    def __init__(self, config):
        cfg = config.factors
        self.weights = {
            "cmf_turn": cfg.cmf_turn,
            "vol_20d": cfg.vol_20d,
            "mom_20d": cfg.mom_20d,
            "overnight_ret": cfg.overnight_ret,
            "vol_rs": cfg.vol_rs,
            "cmf_20": cfg.cmf_20,
            "illiq": cfg.illiq,
        }
        # 因子方向: +1=正向(越大越好), -1=负向(越小越好)
        self.directions = {
            "cmf_turn": +1,
            "vol_20d": -1,
            "mom_20d": -1,
            "overnight_ret": -1,  # A股隔夜高开常被砸盘, 负向
            "vol_rs": -1,
            "cmf_20": -1,
            "illiq": +1,
        }
        # 趋势行情 (bull) 因子方向: 动量正向 + 高波 (追涨杀跌)
        self.trend_directions = {
            "cmf_turn": +1,
            "vol_20d": +1,         # 趋势中高波强
            "mom_20d": +1,         # 趋势中动量延续
            "overnight_ret": +1,
            "vol_rs": +1,          # 放量好
            "cmf_20": +1,          # CMF正向
            "illiq": +1,
        }
        self.max_holdings = config.fusion.max_holdings
        self.max_per_sector = config.fusion.max_per_sector

    def _get_directions(self, regime: str) -> Dict[str, int]:
        """根据市场状态返回因子方向

        bull: 趋势追涨, 动量正向 + 高波偏好
        其他: 反转+低波 (原配置)
        """
        if regime and 'bull' in regime.lower():
            return self.trend_directions
        return self.directions

    def cross_sectional_rank(self,
                             factor_dict: Dict[str, Dict[str, float]],
                             regime: str = "") -> pd.DataFrame:
        """截面rank打分

        Args:
            factor_dict: {symbol: {factor_name: raw_value, ...}, ...}

        Returns:
            DataFrame: index=symbol, columns=factor scores + total_score

        算法:
            1. 对每个因子, 按所有股票的值排序
            2. 得分 = rank百分位 (0~1)
            3. 乘以方向 (+1/-1) 和权重
            4. total_score = sum(weighted_scores)
        """
        if not factor_dict:
            return pd.DataFrame()

        symbols = list(factor_dict.keys())
        factor_names = list(self.weights.keys())

        # 提取因子值矩阵
        factor_matrix = {}
        for fn in factor_names:
            values = {}
            for sym in symbols:
                v = factor_dict[sym].get(fn, 0.0)
                values[sym] = v if not pd.isna(v) else 0.0
            factor_matrix[fn] = pd.Series(values)

        # 截面rank
        rank_scores = pd.DataFrame(index=symbols, columns=factor_names + ["total_score"])
        rank_scores["total_score"] = 0.0

        for fn in factor_names:
            values = factor_matrix[fn]
            if values.std() < 1e-12:
                # 所有值相同, 不给分
                rank_scores[fn] = 0.5
                continue

            # rank: 0=最低, 1=最高
            ranks = values.rank(pct=True)

            # 应用方向 (regime-aware)
            directions = self._get_directions(regime)
            if directions.get(fn, 1) == -1:
                ranks = 1.0 - ranks

            rank_scores[fn] = ranks
            rank_scores["total_score"] += ranks * self.weights[fn]

        return rank_scores.sort_values("total_score", ascending=False)

    def dual_verification(self, stock_data: pd.DataFrame,
                          position_ratio: float) -> bool:
        """双验证过滤

        高仓位 (>0.5): 价格在MA20上方 AND CMF为正
        低仓位 (<=0.5): 价格在MA20上方 OR CMF为正
        """
        if stock_data is None or len(stock_data) < 20:
            return False

        close = stock_data["close"]
        # 价格在MA20上方
        ma20 = close.rolling(20).mean().iloc[-1]
        above_ma = close.iloc[-1] > ma20

        # CMF
        cmf_series = CMFCalculator.compute(stock_data)
        cmf_positive = cmf_series.iloc[-1] > 0 if len(cmf_series) > 0 else False

        if position_ratio > 0.5:
            # 高仓位 AND 验证
            return above_ma and cmf_positive
        else:
            # 低仓位 OR 验证
            return above_ma or cmf_positive

    def pick_stocks(self, stock_pool: List[str],
                    stock_data: Dict[str, pd.DataFrame],
                    position_ratio: float,
                    stock_industry: Optional[Dict[str, str]] = None,
                    selected_sectors: Optional[List[str]] = None,
                    regime: str = "") -> List[StockScore]:
        """从候选池选股

        Args:
            stock_pool:      候选股票列表
            stock_data:      {symbol: daily_data}
            position_ratio:  L1宏观仓位 (决定双验证模式)
            stock_industry:  个股->行业映射
            selected_sectors: L2筛选出的板块 (板块集中度约束)

        Returns:
            按得分排序的 StockScore 列表
        """
        if not stock_pool:
            return []

        # 1. 计算每只股票的因子值
        factor_dict = {}
        factor_calc = FactorCalculator()
        for sym in stock_pool:
            df = stock_data.get(sym)
            if df is None or len(df) < 60:
                continue
            factors = factor_calc.compute_all(df)
            if factors:
                factor_dict[sym] = factors

        if not factor_dict:
            return []

        # 2. 截面rank (根据市场状态切换因子方向)
        rank_df = self.cross_sectional_rank(factor_dict, regime)

        # 3. 双验证过滤 + 板块集中度约束
        sector_count: Dict[str, int] = {}
        results = []

        for sym in rank_df.index:
            df = stock_data.get(sym)
            if df is None:
                continue

            # 双验证
            if not self.dual_verification(df, position_ratio):
                continue

            # 板块集中度
            if stock_industry and selected_sectors:
                ind = stock_industry.get(sym, "")
                if ind not in selected_sectors:
                    continue
                if sector_count.get(ind, 0) >= self.max_per_sector:
                    continue

            total_score = float(rank_df.loc[sym, "total_score"])
            factor_scores = {fn: float(rank_df.loc[sym, fn])
                           for fn in self.weights.keys() if fn in rank_df.columns}

            results.append(StockScore(
                symbol=sym,
                total_score=total_score,
                factor_scores=factor_scores,
                is_selected=True,
            ))

            if stock_industry and selected_sectors:
                ind = stock_industry.get(sym, "")
                sector_count[ind] = sector_count.get(ind, 0) + 1

            if len(results) >= self.max_holdings:
                break

        return results
