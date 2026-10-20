# -*- coding: utf-8 -*-
"""
资金流分析器 (CapitalFlowAnalyzer)

核心功能:
    1. CMF (Chaikin Money Flow)          -- 柴金资金流, 量价结合
    2. OBV (On-Balance Volume)           -- 能量潮
    3. 资金拐头信号                         -- CMF短期/长期对比
    4. 微观结构因子 (OBI/VPIN/Amihud/波动率) -- 分钟级因子
    5. 行业资金流面板分析                    -- 板块资金流入流出
    6. 量价背离检测                          -- 价格与资金流方向偏离

所有算法继承自 V2 策略 (D:\开发\quant_strategy\backtest\) 的验证逻辑:
    - compute_cmf / compute_cmf_turn 来自 backtest_v2_gm.py
    - extract_micro_factors 来自 micro_factor_ic.py
    - 因子方向来自 factor_ic.py 的IC分析结果
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# ============================================================
# 结果数据类型
# ============================================================

@dataclass
class CapitalFlowResult:
    """个股资金流分析结果"""
    symbol: str
    date: str
    cmf: float                     # CMF值 [-1, 1]
    cmf_turn_signal: float         # 资金拐头信号 [+1=回流, -1=流出]
    obv: float                     # OBV累积值
    obv_change_pct: float          # OBV 5日变化率
    flow_score: float              # 综合资金流得分 [-1, 1]
    flow_direction: str            # 'inflow' / 'outflow' / 'neutral'


# ============================================================
# CMF 计算器
# ============================================================

class CMFCalculator:
    """CMF (Chaikin Money Flow) 计算器

    公式:
        MFM = ((C - L) - (H - C)) / (H - L)     资金流乘数
        MFV = MFM * Volume                       资金流量
        CMF = SUM(MFV, n) / SUM(Volume, n)       n日资金流比率

    参考: D:\开发\quant_strategy\backtest\backtest_v2_gm.py: compute_cmf()
    """

    @staticmethod
    def compute(stock_data: pd.DataFrame, period: int = 20) -> pd.Series:
        """计算CMF指标

        Args:
            stock_data: 含 [high, low, close, volume] 的DataFrame
            period: 计算周期

        Returns:
            CMF序列 (index与stock_data对齐)
        """
        high, low, close, volume = (
            stock_data["high"], stock_data["low"],
            stock_data["close"], stock_data["volume"]
        )

        # 1. 资金流乘数 MFM
        hl_range = (high - low).replace(0, np.nan)
        mfm = ((close - low) - (high - close)) / hl_range
        mfm = mfm.fillna(0)

        # 2. 资金流量 MFV
        mfv = mfm * volume

        # 3. CMF = sum(MFV, n) / sum(V, n)
        cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
        cmf = cmf.fillna(0).clip(-1, 1)

        return cmf

    @staticmethod
    def compute_turn_signal(stock_data: pd.DataFrame) -> pd.Series:
        """计算资金拐头信号

        cmf_turn = CMF_short_mean - CMF_long_mean
        正值表示短期资金开始回流，负值表示资金正在加速流出。

        参考: backtest_v2_gm.py: compute_cmf_turn()
        """
        cmf = CMFCalculator.compute(stock_data)

        short = cmf.rolling(2).mean()
        long_ = cmf.rolling(5).mean()

        turn = short - long_
        return turn.fillna(0)


# ============================================================
# OBV 计算器
# ============================================================

class OBVCalculator:
    """OBV (On-Balance Volume) 计算器

    规则:
        close > close_prev  -> OBV += volume
        close < close_prev  -> OBV -= volume
        close == close_prev -> OBV 不变
    """

    @staticmethod
    def compute(stock_data: pd.DataFrame) -> pd.Series:
        """计算OBV"""
        close = stock_data["close"]
        volume = stock_data["volume"]

        direction = np.sign(close.diff())
        direction.iloc[0] = 0

        obv = (direction * volume).cumsum()
        return obv

    @staticmethod
    def change_pct(obv: pd.Series, period: int = 5) -> pd.Series:
        """OBV N日变化率"""
        return obv.pct_change(period).fillna(0)


# ============================================================
# 微观因子提取器
# ============================================================

class MicroFactorExtractor:
    """微观结构因子提取器

    基于分钟K线数据, 提取高频微观因子。
    所有算法来自 D:\开发\quant_strategy\backtest\micro_factor_ic.py

    因子列表:
        obi            订单簿不平衡 (全天)
        obi_open       开盘OBI (前15%分钟)
        obi_close      尾盘OBI (后15%分钟, ICIR最强=-0.772)
        vpin_mean      VPIN均值 (信息交易概率代理)
        vpin_std       VPIN标准差
        rv             已实现波动率 (年化)
        rv_skew        收益率偏度
        rv_kurt        收益率峰度 (IC=-0.042)
        amihud_intra   日内非流动性 (IC=+0.058)
        vol_top10_pct  成交量集中度
        open_gap       开盘跳空
    """

    @staticmethod
    def extract_all(minute_data: pd.DataFrame,
                    daily_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """一次性提取所有微观因子

        Args:
            minute_data: 当天分钟K线 (index=eob, 含 close, volume, open, high, low)
            daily_data:  日线数据 (用于计算开盘跳空), 可选

        Returns:
            {因子名: 值} 字典, 至少100条分钟线才计算
        """
        if minute_data is None or len(minute_data) < 100:
            return {}

        df = minute_data.copy()
        rets = df["close"].pct_change().fillna(0)
        vol = df["volume"]
        n = len(df)

        # ---- OBI: 上涨量 - 下跌量 / 总量 ----
        up_vol = vol[rets > 0].sum()
        down_vol = vol[rets < 0].sum()
        total_vol = up_vol + down_vol
        obi = (up_vol - down_vol) / (total_vol + 1e-6) if total_vol > 0 else 0.0

        # ---- OBI 开盘/尾盘 ----
        early_n = max(int(n * 0.15), 1)
        late_n = max(int(n * 0.15), 1)
        early = df.iloc[:early_n]
        late = df.iloc[-late_n:]

        def _calc_obi(seg):
            r = seg["close"].pct_change().fillna(0)
            v = seg["volume"]
            uv = v[r > 0].sum()
            dv = v[r < 0].sum()
            tv = uv + dv
            return (uv - dv) / (tv + 1e-6) if tv > 0 else 0.0

        obi_open = _calc_obi(early)
        obi_close = _calc_obi(late)

        # ---- VPIN 代理: 分桶OBI标准差 ----
        nb = min(30, n // 5)
        obi_buckets = []
        indices = np.array_split(np.arange(n), nb)
        for idx in indices:
            seg = df.iloc[idx]
            obi_buckets.append(_calc_obi(seg))

        vpin_mean = float(np.mean(obi_buckets)) if obi_buckets else 0.0
        vpin_std = float(np.std(obi_buckets)) if len(obi_buckets) > 1 else 0.0

        # ---- 实现波动率 & 高阶矩 ----
        rets_min = df["close"].pct_change().dropna()
        rv = float(rets_min.std() * np.sqrt(240)) if len(rets_min) > 0 else 0.0
        rv_skew = float(rets_min.skew()) if len(rets_min) > 20 else 0.0
        rv_kurt = float(rets_min.kurtosis()) if len(rets_min) > 20 else 0.0

        # ---- 日内 Amihud ----
        if len(rets_min) > 0 and "volume" in df.columns:
            amihud = (rets_min.abs() / df["volume"].iloc[1:].clip(lower=1)).mean()
        else:
            amihud = 0.0

        # ---- 成交量集中度 ----
        vol_sorted = df["volume"].sort_values(ascending=False)
        vol_top10 = float(vol_sorted.iloc[:10].sum() / vol_sorted.sum()) if len(vol_sorted) >= 10 else 0.0

        # ---- 开盘跳空 ----
        open_gap = 0.0
        if daily_data is not None and len(daily_data) > 0:
            prev_close_val = daily_data["close"].iloc[-1] if "close" in daily_data.columns else None
            if prev_close_val is not None and prev_close_val > 0:
                open_gap = float(df["open"].iloc[0] / prev_close_val - 1)

        return {
            "obi": obi,
            "obi_open": obi_open,
            "obi_close": obi_close,
            "vpin_mean": vpin_mean,
            "vpin_std": vpin_std,
            "rv": rv,
            "rv_skew": rv_skew,
            "rv_kurt": rv_kurt,
            "amihud_intra": float(amihud),
            "vol_top10_pct": vol_top10,
            "open_gap": open_gap,
        }


# ============================================================
# 行业资金流分析
# ============================================================

class SectorFlowAnalyzer:
    """行业资金流分析器"""

    def __init__(self):
        pass

    @staticmethod
    def compute_sector_scores(flow_matrix: pd.DataFrame,
                              date: pd.Timestamp) -> Tuple[List[str], List[str], Dict[str, float]]:
        """计算行业资金流评分

        参考: backtest_v2.py 的 Layer2Sector.compute_scores()

        方法:
            对每个行业, 取最近10日CMF, 计算:
            - recent_4 = 近4日均值
            - prior_4  = 前4日均值 (5~8日前)
            - delta = recent_4 - prior_4       (趋势)
            - accel = 近2日 - 近4日             (加速度)
            - score = delta * 0.6 + accel * 0.4

        Returns:
            (流入板块列表, 流出板块列表, {行业: 评分})
        """
        if flow_matrix is None or len(flow_matrix) == 0:
            return [], [], {}

        date = pd.to_datetime(date).normalize().tz_localize(None)

        # 找到最近日期
        if date not in flow_matrix.index:
            available = flow_matrix.index[flow_matrix.index <= date]
            if len(available) == 0:
                return [], [], {}
            date = available[-1]

        recent = flow_matrix.loc[:date]
        if len(recent) < 10:
            return [], [], {}

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

        if not scores:
            return [], [], {}

        # 排序
        sorted_scores = pd.Series(scores).sort_values(ascending=False)

        inflow = sorted_scores[sorted_scores > 0].head(5).index.tolist()
        outflow = sorted_scores[sorted_scores < 0].tail(5).index.tolist()

        return inflow, outflow, scores

    @staticmethod
    def compute_cmf_zscore(flow_matrix: pd.DataFrame,
                           window: int = 20) -> pd.DataFrame:
        """计算行业CMF Z-score (拥挤度指标)

        z_score = (CMF - MA20) / STD20
        高位表示资金过度拥挤，存在反转风险
        """
        if flow_matrix is None or len(flow_matrix) < window:
            return pd.DataFrame()

        ma = flow_matrix.rolling(window).mean()
        std = flow_matrix.rolling(window).std().clip(lower=1e-8)
        zscore = (flow_matrix - ma) / std
        return zscore.fillna(0)


# ============================================================
# 因子计算器 (用于截面rank)
# ============================================================

class FactorCalculator:
    """多因子计算器

    计算选股所需的全部7个日线因子。

    因子定义:
        cmf_20      CMF 20日均值 (IC=-0.030, 负向)
        cmf_turn    CMF 近2日 - 近5日均值 (IC=+0.020, 正向)
        mom_20d     20日收益率 (IC=-0.062, 反转, 负向)
        vol_20d     20日波动率 (IC=-0.086, 负向, 最强)
        illiq       Amihud非流动性 (IC=+0.071, 正向)
        vol_rs      相对成交量 (IC=-0.029, 负向)
        overnight_ret 隔夜收益率
    """

    @staticmethod
    def compute_all(stock_data: pd.DataFrame) -> Dict[str, float]:
        """计算单只股票的全部日线因子

        Args:
            stock_data: 含 [open, high, low, close, volume, amount] 的DataFrame
                        至少需要60行数据

        Returns:
            {因子名: 最新值}
        """
        if stock_data is None or len(stock_data) < 60:
            return {}

        close = stock_data["close"]
        volume = stock_data["volume"]
        ret = close.pct_change()

        factors = {}

        # ---- CMF ----
        cmf_series = CMFCalculator.compute(stock_data)
        factors["cmf_20"] = float(cmf_series.iloc[-1]) if len(cmf_series) > 0 else 0.0

        # ---- CMF Turn (拐头信号) ----
        cmf_turn_series = CMFCalculator.compute_turn_signal(stock_data)
        factors["cmf_turn"] = float(cmf_turn_series.iloc[-1]) if len(cmf_turn_series) > 0 else 0.0

        # ---- 20日动量 ----
        if len(close) >= 20:
            factors["mom_20d"] = float(close.iloc[-1] / close.iloc[-20] - 1)

        # ---- 20日波动率 ----
        if len(ret.dropna()) >= 20:
            factors["vol_20d"] = float(ret.rolling(20).std().iloc[-1])

        # ---- Amihud非流动性 ----
        if "amount" in stock_data.columns and len(ret.dropna()) >= 20:
            illiq_series = (ret.abs() / stock_data["amount"]).rolling(20).mean()
            factors["illiq"] = float(illiq_series.iloc[-1]) if len(illiq_series) > 0 else 0.0
        else:
            factors["illiq"] = 0.0

        # ---- 相对成交量 ----
        if len(volume) >= 20:
            vol_ma20 = volume.rolling(20).mean()
            vol_rs = volume.iloc[-1] / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] > 0 else 1.0
            factors["vol_rs"] = float(vol_rs)

        # ---- 隔夜收益率 (机构买入代理) ----
        if "open" in stock_data.columns and len(stock_data) >= 2:
            overnight = stock_data["open"] / stock_data["close"].shift(1) - 1
            factors["overnight_ret"] = float(overnight.iloc[-1]) if not pd.isna(overnight.iloc[-1]) else 0.0

        # 填充缺失值
        for k in ["cmf_20", "cmf_turn", "mom_20d", "vol_20d",
                   "illiq", "vol_rs", "overnight_ret"]:
            if k not in factors or pd.isna(factors[k]):
                factors[k] = 0.0

        return factors

    @staticmethod
    def winsorize(factor_values: pd.Series, limits: Tuple[float, float] = (0.01, 0.99)) -> pd.Series:
        """因子去极值 (Winsorize)

        将超出上下分位数的值截断到分位数
        """
        lower = factor_values.quantile(limits[0])
        upper = factor_values.quantile(limits[1])
        return factor_values.clip(lower, upper)

    @staticmethod
    def standardize(factor_values: pd.Series) -> pd.Series:
        """因子标准化 (Z-score)"""
        mean = factor_values.mean()
        std = factor_values.std()
        if std < 1e-8:
            return factor_values * 0
        return (factor_values - mean) / std


# ============================================================
# 主资金流分析器
# ============================================================

class CapitalFlowAnalyzer:
    """资金流分析主引擎

    整合CMF/OBV/微观因子/行业资金流, 输出综合分析结果。

    使用示例:
        >>> analyzer = CapitalFlowAnalyzer(config)
        >>> result = analyzer.analyze_stock(stock_data, minute_data)
        >>> print(f"CMF={result.cmf:.3f}, Score={result.flow_score:.3f}")
    """

    def __init__(self, config):
        self.config = config
        self.cmf_calc = CMFCalculator()
        self.obv_calc = OBVCalculator()
        self.micro_extractor = MicroFactorExtractor()
        self.sector_analyzer = SectorFlowAnalyzer()
        self.factor_calc = FactorCalculator()

    def analyze_stock(self, stock_data: pd.DataFrame,
                      minute_data: Optional[pd.DataFrame] = None,
                      date: Optional[str] = None) -> CapitalFlowResult:
        """分析单只股票的资金流状况

        Args:
            stock_data: 个股日线数据 (含 OHLCV)
            minute_data: 当天分钟K线 (可选)
            date: 分析日期 (用于结果标记)

        Returns:
            CapitalFlowResult
        """
        if stock_data is None or len(stock_data) < 20:
            return CapitalFlowResult(
                symbol="", date=date or "",
                cmf=0.0, cmf_turn_signal=0.0,
                obv=0.0, obv_change_pct=0.0,
                flow_score=0.0, flow_direction="neutral"
            )

        # CMF
        cmf_val = float(self.cmf_calc.compute(stock_data).iloc[-1])
        # CMF拐头
        cmf_turn = float(self.cmf_calc.compute_turn_signal(stock_data).iloc[-1])
        # OBV
        obv_series = self.obv_calc.compute(stock_data)
        obv_val = float(obv_series.iloc[-1])
        obv_chg = float(self.obv_calc.change_pct(obv_series, 5).iloc[-1])

        # 综合资金流得分
        # cmf (40%) + cmf_turn (30%) + obv_chg (30%)
        # 各分量归一化到 [-1, 1]
        cmf_norm = np.clip(cmf_val, -1, 1)
        cmf_turn_norm = np.clip(cmf_turn * 5, -1, 1)  # cmf_turn 值域较小
        obv_norm = np.clip(obv_chg * 5, -1, 1)

        flow_score = float(0.4 * cmf_norm + 0.3 * cmf_turn_norm + 0.3 * obv_norm)
        flow_score = np.clip(flow_score, -1, 1)

        # 方向
        if flow_score > 0.1:
            direction = "inflow"
        elif flow_score < -0.1:
            direction = "outflow"
        else:
            direction = "neutral"

        return CapitalFlowResult(
            symbol="",
            date=date or "",
            cmf=cmf_val,
            cmf_turn_signal=cmf_turn,
            obv=obv_val,
            obv_change_pct=obv_chg,
            flow_score=flow_score,
            flow_direction=direction,
        )

    def analyze_sector(self, flow_matrix: pd.DataFrame,
                       date: pd.Timestamp) -> Tuple[List[str], List[str], Dict[str, float]]:
        """分析行业板块资金流"""
        return self.sector_analyzer.compute_sector_scores(flow_matrix, date)

    def extract_micro_factors(self, minute_data: pd.DataFrame,
                              daily_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """提取微观结构因子"""
        return self.micro_extractor.extract_all(minute_data, daily_data)

    def compute_factors(self, stock_data: pd.DataFrame) -> Dict[str, float]:
        """计算全部日线因子"""
        return self.factor_calc.compute_all(stock_data)

    def detect_flow_divergence(self, stock_data: pd.DataFrame) -> Dict[str, bool]:
        """检测量价背离

        Returns:
            {
                'price_up_flow_down': bool,  价格创新高但CMF下降
                'price_down_flow_up': bool,  价格创新低但CMF上升
                'volume_price_div': bool,    放量下跌 (量价背离)
            }
        """
        if stock_data is None or len(stock_data) < 60:
            return {"price_up_flow_down": False,
                    "price_down_flow_up": False,
                    "volume_price_div": False}

        close = stock_data["close"]
        cmf = self.cmf_calc.compute(stock_data)

        # 价格60日新高 + CMF下降
        price_high = close.iloc[-1] >= close.tail(60).max()
        cmf_down = cmf.iloc[-1] < cmf.iloc[-5]

        # 价格60日新低 + CMF上升
        price_low = close.iloc[-1] <= close.tail(60).min()
        cmf_up = cmf.iloc[-1] > cmf.iloc[-5]

        # 放量下跌
        volume = stock_data["volume"]
        vol_high = volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.5
        price_down = close.iloc[-1] < close.iloc[-1] * 0.98 if len(close) >= 2 else False

        return {
            "price_up_flow_down": price_high and cmf_down,
            "price_down_flow_up": price_low and cmf_up,
            "volume_price_div": vol_high and price_down,
        }
