# -*- coding: utf-8 -*-
"""
全局配置管理模块

从 config.json 读取系统参数, 避免在代码中硬编码配置信息。
所有模块通过 get_config() 统一获取配置。

与掘金量化统一配置文件 (D:\\开发\\quant_strategy\\gm_config.json) 的关系:
    - gm_config.json 管理掘金 Token 和回测参数 (敏感信息)
    - 本模块管理本系统的业务参数 (数据路径、因子权重、图表样式等)
    - 本模块读取 gm_config.json 获取 Token, 不重复存储
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# 配置数据类
# ============================================================

@dataclass
class DataConfig:
    """数据路径配置"""
    stock_daily_dir: str = "D:/开发/训练数据/stock_daily"
    money_flow_dir: str = "D:/开发/训练数据/money_flow"
    sector_flow_dir: str = "D:/开发/训练数据/sector_money_flow"
    north_bound_dir: str = "D:/开发/训练数据/north_bound"
    margin_trade_dir: str = "D:/开发/训练数据/margin_trade"
    minute_kline_dir: str = "D:/开发/训练数据/minute_kline"
    index_daily_dir: str = "D:/开发/训练数据/index_daily"
    news_data_dir: str = "D:/开发/训练数据/news_data"
    # Parquet 落库目录 (优先读取, CSV 降级)
    stock_daily_pq_dir: str = "D:/开发/训练数据/stock_daily_pq"
    index_daily_pq_dir: str = "D:/开发/训练数据/index_daily_pq"
    north_bound_pq_dir: str = "D:/开发/训练数据/north_bound_pq"
    margin_trade_pq_dir: str = "D:/开发/训练数据/margin_trade_pq"
    # 财务与成分股
    financial_dir: str = "D:/开发/训练数据/financial"
    constituents_dir: str = "D:/开发/训练数据/constituents"
    # 数据质量日志
    data_quality_log_dir: str = "D:/开发/训练数据/logs"
    meetings_index_path: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output", "data", "meeting_minutes_index.pkl")
    output_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    gm_config_path: str = "D:/开发/quant_strategy/gm_config.json"


@dataclass
class DataQualityConfig:
    """数据质量 / 交叉验证配置"""
    # AKShare vs BaoStock 收盘价差异告警阈值 (0.5%)
    diff_threshold: float = 0.005
    enable_baostock: bool = True
    # 抽样验证比例 (全量太慢, 5% 足以发现系统性问题)
    sample_ratio: float = 0.05


@dataclass
class FactorWeights:
    """因子权重配置 (日线因子)"""
    cmf_turn: float = 0.35
    vol_20d: float = 0.18
    mom_20d: float = 0.15
    overnight_ret: float = 0.10
    vol_rs: float = 0.10
    cmf_20: float = 0.07
    illiq: float = 0.05


@dataclass
class SentimentConfig:
    """情绪分析配置"""
    # 情绪数据源
    news_sources: List[str] = field(default_factory=lambda: ["finance_news", "social_media", "analyst_report"])
    # 情绪词典路径 (可替换为中文金融情感词典)
    sentiment_dict_path: str = "D:/开发/训练数据/sentiment_dict.json"
    # 情绪窗口 (交易日)
    sentiment_window: int = 5
    # 情绪信号平滑系数
    sentiment_smooth_alpha: float = 0.3
    # 极端情绪阈值
    extreme_bullish: float = 0.8
    extreme_bearish: float = -0.8


@dataclass
class FusionConfig:
    """融合决策配置"""
    # 信号权重
    sentiment_weight: float = 0.35
    capital_flow_weight: float = 0.45
    macro_weight: float = 0.20
    # 仓位参数
    min_position: float = 0.00
    max_position: float = 1.00
    # 持仓数量
    max_holdings: int = 10
    # 换手率上限
    max_turnover: float = 0.60
    # 板块集中度: 单板块最多持有N只
    max_per_sector: int = 3


@dataclass
class BacktestConfig:
    """回测参数配置"""
    start_date: str = "2019-01-01"
    end_date: str = "2026-05-31"
    initial_cash: float = 10_000_000
    commission_ratio: float = 0.0005      # 单边佣金
    stamp_tax: float = 0.001              # 卖出印花税 (新增字段)
    slippage_ratio: float = 0.001
    rebalance_freq: str = "W"  # 调仓频率: W=周, M=月
    benchmark: str = "SHSE.000300"  # 基准: 沪深300


@dataclass
class BigDataConfig:
    """大数据平台配置 (HDFS / Spark)"""
    hdfs_host: str = "localhost"
    hdfs_rpc_port: int = 9000          # NameNode RPC (fs.defaultFS)
    hdfs_web_port: int = 9870          # WebHDFS REST
    hdfs_user: str = "hadoop"
    spark_master: str = "local[*]"
    # HDFS 路径 (fs.defaultFS 前缀 + 相对路径)
    hdfs_fs: str = "hdfs://localhost:9000"
    hdfs_data_dir: str = "/quant/data"
    hdfs_results_dir: str = "/quant/results"
    hdfs_login_log_dir: str = "/quant/logs/login"


@dataclass
class SystemConfig:
    """系统总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    factors: FactorWeights = field(default_factory=FactorWeights)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    bigdata: BigDataConfig = field(default_factory=BigDataConfig)
    gm_token: str = ""
    tushare_token: str = ""


# ============================================================
# 配置加载与访问
# ============================================================

_config: Optional[SystemConfig] = None


def load_gm_token() -> str:
    """从 gm_config.json 获取掘金 Token"""
    gm_path = os.path.expanduser("D:/开发/quant_strategy/gm_config.json")
    if os.path.exists(gm_path):
        with open(gm_path, "r", encoding="utf-8") as f:
            gm_cfg = json.load(f)
        return gm_cfg.get("token", "")
    return ""


def load_tushare_token() -> str:
    """从环境变量或 gm_config.json 获取 Tushare Token (留空则不使用历史成分股)"""
    tok = os.environ.get("TUSHARE_TOKEN", "")
    if tok:
        return tok
    gm_path = os.path.expanduser("D:/开发/quant_strategy/gm_config.json")
    if os.path.exists(gm_path):
        with open(gm_path, "r", encoding="utf-8") as f:
            gm_cfg = json.load(f)
        return gm_cfg.get("tushare_token", "")
    return ""


def get_config() -> SystemConfig:
    """获取全局配置对象 (首次调用时初始化)"""
    global _config
    if _config is None:
        _config = SystemConfig()
        _config.gm_token = load_gm_token()
        _config.tushare_token = load_tushare_token()
    return _config


def reload_config(config_dict: Optional[Dict[str, Any]] = None) -> SystemConfig:
    """重新加载配置 (用于用户自定义覆盖)

    Args:
        config_dict: 可选, 部分配置覆盖字典, 如 {"backtest": {"start_date": "2020-01-01"}}

    Returns:
        更新后的 SystemConfig 对象
    """
    global _config
    _config = SystemConfig()
    _config.gm_token = load_gm_token()
    _config.tushare_token = load_tushare_token()

    if config_dict:
        for key, value in config_dict.items():
            if hasattr(_config, key):
                sub_config = getattr(_config, key)
                if isinstance(value, dict) and sub_config is not None:
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_config, sub_key):
                            setattr(sub_config, sub_key, sub_value)
                else:
                    setattr(_config, key, value)

    return _config
