# -*- coding: utf-8 -*-
"""
情绪-资金流融合分析系统 MCP 服务器

将本系统的核心能力以 MCP 工具形式暴露给 AI 客户端 (stdio 传输):

    数据查询:
        - list_stocks            候选股票列表
        - get_stock_daily        个股日线 (OHLCV)
        - get_index_daily        沪深300指数日线
    分析引擎:
        - analyze_market_sentiment  市场情绪分析 (SentimentEngine)
        - analyze_stock_flow        个股资金流分析 (CapitalFlowAnalyzer)
        - get_macro_position        宏观择时仓位 (MacroLayer: 北向+两融+LPPL)
        - fuse_signal               情绪-资金流融合决策 (FusionEngine)
    回测:
        - run_backtest           运行完整回测 (数分钟)
        - get_backtest_stats     读取最近一次回测绩效

启动:  python mcp_server.py        (stdio 模式, 由 MCP 客户端拉起)
依赖:  pip install "mcp<2"
"""

import os
import sys
import json
import contextlib
import io
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ------------------------------------------------------------------
# 预初始化: 所有可能向 stdout 打印的构建过程都在重定向下完成,
# 保证 stdio 协议通道干净 (DataReader/jieba 初始化会 print)
# ------------------------------------------------------------------
with contextlib.redirect_stdout(io.StringIO()):
    from src.config import get_config
    from src.data_reader import DataReader
    from src.sentiment_engine import SentimentEngine
    from src.capital_flow import CapitalFlowAnalyzer
    from src.fusion_engine import FusionEngine
    from src.triple_layer import MacroLayer

    _CFG = get_config()
    _READER = DataReader(_CFG)
    _SENTIMENT = SentimentEngine(_CFG)
    _FLOW = CapitalFlowAnalyzer(_CFG)
    _FUSION = FusionEngine(_CFG)
    _MACRO = None  # 延迟构建 (需加载北向/两融/指数全历史)

    try:
        import jieba
        jieba.initialize()  # 预建词典缓存, 避免首次分析时打印
    except Exception:
        pass

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("emotion-flow-model")


# ------------------------------------------------------------------
# 序列化
# ------------------------------------------------------------------

def _json_default(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_json_default, indent=2)


def _df_to_records(df: pd.DataFrame, max_rows: int) -> list:
    """DataFrame(索引为日期) -> records, 只保留最近 max_rows 行"""
    if df is None or len(df) == 0:
        return []
    out = df.tail(max_rows).copy()
    out.index = out.index.astype(str)
    out = out.reset_index().rename(columns={"index": "date"})
    if "eob" in out.columns:
        out["eob"] = out["eob"].astype(str)
    return out.to_dict("records")


def _get_macro() -> MacroLayer:
    """延迟构建宏观层 (全历史北向/两融/指数)"""
    global _MACRO
    if _MACRO is None:
        today = datetime.now().strftime("%Y-%m-%d")
        with contextlib.redirect_stdout(io.StringIO()):
            north = _READER.get_north_bound("2018-01-01", today)
            margin = _READER.get_margin_trade("2018-01-01", today)
            index = _READER.get_index("000300", "2018-01-01", today)
        _MACRO = MacroLayer(north, margin, benchmark=index)
    return _MACRO


# ------------------------------------------------------------------
# 数据查询工具
# ------------------------------------------------------------------

@mcp.tool()
def list_stocks() -> str:
    """列出本地候选股票代码 (沪深300成份股, 含沪/深主板及创业板, 来自 stock_daily 目录)"""
    stocks = _READER.get_stock_list()
    return _json({"count": len(stocks), "stocks": stocks})


@mcp.tool()
def get_stock_daily(symbol: str, start_date: str, end_date: str,
                    max_rows: int = 250) -> str:
    """获取个股日线行情 (前复权 OHLCV)

    Args:
        symbol: 6位股票代码, 如 '600000'
        start_date: 起始日期 'YYYY-MM-DD'
        end_date:   截止日期 'YYYY-MM-DD'
        max_rows:   最多返回行数 (取最近N个交易日)
    """
    df = _READER.get_stock_daily(symbol, start_date, end_date)
    if df is None or len(df) == 0:
        return _json({"error": f"无数据: {symbol} {start_date}~{end_date}"})
    return _json({"symbol": symbol, "rows": len(df),
                  "data": _df_to_records(df, max_rows)})


@mcp.tool()
def get_index_daily(start_date: str, end_date: str, max_rows: int = 250) -> str:
    """获取沪深300指数日线

    Args:
        start_date: 起始日期 'YYYY-MM-DD'
        end_date:   截止日期 'YYYY-MM-DD'
        max_rows:   最多返回行数
    """
    df = _READER.get_index("000300", start_date, end_date)
    if df is None or len(df) == 0:
        return _json({"error": "指数数据缺失"})
    return _json({"index": "000300", "rows": len(df),
                  "data": _df_to_records(df, max_rows)})


# ------------------------------------------------------------------
# 分析引擎工具
# ------------------------------------------------------------------

@mcp.tool()
def analyze_market_sentiment(date: str) -> str:
    """分析指定交易日的市场情绪 (词典法+市场代理, 输出 composite_sentiment∈[-1,1])

    Args:
        date: 交易日期 'YYYY-MM-DD'
    """
    index = _READER.get_index("000300", "2018-01-01", date)
    market = None
    if index is not None and len(index) > 0:
        market = index.loc[:date].tail(20)
    with contextlib.redirect_stdout(io.StringIO()):
        result = _SENTIMENT.analyze_daily(date, [], market)
    return _json(result)


@mcp.tool()
def analyze_stock_flow(symbol: str, date: str) -> str:
    """分析个股在指定日期的资金流强度 (CMF/OBV/量价因子, flow_score∈[-1,1])

    Args:
        symbol: 6位股票代码, 如 '600000'
        date:   分析日期 'YYYY-MM-DD' (自动回溯120天数据)
    """
    start = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=120)
             ).strftime("%Y-%m-%d")
    df = _READER.get_merged_data(symbol, start, date)
    if df is None or len(df) < 60:
        return _json({"error": f"{symbol} 数据不足60个交易日, 无法分析"})
    with contextlib.redirect_stdout(io.StringIO()):
        result = _FLOW.analyze_stock(df, date=date)
    return _json(result)


@mcp.tool()
def get_macro_position(date: str, sentiment_score: float = 0.0) -> str:
    """宏观择时: 基于北向资金+融资融券+LPPL泡沫检测, 输出建议仓位[0,1]与市场状态

    Args:
        date: 交易日期 'YYYY-MM-DD'
        sentiment_score: 可选, 当日情绪得分[-1,1], 参与仓位计算
    """
    macro = _get_macro()
    with contextlib.redirect_stdout(io.StringIO()):
        signal = macro.compute_signal(pd.Timestamp(date),
                                      sentiment_score=sentiment_score)
    return _json(signal)


@mcp.tool()
def fuse_signal(sentiment: float, flow: float, macro_position: float,
                date: str = "") -> str:
    """情绪-资金流融合决策: 输出综合方向信号[-1,1]、建议仓位[0,1]、置信度

    Args:
        sentiment:      情绪得分 [-1,1] (可由 analyze_market_sentiment 获得)
        flow:           资金流得分 [-1,1] (截面流入流出比例)
        macro_position: 宏观建议仓位 [0,1] (可由 get_macro_position 获得)
        date:           可选日期标签 'YYYY-MM-DD'
    """
    result = _FUSION.fuse(sentiment=sentiment, flow=flow,
                          macro_position=macro_position, date=date)
    return _json(result)


# ------------------------------------------------------------------
# 回测工具
# ------------------------------------------------------------------

@mcp.tool()
def run_backtest(start_date: str, end_date: str) -> str:
    """运行完整回测 (周度调仓, 全管道: 情绪+资金流+宏观+选股+模拟交易)。
    耗时约2-5分钟/年, 请确保客户端超时设置足够。

    Args:
        start_date: 回测起点 'YYYY-MM-DD' (建议 >= 2025-01-01, 个股数据自2024-06起)
        end_date:   回测终点 'YYYY-MM-DD' (建议 <= 2026-01-31)
    """
    import main as backtest_main

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            backtest_main.run_backtest(start_date, end_date, resume=False)
    except Exception as e:
        return _json({"error": f"回测失败: {e}",
                      "log_tail": buf.getvalue()[-2000:]})

    stats_path = os.path.join(_CFG.data.output_dir, "data",
                              "performance_stats.json")
    stats = None
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    return _json({"stats": stats,
                  "output_dir": _CFG.data.output_dir,
                  "log_tail": buf.getvalue()[-2000:]})


@mcp.tool()
def get_backtest_stats() -> str:
    """读取最近一次回测的绩效统计 (总收益/夏普/最大回撤/胜率等) 与输出文件清单"""
    stats_path = os.path.join(_CFG.data.output_dir, "data",
                              "performance_stats.json")
    if not os.path.exists(stats_path):
        return _json({"error": "尚无回测结果, 请先调用 run_backtest"})
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    charts_dir = os.path.join(_CFG.data.output_dir, "charts")
    charts = sorted(os.listdir(charts_dir)) if os.path.isdir(charts_dir) else []
    return _json({"stats": stats, "charts": charts,
                  "output_dir": _CFG.data.output_dir})


if __name__ == "__main__":
    mcp.run()  # stdio 传输
