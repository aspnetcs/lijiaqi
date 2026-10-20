# -*- coding: utf-8 -*-
"""
A股投资者情绪-资金流融合分析系统 -- 主入口

功能: 串联所有模块, 执行完整的"数据加载 -> 分析 -> 决策 -> 输出"流程

运行方式:
    python main.py                          # 默认: 完整回测 (2019~2026)
    python main.py --start 2024-01-01       # 指定起止日期
    python main.py --mode daily             # 单日分析模式
    python main.py --mode test              # 运行单元测试

系统架构:
    DataReader -> SentimentEngine        —+
    DataReader -> CapitalFlowAnalyzer    —+—> FusionEngine -> ResultExporter
    DataReader -> MacroLayer             —+
"""

import sys
import os
import json
import pickle
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import get_config, reload_config
from src.data_reader import DataReader
from src.sentiment_engine import SentimentEngine
from src.capital_flow import CapitalFlowAnalyzer
from src.triple_layer import MacroLayer, SectorLayer, StockLayer, StockScore
from src.fusion_engine import FusionEngine
from src.result_exporter import ResultExporter


# ============================================================
# 断点续传 (Checkpoint)
# ============================================================

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output", "checkpoint"
)


def _checkpoint_path(start_date: str, end_date: str) -> str:
    """根据回测区间生成 checkpoint 文件路径"""
    return os.path.join(CHECKPOINT_DIR, f"backtest_{start_date}_{end_date}.pkl")


def _find_latest_checkpoint(start_date: str, end_date: str) -> str | None:
    """查找是否存在可用的 checkpoint 文件"""
    path = _checkpoint_path(start_date, end_date)
    if os.path.exists(path):
        return path
    return None


def _save_checkpoint(start_date: str, end_date: str, state: dict) -> None:
    """保存回测状态到 checkpoint 文件

    state 包含:
        - cash, holdings, cost_basis
        - nav_history, sentiment_history, fusion_results, position_history
        - reb_idx (已完成到第几个调仓日)
        - day_i (已完成到第几个交易日)
        - timestamp (保存时间)

    使用 pickle 直接序列化 dataclass 对象, 无需手动转字典。
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = _checkpoint_path(start_date, end_date)

    state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"      [checkpoint] 已保存 -> reb_idx={state.get('reb_idx', '?')}/{state.get('total_reb', '?')}", flush=True)


def _load_checkpoint(path: str) -> dict:
    """从 checkpoint 文件恢复回测状态"""
    with open(path, "rb") as f:
        state = pickle.load(f)

    saved_time = state.get("timestamp", "unknown")
    print(f"\n  [checkpoint] 从 {path} 恢复")
    print(f"  [checkpoint] 保存时间: {saved_time}")
    print(f"  [checkpoint] 已完成交易日: {state.get('day_i', 0)}/{state.get('total_days', '?')}")
    print(f"  [checkpoint] 已完成调仓: {state.get('reb_idx', 0)}/{state.get('total_reb', '?')}")
    print(f"  [checkpoint] 当前现金: {state.get('cash', 0):,.0f}")
    print(f"  [checkpoint] 当前持仓数: {len(state.get('holdings', {}))}")

    return state


def _clear_checkpoint(start_date: str, end_date: str) -> None:
    """回测完成后清除 checkpoint"""
    path = _checkpoint_path(start_date, end_date)
    if os.path.exists(path):
        os.remove(path)
        print(f"  [checkpoint] 回测完成, 已清除: {path}")


def _get_close(reader, symbol, target_date_str):
    """获取股票在目标日期(含)之前的最新收盘价

    因为调仓日不一定是交易日, 用宽窗口向前查找。
    """
    lookback_start = (pd.to_datetime(target_date_str) - timedelta(days=10)).strftime('%Y-%m-%d')
    df = reader.get_stock_daily(symbol, lookback_start, target_date_str)
    if df is not None and len(df) > 0:
        return float(df['close'].iloc[-1])
    return None


def _build_trade_calendar(index_df, start_date, end_date):
    """从指数日线中提取交易日历 (index_df 的 index 即为交易日)"""
    mask = (index_df.index >= pd.to_datetime(start_date)) & (index_df.index <= pd.to_datetime(end_date))
    return index_df.loc[mask].index.tolist()


# ============================================================
# 完整回测
# ============================================================

def run_backtest(start_date: str, end_date: str, resume: bool = False):
    """运行完整回测

    按周调仓, 每个调仓日执行:
        1. 加载最新数据
        2. SentimentEngine 计算情绪
        3. CapitalFlowAnalyzer 计算资金流
        4. MacroLayer 决定仓位
        5. SectorLayer 筛选板块
        6. StockLayer 截面选股
        7. FusionEngine 融合决策
        8. 模拟交易, 记录收益
    回测结束后输出净值曲线、绩效统计、情绪图表等。

    支持断点续传: 每个调仓日结束后自动保存状态,
    下次运行时加 --resume 即可从断点继续。
    """
    config = reload_config({
        'backtest': {'start_date': start_date, 'end_date': end_date}
    })

    print("=" * 60)
    print("  A股投资者情绪-资金流融合分析系统")
    print(f"  回测区间: {start_date} ~ {end_date}")
    if resume:
        print("  [断点续传模式]")
    print("=" * 60)

    # ---- 1. 初始化数据读取 ----
    reader = DataReader(config)

    # ---- 2. 加载宏观数据 ----
    print("\n[Step 1/6] 加载宏观数据...")
    north_df = reader.get_north_bound(start_date, end_date)
    margin_df = reader.get_margin_trade(start_date, end_date)
    index_df = reader.get_index('000300', '2017-01-01', end_date)

    if index_df is None:
        print("[ERROR] 沪深300指数数据缺失, 无法运行回测")
        print("[INFO] 请先运行: python data/download_data.py --data-types index_daily")
        return

    print(f"  沪深300: {len(index_df)} 行")
    print(f"  北向资金: {len(north_df) if north_df is not None else 0} 行")
    print(f"  融资融券: {len(margin_df) if margin_df is not None else 0} 行")

    # ---- 3. 初始化分析模块 ----
    print("\n[Step 2/6] 初始化分析引擎...")
    sentiment_engine = SentimentEngine(config)
    flow_analyzer = CapitalFlowAnalyzer(config)
    macro_layer = MacroLayer(north_df, margin_df, benchmark=index_df)
    sector_layer = SectorLayer()
    stock_layer = StockLayer(config)
    fusion_engine = FusionEngine(config)
    exporter = ResultExporter(config)

    # ---- 4. 加载股票列表 ----
    print("\n[Step 3/6] 加载股票列表...")
    all_stocks = reader.get_stock_list(exclude_kcb=True, exclude_bj=True)
    print(f"  候选股票: {len(all_stocks)} 只")

    # 加载行业映射
    stock_industry = reader.industry_map.load()
    print(f"  行业映射: {len(stock_industry)} 只")

    # ---- 5. 调仓日历 (在主循环里基于交易日历构建, 这里只打印) ----
    print("\n[Step 4/6] 调仓日历将在主循环中基于交易日历生成")

    # ---- 6. 加载板块资金流矩阵 ----
    print("\n[Step 5/6] 加载行业资金流数据...")
    sector_flow_matrix = reader.get_sector_flow_matrix(
        '2018-01-01', end_date)
    print(f"  行业资金流: {len(sector_flow_matrix)} 天 x "
          f"{len(sector_flow_matrix.columns)} 行业")

    # ---- 7. 构建交易日历 + 调仓日历 ----
    print("\n[Step 6/6] 开始回测...")
    trade_days = _build_trade_calendar(index_df, start_date, end_date)
    print(f"  交易日总数: {len(trade_days)}")

    # 调仓日: 每周第一个交易日
    rebalance_set = set()
    prev_week = None
    for td in trade_days:
        iso_week = td.isocalendar()[1]
        if iso_week != prev_week:
            rebalance_set.add(td.normalize())
            prev_week = iso_week
    rebalance_dates_sorted = [d for d in trade_days if d.normalize() in rebalance_set]
    print(f"  调仓日: {len(rebalance_dates_sorted)} 个")

    # ---- 检查断点续传 ----
    checkpoint = None
    if resume:
        ckpt_path = _find_latest_checkpoint(start_date, end_date)
        if ckpt_path:
            checkpoint = _load_checkpoint(ckpt_path)
        else:
            print("  [checkpoint] 未找到可恢复的断点, 从头开始回测")

    # ---- 初始化状态 (或从断点恢复) ----
    if checkpoint:
        cash = checkpoint["cash"]
        holdings = checkpoint["holdings"]
        cost_basis = checkpoint["cost_basis"]
        nav_history = checkpoint["nav_history"]
        sentiment_history = checkpoint["sentiment_history"]
        fusion_results = checkpoint["fusion_results"]
        position_history = checkpoint["position_history"]
        prev_smoothed_pos = checkpoint.get("prev_smoothed_pos")
        start_reb_idx = checkpoint.get("reb_idx", 0)
        start_day_i = checkpoint.get("day_i", 0)
        # 清除上一轮的 holdings_data_cache, 重新加载
        holdings_data_cache = {}
    else:
        cash = config.backtest.initial_cash
        holdings = {}          # {symbol: shares}
        cost_basis = {}        # {symbol: avg_cost_per_share} 用于止损
        nav_history = []       # [{'date': ..., 'nav': ...}, ...]
        sentiment_history = [] # [SentimentResult, ...]
        fusion_results = []    # [FusionResult, ...]
        position_history = []  # [{'date': ..., 'position': ...}, ...]
        prev_smoothed_pos = None
        start_reb_idx = 0
        start_day_i = 0

    stop_loss_pct = 0.08   # 单股止损线: 跌破成本价8%


    # 用于按交易日估值: 缓存每只持仓股的日线数据
    holdings_data_cache = {}

    n_reb = len(rebalance_dates_sorted)
    reb_idx = start_reb_idx

    # ---- 主循环: 按交易日 ----
    for day_i, trade_date in enumerate(trade_days):
        # 跳过已完成的交易日 (断点续传)
        if day_i < start_day_i:
            continue

        date_str = trade_date.strftime('%Y-%m-%d')

        # ====== (A) 调仓日: 止损检查 + 执行决策与换仓 ======
        if reb_idx < n_reb and trade_date.normalize() == rebalance_dates_sorted[reb_idx].normalize():
            print(f"  [调仓 {reb_idx + 1}/{n_reb}] {date_str}", flush=True)

            # ---- A0. 止损检查: 卖出跌幅超过8%的持仓 ----
            stopped_out = []
            for sym in list(holdings.keys()):
                price = _get_close(reader, sym, date_str)
                if price is None or sym not in cost_basis:
                    continue
                pct_loss = (price - cost_basis[sym]) / cost_basis[sym]
                if pct_loss < -stop_loss_pct:
                    proceeds = holdings[sym] * price * (
                        1 - config.backtest.commission_ratio - config.backtest.stamp_tax)
                    cash += proceeds
                    stopped_out.append(f"{sym}({pct_loss:+.1%})")
                    del holdings[sym]
                    del cost_basis[sym]
            if stopped_out:
                print(f"      止损: {', '.join(stopped_out)}", flush=True)

            # ---- a. 情绪分析 (先做, 因为宏观需要情绪输入) ----
            print(f"      情绪分析...", end=" ", flush=True)

            # 使用调研纪要+投资者互动: 取最近7天的文本 (日均~2100条)
            meeting_texts = reader.get_recent_minutes(
                date_str, lookback_days=7, max_texts=5000
            )
            if meeting_texts:
                day_news_texts = meeting_texts
            else:
                # 备用: 传统新闻源
                news_df = reader.get_news(
                    (trade_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                    date_str
                )
                day_news_texts = []
                if news_df is not None and len(news_df) > 0:
                    for _, row in news_df.tail(5).iterrows():
                        for col in ['title', 'content']:
                            if col in row and pd.notna(row[col]):
                                day_news_texts.append(str(row[col]))

            try:
                day_market = index_df.loc[:trade_date].tail(20)
            except Exception:
                day_market = None

            sentiment_result = sentiment_engine.analyze_daily(
                date_str, day_news_texts, day_market
            )
            sentiment_history.append(sentiment_result)
            print(f"情绪={sentiment_result.composite_sentiment:+.2f}", flush=True)

            # ---- b. 宏观信号 (传入情绪得分, 参与仓位计算) ----
            print(f"      宏观择时...", end=" ", flush=True)
            macro_signal = macro_layer.compute_signal(
                trade_date, sentiment_score=sentiment_result.composite_sentiment
            )
            raw_macro_pos = macro_signal.position_ratio

            # 宏观信号直接使用, 不再做 EMA 平滑 (北向 MA + 两融变化率本身已经平滑)
            print(f"仓位={raw_macro_pos:.2f} 状态={macro_signal.market_regime}", flush=True)

            # ---- c. 板块筛选 (先选板块, 用于后续资金流计算和选股) ----
            print(f"      板块筛选...", end=" ", flush=True)
            inflow_sectors = []
            if sector_flow_matrix is not None and len(sector_flow_matrix) > 0:
                try:
                    sector_signal = sector_layer.select_sectors(
                        sector_flow_matrix, trade_date)
                    inflow_sectors = sector_signal.inflow_sectors
                except Exception:
                    pass
            print(f"流入板块{len(inflow_sectors)}个", flush=True)

            # ---- d. 候选股票池构建 ----
            # 优先取流入板块内的股票; 若为空则全市场
            stock_pool = []
            if inflow_sectors:
                for ind in inflow_sectors:
                    stocks = reader.get_stocks_in_industry(ind)
                    stock_pool.extend(stocks)
            if not stock_pool or len(stock_pool) < 50:
                # 候选池过小, 退化为全市场抽样
                stock_pool = all_stocks[:]

            # 加载数据 (最多800只)
            print(f"      加载数据({min(len(stock_pool), 800)}/{len(stock_pool)})...", end=" ", flush=True)
            stock_data = {}
            lookback_start = (trade_date - timedelta(days=120)).strftime('%Y-%m-%d')
            for sym in stock_pool[:800]:
                df = reader.get_merged_data(sym, lookback_start, date_str)
                if df is not None and len(df) >= 60:
                    stock_data[sym] = df
            print(f"有效{len(stock_data)}只", flush=True)

            # ---- e. 资金流得分: 每板块等比例取样，覆盖全部候选股 ----
            print(f"      资金流...", end=" ", flush=True)

            # 按板块分组股票 (确保每个流入板块都有发言权)
            sector_stocks = {}  # {sector_name: [symbol, ...]}
            no_sector = []      # 无行业映射的股票
            for sym in stock_data:
                ind = stock_industry.get(sym, "")
                if ind and ind in inflow_sectors:
                    sector_stocks.setdefault(ind, []).append(sym)
                else:
                    no_sector.append(sym)

            flow_scores = []
            inflow_count = 0
            outflow_count = 0

            if sector_stocks:
                # 每板块等比例取样: 总数不超过总股票数, 每板块至少取 min(sector_size, per_sector)
                n_sectors = len(sector_stocks)
                per_sector = max(1, len(stock_data) // n_sectors)
                for ind, syms in sector_stocks.items():
                    take_n = min(len(syms), per_sector)
                    for sym in syms[:take_n]:
                        result = flow_analyzer.analyze_stock(stock_data[sym], date=date_str)
                        fs = result.flow_score
                        flow_scores.append(fs)
                        if fs > 0.05:
                            inflow_count += 1
                        elif fs < -0.05:
                            outflow_count += 1
            else:
                # 无板块时退化为全量计算
                for sym, df in stock_data.items():
                    result = flow_analyzer.analyze_stock(df, date=date_str)
                    fs = result.flow_score
                    flow_scores.append(fs)
                    if fs > 0.05:
                        inflow_count += 1
                    elif fs < -0.05:
                        outflow_count += 1

            if flow_scores:
                total = len(flow_scores)
                flow_score = (inflow_count - outflow_count) / max(total, 1)
            else:
                flow_score = 0.0
                total = 0
            print(f"得分={flow_score:+.2f} (流入{inflow_count}/{total})", flush=True)

            # ---- f. 融合决策 ----
            print(f"      融合...", end=" ", flush=True)
            fusion_result = fusion_engine.fuse(
                sentiment=sentiment_result.composite_sentiment,
                flow=flow_score,
                macro_position=macro_signal.position_ratio,
                date=date_str,
            )

            fusion_results.append(fusion_result)
            position_history.append({
                'date': trade_date,
                'position': fusion_result.position_ratio,
            })
            print(f"仓位={fusion_result.position_ratio:.2f} 置信={fusion_result.confidence:.0%}", flush=True)

            # ---- g. 选股 (v4: BULL模式加宽选股池和持股数) ----
            print(f"      选股...", end=" ", flush=True)

            is_bull = (fusion_result.regime.value == "bull")

            if is_bull:
                # BULL: 不限板块, 增加持股数和板块集中度
                stock_layer.max_holdings = 15
                stock_layer.max_per_sector = 5
                pick_sectors = None
            else:
                stock_layer.max_holdings = config.fusion.max_holdings
                stock_layer.max_per_sector = config.fusion.max_per_sector
                pick_sectors = inflow_sectors if inflow_sectors else None

            selected = stock_layer.pick_stocks(
                stock_pool=list(stock_data.keys()),
                stock_data=stock_data,
                position_ratio=fusion_result.position_ratio,
                stock_industry=stock_industry,
                selected_sectors=pick_sectors,
                regime=fusion_result.regime.value,
            )
            # 若选股结果为空 (被板块过滤清空), 放宽限制重选一次
            if not selected and stock_data:
                print(f"(放宽板块)", end=" ", flush=True)
                selected = stock_layer.pick_stocks(
                    stock_pool=list(stock_data.keys()),
                    stock_data=stock_data,
                    position_ratio=fusion_result.position_ratio,
                    stock_industry=None,        # 不做板块过滤
                    selected_sectors=None,
                    regime=fusion_result.regime.value,
                )
            print(f"选中{len(selected)}只", flush=True)

            # ---- h. 调仓模拟 ----
            target_symbols = [s.symbol for s in selected]
            target_weights = {}
            if target_symbols:
                base_weight = fusion_result.position_ratio / len(target_symbols)
                for sym in target_symbols:
                    target_weights[sym] = base_weight

            # 清仓不在目标中的
            current_value = 0.0
            for sym in list(holdings.keys()):
                price = _get_close(reader, sym, date_str)
                if price is None:
                    price = 0.0
                if sym not in target_weights:
                    # 卖出: 佣金 + 印花税
                    proceeds = holdings[sym] * price * (1 - config.backtest.commission_ratio
                                                        - config.backtest.stamp_tax)
                    cash += proceeds
                    del holdings[sym]
                    cost_basis.pop(sym, None)
                else:
                    current_value += holdings[sym] * price

            # 买入/调整目标仓位
            total_equity = cash + current_value
            for sym, w in target_weights.items():
                price = _get_close(reader, sym, date_str)
                if price is None or price <= 0:
                    continue
                target_value = total_equity * w
                current_shares = holdings.get(sym, 0)
                current_val = current_shares * price
                delta = target_value - current_val

                if delta > 0:
                    shares_to_buy = int(delta / price)
                    # 买入: 仅佣金
                    cost = shares_to_buy * price * (1 + config.backtest.commission_ratio)
                    if cost <= cash:
                        cash -= cost
                        # 更新持仓成本
                        if sym in cost_basis and holdings.get(sym, 0) > 0:
                            total_val = cost_basis[sym] * holdings[sym] + shares_to_buy * price
                            total_shares = holdings[sym] + shares_to_buy
                            cost_basis[sym] = total_val / total_shares
                        else:
                            cost_basis[sym] = price
                        holdings[sym] = current_shares + shares_to_buy
                elif delta < 0 and current_shares > 0:
                    shares_to_sell = min(int(-delta / price), current_shares)
                    # 卖出: 佣金 + 印花税
                    proceeds = shares_to_sell * price * (1 - config.backtest.commission_ratio
                                                        - config.backtest.stamp_tax)
                    cash += proceeds
                    remaining = current_shares - shares_to_sell
                    if remaining == 0:
                        holdings.pop(sym)
                        cost_basis.pop(sym, None)
                    else:
                        holdings[sym] = remaining

            # 预加载新持仓的日线数据 (用于后续日估值)
            holdings_data_cache = {}
            for sym in holdings.keys():
                df = reader.get_stock_daily(sym,
                                            (trade_date - timedelta(days=10)).strftime('%Y-%m-%d'),
                                            end_date)
                if df is not None:
                    holdings_data_cache[sym] = df

            reb_idx += 1

            # ---- 调仓完成后保存 checkpoint ----
            _save_checkpoint(start_date, end_date, {
                "cash": cash,
                "holdings": dict(holdings),
                "cost_basis": dict(cost_basis),
                "nav_history": nav_history,
                "sentiment_history": sentiment_history,
                "fusion_results": fusion_results,
                "position_history": position_history,
                "reb_idx": reb_idx,
                "day_i": day_i,
                "total_days": len(trade_days),
                "total_reb": n_reb,
            })

        # ====== (B) 每个交易日: 估值并记录净值 ======
        total_value = cash
        for sym, shares in holdings.items():
            df = holdings_data_cache.get(sym)
            if df is None:
                df = reader.get_stock_daily(sym,
                                            (trade_date - timedelta(days=10)).strftime('%Y-%m-%d'),
                                            date_str)
                if df is not None:
                    holdings_data_cache[sym] = df

            if df is not None and len(df) > 0:
                # 找到 <= trade_date 的最后一行
                valid = df[df.index <= trade_date]
                if len(valid) > 0:
                    price = float(valid['close'].iloc[-1])
                    total_value += shares * price

        nav_history.append({
            'date': trade_date,
            'nav': total_value / config.backtest.initial_cash,
        })

    # ---- 8. 回测结束, 统计与输出 ----
    print("\n" + "=" * 60)
    print("  回测完成, 正在生成报告...")
    print("=" * 60)

    # 清除 checkpoint (回测已完整结束)
    _clear_checkpoint(start_date, end_date)

    # 净值序列
    nav_df = pd.DataFrame(nav_history).set_index('date')['nav']
    nav_df = nav_df.sort_index()

    # 基准净值 (沪深300)
    bench_nav = None
    if index_df is not None and 'close' in index_df.columns:
        # 基准: 在交易日历上构建, 然后对齐到策略净值日期
        bench_full = index_df['close'] / index_df['close'].iloc[0]
        # 先取与策略净值相同日期区间的基准
        bench_nav = bench_full.reindex(nav_df.index, method='ffill')

    # 情绪序列
    sentiment_df = pd.DataFrame([{
        'date': s.date,
        'composite_sentiment': s.composite_sentiment,
        'text_sentiment': s.text_sentiment,
        'market_sentiment': s.market_sentiment,
    } for s in sentiment_history])
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    sentiment_df = sentiment_df.set_index('date').sort_index()

    # 融合结果
    fusion_df = pd.DataFrame([{
        'date': f.date,
        'composite_signal': f.composite_signal,
        'position_ratio': f.position_ratio,
        'sentiment_contrib': f.sentiment_contrib,
        'flow_contrib': f.flow_contrib,
        'macro_contrib': f.macro_contrib,
        'regime': f.regime.value,
        'conflict': f.conflict,
        'confidence': f.confidence,
    } for f in fusion_results])
    if len(fusion_df) > 0:
        fusion_df['date'] = pd.to_datetime(fusion_df['date'])
        fusion_df = fusion_df.set_index('date')

    # 仓位序列
    pos_series = None
    if position_history:
        pos_df = pd.DataFrame(position_history).set_index('date')['position']

    # 输出结果
    files = exporter.export_all(
        strategy_nav=nav_df,
        benchmark_nav=bench_nav,
        sentiment_df=sentiment_df,
        flow_matrix=sector_flow_matrix if sector_flow_matrix is not None and len(sector_flow_matrix) > 0 else None,
        fusion_df=fusion_df if len(fusion_df) > 0 else None,
        positions=pos_df if position_history else None,
    )

    # 打印摘要
    stats = exporter._compute_stats(nav_df, bench_nav)
    print(f"\n  总收益:     {stats['total_return_pct']:+.2f}%")
    print(f"  年化收益:   {stats['annual_return_pct']:+.2f}%")
    print(f"  夏普比率:   {stats['sharpe_ratio']:.2f}")
    print(f"  最大回撤:   {stats['max_drawdown_pct']:.2f}%")
    print(f"  胜率:       {stats['win_rate_pct']:.1f}%")
    if 'benchmark_return_pct' in stats:
        print(f"  基准收益:   {stats['benchmark_return_pct']:+.2f}%")
        print(f"  超额收益:   {stats['excess_return_pct']:+.2f}%")

    print(f"\n  输出文件:")
    for name, path in files.items():
        if isinstance(path, str):
            print(f"    {name}: {path}")

    print(f"\n[INFO] 回测完成, 结果输出目录: {config.data.output_dir}")


# ============================================================
# 单日分析
# ============================================================

def run_daily_analysis(date_str: str):
    """运行单日分析 (用于实际交易日快速评估)"""
    config = get_config()

    print(f"[INFO] 单日分析: {date_str}")

    reader = DataReader(config)
    index_df = reader.get_index('000300',
                                (pd.to_datetime(date_str) - timedelta(days=400)).strftime('%Y-%m-%d'),
                                date_str)
    if index_df is None or len(index_df) < 60:
        print("[ERROR] 沪深300数据不足")
        return

    north_df = reader.get_north_bound(
        (pd.to_datetime(date_str) - timedelta(days=400)).strftime('%Y-%m-%d'),
        date_str)
    margin_df = reader.get_margin_trade(
        (pd.to_datetime(date_str) - timedelta(days=400)).strftime('%Y-%m-%d'),
        date_str)

    sentiment_engine = SentimentEngine(config)
    flow_analyzer = CapitalFlowAnalyzer(config)
    macro_layer = MacroLayer(north_df, margin_df, benchmark=index_df)
    fusion_engine = FusionEngine(config)

    # 情绪
    sentiment_result = sentiment_engine.analyze_daily(
        date_str, [],
        index_df.loc[pd.to_datetime(date_str):pd.to_datetime(date_str)]
        if index_df is not None else None
    )

    # 宏观
    macro_signal = macro_layer.compute_signal(pd.to_datetime(date_str))

    # 抽样资金流
    stock_list = reader.get_stock_list()
    flow_scores = []
    sample = stock_list[:200]
    for sym in sample:
        df = reader.get_merged_data(sym,
                                   (pd.to_datetime(date_str) - timedelta(days=120)).strftime('%Y-%m-%d'),
                                   date_str)
        if df is not None and len(df) >= 60:
            result = flow_analyzer.analyze_stock(df, date=date_str)
            flow_scores.append(result.flow_score)

    flow_score = float(np.mean(flow_scores)) if flow_scores else 0.0

    # 融合
    fusion_result = fusion_engine.fuse(
        sentiment=sentiment_result.composite_sentiment,
        flow=flow_score,
        macro_position=macro_signal.position_ratio,
        date=date_str,
    )

    print(f"\n=== {date_str} 单日分析 ===")
    print(f"  情绪得分:   {sentiment_result.composite_sentiment:+.3f} "
          f"[{sentiment_result.sentiment_label}]")
    print(f"  资金流得分: {flow_score:+.3f}")
    print(f"  宏观仓位:   {macro_signal.position_ratio:.2f}")
    print(f"  市场状态:   {macro_signal.market_regime}")
    if macro_signal.bubble_risk:
        print(f"  [警告] LPPL 泡沫风险!")
    print(f"  综合信号:   {fusion_result.composite_signal:.3f}")
    print(f"  建议仓位:   {fusion_result.position_ratio:.1%}")
    print(f"  置信度:     {fusion_result.confidence:.1%}")
    if fusion_result.conflict:
        print(f"  [仲裁] {fusion_result.conflict_desc}")


# ============================================================
# 单元测试
# ============================================================

def run_tests():
    """运行单元测试"""
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    print("[INFO] 运行单元测试...")
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', test_dir, '-v', '--tb=short'],
        cwd=os.path.dirname(__file__),
        capture_output=False,
    )
    return result.returncode


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='A股情绪-资金流融合分析系统')
    parser.add_argument('--mode', type=str, default='backtest',
                        choices=['backtest', 'daily', 'test'],
                        help='运行模式: backtest=回测, daily=单日分析, test=单元测试')
    parser.add_argument('--start', type=str, default='2024-01-01',
                        help='起始日期 (默认: 2024-01-01, 建议先用短区间验证)')
    parser.add_argument('--end', type=str, default='2025-01-01',
                        help='截止日期')
    parser.add_argument('--date', type=str, default=None,
                        help='单日分析日期 (仅 --mode daily)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='断点续传: 从上次中断的位置继续回测')
    parser.add_argument('--clean', action='store_true', default=False,
                        help='清除已有的 checkpoint, 从头开始回测')

    args = parser.parse_args()

    if args.mode == 'backtest':
        if args.clean:
            _clear_checkpoint(args.start, args.end)
            print("[INFO] 已清除 checkpoint, 从头开始")
        run_backtest(args.start, args.end, resume=args.resume)
    elif args.mode == 'daily':
        run_daily_analysis(args.date or datetime.now().strftime('%Y-%m-%d'))
    elif args.mode == 'test':
        sys.exit(run_tests())


if __name__ == '__main__':
    main()
