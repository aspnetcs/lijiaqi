# -*- coding: utf-8 -*-
"""
A股投资者情绪-资金流融合分析系统

模块说明:
    data_reader      -- 数据读取层 (DataReader), 从本地CSV/HDF5加载数据
    sentiment_engine -- 投资者情绪分析引擎 (SentimentEngine), 基于NLP+规则的情绪指数构建
    capital_flow     -- 资金流分析器 (CapitalFlowAnalyzer), CMF/OBV/主力资金计算
    macro_layer      -- L1宏观择时层, 仓位信号生成
    sector_layer     -- L2板块筛选层, 行业资金流+因果网络
    stock_layer      -- L3个股选股层, 多因子截面rank打分
    fusion_engine    -- 情绪-资金流融合决策引擎, 多信号加权融合
    result_exporter  -- 结果可视化与导出, 图表生成+文件输出
    config           -- 全局配置管理, 从config.json读取参数
"""

__version__ = "1.0.0"
__author__ = "实验项目"
