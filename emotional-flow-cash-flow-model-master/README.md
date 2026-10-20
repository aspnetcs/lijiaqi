# A股投资者情绪-资金流融合分析系统

基于面向对象设计的完整AI应用系统，从数据读取、投资者情绪分析、资金流计算、三层递进决策到结果可视化输出。

## 系统架构

```
DataReader --→ SentimentEngine ----┐
DataReader --→ CapitalFlowAnalyzer --→ FusionEngine → ResultExporter
DataReader --→ MacroLayer ---------┘
                    ↓
           SectorLayer → StockLayer
```

## 模块说明

| 模块 | 类 | 职责 |
|------|-----|------|
| data_reader | DataReader + 5个子Reader | 从本地CSV加载日线/资金流/指数/分钟/新闻数据 |
| sentiment_engine | LexiconAnalyzer + MarketSentimentProxy | 中文金融情感词典(60+词) + jieba分词 + 市场代理情绪 |
| capital_flow | CMF/OBV/MicroFactor/Factor计算器 | Chaikin资金流、能量潮、12个微观因子、7个选股因子 |
| triple_layer | MacroLayer/SectorLayer/StockLayer | L1 LPPL泡沫检测 → L2 行业资金流评分 → L3 截面rank选股 |
| fusion_engine | WeightManager/ConflictResolver | 动态权重(4市场状态) + 信号冲突仲裁 + sigmoid仓位映射 |
| result_exporter | ChartGenerator + DataExporter | 净值曲线/热力图/情绪/资金流/融合分析6类图表 |

## 快速开始

### 环境要求

Python >= 3.9, 主要依赖: pandas, numpy, scipy, matplotlib, jieba

```bash
pip install -r requirements.txt
```

### 单元测试

单元测试使用模拟数据，不需要外部数据文件即可运行：

```bash
pytest tests/ -v
```

## 数据依赖说明

**重要：本系统依赖外部金融数据文件，这些文件因体量过大（数十GB）未上传至Git仓库。**

### 所需数据源

| 数据 | 来源 | 路径 | 获取方式 |
|------|------|------|----------|
| 个股日线 (5000+只) | 掘金量化 | `D:/开发/训练数据/stock_daily/` | 掘金SDK下载 |
| 北向资金 | 掘金量化 | `D:/开发/训练数据/north_bound/` | 掘金SDK下载 |
| 融资融券余额 | 国泰安CSMAR | `D:/开发/训练数据/margin_trade/` | 国泰安ZIP → 运行 scripts/preprocess_margin.py |
| 调研纪要文本 | 国泰安CSMAR | Pickle索引 | 国泰安CSV → 运行 scripts/preprocess_meetings.py |
| 投资者互动文本 | 国泰安CSMAR | 并入情绪库 | 运行 scripts/preprocess_interactions.py |
| 行业资金流矩阵 | V2策略生成 | `D:/开发/训练数据/sector_money_flow/` | 掘金SDK + 计算脚本 |
| 行业分类 | westock-data | `src/` 自动生成 | 运行 scripts/build_sw2_industry_map.py |
| 沪深300日线 | 掘金量化 | `D:/开发/训练数据/index_daily/` | 掘金SDK下载 |
| 掘金Token | 用户自行申请 | `D:/开发/quant_strategy/gm_config.json` | 掘金量化官网注册 |

### 数据获取步骤（完整流程）

1. **注册掘金量化**（https://www.myquant.cn）获取 Token，保存至 `D:/开发/quant_strategy/gm_config.json`
2. **下载个股日线/北向/指数数据**：运行 `data/download_data.py`
3. **获取国泰安数据**：通过学校CSMAR数据库下载融资融券和调研纪要原始文件
4. **预处理**：运行 `scripts/` 下的预处理脚本进行格式转换和索引

### 注意事项

- 单元测试（`pytest tests/ -v`）无需数据文件，可用于验证代码逻辑正确性
- 完整回测（`python main.py --start 2025-01-01 --end 2026-01-01`）需要全部数据就位
- 掘金SDK需要有效的Token和网络连接
- 国泰安数据需要高校数据库访问权限

## 回测结果 (2025年)

| 指标 | 策略 | 基准(沪深300) |
|------|:----:|:------------:|
| 总收益 | **+6.32%** | +21.19% |
| 超额收益 | -14.87% | - |
| 夏普比率 | **0.12** | - |
| 最大回撤 | **-5.55%** | -11.20% |
| 胜率 | 54.1% | - |

## 学术参考

- Fama E F, French K R. A Five-Factor Asset Pricing Model. JFE, 2015.
- Jegadeesh N, Titman S. Returns to Buying Winners and Selling Losers. JF, 1993.
- Sornette D, Johansen A. Large Financial Crashes. Physica A, 1997.
- Chaikin M. Chaikin Money Flow (CMF) Indicator.
- Singha M et al. Forecast-to-Fill (arXiv:2511.08571), 2025.
- Singha M. 13-Sharpe OOS Factor (arXiv:2511.12490), 2025.

完整引用清单见实验报告。

## 目录结构

```
sentiment_capital_system/
├── src/                    # 核心源码 (10个模块)
│   ├── config.py           # 配置管理
│   ├── data_reader.py      # 数据读取层 (6个子Reader)
│   ├── sentiment_engine.py # 情绪分析引擎 (词典+代理)
│   ├── capital_flow.py     # 资金流分析器 (CMF/因子)
│   ├── triple_layer.py     # 三层递进决策
│   ├── fusion_engine.py    # 融合决策引擎 (核心创新)
│   └── result_exporter.py  # 结果可视化与导出
├── scripts/                # 数据预处理工具
├── tests/                  # 单元测试
├── docs/                   # 系统文档
├── output/                 # 回测输出
├── main.py                 # 主入口
└── requirements.txt
```

## 开源许可证

MIT License - 见 [LICENSE](LICENSE) 文件
