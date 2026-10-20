# -*- coding: utf-8 -*-
"""
结果可视化与导出 (ResultExporter)

生成图表:
    1. 净值曲线对比 (策略 vs 基准)
    2. 月度收益热力图
    3. 情绪指数走势图
    4. 行业资金流热力图
    5. 仓位变化图

遵循中国股市可视化规范: 红涨绿跌
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.dates import DateFormatter
from typing import List, Dict, Optional, Tuple

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 中国股市配色
COLOR_UP = "#E60000"
COLOR_DOWN = "#009900"
COLOR_NAV = "#CC0000"
COLOR_BENCH = "#888888"
COLOR_SENT = "#FF6B35"
COLOR_FLOW = "#0044CC"
COLOR_NEUTRAL = "#666666"


class ChartGenerator:
    """图表生成器"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save_and_close(self, fig, save_name: str) -> str:
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    # ---- 1. 净值曲线 ----

    def plot_nav_curve(self, strategy_nav: pd.Series,
                       benchmark_nav: Optional[pd.Series] = None,
                       save_name: str = "01_nav_curve.png") -> str:
        """净值曲线 + 回撤阴影"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                        gridspec_kw={"height_ratios": [2.5, 1]},
                                        sharex=True)

        # 上: 净值曲线
        ax1.plot(strategy_nav.index, strategy_nav.values,
                color=COLOR_NAV, linewidth=1.5, label="strategy")
        ax1.fill_between(strategy_nav.index, 1.0, strategy_nav.values,
                        where=strategy_nav.values >= 1.0,
                        color=COLOR_UP, alpha=0.08)
        ax1.fill_between(strategy_nav.index, strategy_nav.values, 1.0,
                        where=strategy_nav.values < 1.0,
                        color=COLOR_DOWN, alpha=0.08)

        if benchmark_nav is not None:
            # 对齐索引
            common_idx = strategy_nav.index.intersection(benchmark_nav.index)
            ax1.plot(common_idx, benchmark_nav.loc[common_idx].values,
                    color=COLOR_BENCH, linewidth=1, linestyle="--", alpha=0.7,
                    label="benchmark")

        ax1.axhline(y=1.0, color="gray", linewidth=0.5, linestyle=":")
        ax1.set_ylabel("Net Value", fontsize=12)
        ax1.legend(loc="upper left", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 统计文本框
        total_ret = (strategy_nav.iloc[-1] / strategy_nav.iloc[0] - 1) * 100
        ann_ret = ((strategy_nav.iloc[-1] / strategy_nav.iloc[0]) **
                   (252 / max(len(strategy_nav), 1)) - 1) * 100
        max_dd = ((strategy_nav / strategy_nav.cummax() - 1).min()) * 100

        text = f"Total: {total_ret:+.1f}%  |  Ann: {ann_ret:+.1f}%  |  MaxDD: {max_dd:.1f}%"
        ax1.text(0.02, 0.95, text, transform=ax1.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # 下: 回撤
        drawdown = strategy_nav / strategy_nav.cummax() - 1
        ax2.fill_between(strategy_nav.index, 0, drawdown.values * 100,
                        color=COLOR_DOWN, alpha=0.3)
        ax2.plot(strategy_nav.index, drawdown.values * 100,
                color=COLOR_DOWN, linewidth=0.8)
        ax2.set_ylabel("Drawdown (%)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(DateFormatter("%Y"))

        fig.suptitle("Strategy Net Value & Drawdown", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._save_and_close(fig, save_name)

    # ---- 2. 月度收益热力图 ----

    def plot_heatmap(self, daily_nav: pd.Series,
                     save_name: str = "02_monthly_heatmap.png") -> str:
        """月度收益热力图"""
        monthly = daily_nav.resample("ME").last().pct_change().dropna()
        monthly.index = monthly.index.to_period("M")

        # 构建矩阵
        years = sorted(set(m.year for m in monthly.index))
        months = range(1, 13)
        matrix = pd.DataFrame(index=years, columns=list(months), dtype=float)

        for m, val in monthly.items():
            if m.year in matrix.index:
                matrix.loc[m.year, m.month] = val * 100

        fig, ax = plt.subplots(figsize=(14, len(years) * 0.6 + 2))

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("red_green",
                                                  [COLOR_DOWN, "#FFFFFF", COLOR_UP], N=256)

        im = ax.imshow(matrix.values, cmap=cmap, aspect="auto",
                      vmin=-15, vmax=15)

        # 标注数字
        for i in range(len(years)):
            for j in range(12):
                val = matrix.iloc[i, j]
                if not pd.isna(val):
                    text_color = "white" if abs(val) > 8 else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                           fontsize=7, color=text_color, fontweight="bold")

        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)

        # 年度收益
        for i, yr in enumerate(years):
            yr_ret = matrix.loc[yr].sum()
            ax.text(12.3, i, f"{yr_ret:+.1f}%", va="center", fontsize=9,
                   color=COLOR_UP if yr_ret > 0 else COLOR_DOWN, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Monthly Return (%)")
        ax.set_title("Monthly Return Heatmap", fontsize=14, fontweight="bold")

        return self._save_and_close(fig, save_name)

    # ---- 3. 情绪指数图 ----

    def plot_sentiment(self, sentiment_df: pd.DataFrame,
                       save_name: str = "03_sentiment_index.png") -> str:
        """情绪指数走势图

        Args:
            sentiment_df: index=date, columns=[composite_sentiment]
        """
        fig, ax = plt.subplots(figsize=(16, 6))

        sentiment = sentiment_df["composite_sentiment"]
        dates = sentiment.index

        # 填充正负区域
        ax.fill_between(dates, 0, sentiment.values,
                       where=sentiment.values >= 0,
                       color=COLOR_UP, alpha=0.15)
        ax.fill_between(dates, sentiment.values, 0,
                       where=sentiment.values < 0,
                       color=COLOR_DOWN, alpha=0.15)

        ax.plot(dates, sentiment.values, color=COLOR_SENT, linewidth=1.2)

        # 零线
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

        # 极端区域
        ax.axhspan(0.7, 1.0, color=COLOR_UP, alpha=0.05)
        ax.axhspan(-1.0, -0.7, color=COLOR_DOWN, alpha=0.05)

        ax.set_ylabel("Sentiment Score", fontsize=12)
        ax.set_title("Investor Sentiment Index", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

        return self._save_and_close(fig, save_name)

    # ---- 4. 行业资金流热力图 ----

    def plot_sector_flow(self, flow_matrix: pd.DataFrame,
                         top_n: int = 15,
                         save_name: str = "04_sector_flow.png") -> str:
        """行业资金流热力图

        Args:
            flow_matrix: index=date, columns=industry, values=CMF
        """
        if flow_matrix.empty or len(flow_matrix.columns) == 0:
            return ""

        # 取最近活跃的top_n行业
        recent_std = flow_matrix.tail(60).std().sort_values(ascending=False)
        top_industries = recent_std.head(top_n).index.tolist()
        subset = flow_matrix[top_industries].tail(120)

        fig, ax = plt.subplots(figsize=(16, max(8, len(top_industries) * 0.3)))

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("flow_cmap",
                                                  [COLOR_DOWN, "#FFFFFF", COLOR_UP], N=256)

        im = ax.imshow(subset.T.values, cmap=cmap, aspect="auto",
                      vmin=-0.1, vmax=0.1)

        ax.set_yticks(range(len(top_industries)))
        ax.set_yticklabels(top_industries, fontsize=9)

        # X轴日期标注
        n_dates = len(subset)
        xtick_positions = np.linspace(0, n_dates - 1, min(12, n_dates)).astype(int)
        xtick_labels = [subset.index[i].strftime("%Y-%m") for i in xtick_positions]
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=45, fontsize=8)

        plt.colorbar(im, ax=ax, label="CMF")
        ax.set_title("Sector Capital Flow Heatmap", fontsize=14, fontweight="bold")

        return self._save_and_close(fig, save_name)

    # ---- 5. 仓位变化图 ----

    def plot_position(self, positions: pd.Series,
                      save_name: str = "05_position_history.png") -> str:
        """仓位变化图"""
        fig, ax = plt.subplots(figsize=(16, 4))

        ax.fill_between(positions.index, 0, positions.values,
                       color=COLOR_NAV, alpha=0.2)
        ax.plot(positions.index, positions.values,
               color=COLOR_NAV, linewidth=1.2)

        ax.set_ylabel("Position Ratio", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_title("Position History", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        return self._save_and_close(fig, save_name)

    # ---- 6. 融合分析图 ----

    def plot_fusion_analysis(self, fusion_df: pd.DataFrame,
                             save_name: str = "06_fusion_analysis.png") -> str:
        """融合决策分析: 三通道贡献 + 仓位 + 冲突标记"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

        # 上: 三通道贡献堆叠
        ax1 = axes[0]
        ax1.fill_between(fusion_df.index, 0, fusion_df["sentiment_contrib"],
                        color=COLOR_SENT, alpha=0.4, label="Sentiment")
        ax1.fill_between(fusion_df.index,
                        fusion_df["sentiment_contrib"],
                        fusion_df["sentiment_contrib"] + fusion_df["flow_contrib"],
                        color=COLOR_FLOW, alpha=0.4, label="Capital Flow")
        ax1.fill_between(fusion_df.index,
                        fusion_df["sentiment_contrib"] + fusion_df["flow_contrib"],
                        fusion_df["sentiment_contrib"] + fusion_df["flow_contrib"] + fusion_df["macro_contrib"],
                        color=COLOR_NEUTRAL, alpha=0.4, label="Macro")
        ax1.set_ylabel("Contributions", fontsize=11)
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 中: 仓位
        ax2 = axes[1]
        ax2.fill_between(fusion_df.index, 0, fusion_df["position_ratio"],
                        color=COLOR_NAV, alpha=0.2)
        ax2.plot(fusion_df.index, fusion_df["position_ratio"],
                color=COLOR_NAV, linewidth=1.2)

        # 冲突标记
        conflicts = fusion_df[fusion_df["conflict"]]
        for idx in conflicts.index:
            ax2.axvline(x=idx, color=COLOR_DOWN, alpha=0.3, linewidth=0.8)

        ax2.set_ylabel("Position", fontsize=11)
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)

        # 下: 置信度
        ax3 = axes[2]
        ax3.plot(fusion_df.index, fusion_df["confidence"],
                color=COLOR_FLOW, linewidth=1)
        ax3.set_ylabel("Confidence", fontsize=11)
        ax3.set_ylim(0, 1.05)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

        fig.suptitle("Fusion Decision Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        return self._save_and_close(fig, save_name)


# ============================================================
# 数据导出器
# ============================================================

class DataExporter:
    """数据文件导出"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_csv(self, df: pd.DataFrame, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, encoding="utf-8-sig")
        return path

    def export_json(self, data: dict, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path


# ============================================================
# 结果导出主控
# ============================================================

class ResultExporter:
    """结果展示与导出主控"""

    def __init__(self, config):
        output_dir = config.data.output_dir
        self.charts = ChartGenerator(os.path.join(output_dir, "charts"))
        self.data_exporter = DataExporter(os.path.join(output_dir, "data"))
        self.output_dir = output_dir

    def export_all(self,
                   strategy_nav: pd.Series,
                   benchmark_nav: Optional[pd.Series] = None,
                   sentiment_df: Optional[pd.DataFrame] = None,
                   flow_matrix: Optional[pd.DataFrame] = None,
                   fusion_df: Optional[pd.DataFrame] = None,
                   positions: Optional[pd.Series] = None) -> Dict[str, str]:
        """导出所有结果

        Returns:
            {描述: 文件路径}
        """
        files = {}

        # 净值曲线
        if strategy_nav is not None:
            path = self.charts.plot_nav_curve(strategy_nav, benchmark_nav)
            files["nav_curve"] = path

            # 月度热力
            path = self.charts.plot_heatmap(strategy_nav)
            files["monthly_heatmap"] = path

        # 情绪
        if sentiment_df is not None:
            path = self.charts.plot_sentiment(sentiment_df)
            files["sentiment"] = path

        # 行业资金流
        if flow_matrix is not None and not flow_matrix.empty:
            path = self.charts.plot_sector_flow(flow_matrix)
            if path:
                files["sector_flow"] = path

        # 仓位
        if positions is not None:
            path = self.charts.plot_position(positions)
            files["positions"] = path

        # 融合分析
        if fusion_df is not None:
            path = self.charts.plot_fusion_analysis(fusion_df)
            files["fusion_analysis"] = path

        # 导出CSV
        if strategy_nav is not None:
            nav_df = pd.DataFrame({"nav": strategy_nav})
            if benchmark_nav is not None:
                common = strategy_nav.index.intersection(benchmark_nav.index)
                nav_df["benchmark"] = benchmark_nav.loc[common]
            self.data_exporter.export_csv(nav_df, "daily_nav.csv")
            files["daily_nav_csv"] = os.path.join(self.output_dir, "data", "daily_nav.csv")

        if fusion_df is not None:
            self.data_exporter.export_csv(fusion_df.reset_index(), "fusion_results.csv")
            files["fusion_csv"] = os.path.join(self.output_dir, "data", "fusion_results.csv")

        # 性能统计
        if strategy_nav is not None:
            stats = self._compute_stats(strategy_nav, benchmark_nav)
            self.data_exporter.export_json(stats, "performance_stats.json")
            files["stats"] = os.path.join(self.output_dir, "data", "performance_stats.json")

        return files

    @staticmethod
    def _compute_stats(nav: pd.Series,
                       bench_nav: Optional[pd.Series] = None) -> dict:
        """计算策略统计指标"""
        returns = nav.pct_change().dropna()

        total_ret = (nav.iloc[-1] / nav.iloc[0] - 1)
        n_years = len(returns) / 252
        ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.03) / (ann_vol + 1e-8)
        max_dd = (nav / nav.cummax() - 1).min()
        win_rate = (returns > 0).sum() / len(returns)

        stats = {
            "total_return_pct": round(total_ret * 100, 2),
            "annual_return_pct": round(ann_ret * 100, 2),
            "annual_volatility_pct": round(ann_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate_pct": round(win_rate * 100, 2),
            "n_days": len(returns),
            "n_years": round(n_years, 1),
        }

        if bench_nav is not None:
            common = nav.index.intersection(bench_nav.index)
            bench_aligned = bench_nav.loc[common]
            strat_aligned = nav.loc[common]
            bench_total = bench_aligned.iloc[-1] / bench_aligned.iloc[0] - 1
            stats["benchmark_return_pct"] = round(bench_total * 100, 2)
            stats["excess_return_pct"] = round(total_ret * 100 - bench_total * 100, 2)

        return stats
