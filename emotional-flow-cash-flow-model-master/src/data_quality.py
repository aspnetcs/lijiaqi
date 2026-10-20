# -*- coding: utf-8 -*-
"""
数据质量交叉验证模块

用 BaoStock (免费、无需 token) 作为第二数据源, 与 AKShare 下载的前复权收盘价对比.
差异超过阈值 (默认 0.5%) 的日期记入日志, 用于发现数据质量问题.

注意: AKShare 与 BaoStock 前复权基准日可能不同, 因此仅比较相对涨跌幅, 首日差异忽略.
"""
import os
import logging
import pandas as pd
from datetime import datetime


class CrossValidator:
    def __init__(self, threshold: float = 0.005, log_dir: str = None):
        self.threshold = threshold
        self._bs_logged_in = False
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"data_quality_{datetime.now().strftime('%Y%m%d')}.log")
            self.logger = logging.getLogger("data_quality")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                fh = logging.FileHandler(log_path, encoding="utf-8")
                fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
                self.logger.addHandler(fh)
        else:
            self.logger = logging.getLogger("data_quality")

    def _login(self) -> bool:
        if self._bs_logged_in:
            return True
        try:
            import baostock as bs
            bs.login()
            self._bs_logged_in = True
            return True
        except Exception as e:
            self.logger.warning("BaoStock 登录失败, 跳过验证: %s", e)
            return False

    def _to_bs_code(self, code: str) -> str:
        """600000 -> sh.600000, 000001 -> sz.000001"""
        code = str(code).zfill(6)
        if code.startswith(("6", "9", "5")):
            return f"sh.{code}"
        return f"sz.{code}"

    def validate_stock(self, code: str, ak_df: pd.DataFrame) -> pd.DataFrame:
        """对比 AKShare 与 BaoStock 收盘价, 返回差异报告"""
        if not self._login():
            return pd.DataFrame()
        import baostock as bs
        bs_code = self._to_bs_code(code)
        start = pd.to_datetime(ak_df["eob"].min()).strftime("%Y-%m-%d")
        end = pd.to_datetime(ak_df["eob"].max()).strftime("%Y-%m-%d")
        try:
            rs = bs.query_history_k_data_plus(
                bs_code, "date,close",
                start_date=start, end_date=end,
                frequency="d", adjustflag="2",  # 前复权
            )
            rows = []
            while (rs.error_code == "0") and rs.next():
                rows.append(rs.get_row_data())
            bs_df = pd.DataFrame(rows, columns=rs.fields)
            if len(bs_df) == 0:
                return pd.DataFrame()
            bs_df["close"] = pd.to_numeric(bs_df["close"], errors="coerce")
            bs_df["date"] = pd.to_datetime(bs_df["date"]).dt.strftime("%Y-%m-%d")
            ak = ak_df[["eob", "close"]].copy()
            ak["eob"] = pd.to_datetime(ak["eob"]).dt.strftime("%Y-%m-%d")
            merged = ak.merge(bs_df, left_on="eob", right_on="date", suffixes=("_ak", "_bs"))
            # 比较相对涨跌幅 (消除复权基准日差异)
            merged["ret_ak"] = merged["close_ak"].pct_change()
            merged["ret_bs"] = merged["close_bs"].pct_change()
            merged["diff"] = (merged["ret_ak"] - merged["ret_bs"]).abs()
            merged["is_alert"] = merged["diff"] > self.threshold
            alerts = merged[merged["is_alert"]]
            if len(alerts) > 0:
                self.logger.warning("[%s] %d 个日期涨跌幅差异>%.1f%%, 示例: %s",
                                    code, len(alerts), self.threshold * 100,
                                    alerts[["eob", "ret_ak", "ret_bs"]].head(3).to_dict("records"))
            return merged
        except Exception as e:
            self.logger.warning("[%s] BaoStock 查询失败: %s", code, e)
            return pd.DataFrame()

    def run_batch(self, codes: list) -> dict:
        """批量验证, 返回统计 {total, with_alerts, total_alert_rows}"""
        if not self._login():
            return {"total": 0, "with_alerts": 0, "total_alert_rows": 0}
        stats = {"total": 0, "with_alerts": 0, "total_alert_rows": 0}
        import baostock as bs
        for code in codes:
            pq_path = os.path.join("D:/开发/训练数据/stock_daily_pq", f"{code}.parquet")
            if not os.path.exists(pq_path):
                continue
            try:
                ak_df = pd.read_parquet(pq_path)
                if "eob" not in ak_df.columns or "close" not in ak_df.columns:
                    continue
                report = self.validate_stock(code, ak_df)
                stats["total"] += 1
                if len(report) > 0 and report["is_alert"].any():
                    stats["with_alerts"] += 1
                    stats["total_alert_rows"] += int(report["is_alert"].sum())
            except Exception as e:
                self.logger.warning("[%s] 验证异常: %s", code, e)
        self.logger.info("验证完成: 共%d只, 有差异%d只, 差异日期%d行",
                         stats["total"], stats["with_alerts"], stats["total_alert_rows"])
        try:
            bs.logout()
        except Exception:
            pass
        return stats
