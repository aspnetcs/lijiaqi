# -*- coding: utf-8 -*-
"""
指数成分股读取器

优先用 Tushare 历史权重 (避免幸存者偏差), 无 token 时降级 AKShare 当前成分股并告警.
"""
import os
import pandas as pd
from typing import List, Optional


class ConstituentsReader:
    def __init__(self, constituents_dir: str, tushare_token: str = ""):
        self.dir = constituents_dir
        self.tushare_token = tushare_token
        self._current = None
        self._weight = None  # Tushare 历史权重

    @property
    def current(self) -> pd.DataFrame:
        """当前沪深300成分股 (AKShare index_stock_cons_csindex)"""
        if self._current is None:
            path = os.path.join(self.dir, "hs300_current.csv")
            if os.path.exists(path):
                self._current = pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str})
            else:
                self._current = pd.DataFrame(columns=["code", "name"])
        return self._current

    @property
    def has_historical(self) -> bool:
        """是否有历史成分股数据 (BaoStock 或 Tushare 下载到 hs300_weight/)"""
        weight_dir = os.path.join(self.dir, "hs300_weight")
        if not os.path.isdir(weight_dir):
            return False
        return any(f.endswith(".parquet") for f in os.listdir(weight_dir))

    def _load_weight(self) -> Optional[pd.DataFrame]:
        if self._weight is None and self.has_historical:
            import glob
            files = glob.glob(os.path.join(self.dir, "hs300_weight", "*.parquet"))
            if files:
                self._weight = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        return self._weight

    def get_hs300(self, date: str) -> List[str]:
        """获取指定日期的沪深300成分股代码列表

        优先级:
          1. Tushare 历史权重 (取 <=date 的最新成分)
          2. 降级: 当前成分股 (存在幸存者偏差)
        """
        w = self._load_weight()
        if w is not None and len(w) > 0:
            w = w.copy()
            w["trade_date"] = pd.to_datetime(w["trade_date"]).dt.strftime("%Y-%m-%d")
            sub = w[w["trade_date"] <= date]
            if len(sub) > 0:
                latest = sub["trade_date"].max()
                return sub[sub["trade_date"] == latest]["con_code"].astype(str).str.zfill(6).tolist()
        # 降级
        codes = self.current["code"].astype(str).str.zfill(6).tolist() if "code" in self.current.columns else []
        if codes:
            import logging
            logging.getLogger("constituents").warning(
                "使用当前成分股回测 %s, 存在幸存者偏差 (建议配置 Tushare token 获取历史成分)", date)
        return codes

    def get_current(self) -> List[str]:
        """返回当前成分股代码列表"""
        if "code" in self.current.columns:
            return self.current["code"].astype(str).str.zfill(6).tolist()
        return []

    @staticmethod
    def fetch_tushare_weight(token: str, start: str = "20190101", end: str = None) -> str:
        """从 Tushare 下载沪深300历史权重 (需 token), 保存到 constituents/hs300_weight/

        Returns: 保存的文件路径
        """
        import tushare as ts
        ts.set_token(token)
        pro = ts.pro_api()
        if end is None:
            end = pd.Timestamp.now().strftime("%Y%m%d")
        df = pro.index_weight(index_code="000300.SH", start_date=start, end_date=end)
        out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "constituents", "hs300_weight")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"hs300_weight_{start}_{end}.parquet")
        df.to_parquet(out, index=False)
        return out
