# -*- coding: utf-8 -*-
"""
PySpark 批处理作业 (local 模式, 数据源为 HDFS)

Job1 全市场因子计算: 20日动量 / 20日波动率 / 量比
Job2 数据质量聚合: 每只股票的行数、填充数、停牌数、越界涨跌幅计数
Job3 登录日志聚合: 按日期统计登录次数/成功率

用法: python -m src.bigdata.spark_jobs --job all
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime

log = logging.getLogger(__name__)

_SUMMARY_FACTORS = "/quant/results/factors/_summary.json"
_SUMMARY_QUALITY = "/quant/results/quality/_summary.json"


def get_spark():
    """SparkSession 单例 (local[*])"""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from src.config import get_config
    bd = get_config().bigdata
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder
             .master(bd.spark_master)
             .appName("emotion-flow-bigdata")
             .config("spark.hadoop.fs.defaultFS", bd.hdfs_fs)
             # umask=000: Spark 重建输出目录/文件为 777/666, 避免 WebHDFS (hadoop 用户) 写汇总 JSON 时 403
             .config("spark.hadoop.fs.permissions.umask-mode", "000")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.driver.memory", "2g")
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _input_df(spark, parquet_dir: str):
    """读目录下全部 parquet, 从文件名提取 symbol 列"""
    from pyspark.sql import functions as F
    df = (spark.read.parquet(parquet_dir)
          .withColumn("symbol",
                      F.regexp_extract(F.input_file_name(), r"([^/\\]+)\.parquet$", 1)))
    return df


# ------------------------------------------------------------------
# Job1: 全市场批量因子计算
# ------------------------------------------------------------------

def run_factor_job(spark) -> dict:
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    from src.config import get_config
    bd = get_config().bigdata
    src = f"{bd.hdfs_fs}{bd.hdfs_data_dir}/stock_daily_pq"
    out = f"{bd.hdfs_fs}{bd.hdfs_results_dir}/factors"

    df = _input_df(spark, src).select("symbol", "eob", "close", "volume")
    w = Window.partitionBy("symbol").orderBy("eob")
    w20 = w.rowsBetween(-19, 0)

    ret = F.col("close") / F.lag("close", 1).over(w) - 1.0
    factors = (df
               .withColumn("ret", ret)
               .withColumn("mom_20d", F.col("close") / F.lag("close", 20).over(w) - 1.0)
               .withColumn("vol_20d", F.stddev_samp("ret").over(w20))
               .withColumn("vol_avg_20d", F.avg("volume").over(w20))
               .withColumn("volume_ratio", F.col("volume") / F.avg("volume").over(w20))
               .na.drop(subset=["mom_20d", "vol_20d", "volume_ratio"]))

    factors.coalesce(1).write.mode("overwrite").parquet(out)

    # 汇总: 最新一日动量 Top10
    latest = factors.agg(F.max("eob").alias("max_eob")).collect()[0]["max_eob"]
    top = (factors.filter(F.col("eob") == latest)
           .orderBy(F.desc("mom_20d")).limit(10)
           .select("symbol", "close", "mom_20d")
           .collect())
    summary = {
        "job": "factors", "finished_at": datetime.now().isoformat(timespec="seconds"),
        "total_rows": factors.count(),
        "symbols": factors.select("symbol").distinct().count(),
        "latest_date": str(latest),
        "top_momentum": [{"symbol": r["symbol"], "close": round(r["close"], 2),
                          "mom_20d": round(r["mom_20d"], 4)} for r in top],
    }
    from src.bigdata.hdfs_client import get_client
    if not get_client().write(_SUMMARY_FACTORS, json.dumps(summary, ensure_ascii=False, indent=1)):
        log.warning("因子汇总 JSON 写入 HDFS 失败 (检查 /quant 权限)")
    return summary


# ------------------------------------------------------------------
# Job2: 数据质量聚合
# ------------------------------------------------------------------

def run_quality_job(spark) -> dict:
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    from src.config import get_config
    bd = get_config().bigdata
    src = f"{bd.hdfs_fs}{bd.hdfs_data_dir}/stock_daily_pq"
    out = f"{bd.hdfs_fs}{bd.hdfs_results_dir}/quality"

    df = _input_df(spark, src)
    ret = F.col("close") / F.lag("close", 1).over(
        Window.partitionBy("symbol").orderBy("eob")) - 1.0
    df = df.withColumn("ret", ret)

    quality = (df.groupBy("symbol")
               .agg(F.count("*").alias("rows"),
                    F.sum(F.when(F.col("was_filled") == 1, 1).otherwise(0)).alias("filled"),
                    F.sum(F.when(F.col("is_suspended") == 1, 1).otherwise(0)).alias("suspended"),
                    F.sum(F.when(F.col("is_limit_up") == 1, 1).otherwise(0)).alias("limit_up"),
                    F.sum(F.when(F.col("is_limit_down") == 1, 1).otherwise(0)).alias("limit_down"),
                    F.sum(F.when(F.abs(F.col("ret")) > 0.21, 1).otherwise(0)).alias("abnormal_ret"))
               .orderBy("symbol"))

    quality.coalesce(1).write.mode("overwrite").parquet(out)

    total = quality.agg(F.sum("rows").alias("rows"), F.sum("filled").alias("filled"),
                        F.sum("suspended").alias("suspended"),
                        F.sum("abnormal_ret").alias("abnormal")).collect()[0]
    worst = (quality.orderBy(F.desc("abnormal_ret")).limit(5).collect())
    summary = {
        "job": "quality", "finished_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": quality.count(),
        "total_rows": total["rows"], "filled_rows": total["filled"],
        "suspended_rows": total["suspended"], "abnormal_ret_rows": total["abnormal"],
        "worst_symbols": [{"symbol": r["symbol"], "abnormal_ret": r["abnormal_ret"]} for r in worst],
    }
    from src.bigdata.hdfs_client import get_client
    if not get_client().write(_SUMMARY_QUALITY, json.dumps(summary, ensure_ascii=False, indent=1)):
        log.warning("质量汇总 JSON 写入 HDFS 失败 (检查 /quant 权限)")
    return summary


# ------------------------------------------------------------------
# Job3: 登录日志聚合
# ------------------------------------------------------------------

def run_login_stats(spark) -> dict:
    from pyspark.sql import functions as F

    from src.config import get_config
    bd = get_config().bigdata
    src = f"{bd.hdfs_fs}{bd.hdfs_login_log_dir}"

    df = spark.read.text(src)
    parsed = (df.select(F.from_json(F.col("value"),
                                    "ts STRING, username STRING, ip STRING, success BOOLEAN"
                                    ).alias("j"))
              .select("j.*").na.drop(subset=["username"]))
    stats = (parsed.withColumn("date", F.substring("ts", 1, 10))
             .groupBy("date")
             .agg(F.count("*").alias("logins"),
                  F.sum(F.when(F.col("success"), 1).otherwise(0)).alias("success"))
             .orderBy("date"))
    rows = [r.asDict() for r in stats.collect()]
    summary = {"job": "login_stats", "finished_at": datetime.now().isoformat(timespec="seconds"),
               "days": rows}
    from src.bigdata.hdfs_client import get_client
    get_client().write("/quant/results/login_stats/_summary.json",
                       json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


# ------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="PySpark 大数据作业")
    ap.add_argument("--job", choices=["factors", "quality", "login", "all"], default="all")
    args = ap.parse_args()

    spark = get_spark()
    try:
        if args.job in ("factors", "all"):
            log.info("=== Job1 全市场因子计算 ===")
            log.info(json.dumps(run_factor_job(spark), ensure_ascii=False)[:500])
        if args.job in ("quality", "all"):
            log.info("=== Job2 数据质量聚合 ===")
            log.info(json.dumps(run_quality_job(spark), ensure_ascii=False)[:500])
        if args.job in ("login", "all"):
            log.info("=== Job3 登录日志聚合 ===")
            try:
                log.info(json.dumps(run_login_stats(spark), ensure_ascii=False)[:500])
            except Exception as e:  # 日志目录可能为空
                log.warning("登录日志聚合跳过: %s", e)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
