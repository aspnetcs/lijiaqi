# -*- coding: utf-8 -*-
"""
本地数据集上传到 HDFS

用法:
    python data/upload_to_hdfs.py            # 上传全部缺失/过期的数据目录
    python data/upload_to_hdfs.py --force    # 强制重传 (先删远端)
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.config import get_config  # noqa: E402


def _hadoop_fs(*args) -> bool:
    """调用 hadoop fs 命令 (依赖 HADOOP_HOME)"""
    hadoop = os.path.join(os.environ.get("HADOOP_HOME", "D:/hadoop-3.3.6"), "bin", "hadoop.cmd")
    if not os.path.exists(hadoop):
        print(f"[错误] 未找到 {hadoop}, 请确认 HADOOP_HOME")
        return False
    env = dict(os.environ, JAVA_HOME=os.environ.get("JAVA_HOME", "F:/jdk11.0.23"))
    r = subprocess.run([hadoop, "fs", *args], env=env, capture_output=True, text=True, shell=True)
    if r.returncode != 0:
        print(f"[错误] hadoop fs {' '.join(args)} 失败:\n{r.stderr[-800:]}")
        return False
    return True


def upload_all(force: bool = False):
    cfg = get_config()
    bd = cfg.bigdata
    mapping = {
        "stock_daily_pq": cfg.data.stock_daily_pq_dir,
        "index_daily_pq": cfg.data.index_daily_pq_dir,
        "north_bound_pq": cfg.data.north_bound_pq_dir,
        "margin_trade_pq": cfg.data.margin_trade_pq_dir,
        "financial": cfg.data.financial_dir,
        "constituents": cfg.data.constituents_dir,
    }
    if force:
        print("[清理] 删除远端", bd.hdfs_data_dir)
        _hadoop_fs("-rm", "-r", bd.hdfs_data_dir)
    # 先确保父目录存在 (目标存在时 put 才会创建同名子目录; 不存在则变成重命名/拍平)
    _hadoop_fs("-mkdir", "-p", bd.hdfs_data_dir)
    for local_dir in mapping.values():
        if not os.path.isdir(local_dir):
            print(f"[跳过] 本地不存在: {local_dir}")
            continue
        print(f"[上传] {local_dir} -> {bd.hdfs_data_dir}/")
        if _hadoop_fs("-put", local_dir.replace("/", "\\"), bd.hdfs_data_dir):
            print(f"[完成] {os.path.basename(local_dir.rstrip('/'))}")

    # 初始化日志与结果目录
    for d in (bd.hdfs_login_log_dir, bd.hdfs_results_dir):
        _hadoop_fs("-mkdir", "-p", d)
    print("[完成] 目录初始化:", bd.hdfs_login_log_dir, bd.hdfs_results_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="上传本地量化数据到 HDFS")
    ap.add_argument("--force", action="store_true", help="强制重传")
    args = ap.parse_args()
    upload_all(force=args.force)
