# -*- coding: utf-8 -*-
"""
登录审计: 登录/注册事件追加写入 HDFS (JSON Lines)

HDFS 是旁路日志通道, 写入失败只告警、绝不阻塞登录流程。
"""
import json
import logging
from datetime import datetime

log = logging.getLogger(__name__)


def log_login_event(username: str, ip: str = "", success: bool = True) -> None:
    """追加一条登录事件到 HDFS /quant/logs/login/login-{YYYYMM}.jsonl"""
    try:
        from src.bigdata.hdfs_client import get_client
        client = get_client()
        if not client._alive():
            log.warning("HDFS 不可达, 登录日志本次未写入 (%s)", username)
            return
        base = get_log_dir()
        if not client.exists(base):
            client.mkdir(base)
        fname = f"{base}/login-{datetime.now():%Y%m}.jsonl"
        event = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "username": username,
            "ip": ip,
            "success": bool(success),
        }
        ok = client.append(fname, json.dumps(event, ensure_ascii=False) + "\n")
        if not ok:
            log.warning("登录日志写入 HDFS 失败 (%s)", username)
    except Exception as e:  # noqa: BLE001 任何异常都不能影响登录主流程
        log.warning("登录审计异常: %s", e)


def get_log_dir() -> str:
    from src.config import get_config
    return get_config().bigdata.hdfs_login_log_dir
