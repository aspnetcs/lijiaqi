# -*- coding: utf-8 -*-
"""
WebHDFS REST 客户端

通过 HTTP (NameNode 9870 端口) 读写 HDFS, 避免 Windows 上 libhdfs 原生依赖。
协议文档: https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/WebHDFS.html
"""
import io
import json
import logging
from typing import Optional, List, Dict, Any

import requests

log = logging.getLogger(__name__)


class HdfsClient:
    """WebHDFS 封装: write/read/append/exists/list/delete/upload"""

    def __init__(self, host: str = "localhost", port: int = 9870,
                 user: str = "hadoop", timeout: int = 10):
        self.base = f"http://{host}:{port}/webhdfs/v1"
        self.user = user
        self.timeout = timeout
        # trust_env=False: 绕过系统代理 (localhost HDFS 流量不走 127.0.0.1 代理)
        self.session = requests.Session()
        self.session.trust_env = False

    def _url(self, path: str, op: str, **params) -> str:
        p = {"op": op, "user.name": self.user}
        p.update(params)
        qs = "&".join(f"{k}={requests.utils.quote(str(v), safe='')}" for k, v in p.items())
        return f"{self.base}{path}?{qs}"

    def _alive(self) -> bool:
        try:
            return self.session.get(self._url("/", "GETFILESTATUS"), timeout=self.timeout).status_code < 500
        except requests.RequestException:
            return False

    # ---------- 基础操作 ----------

    def exists(self, path: str) -> bool:
        try:
            return self.session.get(self._url(path, "GETFILESTATUS"), timeout=self.timeout).status_code == 200
        except requests.RequestException:
            return False

    def is_dir(self, path: str) -> bool:
        try:
            r = self.session.get(self._url(path, "GETFILESTATUS"), timeout=self.timeout)
            return r.status_code == 200 and r.json().get("FileStatus", {}).get("type") == "DIRECTORY"
        except requests.RequestException:
            return False

    def mkdir(self, path: str) -> bool:
        """建目录 (含父目录, 已存在不报错)"""
        try:
            r = self.session.put(self._url(path, "MKDIRS", permission="755"), timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException as e:
            log.warning("HDFS mkdir %s 失败: %s", path, e)
            return False

    def write(self, path: str, content: str, overwrite: bool = True) -> bool:
        """两段式写文件: 重定向到 DataNode 后 PUT 数据"""
        try:
            op = "CREATE" if overwrite else "APPEND"
            r1 = self.session.put(self._url(path, op, overwrite="true" if overwrite else "false",
                                        noredirect="true"), timeout=self.timeout)
            if r1.status_code not in (200, 201):
                log.warning("HDFS write %s 第一段失败: %s %s", path, r1.status_code, r1.text[:200])
                return False
            dn_url = r1.json()["Location"]
            r2 = self.session.put(dn_url, data=content.encode("utf-8"),
                              headers={"Content-Type": "application/octet-stream"},
                              timeout=self.timeout)
            return r2.status_code in (200, 201)
        except requests.RequestException as e:
            log.warning("HDFS write %s 失败: %s", path, e)
            return False

    def append(self, path: str, content: str) -> bool:
        """追加分片 (文件不存在时自动建)。用于登录审计日志等追加场景"""
        if not self.exists(path):
            return self.write(path, content, overwrite=True)
        try:
            r1 = self.session.post(self._url(path, "APPEND", noredirect="true"), timeout=self.timeout)
            if r1.status_code != 200 or "Location" not in r1.json():
                return False
            dn_url = r1.json()["Location"]
            r2 = self.session.post(dn_url, data=content.encode("utf-8"),
                               headers={"Content-Type": "application/octet-stream"},
                               timeout=self.timeout)
            return r2.status_code in (200, 201)
        except requests.RequestException as e:
            log.warning("HDFS append %s 失败: %s", path, e)
            return False

    def read(self, path: str) -> Optional[str]:
        try:
            r1 = self.session.get(self._url(path, "OPEN", noredirect="true"), timeout=self.timeout)
            if r1.status_code != 200 or "Location" not in r1.json():
                return None
            r2 = self.session.get(r1.json()["Location"], timeout=self.timeout)
            return r2.text if r2.status_code == 200 else None
        except requests.RequestException as e:
            log.warning("HDFS read %s 失败: %s", path, e)
            return None

    def list(self, path: str) -> List[Dict[str, Any]]:
        """列目录, 返回 FileStatus 列表 (失败返回空列表)"""
        try:
            r = self.session.get(self._url(path, "LISTSTATUS"), timeout=self.timeout)
            if r.status_code != 200:
                return []
            return r.json().get("FileStatuses", {}).get("FileStatus", [])
        except requests.RequestException:
            return []

    def delete(self, path: str, recursive: bool = False) -> bool:
        try:
            r = self.session.delete(self._url(path, "DELETE", recursive="true" if recursive else "false"),
                                timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException as e:
            log.warning("HDFS delete %s 失败: %s", path, e)
            return False

    # ---------- 汇总信息 (dashboard 面板用) ----------

    def dir_stats(self, path: str = "/quant/data") -> Dict[str, Any]:
        """递归统计目录: 文件数/总字节/子目录名"""
        files = 0
        total = 0
        subdirs = []
        stack = [path]
        while stack:
            cur = stack.pop()
            for st in self.list(cur):
                full = cur.rstrip("/") + "/" + st["pathSuffix"]
                if st["type"] == "DIRECTORY":
                    subdirs.append(full)
                    stack.append(full)
                else:
                    files += 1
                    total += st["length"]
        return {"path": path, "alive": True, "files": files,
                "bytes": total, "subdirs": subdirs}


def get_client() -> HdfsClient:
    """从系统配置构造客户端"""
    from src.config import get_config
    bd = get_config().bigdata
    return HdfsClient(host=bd.hdfs_host, port=bd.hdfs_web_port, user=bd.hdfs_user)
