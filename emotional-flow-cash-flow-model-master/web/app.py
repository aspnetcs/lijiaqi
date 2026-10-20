# -*- coding: utf-8 -*-
"""情绪-资金流系统 Web 应用: 登录/注册/验证码 + 受保护的分析接口"""
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime

# 保证能 import 到 src 包 (uvicorn 从项目根启动时已有, 此处兜底)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from itsdangerous import URLSafeSerializer, BadSignature

from web import db
from web.auth import hash_password, verify_password, gen_captcha

app = FastAPI(title="情绪-资金流融合分析系统")
SECRET_KEY = "emotion-flow-secret-key-change-in-production"
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=3600)
_signer = URLSafeSerializer(SECRET_KEY)

# 模板与静态资源
_TEMPLATES = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# 延迟加载分析引擎 (首次受保护接口调用时初始化, 避免启动慢)
_ENGINES = {}


def _engines():
    if _ENGINES.get("ready"):
        return _ENGINES
    from src.config import get_config
    from src.data_reader import DataReader
    from src.sentiment_engine import SentimentEngine
    from src.capital_flow import CapitalFlowAnalyzer
    from src.fusion_engine import FusionEngine
    from src.triple_layer import MacroLayer

    cfg = get_config()
    reader = DataReader(cfg)
    sentiment = SentimentEngine(cfg)
    flow = CapitalFlowAnalyzer(cfg)
    fusion = FusionEngine(cfg)
    # 宏观层需要全历史北向/两融/指数
    today = datetime.now().strftime("%Y-%m-%d")
    north = reader.get_north_bound("2018-01-01", today)
    margin = reader.get_margin_trade("2018-01-01", today)
    index = reader.get_index("000300", "2018-01-01", today)
    macro = MacroLayer(north, margin, benchmark=index)

    _ENGINES.update(cfg=cfg, reader=reader, sentiment=sentiment,
                    flow=flow, fusion=fusion, macro=macro, ready=True)
    return _ENGINES


# ------------------------------------------------------------------
# 鉴权
# ------------------------------------------------------------------

def current_user(request: Request):
    # 注意: cookie 名不能叫 "session", 那是 SessionMiddleware 存验证码用的
    token = request.cookies.get("auth_token")
    if not token:
        return None
    try:
        payload = _signer.loads(token)
        return db.get_user_by_id(payload["uid"])
    except BadSignature:
        return None


def require_login(request: Request):
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return user


# ------------------------------------------------------------------
# 页面路由
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if current_user(request):
        return RedirectResponse("/dashboard", 303)
    return RedirectResponse("/login", 303)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return _TEMPLATES.TemplateResponse(request, "login.html")


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return _TEMPLATES.TemplateResponse(request, "register.html")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, user=Depends(require_login)):
    return _TEMPLATES.TemplateResponse(request, "dashboard.html",
                                       {"username": user["username"]})


# ------------------------------------------------------------------
# 验证码 (每次请求刷新, 存 session)
# ------------------------------------------------------------------

@app.get("/captcha")
def captcha(request: Request):
    code, buf = gen_captcha()
    request.session["captcha"] = code
    return StreamingResponse(buf, media_type="image/png")


def _check_captcha(request: Request, user_input: str) -> bool:
    expected = request.session.pop("captcha", None)
    if not expected or not user_input:
        return False
    return user_input.upper() == expected


# ------------------------------------------------------------------
# 注册 / 登录 / 登出
# ------------------------------------------------------------------

@app.post("/register")
def register(request: Request, username: str = Form(...), password: str = Form(...), captcha: str = Form("")):
    username = username.strip()
    if len(username) < 3 or len(password) < 6:
        return HTMLResponse("用户名至少3位, 密码至少6位", status_code=400)
    if not _check_captcha(request, captcha):
        return HTMLResponse("验证码错误", status_code=400)
    if db.get_user(username):
        return HTMLResponse("用户名已存在", status_code=400)
    db.create_user(username, hash_password(password))
    return RedirectResponse("/login?registered=1", 303)


# 简单失败计数 (内存, 重启清零)
_FAILS = {}


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...), captcha: str = Form("")):
    user = db.get_user(username.strip())
    if not user:
        return HTMLResponse("用户名或密码错误", status_code=401)

    # 失败锁定: 同一用户连续失败 5 次锁 5 分钟
    fail = _FAILS.get(username, {"count": 0, "locked_until": 0})
    if fail["locked_until"] > time.time():
        return HTMLResponse(f"登录失败过多, 请 {int(fail['locked_until']-time.time())} 秒后再试", status_code=429)

    if not _check_captcha(request, captcha):
        return HTMLResponse("验证码错误", status_code=400)
    if not verify_password(password, user["password_hash"]):
        fail["count"] += 1
        if fail["count"] >= 5:
            fail["locked_until"] = time.time() + 300
            fail["count"] = 0
        _FAILS[username] = fail
        return HTMLResponse("用户名或密码错误", status_code=401)

    db.update_last_login(user["id"])
    # 登录审计 -> HDFS (旁路, 失败不影响登录)
    try:
        from src.bigdata.login_audit import log_login_event
        log_login_event(user["username"],
                        ip=(request.client.host if request.client else ""),
                        success=True)
    except Exception:
        pass
    _FAILS.pop(username, None)
    token = _signer.dumps({"uid": user["id"], "username": user["username"]})
    resp = RedirectResponse("/dashboard", 303)
    resp.set_cookie("auth_token", token, httponly=True, samesite="lax", max_age=3600)
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", 303)
    resp.delete_cookie("auth_token")
    return resp


# ------------------------------------------------------------------
# 受保护的分析接口 (复用现有引擎)
# ------------------------------------------------------------------

def _jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


def _json(obj):
    import json
    def _default(o):
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        return str(o)
    return json.loads(json.dumps(obj, ensure_ascii=False, default=_default))


@app.get("/api/stocks")
def api_stocks(user=Depends(require_login)):
    return _json({"count": len(_engines()["reader"].get_stock_list())})


@app.post("/api/sentiment")
def api_sentiment(payload: dict, user=Depends(require_login)):
    date = payload.get("date")
    e = _engines()
    idx = e["reader"].get_index("000300", "2018-01-01", date)
    market = idx.loc[:date].tail(20) if idx is not None and len(idx) else None
    result = e["sentiment"].analyze_daily(date, [], market)
    return _json(result)


@app.post("/api/flow")
def api_flow(payload: dict, user=Depends(require_login)):
    import io as _io, contextlib
    symbol = payload["symbol"]
    date = payload["date"]
    start = (datetime.strptime(date, "%Y-%m-%d") - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    df = _engines()["reader"].get_merged_data(symbol, start, date)
    if df is None or len(df) < 60:
        raise HTTPException(400, "数据不足60个交易日")
    with contextlib.redirect_stdout(_io.StringIO()):
        result = _engines()["flow"].analyze_stock(df, date=date)
    return _json(result)


@app.post("/api/macro")
def api_macro(payload: dict, user=Depends(require_login)):
    import contextlib, io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        sig = _engines()["macro"].compute_signal(
            pd.Timestamp(payload["date"]), sentiment_score=payload.get("sentiment_score", 0.0))
    return _json(sig)


@app.post("/api/fuse")
def api_fuse(payload: dict, user=Depends(require_login)):
    result = _engines()["fusion"].fuse(
        sentiment=payload["sentiment"],
        flow=payload["flow"],
        macro_position=payload["macro_position"],
        date=payload.get("date", ""))
    return _json(result)


@app.post("/api/backtest")
def api_backtest(payload: dict, user=Depends(require_login)):
    import contextlib, io as _io, json as _json_mod
    import main as backtest_main
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        backtest_main.run_backtest(payload["start_date"], payload["end_date"], resume=False)
    stats_path = os.path.join(_engines()["cfg"].data.output_dir, "data", "performance_stats.json")
    stats = _json_mod.load(open(stats_path, "r", encoding="utf-8")) if os.path.exists(stats_path) else None
    return {"stats": stats, "log_tail": buf.getvalue()[-2000:]}


@app.get("/api/stats")
def api_stats(user=Depends(require_login)):
    import json as _json_mod
    cfg = _engines()["cfg"]
    stats_path = os.path.join(cfg.data.output_dir, "data", "performance_stats.json")
    if not os.path.exists(stats_path):
        return {"error": "暂无回测结果"}
    return _json_mod.load(open(stats_path, "r", encoding="utf-8"))


# ------------------------------------------------------------------
# 大数据平台接口 (HDFS + Spark)
# ------------------------------------------------------------------

def _hdfs_json(client, path: str):
    """读取 HDFS 上的 JSON 汇总文件, 失败返回 None"""
    import json as _json_mod
    raw = client.read(path)
    if not raw:
        return None
    try:
        return _json_mod.loads(raw)
    except Exception:
        return None


@app.get("/api/bigdata/summary")
def api_bigdata_summary(user=Depends(require_login)):
    from src.bigdata.hdfs_client import get_client
    cfg = _engines()["cfg"].bigdata
    client = get_client()
    if not client._alive():
        return {"alive": False,
                "hint": "HDFS 未启动, 请运行 D:/hadoop-3.3.6/sbin/start-dfs.cmd"}
    data_stats = client.dir_stats(cfg.hdfs_data_dir)
    factors = _hdfs_json(client, f"{cfg.hdfs_results_dir}/factors/_summary.json")
    quality = _hdfs_json(client, f"{cfg.hdfs_results_dir}/quality/_summary.json")
    return {"alive": True, "data": data_stats,
            "factors": factors, "quality": quality}


@app.get("/api/bigdata/logs")
def api_bigdata_logs(user=Depends(require_login)):
    from src.bigdata.hdfs_client import get_client
    cfg = _engines()["cfg"].bigdata
    client = get_client()
    if not client._alive():
        return {"alive": False, "hint": "HDFS 未启动"}
    files = [{"file": st["pathSuffix"], "bytes": st["length"]}
             for st in client.list(cfg.hdfs_login_log_dir)
             if st["type"] == "FILE"]
    stats = _hdfs_json(client, f"{cfg.hdfs_results_dir}/login_stats/_summary.json")
    return {"alive": True, "log_files": files, "stats": stats}


# 启动时建表
db.init_db()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
