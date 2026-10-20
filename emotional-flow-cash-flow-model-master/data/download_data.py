# -*- coding: utf-8 -*-
"""
数据预下载脚本

功能: 从掘金量化平台下载所需历史数据到本地磁盘, 供后续离线使用。

使用方法:
    python data/download_data.py --start 2019-01-01 --end 2026-05-31
    python data/download_data.py --mock                    # 模拟模式, 仅测试流程

数据下载清单:
    1. 全A股日K线 (OHLCV)         -> D:/开发/训练数据/stock_daily/
    2. 沪深300指数日K线            -> D:/开发/训练数据/index_daily/
    3. 行业板块日K线               -> D:/开发/训练数据/sector_daily/
    4. 北向资金每日净买卖            -> D:/开发/训练数据/north_bound/
    5. 融资融券余额日度数据          -> D:/开发/训练数据/margin_trade/

依赖:
    - 掘金量化 SDK (pip install gm)
    - 有效的掘金 Token (从 D:/开发/quant_strategy/gm_config.json 读取)
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime, timedelta
from typing import List

import pandas as pd

# 掘金SDK需要手动导入
_GM_AVAILABLE = False
_GM_TOKEN_SET = False
gm = None

try:
    import gm.api as gm
    _GM_AVAILABLE = True
except Exception as e:
    print(f"[ERROR] 掘金SDK导入失败: {e}")
    print("[INFO] 请确认掘金终端已打开并登录, 然后重试")
    print("[INFO] 将退化为模拟模式")


# ============================================================
# 配置
# ============================================================

GM_CONFIG_PATH = "D:/开发/quant_strategy/gm_config.json"
DATA_ROOT = "D:/开发/训练数据"

# 各数据类型存储子目录
DATA_SUBDIRS = {
    'stock_daily': 'stock_daily',
    'index_daily': 'index_daily',
    'north_bound': 'north_bound',
    'margin_trade': 'margin_trade',
}

# 指数代码映射
INDEX_CODES = {
    '000300': 'SHSE.000300',   # 沪深300
    '000001': 'SHSE.000001',   # 上证指数
    '399001': 'SZSE.399001',   # 深证成指
}


def load_gm_token() -> str:
    """从 gm_config.json 加载掘金Token"""
    if os.path.exists(GM_CONFIG_PATH):
        with open(GM_CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        return cfg.get('token', '')
    return ''


def setup_gm() -> bool:
    """初始化掘金SDK, 返回是否成功"""
    global _GM_TOKEN_SET
    if not _GM_AVAILABLE:
        return False
    token = load_gm_token()
    if not token:
        print("[ERROR] 未找到掘金Token, 请检查 D:/开发/quant_strategy/gm_config.json")
        return False
    gm.set_token(token)
    _GM_TOKEN_SET = True
    print("[INFO] 掘金SDK初始化成功")
    return True


# ============================================================
# 辅助函数
# ============================================================

def _format_end_time(end_date: str) -> str:
    """掘金API end_time 格式: '2026-05-31 16:00:00'"""
    return f"{end_date} 16:00:00"


def _should_skip(output_dir: str, code: str, incremental: bool) -> bool:
    """增量模式: 如果文件已存在且非空, 跳过"""
    if not incremental:
        return False
    path = os.path.join(output_dir, f"{code}.csv")
    return os.path.exists(path) and os.path.getsize(path) > 100


# ============================================================
# 数据下载函数
# ============================================================

def download_stock_list() -> List[str]:
    """获取全A股股票列表

    通过掘金 gm.get_symbols() 获取, 过滤科创板/北交所。
    如掘金不可用, 返回本地 stock_daily/ 目录下已有文件列表。
    """
    if not _GM_AVAILABLE or not _GM_TOKEN_SET:
        # 降级: 扫描本地已下载的股票
        stock_dir = os.path.join(DATA_ROOT, 'stock_daily')
        if os.path.isdir(stock_dir):
            codes = []
            for f in os.listdir(stock_dir):
                if f.endswith('.csv') and not f.startswith('_'):
                    code = f.replace('.csv', '')
                    base = code.split('.')[-1] if '.' in code else code
                    if base.isdigit() and len(base) == 6:
                        if not (base.startswith('688') or base.startswith('689')
                                or base.startswith('4') or base.startswith('8')):
                            codes.append(code)
            if codes:
                print(f"[INFO] 从本地 stock_daily/ 找到 {len(codes)} 只股票")
                return sorted(codes)

        print("[MOCK] 返回示例股票列表 (掘金不可用且本地无数据)")
        return ['SHSE.600000', 'SHSE.600036', 'SZSE.000001']

    try:
        # 获取A股全部标的
        symbols = gm.get_symbols(sec_type1=1010)
        if symbols is None or len(symbols) == 0:
            # 回退: 逐市场获取
            all_syms = []
            for exchange in ['SHSE', 'SZSE']:
                syms = gm.get_symbols(exchange=exchange)
                if syms:
                    all_syms.extend(syms)
            symbols = all_syms

        if symbols is None or len(symbols) == 0:
            print("[ERROR] 掘金 get_symbols() 返回空, 请检查Token和网络")
            return []

        codes = []
        for s in symbols:
            if isinstance(s, dict):
                code = s.get('symbol', '')
                sec_type = str(s.get('sec_type', ''))
            else:
                code = str(s)
                sec_type = ''

            if not code:
                continue

            # 过滤: 只保留主板+创业板+科创板股票
            base = code.split('.')[-1] if '.' in code else code
            if not base.isdigit() or len(base) != 6:
                continue

            # 排除北交所(4/8开头), 保留科创板(688)供选项
            if base.startswith('4'):
                continue
            if base.startswith('8') and len(base) == 6 and not base.startswith('8'):
                continue

            codes.append(code)

        print(f"[INFO] 从掘金获取到 {len(codes)} 只A股 (已过滤北交所)")
        return sorted(codes)

    except Exception as e:
        print(f"[ERROR] 获取股票列表失败: {e}")
        return []


def download_stock_daily(symbols: List[str], start_date: str, end_date: str,
                         output_dir: str, incremental: bool = False):
    """下载个股日K线数据

    使用 gm.history() 批量下载, 每只股票存为一个CSV文件。
    支持断点续传 (incremental模式跳过已有文件)。
    """
    if not symbols:
        print("[WARN] 股票列表为空, 跳过下载")
        return

    os.makedirs(output_dir, exist_ok=True)
    start_time = f"{start_date} 00:00:00"
    end_time = f"{end_date} 16:00:00"
    total = len(symbols)
    success = 0
    skipped = 0
    failed = 0
    first_errors = []  # 记录前3个错误供诊断

    for i, code in enumerate(symbols):
        if _should_skip(output_dir, code, incremental):
            skipped += 1
            continue

        try:
            df = gm.history(
                symbol=code,
                frequency='1d',
                start_time=start_date,
                end_time=end_time,
                fields='eob,open,high,low,close,volume,amount',
                adjust=1,  # ADJUST_PREV
                df=True,
            )

            if df is None or len(df) == 0:
                failed += 1
                continue

            # 统一列名
            if 'eob' in df.columns:
                df['eob'] = pd.to_datetime(df['eob'])

            out_path = os.path.join(output_dir, f"{code}.csv")
            df.to_csv(out_path, index=False, encoding='utf-8-sig')
            success += 1

        except Exception as e:
            failed += 1
            if len(first_errors) < 3:
                first_errors.append(f"  [{code}] {e}")

        # 进度提示
        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"  进度: {i + 1}/{total}  success={success}  skipped={skipped}  failed={failed}")

        # 请求限速 (避免触发掘金频率限制)
        if (i + 1) % 50 == 0:
            time.sleep(0.5)

    print(f"  完成: success={success}  skipped={skipped}  failed={failed}")
    if first_errors:
        print("  前3个错误示例:")
        for err_msg in first_errors:
            print(err_msg)


def download_index_daily(start_date: str, end_date: str, output_dir: str):
    """下载指数日K线 (沪深300, 上证指数, 深证成指)"""
    if not _GM_AVAILABLE or not _GM_TOKEN_SET:
        print("[MOCK] 跳过指数下载 (掘金不可用)")
        return

    os.makedirs(output_dir, exist_ok=True)
    end_time = _format_end_time(end_date)

    for name, code in INDEX_CODES.items():
        out_path = os.path.join(output_dir, f"{name}.csv")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            print(f"  指数 {name} 已存在, 跳过")
            continue

        try:
            df = gm.history(
                symbol=code,
                frequency='1d',
                start_time=start_date,
                end_time=end_time,
                fields='eob,open,high,low,close,volume,amount',
                adjust=1,
                df=True,
            )

            if df is not None and len(df) > 0:
                if 'eob' in df.columns:
                    df['eob'] = pd.to_datetime(df['eob'])
                df.to_csv(out_path, index=False, encoding='utf-8-sig')
                print(f"  指数 {name} ({code}) 下载完成: {len(df)} 行")
            else:
                print(f"  指数 {name} ({code}) 无数据")
        except Exception as e:
            print(f"  指数 {name} ({code}) 下载失败: {e}")


def download_north_bound(start_date: str, end_date: str, output_dir: str):
    """下载北向资金数据

    使用 akshare 的 stock_hsgt_hist_em() 获取沪股通/深股通历史数据。
    如果 akshare 不可用, 尝试掘金 gm.history 获取。

    数据列: 日期, 当日成交净买额
    """
    os.makedirs(output_dir, exist_ok=True)

    # 尝试使用 akshare
    try:
        import akshare as ak

        for market, prefix in [("沪股通", "沪"), ("深股通", "深")]:
            out_path = os.path.join(output_dir, f"hsgt_{prefix}.csv")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
                print(f"  北向{prefix} 已存在, 跳过")
                continue

            try:
                df = ak.stock_hsgt_hist_em(symbol=market)
                if df is not None and len(df) > 0:
                    df = df.rename(columns={
                        '日期': '日期',
                        '当日成交净买额': '当日成交净买额',
                    })
                    # 过滤日期范围
                    if '日期' in df.columns:
                        df['日期'] = pd.to_datetime(df['日期'])
                        mask = (df['日期'] >= start_date) & (df['日期'] <= end_date)
                        df = df[mask]
                    df.to_csv(out_path, index=False, encoding='utf-8-sig')
                    print(f"  北向{prefix} 下载完成: {len(df)} 行 (akshare)")
            except Exception:
                print(f"  北向{prefix} akshare 下载失败, 将创建空模板")

                # 创建空模板 (列名与V2兼容)
                empty_df = pd.DataFrame(columns=['日期', '当日成交净买额'])
                empty_df.to_csv(out_path, index=False, encoding='utf-8-sig')

        return

    except ImportError:
        print("[INFO] akshare 未安装, 将创建北向资金空模板文件")

    # 降级: 创建空模板
    for prefix in ["沪", "深"]:
        out_path = os.path.join(output_dir, f"hsgt_{prefix}.csv")
        if not os.path.exists(out_path):
            empty_df = pd.DataFrame(columns=['日期', '当日成交净买额'])
            empty_df.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"  北向{prefix} 空模板已创建")

    print("[WARN] 北向资金数据需要手动填充或使用 akshare 自动下载")
    print("[INFO] 安装aksahre: pip install akshare")


def download_margin_trade(start_date: str, end_date: str, output_dir: str):
    """下载融资融券余额

    使用 akshare 的 stock_margin_sse() 获取上交所融资融券数据。
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "margin_sse.csv")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
        print("  融资融券数据已存在, 跳过")
        return

    try:
        import akshare as ak

        df = ak.stock_margin_sse(start_date=start_date.replace('-', ''),
                                end_date=end_date.replace('-', ''))
        if df is not None and len(df) > 0:
            # 适配V2列名
            col_map = {}
            for c in df.columns:
                if '信用交易日期' in str(c) or '日期' in str(c):
                    col_map[c] = '信用交易日期'
                elif '融资融券余额' in str(c) or '余额' in str(c):
                    col_map[c] = '融资融券余额'
                elif '融资余额' in str(c):
                    col_map[c] = '融资余额'
            if col_map:
                df = df.rename(columns=col_map)
            df.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"  融资融券 下载完成: {len(df)} 行 (akshare)")
            return

    except ImportError:
        print("[INFO] akshare 未安装")
    except Exception as e:
        print(f"  融资融券 akshare 下载失败: {e}")

    # 降级: 创建空模板
    empty_df = pd.DataFrame(columns=['信用交易日期', '融资融券余额', '融资余额'])
    empty_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print("  融资融券 空模板已创建")
    print("[WARN] 融资融券数据需要手动填充或使用 akshare 自动下载")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='A股数据预下载工具')
    parser.add_argument('--start', type=str, default='2019-01-01',
                        help='起始日期 (默认: 2019-01-01)')
    parser.add_argument('--end', type=str, default='2026-05-31',
                        help='截止日期 (默认: 2026-05-31)')
    parser.add_argument('--incremental', action='store_true',
                        help='增量更新模式 (跳过已存在的文件)')
    parser.add_argument('--data-types', type=str, nargs='+',
                        default=['stock_daily', 'index_daily'],
                        help='要下载的数据类型 (默认: stock_daily index_daily)')
    parser.add_argument('--mock', action='store_true',
                        help='模拟模式 (不实际下载, 仅测试流程)')

    args = parser.parse_args()

    # 初始化掘金
    if not args.mock:
        if not setup_gm():
            print("[ERROR] 掘金SDK未就绪, 无法下载数据")
            print("[INFO] 可以使用 --mock 参数测试下载流程")
            sys.exit(1)

    # 创建数据目录
    for subdir in DATA_SUBDIRS.values():
        os.makedirs(os.path.join(DATA_ROOT, subdir), exist_ok=True)

    print("=" * 60)
    print(f"  A股数据预下载工具")
    print(f"  区间: {args.start} ~ {args.end}")
    print(f"  类型: {args.data_types}")
    print(f"  模式: {'模拟' if args.mock else '真实下载'}")
    print("=" * 60)

    # 获取股票列表
    symbols = download_stock_list()
    if not symbols:
        print("[ERROR] 无法获取股票列表, 退出")
        sys.exit(1)
    print(f"[INFO] 共 {len(symbols)} 只股票待下载")

    # 按类型下载
    for data_type in args.data_types:
        output_dir = os.path.join(DATA_ROOT, DATA_SUBDIRS.get(data_type, data_type))
        print(f"\n[INFO] 正在下载: {data_type} -> {output_dir}")

        if args.mock:
            print(f"  [MOCK] 跳过实际下载 (模拟模式)")
            continue

        if data_type == 'stock_daily':
            download_stock_daily(symbols, args.start, args.end,
                                output_dir, args.incremental)
        elif data_type == 'index_daily':
            download_index_daily(args.start, args.end, output_dir)
        elif data_type == 'north_bound':
            download_north_bound(args.start, args.end, output_dir)
        elif data_type == 'margin_trade':
            download_margin_trade(args.start, args.end, output_dir)
        else:
            print(f"  [WARN] 未知数据类型: {data_type}")

    print("\n" + "=" * 60)
    print("  数据下载完成!")
    print(f"  数据目录: {DATA_ROOT}")
    print("=" * 60)


if __name__ == '__main__':
    main()
