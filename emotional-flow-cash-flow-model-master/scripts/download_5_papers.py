"""
Download 5 ArXiv papers as PDF to D:\股票知识库\ArXiv论文\量化交易策略\
Papers:
1. E-TRENDS (2603.14453) -> E-TRENDS_Enhanced_LSTM_Trend_Forecasting.pdf
2. Overreaction as Momentum Indicator (2602.18912) -> Overreaction_as_Momentum_Indicator.pdf
3. Discovery of 13-Sharpe OOS Factor (2511.12490) -> 13-Sharpe_OOS_Factor_Drift_Regimes.pdf
4. Forecast-to-Fill (2511.08571) -> Forecast-to-Fill_Benchmark_Neutral_Alpha.pdf
5. Follow the Leader (2501.07135) -> Follow_the_Leader_Network_Momentum.pdf
"""
import urllib.request
import ssl
import os

papers = [
    ("2603.14453", "E-TRENDS_Enhanced_LSTM_Trend_Forecasting.pdf"),
    ("2602.18912", "Overreaction_as_Momentum_Indicator.pdf"),
    ("2511.12490", "13-Sharpe_OOS_Factor_Drift_Regimes.pdf"),
    ("2511.08571", "Forecast-to-Fill_Benchmark_Neutral_Alpha.pdf"),
    ("2501.07135", "Follow_the_Leader_Network_Momentum.pdf"),
]

base_url = "https://arxiv.org/pdf/"
dest_dir = r"D:\股票知识库\ArXiv论文\量化交易策略"

os.makedirs(dest_dir, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

for arxiv_id, filename in papers:
    filepath = os.path.join(dest_dir, filename)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
        print(f"[SKIP] {filename} already exists ({os.path.getsize(filepath)} bytes)")
        continue
    url = base_url + arxiv_id
    print(f"[DOWNLOADING] {arxiv_id} -> {filename}")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        resp = urllib.request.urlopen(req, context=ctx, timeout=60)
        data = resp.read()
        with open(filepath, "wb") as f:
            f.write(data)
        print(f"  [OK] {len(data)} bytes saved to {filepath}")
    except Exception as e:
        print(f"  [ERROR] {e}")

print("\nDone. Files in directory:")
for f in os.listdir(dest_dir):
    fp = os.path.join(dest_dir, f)
    if os.path.isfile(fp):
        print(f"  {f} ({os.path.getsize(fp)} bytes)")
