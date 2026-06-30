"""Quick connectivity test for GDELT (run from YOUR machine, not the sandbox).

    .venv/Scripts/python.exe scripts/gdelt_check.py

If this prints articles, GDELT works from your network and we can build the
historical-news ingestion on it. If it prints 429 / errors, your IP is throttled too
and we'll scope the backtest to a recent window instead.
"""
import json
import urllib.request
import urllib.parse

base = "https://api.gdeltproject.org/api/v2/doc/doc"
params = {
    "query": "Tesco",
    "mode": "artlist",
    "maxrecords": "10",
    "format": "json",
    "startdatetime": "20230101000000",
    "enddatetime": "20230131000000",
    "sort": "datedesc",
}
url = base + "?" + urllib.parse.urlencode(params)
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 SentiFolio"})

try:
    raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    arts = json.loads(raw).get("articles", [])
    print(f"OK - GDELT works from your network. {len(arts)} Tesco articles (Jan 2023):")
    for a in arts[:5]:
        print("  ", a.get("seendate"), "|", a.get("domain"), "|", a.get("title", "")[:70])
except Exception as e:
    print("FAILED:", repr(e))
    print("If this is a 429, your IP is throttled too -> we go with the recent-window plan.")
