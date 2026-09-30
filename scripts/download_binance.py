"""Download Binance spot aggTrades daily archives from data.binance.vision and verify their SHA-256 checksums.

Usage: python scripts/download_binance.py BTCUSDT ETHUSDT --start 2026-09-23 --end 2026-09-29 [--out data/binance]
"""
import argparse
import datetime as dt
import hashlib
import pathlib
import urllib.request

BASE = "https://data.binance.vision/data/spot/daily/aggTrades"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(symbol, day, out):
    """Download one day's archive unless an intact copy is already there."""
    name = f"{symbol}-aggTrades-{day}.zip"
    url = f"{BASE}/{symbol}/{name}"
    path = out / name
    expected = urllib.request.urlopen(url + ".CHECKSUM", timeout=60).read().decode().split()[0]
    if path.exists() and sha256(path) == expected:
        return path, "already present"
    urllib.request.urlretrieve(url, path)
    if sha256(path) != expected:
        path.unlink()
        raise RuntimeError(f"checksum mismatch for {name}")
    return path, "downloaded, checksum OK"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("symbols", nargs="+", help="e.g. BTCUSDT ETHUSDT")
    ap.add_argument("--start", required=True, help="first day, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="last day, YYYY-MM-DD (inclusive)")
    ap.add_argument("--out", default="data/binance")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    for symbol in args.symbols:
        day = start
        while day <= end:
            path, status = fetch(symbol, day.isoformat(), out)
            print(f"{path.name}: {status} ({path.stat().st_size / 2**20:.1f} MB)")
            day += dt.timedelta(days=1)


if __name__ == "__main__":
    main()
